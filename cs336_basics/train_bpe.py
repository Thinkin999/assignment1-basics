import json
import os
import argparse
from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import Dict, List, Tuple

import regex as re

_GPT2_PRETOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _mp_count_pretokens_worker(args: tuple[str, int, int, List[str]]) -> Counter[str]:
    """
    multiprocessing worker（必须是模块顶层函数，才能被 pickle）。
    进程内独立打开文件、读取 [start, end) 字节，并在 chunk 内按 special tokens split 后做预分词计数。
    """
    import regex as _re

    path, start, end, special_tokens = args
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    pat = _re.compile(_GPT2_PRETOKENIZE_PATTERN)
    c: Counter[str] = Counter()

    if special_tokens:
        split_pat = "|".join(_re.escape(t) for t in special_tokens)
        pieces = _re.split(split_pat, chunk)
    else:
        pieces = [chunk]

    for piece in pieces:
        if not piece:
            continue
        for m in pat.finditer(piece):
            c[m.group()] += 1
    return c

def train_bpe(
    input_path: str, 
    vocab_size: int=10000, 
    special_tokens: List[str]=["<|endoftext|>"],
    num_processes: int = 1,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:

    # 1. 初始化基础词表 (0-255)
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    
    gpt2_pat = _GPT2_PRETOKENIZE_PATTERN
    full_pat = re.compile(gpt2_pat)

    def _count_pretokens_in_text(text_chunk: str) -> Counter[str]:
        """对单个文本 chunk 做预分词计数（chunk 内按 special tokens 再 split，避免跨边界相邻）。"""
        c: Counter[str] = Counter()
        if special_tokens:
            split_pat = "|".join(re.escape(t) for t in special_tokens)
            pieces = re.split(split_pat, text_chunk)
        else:
            pieces = [text_chunk]
        for piece in pieces:
            if not piece:
                continue
            for m in full_pat.finditer(piece):
                c[m.group()] += 1
        return c

    def _find_chunk_boundaries_multi(
        file, desired_num_chunks: int, split_tokens: List[bytes]
    ) -> List[int]:
        """
        在二进制文件中为多进程切块找边界，尽量对齐到任一 special token 的起始位置，
        避免把 token 切断导致跨段统计错误。
        """
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        if desired_num_chunks <= 1:
            return [0, file_size]

        chunk_size = file_size // desired_num_chunks
        boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        boundaries[-1] = file_size

        if not split_tokens:
            return sorted(set(boundaries))

        max_len = max(len(t) for t in split_tokens)
        mini_chunk_size = 4096

        for bi in range(1, len(boundaries) - 1):
            guess = boundaries[bi]
            # 向后回退一些字节，避免 guess 落在 token 中间导致 token 被切断
            initial_position = max(0, guess - (max_len - 1))
            file.seek(initial_position)
            pos = initial_position
            while True:
                buf = file.read(mini_chunk_size)
                if buf == b"":
                    boundaries[bi] = file_size
                    break

                # 找到本 buf 内“最早出现”的任一 token
                found_positions = [buf.find(t) for t in split_tokens]
                found_positions = [p for p in found_positions if p != -1]
                if found_positions:
                    boundaries[bi] = pos + min(found_positions)
                    break

                pos += mini_chunk_size

        return sorted(set(boundaries))

    # --- 计算 raw_counts（可选并行） ---
    raw_counts: Counter[str] = Counter()
    if num_processes and num_processes > 1:
        split_tokens_bytes = [t.encode("utf-8") for t in special_tokens] if special_tokens else []

        with open(input_path, "rb") as f:
            boundaries = _find_chunk_boundaries_multi(f, num_processes, split_tokens_bytes)

        tasks = [(input_path, s, e, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:]) if e > s]
        procs = max(1, min(int(num_processes), len(tasks)))
        with Pool(processes=procs) as pool:
            for partial in pool.imap_unordered(_mp_count_pretokens_worker, tasks, chunksize=1):
                raw_counts.update(partial)
    else:
        # 串行：读取全文件，再按 special token 切段后预分词计数
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        raw_counts = _count_pretokens_in_text(text)

    # 处理 words
    bytes_list = []
    bytes_counts = []
    for word, count in raw_counts.items():
        # 将单词切分为字节列表
        bytes_from_word = [bytes([b]) for b in word.encode("utf-8")]
        bytes_list.append(bytes_from_word)
        bytes_counts.append(count)

    # 统计 Pair 频率（注意：indices 只能表示“某个 word 是否包含该 pair”，不能表达出现次数；
    # 因此在更新时必须以“整段移除该 word 对所有 pairs 的贡献 -> merge -> 整段加回贡献”的方式维护）
    bytes_pairs: Counter[Tuple[bytes, bytes]] = Counter()
    bytes_pairs_indices: defaultdict[Tuple[bytes, bytes], set[int]] = defaultdict(set)

    def _pairs_in_word(tokens: List[bytes]) -> List[Tuple[bytes, bytes]]:
        return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]

    for i, word in enumerate(bytes_list):
        count = bytes_counts[i]
        pairs = _pairs_in_word(word)
        for pair in pairs:
            bytes_pairs[pair] += count
        for pair in set(pairs):
            bytes_pairs_indices[pair].add(i)

    # 计算需要合并的次数
    num_merges = vocab_size - 256 - len(special_tokens)
    
    for _ in range(num_merges):
        if not bytes_pairs:
            break
        
        # 选择要 merge 的 pair：
        # - 频率最高优先
        # - 频率相同按 pair 的字典序（tuple[bytes, bytes]）取更大者
        # 该 tie-break 与 reference merges 对齐（例如在 index=31 的平局点）。
        best_pair = max(
            (p for p, c in bytes_pairs.items() if c > 0),
            key=lambda p: (bytes_pairs[p], p),
            default=None,
        )
        
        if best_pair is None or bytes_pairs[best_pair] <= 0:
            break

        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        # 找到所有包含 best_pair 的 words（copy 一份，避免我们更新 indices 时迭代集合出问题）
        related_indices = list(bytes_pairs_indices.get(best_pair, set()))

        # 对每个受影响的 word：先移除旧贡献 -> merge -> 加回新贡献
        for index in related_indices:
            cur_word = bytes_list[index]
            count = bytes_counts[index]

            old_pairs = _pairs_in_word(cur_word)
            for p in old_pairs:
                bytes_pairs[p] -= count
            for p in set(old_pairs):
                bytes_pairs_indices[p].discard(index)

            # merge（左到右，非重叠）
            new_word: List[bytes] = []
            j = 0
            while j < len(cur_word):
                if j < len(cur_word) - 1 and (cur_word[j], cur_word[j + 1]) == best_pair:
                    new_word.append(cur_word[j] + cur_word[j + 1])
                    j += 2
                else:
                    new_word.append(cur_word[j])
                    j += 1

            bytes_list[index] = new_word

            new_pairs = _pairs_in_word(new_word)
            for p in new_pairs:
                bytes_pairs[p] += count
            for p in set(new_pairs):
                bytes_pairs_indices[p].add(index)

        # best_pair 已经被所有相关 word 的 old_pairs 扣掉，并且 merge 后不会再出现
        # 为了稳妥起见，防止出现极小的负数或残留，我们直接把它清零并移除 indices
        bytes_pairs[best_pair] = 0
        bytes_pairs_indices[best_pair].clear()

    # 最后添加 special tokens 到词表
    for t in special_tokens:
        vocab[len(vocab)] = t.encode("utf-8")

    return vocab, merges
def get_encoder_dict():
    """
    创建一个从字节到 Unicode 字符的映射。
    参考 GPT-2 官方实现，将不可见字节映射到更高位的 Unicode 区块。
    """
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    
    cs = bs[:] # 拷贝一份，作为对应的字符编码
    
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)#也就是不可见的
            cs.append(256 + n)#bs里面刚开始是可见的，然后把那些256以内的不可见的转化成可见的
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)} #变成int -> 可见字符的映射

def save_tokenizer(vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], save_dir: str):
    """
    保存分词器模型到指定目录。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    byte_encoder = get_encoder_dict()

    # 1. 转换并保存 vocab.json
    # 我们的 vocab 目前是 {id: bytes}，需要转成 {string: id}
    # 因为 JSON 的 key 必须是字符串
    token_to_id = {}
    for idx, b_seq in vocab.items():#str->int
        # 将每个字节序列转为对应的 Unicode 字符串
        # 比如 b'a\xff' -> 'a' + byte_encoder[255]
        token_str = "".join(byte_encoder[b] for b in b_seq)
        token_to_id[token_str] = idx

    with open(os.path.join(save_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(token_to_id, f, indent=4, ensure_ascii=False)

    # 2. 转换并保存 merges.txt
    with open(os.path.join(save_dir, "merges.txt"), "w", encoding="utf-8") as f:
        # 写入一个版本标记（可选）
        for pair in merges:#list(tupe(bytes,bytes))
            # pair 是 (bytes, bytes)，也要转换成映射后的字符串
            p0 = "".join(byte_encoder[b] for b in pair[0])
            p1 = "".join(byte_encoder[b] for b in pair[1])
            f.write(f"{p0} {p1}\n")

    print(f"Tokenizer 已保存至: {save_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer and save vocab/merges.")
    parser.add_argument(
        "input_path",
        type=str,
        nargs="?",
        default="/home/pkuhetu/lqs/cs336_data/owt_valid.txt",
        help="训练语料路径（文本文件，utf-8）。不传则使用默认路径。",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="词表大小（包含 256 bytes + special tokens）。默认 10000。",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="OpenWebText",
        help="输出目录，写入 vocab.json 与 merges.txt。默认 bpe_tokenizer_out。",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=32,
        help="预分词计数使用的进程数（>1 启用多进程）。默认 1。",
    )
    parser.add_argument(
        "--special_tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help="special tokens 列表（用空格分隔）。默认：<|endoftext|>。",
    )
    parser.add_argument(
        "--no_special_tokens",
        action="store_true",
        help="禁用 special tokens（等价于 special_tokens=[]）。",
    )

    args = parser.parse_args()
    special_tokens = [] if args.no_special_tokens else list(args.special_tokens)

    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        num_processes=args.num_processes,
    )
    save_tokenizer(vocab, merges, args.save_dir)

if __name__ == "__main__":
    main()