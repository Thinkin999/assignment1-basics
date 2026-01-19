import json
from typing import Any, List, Tuple, Optional, Dict
import regex as re
from collections import Counter, defaultdict
import os
import json
def train_bpe(
    input_path: str, 
    vocab_size: int=10000, 
    special_tokens: List[str]=["<|endoftext|>"]
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:

    #-----vocab initialization-----
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    
    #-----read text-----
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    #-----pattern for matching words-----
    gpt2_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    #TODO:在这里cs336推荐的方式，先根据endoftext，使用re.split把text切割为多个分块对每个分块进行处理
    if special_tokens:
        special_pat = "|".join(re.escape(t) for t in special_tokens)
        full_pat = f"{special_pat}|{gpt2_pat}"
    else:
        full_pat = gpt2_pat
    full_pat = re.compile(full_pat)

    raw_counts = Counter()
    for match in full_pat.finditer(text):
        word = match.group()
        raw_counts[word] += 1

    bytes_list = []
    bytes_counts = []
    for word, count in raw_counts.items():
        if word not in special_tokens:
            bytes_from_word = [bytes([b]) for b in word.encode("utf-8")]
            bytes_list.append(bytes_from_word)
            bytes_counts.append(count)

    #-----bytes_pairs initialization-----
    bytes_pairs = defaultdict[tuple, int](int)#把这一部分变成最大堆
    bytes_pairs_indices = defaultdict(set)
    for i, word in enumerate(bytes_list):
        for j in range(len(word) - 1):
            bytes_pairs[(word[j], word[j + 1])] += bytes_counts[i]
            bytes_pairs_indices[(word[j], word[j + 1])].add(i)
    #进行正式的合并
    num_merges = vocab_size - 256 - len(special_tokens)
    for _ in range(num_merges):
        if not bytes_pairs:
            break
        best_pair = max(bytes_pairs.items(), key=lambda x: (x[1], x[0]))[0]#在这里是不是可以使用最大堆
        if bytes_pairs[best_pair] <= 0:
            break
        merges.append(best_pair)
        bytes_pairs[best_pair] = 0
        related_indices = bytes_pairs_indices[best_pair]
        for index in list(related_indices):
            cur_word = bytes_list[index]
            new_word = []
            j = 0
            while j < len(cur_word):
                if j < len(cur_word) - 1 and (cur_word[j], cur_word[j + 1]) == best_pair:
                    new_bytes = cur_word[j] + cur_word[j + 1]
                    new_word.append(new_bytes)
                    if j > 0:
                        old_prev_pair = (cur_word[j - 1], cur_word[j])
                        bytes_pairs[old_prev_pair] -= bytes_counts[i]
                        bytes_pairs_indices[old_prev_pair].discard(index)

                        new_prev_pair = (cur_word[j - 1], new_bytes)
                        bytes_pairs[new_prev_pair] += bytes_counts[i]
                        bytes_pairs_indices[new_prev_pair].add(index)
                    if j < len(cur_word) - 2:
                        old_next_pair = (cur_word[j + 1], cur_word[j + 2])
                        bytes_pairs[old_next_pair] -= bytes_counts[i]
                        bytes_pairs_indices[old_next_pair].discard(index)

                        new_next_pair = (new_bytes, cur_word[j + 2])
                        bytes_pairs[new_next_pair] += bytes_counts[i]
                        bytes_pairs_indices[new_next_pair].add(index)
                    j += 2
                else:
                    new_word.append(cur_word[j])
                    j += 1
            bytes_list[i] = new_word

    for bytes_pair in merges:
        vocab[len(vocab)] = bytes_pair[0] + bytes_pair[1]
    for t in special_tokens:
        vocab[len(vocab)]  = t.encode("utf-8")   
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
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}      

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
    for idx, b_seq in vocab.items():
        # 将每个字节序列转为对应的 Unicode 字符串
        # 比如 b'a\xff' -> 'a' + byte_encoder[255]
        token_str = "".join(byte_encoder[b] for b in b_seq)
        token_to_id[token_str] = idx

    with open(os.path.join(save_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(token_to_id, f, indent=4, ensure_ascii=False)

    # 2. 转换并保存 merges.txt
    with open(os.path.join(save_dir, "merges.txt"), "w", encoding="utf-8") as f:
        # 写入一个版本标记（可选）
        for pair in merges:
            # pair 是 (bytes, bytes)，也要转换成映射后的字符串
            p0 = "".join(byte_encoder[b] for b in pair[0])
            p1 = "".join(byte_encoder[b] for b in pair[1])
            f.write(f"{p0} {p1}\n")

    print(f"Tokenizer 已保存至: {save_dir}")
    
def main():
    input_path = "/home/pkuhetu/lqs/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt"
    train_bpe(input_path)

if __name__ == "__main__":
    main()