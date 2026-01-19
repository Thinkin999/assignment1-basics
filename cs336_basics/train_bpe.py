from re import Pattern
from typing import Any, List, Tuple, Optional, Dict
import regex as re
from collections import Counter, defaultdict

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
        

def main():
    input_path = "/home/pkuhetu/lqs/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt"
    train_bpe(input_path)

if __name__ == "__main__":
    main()