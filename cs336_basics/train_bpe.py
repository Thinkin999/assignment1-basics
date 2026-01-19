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

    num_merges = vocab_size - 256 - len(special_tokens)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    print("text is", text)

    gpt2_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    if special_tokens:
        special_pat = "|".join(re.escape(t) for t in special_tokens)
        full_pat = f"{special_pat}|{gpt2_pat}"
    else:
        full_pat = gpt2_pat
    full_pat = re.compile(full_pat)

    raw_counts = Counter()
    bytes_list = []
    bytes_counts = []
    bytes_pairs = defaultdict[tuple, int](int)#把这一部分变成最大堆
    bytes_pairs_indices = defaultdict(set)
    for match in full_pat.finditer(text):
        word = match.group()
        raw_counts[word] += 1
    for word, count in raw_counts.items():
        if word not in special_tokens:
            bytes_from_word = [bytes([b]) for b in word.encode("utf-8")]
            bytes_list.append(bytes_from_word)
            bytes_counts.append(count)
    for i, word in enumerate(bytes_list):
        for j in range(len(word) - 1):
            bytes_pairs[(word[j], word[j + 1])] += bytes_counts[i]
            bytes_pairs_indices[(word[j], word[j + 1])].add(i)
    for _ in range(num_merges):
        if not bytes_pairs:
            break
        best_pair = max(bytes_pairs.items(), key=lambda x: (x[1], x[0]))[0]
        if bytes_pairs[best_pair] <= 0:
            break
        merges.append(best_pair)
        bytes_pairs[best_pair] = 0
        related_indices = bytes_pairs_indices[best_pair]
        for i in range(len(related_indices)):
            cur_word = bytes_list[i]
            new_word = list(cur_word)

            for j in range(len(cur_word) - 1):
                if cur_word[j] == best_pair[0] and cur_word[j + 1] == best_pair[1]:
                    new_word[j] = new_word[j] + new_word[j + 1]
                    del new_word[j + 1]
                    if j >= 1:
                        prev_pair = (cur_word[j - 1], cur_word[j])
                        bytes_pairs[prev_pair] -= bytes_counts[i]
                    if j + 2 <= len(cur_word) - 1:
                        next_pair = (cur_word[j + 1], cur_word[j + 2])
                        bytes_pairs[next_pair] -= bytes_counts[i]
            bytes_list[i] = new_word
    for bytes_pair in merges:
        vocab[len(vocab)] = bytes_pair[0] + bytes_pair[1]
    for t in special_tokens:
        vocab[len(vocab)]  = t.encode("utf-8")   
    return vocab, merges
        

def main():
    input_path = "/home/pkuhetu/lqs/assignment1-basics/cs336_basics/simple_test.txt"
    train_bpe(input_path)

if __name__ == "__main__":
    main()