from re import Pattern
from typing import Any, List, Tuple, Optional, Dict
import regex as re
from collections import Counter, defaultdict

def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: List[str]
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]
    """
    Used to train a tokenizer.
    Args:

    Return:
            vocab:dict[int, bytes],用来从token转换为字节 进行decode的操作
            merges:List[Tuple[bytes, bytes]] :合并的规则，但是这样去tokenize的话也太慢了吧
            看起来也是先转换为字节流，然后再进行合并，按照顺序进行排列
    """
    #-----vocab initialization-----
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    num_merges = vocab_size - 256 - len(special_tokens)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    gpt2_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    if special_tokens:
        special_pat = "|".join(re.escape(t) for t in special_tokens)
        full_pat = f"{special_pat}|{gpt2_pat}"
    else:
        full_pat = gpt2_pat
    full_pat = re.compile(full_pat)

    raw_counts = Counter()#记录切分出来的word和数量的对应关系
    bytes_list = []#存储bytes形式的word
    bytes_counts = []#存储对应bytes形式的word的数量
    bytes_pairs = defaultdict[tuple, int](int)#存储bytes pair及其对应的数量，这里是不是可以变成最大堆，始终把最大的数目放在上面，然后再拿走
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
        best_pair = max(bytes_pairs.item(), key=lambda x: (x[1], x[0]))[0]
        if bytes_pairs[best_pair] <= 0:
            break
        merges.append(best_pair)
        bytes_pairs[best_pair] = 0
        
        related_indices = bytes_pairs_indices[best_pair]
        for i in range(len(related_indices)):
            cur_word = bytes_list[i]
            new_word = []
            occur_list = []
            for j in range(len(cur_word) - 1):
                if cur_word[j] == best_pair[0] and cur_word[j + 1] == best_pair[1]:
                    if j >= 1:#左边还有
                        prev_pair = (cur_word[j - 1], cur_word[j])
                        bytes_pairs[prev_pair] -= bytes_counts[i]
                    if j + 2 <= len(cur_word) - 1:
                        next_pair = (cur_word[j + 1], cur_word[j + 2])
                        bytes_pairs[next_pair] -= bytes_counts[i]

            #前面减少，后面减少

        
    #---3.合并计算---
    #我们现在要进行统计计算了，那么应该怎么做呢，
    #遍历word count的每一个key value对，对每一个key遍历所有的bytes pair 然后进行加和
    #然后记录这些数量然后求一个max，求完一个max之后找到那些有这
   

    #3. 进行计算 找到那个最多的？那我们要怎么知道，那个最多的在哪里呢，还是要再走一遍，
    #       

    #然后进行合并，遍历里面的每一个然后进行模式的收集但是可以找到那个最大的，但是要怎么进行合并呢

    #我想检验一下，这个正则表达式会怎么进行划分
    return NotImplementedError