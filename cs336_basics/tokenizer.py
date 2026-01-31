from typing import Iterable, Iterator
from .train_bpe import get_encoder_dict
import regex as re
import json
class Tokenizer:
    """
    建立清晰的逻辑
    1.文本正则化切割 str -> list[str]
    2.str-> list[bytes]
    3.merge
    4.list[bytes] -> vocab

    想清楚不同的层级
    merge iterator id映射表示层
    """
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None):

        self.vocab = vocab
        self.bytes_to_id = {v: k for k, v in vocab.items()}

        self.merges = merges
        self.special_tokens = special_tokens or []
      
        self.gpt_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: list[str] | None = None):
       
        special_tokens = special_tokens or []
        bytes_encoder = get_encoder_dict()#int -> str
        bytes_decoder = {v: k for k, v in bytes_encoder.items()}

        with open(vocab_filepath, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
        vocab = {v: bytes([bytes_decoder[s] for s in k]) 
            if k not in special_tokens else k.encode("utf-8") 
            for k, v in token_to_id.items()}

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()

                p0 = bytes([bytes_decoder[s] for s in parts[0]])
                p1 = bytes([bytes_decoder[s] for s in parts[1]])

                merges.append((p0, p1))

        return cls(vocab, merges, special_tokens)

    def encode(
        self, 
        text: str) -> list[int]:
    
        if self.special_tokens:
            sp_pat = "|".join(re.escape(t) for t in self.special_tokens)
            full_pat = sp_pat + "|" + self.gpt_pat
            full_pat = re.compile(full_pat)
        else:
            full_pat = re.compile(self.gpt_pat)

        pieces = re.findall(full_pat, text)
        #list[str] -> list[list[bytes]]
        sp_token_utf8 = [t.encode("utf-8") for t in self.special_tokens]
        words_list = []
        for p in pieces:
            p_bytes = p.encode("utf-8")
            bytes_from_word = [p_bytes] if p_bytes in sp_token_utf8 else [bytes([p]) for p in p_bytes]
            words_list.append(bytes_from_word)
      
        for merge in self.merges:
            for i in range(len(words_list)):
                cur = words_list[i]
                if cur[0] in sp_token_utf8:
                    words_list[i] = cur
                    continue
                new = []
                l = len(cur)
                j = 0
                while j < l:
                    if j < l - 1 and (cur[j], cur[j + 1]) == merge:
                        new.append(cur[j] + cur[j + 1])
                        j += 2
                    else:
                        new.append(cur[j])
                        j += 1
                words_list[i] = new
     
        res = []
        for bytes_list in words_list:
            for b in bytes_list:
                res.append(self.bytes_to_id[b])
        return res          

    def encode_iterable(
        self, 
        iterable: Iterable[str]) -> Iterator[int]:
       
        if self.special_tokens:
            sp_pat = "|".join(re.escape(t) for t in self.special_tokens)
            full_pat = sp_pat + "|" + self.gpt_pat
            full_pat = re.compile(full_pat)
        else:
            full_pat = re.compile(self.gpt_pat)
        sp_token_utf8 = [t.encode("utf-8") for t in self.special_tokens]

        token_ids = []

        for text in iterable:
            words = full_pat.finditer(text)
        
            for match in words:
                word = match.group(
                )
                word_bytes = word.encode("utf-8")
                if word_bytes in sp_token_utf8:
                    yield self.bytes_to_id[word_bytes]
                    continue
                       
                bytes_list = [bytes([b]) for b in word_bytes]

                for merge in self.merges:
                    new = []
                    l = len(bytes_list)
                    j = 0
                    while j < l:
                        if j < l - 1 and (bytes_list[j], bytes_list[j + 1]) == merge:
                            new.append(bytes_list[j] + bytes_list[j + 1])
                            j += 2
                        else:
                            new.append(bytes_list[j])#这里存在可以优化的逻辑
                            j += 1
                    for b in bytes_list:
                        yield self.bytes_to_id[b]
        


    def decode(self, ids: list[int]) -> str:
       
        bytes_list = [self.vocab[id] for id in ids]
        full_bytes = b''.join(bytes_list)
        return full_bytes.decode("utf-8", errors="replace")

