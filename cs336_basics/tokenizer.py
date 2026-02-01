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
    text (str)
    ↓
    pieces (list[str])
    ↓
    token-level bytes units (list[bytes])
    ↓
    token ids (list[int])
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
        self.special_tokens_utf8 = {t.encode("utf-8") for t in self.special_tokens}#转化成set
      
        self.gpt_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        


    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: list[str] | None = None):
       
        special_tokens = special_tokens or []
        bytes_encoder = get_encoder_dict()#int -> str
        bytes_decoder = {v: k for k, v in bytes_encoder.items()}#str -> int

        with open(vocab_filepath, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
        vocab = {v: bytes([bytes_decoder[s] for s in k]) #id -> bytes
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

        return cls(vocab, merges, special_tokens)#写的很棒 没有问题 
        #相当于，从str和id的关系，到了bytes和id的关系

    def _split_to_pieces(self, text: str) -> list[str]:
        """
        Split raw text into regex-level pieces.
        BPE must not cross piece boundaries.

        1. 先按 special tokens 切分（不丢失 special token）
        2. 普通文本再用 GPT regex 切分
        """

        # 1️⃣ special token：长度倒序，防止前缀吞噬
        specials = sorted(self.special_tokens, key=len, reverse=True)

        # 2️⃣ GPT 正文 regex：必须是“无捕获组 / 全非捕获组”
        gpt_pat = self.gpt_pat
        if isinstance(gpt_pat, str):
            gpt_pat = re.compile(gpt_pat)

        pieces: list[str] = []

        # ✅ 关键：special_tokens 为空时，不能构造 "()" 这种会匹配空串的 regex。
        # `re.split("()", text)` 会在每个字符边界产生切分，导致 piece 被拆成单字符，从而 token 数量暴涨。
        if not specials:
            for m in gpt_pat.finditer(text):
                pieces.append(m.group(0))
            return pieces

        # 3️⃣ 构造 special token 的 regex（必须是捕获组，split 才会保留）
        special_pattern = "(" + "|".join(re.escape(t) for t in specials) + ")"
        special_pat = re.compile(special_pattern)

        # 4️⃣ 第一阶段：按 special token 切 chunk
        for part in special_pat.split(text):
            if not part:
                continue

            if part in self.special_tokens:
                # special token 是 tokenizer 的“结构边界”
                pieces.append(part)
            else:
                # 5️⃣ 普通文本：用 finditer 逐段扫描
                for m in gpt_pat.finditer(part):
                    pieces.append(m.group(0))

        return pieces

    def _bpe(self, 
        bytes_list: list[bytes]) -> list[bytes]:
        """
        对于每一个bytes_list,我们遍历每一个merge规则，但是实际上，可以维护成一个heap，每次合并优先级最高的rank，
        然后修改这个rank
        """
        cur = list(bytes_list)#防御性写法，不要状态共享，防止别人修改你的代码
        for merge in self.merges:
            i = 0
            new = []
            l = len(cur)
            if l < 2: #表达算法不变量
                break
            while i < l:
                if i < l - 1 and (cur[i], cur[i + 1]) == merge:
                    new.append(cur[i] + cur[i + 1])
                    i += 2
                else:
                    new.append(cur[i])
                    i += 1
            cur = new
        return cur

    def _encode_piece(self, piece: str) -> list[bytes]:
        """
        Encode a single regex piece into token-level byte units.
        piece → bytes units → vocab id 我们注释这个是为了确定好数据的流向
        """
        piece_utf8 = piece.encode("utf-8")
        if piece_utf8 in self.special_tokens_utf8:
            return [piece_utf8]
        else:
            bytes_list = [bytes([b]) for b in piece_utf8]
            return self._bpe(bytes_list)
    
    def _bytes_to_ids(self, units: list[bytes]) -> list[int]:
        return [self.bytes_to_id[b] for b in units]


    def encode(
        self, 
        text: str) -> list[int]:
        """
        我们应该明白，encode是把文本变成了token流
        """
        ids = []
        pieces = self._split_to_pieces(text)
        for p in pieces:
            bytes_unit = self._encode_piece(p)
            token_ids = self._bytes_to_ids(bytes_unit)
            ids.extend(token_ids)
        return ids          

    def encode_iterable(
        self, 
        iterable: Iterable[str]) -> Iterator[int]:
        """
        我们写的是chunk safe tokenizer
        状态能否跨越边界连续工作非常重要
        """
        for text in iterable:
            pieces = self._split_to_pieces(text)
            for p in pieces:
                bytes_unit = self._encode_piece(p)
                token_ids = self._bytes_to_ids(bytes_unit)
                for tid in token_ids:
                    yield tid
    #如果使用的是encode，哪怕加上了yield也相当于是一个个消费一个完整的资源
    # iterable是一步步制造并消费   是一步步算结果，对于每一个text片段，我们计算其中的每个
    #piece对应的token ids，然后再一个个的取出

    #find all和find iter只会影响文本怎么来的，而不会影响
    #是否iterable，决定了是怎么计算分词的

    #

    def _ids_to_bytes(self, ids: list[int]) -> bytes:
        return b''.join(self.vocab[id] for id in ids)

    
    def decode(self, ids: list[int]) -> str:
        """
        token ids -> bytes_unit -> bytes_stream -> str text 
        """
        full_bytes = self._ids_to_bytes(ids)
        return full_bytes.decode("utf-8", errors="replace")

