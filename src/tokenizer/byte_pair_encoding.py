import regex as re
from typing import List, Tuple, Dict


def split_on_special_tokens(text: str, special_tokens: List[str]) -> List[str]:
    if not special_tokens:
        return [text]
    delim = "|".join(re.escape(tok) for tok in special_tokens)
    return re.split(delim, text)


special = ["<|endoftext|>", "yes"]
s = "Doc1.<|endoftext|>Doc2."
print(split_on_special_tokens(s, special))  # ['Doc1.', 'Doc2.']

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

pat = re.compile(PAT)


def gpt2_pretokenize(text: str) -> List[str]:
    return [m.group(0) for m in pat.finditer(text)]


print(gpt2_pretokenize("some text that i'll pre-tokenize"))


def pretok_to_byte_tokens(pretok: str) -> Tuple[bytes, ...]:
    b = pretok.encode("utf-8")
    return tuple(bytes([bb]) for bb in b)


print(pretok_to_byte_tokens(" text")[:6])


def count_pretokens(
    text: str, special_tokens: List[str]
) -> Dict[Tuple[bytes, ...], int]:
    counts = {}
    for seg in split_on_special_tokens(text, special_tokens):
        if not seg:
            continue
        for m in pat.finditer(seg):
            seq = pretok_to_byte_tokens(m.group(0))
            counts[seq] = counts.get(seq, 0) + 1
    return counts


demo = "low low lower<|endoftext|>low!"
print(demo)
print(count_pretokens(demo, ["<|endoftext|>"]))


def pair_frequencies(token_seqs: Dict[Tuple[bytes, ...], int]):
    pc = {}
    for seq, c in token_seqs.items():
        for i in range(len(seq) - 1):
            pc[(seq[i], seq[i + 1])] = pc.get((seq[i], seq[i + 1]), 0) + c
    return pc


def best_pair(pair_counts: Dict[Tuple[bytes, bytes], int]) -> Tuple[bytes, bytes]:
    return max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]


def merge_sequence(seq: Tuple[bytes, ...], a: bytes, b: bytes) -> Tuple[bytes, ...]:
    out = []
    i = 0
    ab = a + b
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
            out.append(ab)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return tuple(out)


def apply_merge(
    token_seqs: Dict[Tuple[bytes, ...], int], a: bytes, b: bytes
) -> Dict[Tuple[bytes, ...], int]:
    hm = {}
    for seq, c in token_seqs.items():
        new_seq = merge_sequence(seq, a, b)
        hm[new_seq] = hm.get(new_seq, 0) + c
    return hm


def num_merges_for(vocab_size: int, special_tokens: List[str]) -> int:
    return vocab_size - 256 - len(special_tokens)


def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]):
    vocab: Dict[int, bytes] = {}
    for i, tok in enumerate(special_tokens):
        vocab[i] = tok.encode("utf-8")

    base = len(special_tokens)
    for i in range(256):
        vocab[base + i] = bytes([i])

    merges: List[Tuple[bytes, bytes]] = []
    merges_needed = num_merges_for(vocab_size, special_tokens)
    next_id = 256 + base

    with open(input_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    token_seqs = count_pretokens(text, special_tokens)

    for _ in range(merges_needed):
        pc = pair_frequencies(token_seqs)
        if not pc:
            break
        a, b = best_pair(pc)
        merges.append((a, b))
        vocab[next_id] = a + b
        next_id += 1

        token_seqs = apply_merge(token_seqs, a, b)

    return vocab, merges


import cProfile, pstats

pr = cProfile.Profile()
pr.enable()

vocab, merges = train_bpe("tinystories_train.txt", 10_000, ["<|endoftext|>"])

pr.disable()
pstats.Stats(pr).sort_stats("cumtime").print_stats(25)


# Encoding


def merge_all(seq: List[bytes], a: bytes, b: bytes) -> List[bytes]:
    out: List[bytes] = []
    i = 0
    ab = a + b
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
            out.append(ab)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def apply_merges_in_order(
    seq: List[bytes], merges: List[Tuple[bytes, bytes]]
) -> List[bytes]:
    for a, b in merges:
        if len(seq < 2):
            break

        found = False
        for i in range(len(seq) - 1):
            if seq[i] == a and seq[i + 1] == b:
                found = True
                break
        if found:
            seq = merge_all(seq, a, b)
    return seq


def build_bytes_to_id(vocab: Dict[int, bytes]) -> Dict[bytes, int]:
    return {tok_bytes: tid for tid, tok_bytes in vocab.items()}


def encode_no_special(
    text: str, merges: List[Tuple[bytes, bytes]], bytes_to_id: Dict[bytes, int]
) -> List[int]:
    ids: List[int] = []
    for m in PAT.finditer(text):
        pretok = m.group(0)
        seq = pretok_to_byte_tokens(pretok)  # bytes tokens
        seq = apply_merges_in_order(seq, merges)  # merged tokens
        for tok_bytes in seq:
            ids.append(bytes_to_id[tok_bytes])  # bytes -> id
    return ids


def decode_ids(ids: List[int], vocab: Dict[int, bytes]) -> str:
    b = b"".join(vocab[i] for i in ids)
    return b.decode("utf-8", errors="replace")


class Tokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: List[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = special_tokens or []

        self.bytes_to_id = build_bytes_to_id(self.vocab)

        # Ensure special tokens exist in vocab (append if missing)
        for s in self.special_tokens:
            b = s.encode("utf-8")
            if b not in self.bytes_to_id:
                new_id = max(self.vocab.keys()) + 1 if self.vocab else 0
                self.vocab[new_id] = b
                self.bytes_to_id[b] = new_id

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: List[str] | None = None,
    ) -> "Tokenizer":
        # Assumes base64 JSON format (adjust if you used pickle)
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vobj = json.load(f)
        with open(merges_filepath, "r", encoding="utf-8") as f:
            mobj = json.load(f)

        vocab: Dict[int, bytes] = {
            int(k): base64.b64decode(v) for k, v in vobj["vocab"].items()
        }
        merges: List[Tuple[bytes, bytes]] = [
            (base64.b64decode(a), base64.b64decode(b)) for a, b in mobj["merges"]
        ]
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> List[int]:
        return encode_with_special(
            text, self.merges, self.bytes_to_id, self.special_tokens
        )

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        return encode_iterable(
            iterable, self.merges, self.bytes_to_id, self.special_tokens
        )

    def decode(self, ids: List[int]) -> str:
        return decode_ids(ids, self.vocab)
