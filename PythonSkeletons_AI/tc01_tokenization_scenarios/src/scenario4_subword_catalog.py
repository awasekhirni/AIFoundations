"""
#  Copyright 2025 β ORI Inc.Canada All Rights Reserved.
#  Author: Awase Khirni Syed
#  AlphaFactory Bootcamps: β ORI Inc. AI Foundations - NLP Techniques Part-I
#  Version: 1.2
#  Date: 2025-01-25
#  technical card 01: Tokenization
# scenario4_subword_catalog.py
# Part A: pretrained WordPiece Bidirectional encoder representations from transformers (2018-Google's BERT) — unknown words become meaningful pieces.
# Part B: train your own BPE tokenizer on domain data.
# BPE- byte pair encoding
# pip3 install transformers tokenizers
"""
from typing import Dict, List
from transformers import AutoTokenizer
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


class SubwordTokenizer:
    """Thin wrapper around a pretrained HuggingFace tokenizer."""

    def __init__(self, model_name: str = "bert-base-uncased"):
        # For heavy multilingual catalogs use: "bert-base-multilingual-cased"
        self.model_name = model_name
        self.tk = AutoTokenizer.from_pretrained(model_name)

    @property
    def vocab_size(self) -> int:
        return self.tk.vocab_size

    def subwords(self, text: str) -> List[str]:
        return self.tk.tokenize(text)

    def encode(self, text: str) -> List[int]:
        return self.tk.encode(text, add_special_tokens=True)  # adds [CLS]...[SEP]

    def decode(self, ids: List[int]) -> str:
        return self.tk.decode(ids)

    def explain(self, text: str) -> Dict:
        ids = self.encode(text)
        return {"text": text,
                "tokens": self.tk.convert_ids_to_tokens(ids),
                "ids": ids, "n_tokens": len(ids)}


class OOVComparator:
    """WHY subwords matter: unknown words at word level become meaningful
    pieces at subword level """

    def __init__(self, toy_word_vocab: set, subword: SubwordTokenizer):
        self.word_vocab = toy_word_vocab
        self.subword = subword

    def compare(self, word: str) -> Dict:
        return {"word": word,
                "word_level_token": word if word.lower() in self.word_vocab else "<UNK>",
                "subword_pieces": self.subword.subwords(word)}


class ProductCatalogTokenizer:
    """Turns messy catalog titles into model-ready subword input."""

    def __init__(self, model_name: str = "bert-base-uncased"):
        self.sw = SubwordTokenizer(model_name)

    def process(self, titles: List[str]) -> List[Dict]:
        out = []
        for t in titles:
            e = self.sw.explain(t)
            e["has_UNK"] = "[UNK]" in e["tokens"]
            out.append(e)
        return out

    def summary(self, processed: List[Dict]) -> Dict:
        n = len(processed)
        return {"titles": n,
                "avg_subword_tokens": round(sum(p["n_tokens"] for p in processed) / n, 1),
                "titles_with_UNK": sum(p["has_UNK"] for p in processed)}


class CustomBPETokenizer:
    """Train your own Byte-Pair-Encoding tokenizer on domain data."""

    def __init__(self, vocab_size: int = 300):
        self.vocab_size = vocab_size
        self.tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tok.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tok.decoder = decoders.BPEDecoder()

    def train(self, corpus: List[str]) -> "CustomBPETokenizer":
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size, min_frequency=1,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        self.tok.train_from_iterator(corpus, trainer)
        return self

    def encode(self, text: str) -> Dict:
        enc = self.tok.encode(text)
        return {"tokens": enc.tokens, "ids": enc.ids}

    def decode(self, ids: List[int]) -> str:
        return self.tok.decode(ids)


# ----------------------------- SAMPLE DATA ---------------------------------
PRODUCT_TITLES = [
    "Philips Sonicare DiamondClean Smart 9300 Rechargeable Electric Toothbrush",
    "iPhone15ProMax Clear Case with MagSafe & Kickstand",
    "Unicorn Kids Frappuccino Mug 350ml ☕",
    "Dyson V15 Detect Absolute Staubsauger Akku HEPA",
    "Cafetiere a piston Bodum 34oz 1L Inox",
    "SAMSUNG 55 QLED 4K Smart TV QN90B",
    "Ninja AirFryer XL 5.5L Digital",
    "Logitech MX Master 3S Wireless Mouse",
]
TOY_WORD_VOCAB = {"philips", "smart", "rechargeable", "electric", "toothbrush",
                   "clear", "case", "mug", "samsung", "tv", "wireless", "mouse"}


#----------------ASSSIGNMENT ---look at car specification data ------------------
#----------------ASSSIGNMENT --- look at medical labels data on prescriptions -------

if __name__ == "__main__":
    sw = SubwordTokenizer()
    print(f"Pretrained: {sw.model_name} | vocab={sw.vocab_size:,} tokens")

    print("\n--- OOV comparison (word-level vocab vs subwords) ---")
    cmp = OOVComparator(TOY_WORD_VOCAB, sw)
    for w in ["toothbrush", "Sonicare", "Frappuccino", "AirFryer"]:
        print(" ", cmp.compare(w))

    print("\n--- Catalog pipeline ---")
    pipe = ProductCatalogTokenizer()
    processed = pipe.process(PRODUCT_TITLES)
    for p in processed:
        print(f"  {p['text'][:48]:48} -> {p['n_tokens']:2d} tokens")
    print("  summary:", pipe.summary(processed))

    print("\n--- Round trip ---")
    ids = sw.encode(PRODUCT_TITLES[0])
    print("  ids[:10] =", ids[:10])
    print("  decoded  =", sw.decode(ids))

    print("\n--- Custom BPE trained on the catalog ---")
    bpe = CustomBPETokenizer(300).train(PRODUCT_TITLES * 10)
    print("  learned vocab size:", bpe.tok.get_vocab_size())
    print("  encode unseen query:", bpe.encode("Sonicare DiamondClean 9500"))
