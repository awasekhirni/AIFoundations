"""
#  Copyright 2025 β ORI Inc.Canada All Rights Reserved.
#  Author: Awase Khirni Syed
#  AlphaFactory Bootcamps: β ORI Inc. AI Foundations - NLP Techniques Part-I
#  Version: 1.2
#  Date: 2025-01-25
#  technical card 01: Tokenization
# scenario7_sentencepiece_unigram.py

# End-to-end: train a Unigram model on multilingual support tickets ->
# tokenize with probabilities -> compare vs BPE -> prove losslessness ->
# route unsupported-language tickets using byte-fallback statistics.

# Papers: Kudo & Richardson 2018 (SentencePiece), Kudo 2018 (Subword Regularization).
Note: SentencePieceTrainer prints its own INFO logs to stderr — that is normal.

pip install sentencepiece        # core (required)
pip install transformers         # optional — Part I only


"""
import io
import os
import sys
import tempfile
from typing import Dict, Iterable, List, Tuple

try:
    import sentencepiece as spm
except ImportError:
    sys.exit("Missing dependency -> pip install sentencepiece")


# ----------------------------- SAMPLE DATA ---------------------------------
SAMPLE_TICKETS = [
    {"id": "T-101", "lang": "en", "text": "My order #55821 arrived damaged. Please send a replacement."},
    {"id": "T-102", "lang": "en", "text": "The app crashes when I upload photos on iPhone 15."},
    {"id": "T-103", "lang": "en", "text": "I was charged twice for the monthly subscription."},
    {"id": "T-104", "lang": "ja", "text": "注文#55821の商品が破損して届きました。交換をお願いします。"},
    {"id": "T-105", "lang": "ja", "text": "アプリが起動しません。再インストールしても同じエラーが出ます。"},
    {"id": "T-106", "lang": "ja", "text": "支払いが二重に請求されました。返金をリクエストします。"},
    {"id": "T-107", "lang": "zh", "text": "我的订单#55821收到时已损坏，请寄送替换品。"},
    {"id": "T-108", "lang": "zh", "text": "应用上传照片时闪退，请尽快修复。"},
    {"id": "T-109", "lang": "ko", "text": "주문 #55821 상품이 파손되어 도착했습니다. 교환을 요청합니다."},
    {"id": "T-110", "lang": "ko", "text": "앱 실행 시 오류가 발생합니다."},
    {"id": "T-111", "lang": "th", "text": "สินค้าชำรุดเมื่อได้รับ ขอเปลี่ยนใหม่"},
    {"id": "T-112", "lang": "en", "text": "Refund not received for order #77340, please check."},
    {"id": "T-113", "lang": "zh", "text": "退款尚未到账，请核实订单#77340。"},
]

# Held-out tickets: unseen script (Cyrillic), emoji-stress, in-domain Japanese
UNSEEN_TICKETS = [
    {"id": "U-201", "lang": "?",  "text": "Мой заказ повреждён, пожалуйста, верните деньги."},
    {"id": "U-202", "lang": "?",  "text": "Order #55821 refund 💳💳💳 please!"},
    {"id": "U-203", "lang": "ja", "text": "返金処理がまだ完了していません。"},
]


class MultilingualTicketCorpus:
    """Data provider. Synthetic by default; swap in a real multilingual corpus
    with from_huggingface() (pip install datasets).
    Real-scale sources: HF 'setfit/amazon_reviews_multi' (en/ja/zh/de/es/fr),
    Tatoeba, or Meta FLORES-200 for cross-tokenizer evaluation."""

    def __init__(self, tickets: List[Dict]):
        self.tickets = tickets

    @property
    def languages(self) -> List[str]:
        return sorted({t["lang"] for t in self.tickets})

    def sentences(self) -> List[str]:
        return [t["text"] for t in self.tickets]

    @staticmethod
    def sample() -> "MultilingualTicketCorpus":
        return MultilingualTicketCorpus(SAMPLE_TICKETS)

    @staticmethod
    def from_huggingface(per_lang: int = 500) -> "MultilingualTicketCorpus":
        from datasets import load_dataset          # optional dependency
        rows = []
        for lang in ("en", "ja", "zh"):
            ds = load_dataset("setfit/amazon_reviews_multi", lang, split="train")
            rows += [{"id": f"{lang}-{i}", "lang": lang, "text": r["text"]}
                     for i, r in enumerate(ds.select(range(per_lang)))]
        return MultilingualTicketCorpus(rows)


class UnigramModelTrainer:
    """SentencePiece trainer factory.

    Key flags (and why):
      model_type='unigram'           - probabilistic LM over pieces, Viterbi decode
      character_coverage=1.0         - keep every corpus char (tiny corpus);
                                       use 0.9995 for large CJK corpora
      normalization_rule_name='identity' - LOSSLESS: no Unicode folding, no
                                       whitespace collapsing (the legacy
                                       'nmt_nfkc' default turns ＯＲＤＥＲ into
                                       ORDER — see RoundTripAuditor below)
      byte_fallback=True             - 256 <0xXX> pieces -> ZERO <unk>, ever
      add_dummy_prefix=True          - every text starts with ▁ so word-initial
                                       pieces are learned consistently
      hard_vocab_limit=False         - tolerate tiny corpora that cannot
                                       support the full requested budget
    """

    def __init__(self, vocab_size: int = 800, model_type: str = "unigram",
                 character_coverage: float = 1.0, byte_fallback: bool = True,
                 normalization: str = "identity",
                 remove_extra_whitespaces: bool = False):
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.character_coverage = character_coverage
        self.byte_fallback = byte_fallback
        self.normalization = normalization
        self.remove_extra_whitespaces = remove_extra_whitespaces

    def _flags(self) -> Dict:
        return dict(model_type=self.model_type,
                    vocab_size=self.vocab_size,
                    character_coverage=self.character_coverage,
                    normalization_rule_name=self.normalization,
                    byte_fallback=self.byte_fallback,
                    add_dummy_prefix=True,
                    remove_extra_whitespaces=self.remove_extra_whitespaces,
                    hard_vocab_limit=False)

    def train(self, sentences: Iterable[str]) -> spm.SentencePieceProcessor:
        """Train and return a loaded processor — in memory when the installed
        version supports model_writer, else via temp files (older versions)."""
        sentences = list(sentences)
        flags = self._flags()
        try:
            buf = io.BytesIO()
            spm.SentencePieceTrainer.train(
                sentence_iterator=iter(sentences), model_writer=buf, **flags)
            return spm.SentencePieceProcessor(model_proto=buf.getvalue())
        except TypeError:                              # no iterator/model_writer
            with tempfile.TemporaryDirectory() as tmp:
                src = os.path.join(tmp, "corpus.txt")
                prefix = os.path.join(tmp, "sp")
                with open(src, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(sentences))
                spm.SentencePieceTrainer.train(input=src, model_prefix=prefix, **flags)
                return spm.SentencePieceProcessor(model_file=prefix + ".model")


class UnigramTokenizer:
    """Wrapper exposing the pieces, ids and PROBABILITIES that make the
    Unigram model unique among subword tokenizers."""

    def __init__(self, sp: spm.SentencePieceProcessor, name: str = "unigram"):
        self.sp = sp
        self.name = name

    # ---- basic ------------------------------------------------------------
    def pieces(self, text: str) -> List[str]:
        return self.sp.encode(text, out_type=str)

    def ids(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)

    def decode_pieces(self, pieces: List[str]) -> str:
        return self.sp.decode_pieces(pieces)

    def decode_ids(self, ids: List[int]) -> str:
        return self.sp.decode(ids)

    def detok_join(self, pieces: List[str]) -> str:
        """The ▁ trick: the space lives INSIDE the token, so detokenization
        needs no vocabulary at all — just join and replace."""
        return "".join(pieces).replace("\u2581", " ").lstrip()

    # ---- unigram-LM specifics ----------------------------------------------
    def piece_logprobs(self, text: str) -> List[Tuple[str, float]]:
        return [(p, round(self.sp.get_score(self.sp.piece_to_id(p)), 2))
                for p in self.pieces(text)]

    def text_stats(self, text: str) -> Dict:
        ids = self.ids(text)
        n = max(len(ids), 1)
        n_bytes = sum(self.sp.is_byte(i) for i in ids)
        try:
            avg_lp = round(sum(self.sp.get_score(i) for i in ids) / n, 2)
        except RuntimeError:            # scores only exist for UNIGRAM models
            avg_lp = None
        return {"n_tokens": len(ids),
                "n_byte_pieces": n_bytes,
                "byte_ratio": round(n_bytes / n, 3),
                "avg_logprob": avg_lp,
                "chars_per_token": round(len(text) / n, 2)}

    def vocab_report(self) -> Dict:
        n = self.sp.piece_size()
        usable = [i for i in range(n)
                  if not (self.sp.is_control(i) or self.sp.is_byte(i)
                          or self.sp.is_unknown(i))]
        top = sorted(usable, key=lambda i: self.sp.get_score(i), reverse=True)[:8]
        return {"vocab_size": n,
                "byte_pieces": sum(self.sp.is_byte(i) for i in range(n)),
                "control_pieces": sum(self.sp.is_control(i) for i in range(n)),
                "top_pieces": [self.sp.id_to_piece(i) for i in top]}


class SubwordRegularizer:
    """Unigram's superpower: one text has MANY valid segmentations.
    nbest()  -> deterministic top-k from the Viterbi lattice.
    sample() -> stochastic draws (alpha = diversity; used as data augmentation
                when training MT/LLM models — Kudo 2018)."""

    def __init__(self, tokenizer: UnigramTokenizer, alpha: float = 0.1):
        self.tok = tokenizer
        self.alpha = alpha

    def nbest(self, text: str, k: int = 3) -> List[List[str]]:
        sp = self.tok.sp
        if hasattr(sp, "nbest_encode"):
            return sp.nbest_encode(text, nbest_size=k, out_type=str)
        return sp.nbest_encode_as_pieces(text, nbest_size=k)

    def sample(self, text: str, n_draws: int = 4) -> List[List[str]]:
        return [self.tok.sp.encode(text, out_type=str, enable_sampling=True,
                                   nbest_size=-1, alpha=self.alpha)
                for _ in range(n_draws)]


class UnigramVsBPEComparator:
    """Controlled A/B: same corpus, same flags, same budget — only the
    algorithm differs. (Scenario 4 used *pretrained* WordPiece/BPE.)"""

    def __init__(self, sentences: List[str], vocab_size: int = 800):
        self.sentences = sentences
        self.vocab_size = vocab_size

    def run(self) -> Dict:
        models = {mt: UnigramTokenizer(
                      UnigramModelTrainer(self.vocab_size, model_type=mt)
                      .train(self.sentences), name=mt)
                  for mt in ("unigram", "bpe")}

        report: Dict = {}
        for name, tok in models.items():
            stats = [tok.text_stats(s) for s in self.sentences]
            n = len(stats)
            report[name] = {
                "vocab_size": tok.sp.piece_size(),
                "avg_tokens": round(sum(s["n_tokens"] for s in stats) / n, 1),
                "chars_per_token": round(
                    sum(s["chars_per_token"] for s in stats) / n, 2),
                "lossless_roundtrip": all(
                    tok.decode_pieces(tok.pieces(s)) == s for s in self.sentences),
            }
        sets = [{t.sp.id_to_piece(i) for i in range(t.sp.piece_size())}
                for t in models.values()]
        report["vocab_jaccard_overlap"] = round(
            len(sets[0] & sets[1]) / len(sets[0] | sets[1]), 3)
        return report


class RoundTripAuditor:
    """Encode->decode fidelity. identity + byte_fallback reconstructs the
    input EXACTLY, even for characters never seen in training. The legacy
    default (nmt_nfkc, no byte fallback — what T5 shipped with) is lossy in
    TWO ways: Unicode folding AND <unk>."""

    TEST_TEXTS = [
        "ＯＲＤＥＲ #55821 破損",    # full-width Latin (U+FF2x)
        "two  spaces   inside",       # whitespace runs
        "café ☕ naïve",             # accents + emoji unseen in training
    ]

    def audit(self, tok: UnigramTokenizer) -> List[Dict]:
        rows = []
        for text in self.TEST_TEXTS:
            decoded = tok.decode_pieces(tok.pieces(text))
            rows.append({"text": text, "lossless": decoded == text,
                         "decoded": decoded})
        return rows


class UnsupportedLanguageRouter:
    """End-to-end business logic built purely on TOKENIZER statistics.
    The intent classifier was trained on ja/zh/ko/th/en. A ticket in an
    unseen script tokenizes almost entirely into byte pieces — detect that
    with byte_ratio and route it to translation instead of feeding garbage
    to the classifier. (Byte_ratio also catches emoji-stress tickets.)"""

    def __init__(self, tokenizer: UnigramTokenizer, byte_threshold: float = 0.20):
        self.tok = tokenizer
        self.byte_threshold = byte_threshold

    def calibrate(self, train_sentences: List[str]) -> float:
        worst = max(self.tok.text_stats(s)["byte_ratio"]
                    for s in train_sentences)
        print(f"  calibration: worst training byte_ratio={worst} "
              f"(threshold={self.byte_threshold})")
        return worst

    def route(self, text: str) -> Dict:
        st = self.tok.text_stats(text)
        supported = st["byte_ratio"] < self.byte_threshold
        return {**st, "supported": supported,
                "action": "PROCESS_INTENT_MODEL" if supported
                          else "ROUTE_TO_TRANSLATION"}

    def scan(self, tickets: List[Dict]) -> List[Dict]:
        return [{"ticket_id": t["id"], **self.route(t["text"])}
                for t in tickets]


class PretrainedUnigramModels:
    """Production Unigram tokenizers loadable in one line (transformers).
    t5-base = 32k (English C4); google/mt5-base = 250k (101 languages);
    xlm-roberta-base is SentencePiece-Unigram too (250k, 100 languages)."""

    DEMOS = [
        ("t5-base", "Refund my order #55821"),
        ("google/mt5-base", "注文#55821の商品が破損しています。交換を要求します。"),
    ]

    def run(self) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            print("  [skip] pip install transformers to run this section")
            return
        for name, text in self.DEMOS:
            try:
                tok = AutoTokenizer.from_pretrained(name)
                pieces = tok.tokenize(text)
                tail = "..." if len(pieces) > 9 else ""
                print(f"  {name:18} vocab={tok.vocab_size:>7,} "
                      f"pieces={pieces[:9]}{tail}")
            except Exception as exc:
                print(f"  {name:18} [skip] {type(exc).__name__} (offline?)")


if __name__ == "__main__":
    corpus = MultilingualTicketCorpus.sample()
    print(f"Corpus: {len(corpus.tickets)} tickets | languages={corpus.languages}")
    # corpus = MultilingualTicketCorpus.from_huggingface()   # <- real data

    print("\n--- A. Train SentencePiece Unigram (vocab=800, byte_fallback) ---")
    tok = UnigramTokenizer(UnigramModelTrainer(vocab_size=800)
                           .train(corpus.sentences()))
    rep = tok.vocab_report()
    print(f"  vocab={rep['vocab_size']} byte_pieces={rep['byte_pieces']} "
          f"control={rep['control_pieces']}")
    print(f"  top pieces by log-prob: {rep['top_pieces']}")

    print("\n--- B. '▁' = whitespace INSIDE tokens (U+2581) ---")
    for t in corpus.tickets[:1] + corpus.tickets[3:4]:           # en + ja
        pieces = tok.pieces(t["text"])
        tail = "..." if len(pieces) > 12 else ""
        print(f"  [{t['lang']}] {t['text'][:38]}...")
        print(f"        pieces: {pieces[:12]}{tail}")
        print(f"        join-only detokenization exact: "
              f"{tok.detok_join(pieces) == t['text']}")
    print("  NOTE: no whitespace pre-tokenization -> ja/zh/th need no special mode.")

    print("\n--- C. Every piece carries a log-probability (encode = Viterbi argmax) ---")
    for piece, lp in tok.piece_logprobs(corpus.tickets[3]["text"])[:8]:
        print(f"    {piece!r:12} log P = {lp}")

    print("\n--- D. byte_fallback: unseen script -> byte pieces, never <unk> ---")
    ru = UNSEEN_TICKETS[0]["text"]
    print(f"  unseen Cyrillic: {ru}")
    print(f"  first pieces: {tok.pieces(ru)[:10]}...")
    print(f"  stats: {tok.text_stats(ru)}")
    print(f"  round-trip exact: {tok.decode_pieces(tok.pieces(ru)) == ru}")

    print("\n--- E. Subword regularization: one text, many valid segmentations ---")
    reg = SubwordRegularizer(tok, alpha=0.1)
    target = corpus.tickets[3]["text"]
    for i, seg in enumerate(reg.nbest(target, k=3), 1):
        print(f"  nbest #{i} ({len(seg):2d} tokens): {seg[:10]}...")
    for i, seg in enumerate(reg.sample(target, n_draws=4), 1):
        print(f"  sample #{i} ({len(seg):2d} tokens): {seg[:10]}...")
    print("  (sampled segmentations differ per run — that IS the augmentation)")

    print("\n--- F. Unigram vs BPE, same corpus & budget (controlled A/B) ---")
    for k, v in UnigramVsBPEComparator(corpus.sentences(), 800).run().items():
        print(f"  {k:24} {v}")

    print("\n--- G. Round-trip audit: identity+byte_fallback vs legacy nmt_nfkc ---")
    legacy = UnigramTokenizer(UnigramModelTrainer(
        vocab_size=800, normalization="nmt_nfkc", byte_fallback=False,
        remove_extra_whitespaces=True).train(corpus.sentences()),
        name="nmt_nfkc (legacy)")
    for model in (tok, legacy):
        print(f"  model={model.name}:")
        for row in RoundTripAuditor().audit(model):
            flag = "OK   " if row["lossless"] else "LOSSY"
            print(f"    [{flag}] {row['text']!r:30} -> {row['decoded']!r}")

    print("\n--- H. End-to-end: route unsupported-language tickets ---")
    router = UnsupportedLanguageRouter(tok)
    router.calibrate(corpus.sentences())
    for r in router.scan(UNSEEN_TICKETS):
        print(f"  {r['ticket_id']} byte_ratio={r['byte_ratio']:<6} "
              f"tokens={r['n_tokens']:<3} -> {r['action']}")

    print("\n--- I. Production Unigram tokenizers (T5 / mT5) ---")
    PretrainedUnigramModels().run()
