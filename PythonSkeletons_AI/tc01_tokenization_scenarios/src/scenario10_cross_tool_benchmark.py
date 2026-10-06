"""
#  Copyright 2025 β ORI Inc.Canada All Rights Reserved.
#  Author: Awase Khirni Syed
#  AlphaFactory Bootcamps: β ORI Inc. AI Foundations - NLP Techniques Part-I
#  Version: 1.2
#  Date: 2025-10-05
#  technical card 01: Tokenization

# scenario10_cross_tool_benchmark.py

# CAPSTONE of the tokenization series (S1-S9): one identical corpus, eight
# tokenizers across five tools, one unified, comparable report.

# Adapters (ties back to the whole series):
#   RegexWordAdapter     S1   word (filtered)  stdlib        trained
#   NLTKAdapter          S2   word             nltk          closed-vocab sim
#   SpaCyAdapter         S3   word             spacy         closed-vocab sim
#   HFWordPieceAdapter   S4   subword          transformers  PRETRAINED
#   HFBPETrainedAdapter  S9   subword          tokenizers    trained (byte-level)
#   SentencePieceAdapter S7   subword          sentencepiece trained (unigram)
#   TiktokenAdapter      S5   subword          tiktoken      PRETRAINED
#   CharacterAdapter     S8   character        stdlib        trained

# Design decisions that keep the comparison honest:
#   * every adapter receives the SAME train split and is evaluated on the SAME
#     eval split (verified by an md5 fingerprint printed in the report);
#   * word-level tokenizers cannot emit UNK, so their Out of Vocabulary (OOV) is measured against
#     their own training vocabulary (closed-vocabulary simulation, S8);
#   * byte-fallback pieces (<0xXX>) are NOT counted as Out of Vocabulary (OOV): they lose no
#     information (S7/S8) — they are reported separately as 'coverage';
#   * timing uses each tool's NATIVE hot path (timed_call), min of N repeats
#     after a warmup pass; one-time init cost is timed separately;
#   * missing tools are skipped with a reason — the harness always runs.

# OOV is one the most important concept- it is a unit of text that does not exist in the
# tokenizer's fixed vocabulary. the tokenizer has no ID for it and must do something to fallback with it.

#  creating a unified metric contract ( for evaluation we need a common bench mark/metric)
# Metric , Definition
#
#  `tokens/doc` ,mean tokens per eval document
#  `chars/tok` , total chars ÷ total tokens — compression ,
#  `fertility`  tokens ÷ whitespace-words — the classic multilingual cost metric
#  `OOV%`  ,tokens carrying no recoverable surface: `[UNK]`/`<UNK>` pieces; for open-vocab word tokenizers (NLTK/spaCy/regex), eval tokens unseen in the train vocab (the closed-vocabulary simulation from S8)
# `lossless` , encode→decode reproduces the input exactly
#  `init` / `steady` / `tok/s` , construction cost (load or train) vs. warm steady-state throughput (min-of-repeats, GC-robust)
# pip3 install nltk spacy transformers tokenizers tiktoken sentencepiece
# python3 -m spacy download en_core_web_sm

important: out of vocabulary (oov) is about irrecoverable loss, not just unfamiliarity.
A byte-fallback piece is seen as expensive but fully recoverable; a [unk] is gone forever.
subword tokenization is critical for handling out-of-vocabulary(oov) words.
subword schemes decompose unseen words into known pieces insteaad of collapsing them into [unk], which is
exactly what we have demonstrated in scenarios 4,7,9 and 10 empirically.
"""
import hashlib
import io
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


def _try_import(module: str):
    try:
        return __import__(module)
    except ImportError:
        return None


nltk = _try_import("nltk")
spacy = _try_import("spacy")
transformers = _try_import("transformers")
tokenizers = _try_import("tokenizers")
tiktoken = _try_import("tiktoken")
sentencepiece = _try_import("sentencepiece")


# ----------------------------- IDENTICAL CORPUS -----------------------------
TRAIN_SECTIONS: Dict[str, List[str]] = {
    "clean_en": [
        "The deployment pipeline failed because the container image was missing a required dependency.",
        "Users reported that search results took longer to load after the update.",
        "The engineering team will publish a postmortem next Monday morning.",
        "Latency dropped by forty percent after we enabled response caching.",
        "The mobile app now supports offline reading for premium subscribers.",
        "Data engineers backfilled the events table for the first quarter.",
        "The API returns a signed URL that expires after fifteen minutes.",
        "Customer support escalated two tickets about failed password resets.",
        "The dashboard renders charts asynchronously to keep the main thread free.",
        "We migrated the recommendation service from Python to Rust last spring.",
    ],
    # NOTE: no emojis used in TRAINING — emoji appears only in EVAL, so it acts as a
    # controlled OOV probe for every tokenizer at once.
    "social": [
        "just shipped v2.0 of our CLI - 3x faster installs, release notes at https://example.com/v2",
        "@devops_dan the build broke again, CI logs look fine tho #frustrated",
        "hot take: tabs vs spaces matters less than your linter config",
        "reading the RFC before implementing saved us a whole sprint",
        "our startup's on-call rotation is 2 people for 40 microservices lol",
        "RT if you've ever debugged production on a Friday #devlife",
        "the new hire docs live at docs.example.com/onboarding fyi",
        "asked for a code review, got a philosophy lecture instead",
    ],
    "code": [
        "def parse_config(path): return json.loads(Path(path).read_text())",
        "df = df.dropna(subset=['user_id', 'ts']).sort_values('ts')",
        "result = subprocess.run(cmd, capture_output=True, check=True)",
        "model = AutoModel.from_pretrained('bert-base-uncased', cache_dir=CACHE)",
        "loss = F.cross_entropy(logits.view(-1, V), labels.view(-1))",
        "async with aiohttp.ClientSession() as s: r = await s.get(url)",
        "for k, v in sorted(counter.items(), key=lambda x: -x[1])[:10]: print(k, v)",
        "logger.info('shard %d done in %.2fs', shard_id, elapsed)",
    ],
    "cjk": [
        "モジュールの読み込みに失敗しました。設定ファイルを確認してください。",
        "検索結果の読み込みに時間がかかっています。",
        "キャッシュを有効にした後、レイテンシが40%低下しました。",
        "ユーザーは検索結果の読み込みが遅いと報告しました。",
        "用户的搜索结果加载时间过长。",
        "容器镜像缺少必要的依赖项。部署流水线失败,请检查日志文件。",
        "แอปพลิเคชันรองรับการอ่านแบบออฟไลน์แล้ว",
        "모바일 앱이 자동으로 업데이트되었습니다.",
    ],
}

# The EVAL split : byte-identical input for every adapter, including two
# 'stress' docs (full-width Latin, combining marks, emoji, leetspeak) whose
# characters were deliberately NEVER seen in training.
EVAL_DOCS: List[Tuple[str, str]] = [
    ("clean_en", "The staging cluster was scaled down overnight to reduce costs."),
    ("clean_en", "Analysts expect the migration to finish before the holiday freeze."),
    ("social",   "v2.1 is out! 2x faster cold starts ☕🔥 grab it at https://example.com/v2.1 #release"),
    ("social",   "spend 3 hrs on a bug that was a typo… classic @me #devlife 😭"),
    ("code",     "tokens = tokenizer.encode(text, add_special_tokens=False)[:max_len]"),
    ("code",     "metrics = {'p95': p95, 'p99': p99, 'rps': rps, 'errors': n_err}"),
    ("cjk",      "ログファイルにエラーが記録されました。再試行してください。"),
    ("cjk",      "缓存命中率达到了百分之九十五。"),
    ("stress",   "ＯＲＤＥＲ #55821: W1NN3R! çlığ̈ Űnicode ☕ — rｅfund pls"),
    ("stress",   "stress test: 🚀🚀🚀 full-width ｖ２．０ + leetspeak FR33 CL41M"),
]


class BenchmarkCorpus:
    """The single source of truth. Everyone/every function trains AND evaluates on this."""

    def __init__(self, train_sections: Dict[str, List[str]],
                 eval_docs: List[Tuple[str, str]]):
        self.train_sections = train_sections
        self.eval_docs = eval_docs

    def train_docs(self) -> List[Tuple[str, str]]:
        return [(sec, t) for sec, texts in self.train_sections.items()
                for t in texts]

    def eval_texts(self) -> List[str]:
        return [t for _, t in self.eval_docs]

    def fingerprint(self) -> str:
        """Proof that every adapter saw byte-identical input."""
        blob = "\n".join(self.eval_texts()).encode("utf-8")
        return hashlib.md5(blob).hexdigest()[:10]

    def stats(self) -> Dict:
        train = self.train_docs()
        tr_chars = sum(len(t) for _, t in train)
        ev_chars = sum(len(t) for _, t in self.eval_docs)
        ev_words = sum(len(re.findall(r"\S+", t)) for _, t in self.eval_docs)
        return {"train_docs": len(train), "train_chars": tr_chars,
                "eval_docs": len(self.eval_docs), "eval_chars": ev_chars,
                "eval_ws_words": ev_words,
                "sections": sorted(self.train_sections) + ["stress"],
                "fingerprint": self.fingerprint()}

    @staticmethod
    def from_files(paths: List[str]) -> "BenchmarkCorpus":
        """Real-scale swap-in: point at wikitext / The Stack / FLORES dumps.
        Everything downstream works unchanged."""
        sections: Dict[str, List[str]] = defaultdict(list)
        for p in paths:
            with open(p, encoding="utf-8", errors="ignore") as fh:
                sections[os.path.basename(p)] = [
                    ln for ln in (l.rstrip("\n") for l in fh) if ln]
        return BenchmarkCorpus(dict(sections), EVAL_DOCS)


# ----------------------------- ADAPTER LAYER --------------------------------
class BaseAdapter:
    """The contract every tokenizer must honor to be comparable."""
    name = "base"
    short = "base"
    family = "?"          # word / subword / character
    track = "?"           # pretrained / trained
    deps = "stdlib"

    def __init__(self, train: List[Tuple[str, str]]):
        raise NotImplementedError

    def tokens_of(self, text: str) -> List[str]:
        raise NotImplementedError

    def timed_call(self, text: str) -> int:
        """The tool's NATIVE hot path (used for timing). Default: full path."""
        return len(self.tokens_of(text))

    def vocab_size(self) -> int:
        raise NotImplementedError

    def is_oov(self, token: str) -> bool:
        return False

    def roundtrip(self, text: str) -> Optional[str]:
        """encode -> decode; None when exact reconstruction is unsupported."""
        return None

    @classmethod
    def available(cls) -> Tuple[bool, str]:
        return True, "ready"


class RegexWordAdapter(BaseAdapter):
    """S1 condensed: rule-based, keeps only informative token types."""
    name = "Regex rules, informative types (S1)"
    short = "regex(S1)"
    family = "word"
    track = "trained"

    PAT = re.compile(
        r"https?://\S+|[\w.+-]+@[\w-]+\.[\w.]+|\#\w+|@\w+"
        r"|\w+['’](?:t|re|ve|ll|s|d|m)"
        r"|\$\d+(?:\.\d+)?|\d+(?:\.\d+)?%?"
        r"|[A-Za-z]+|[0-9]+"
        r"|[\U0001F300-\U0001FAFF\u2600-\u27BF]")

    def __init__(self, train):
        toks = [t for _, txt in train for t in self.PAT.findall(txt)]
        self._vocab = {t.lower() for t in toks}

    def tokens_of(self, text):
        return self.PAT.findall(text)

    def vocab_size(self):
        return len(self._vocab)

    def is_oov(self, t):
        return t.lower() not in self._vocab


class NLTKAdapter(BaseAdapter):
    """S2: word_tokenize (Punkt sentence split + Treebank). Open vocabulary,
    so OOV is simulated against its own training vocabulary."""
    name = "NLTK word_tokenize (S2)"
    short = "nltk(S2)"
    family = "word"
    track = "trained"
    deps = "nltk"

    def __init__(self, train):
        for path, pkg in (("tokenizers/punkt", "punkt"),
                          ("tokenizers/punkt_tab", "punkt_tab")):
            try:
                nltk.data.find(path)
            except (LookupError, AttributeError):
                nltk.download(pkg, quiet=True)
        self._vocab = {w.lower() for _, txt in train
                       for w in nltk.tokenize.word_tokenize(txt)}

    def tokens_of(self, text):
        return nltk.tokenize.word_tokenize(text)

    def vocab_size(self):
        return len(self._vocab)

    def is_oov(self, t):
        return t.lower() not in self._vocab

    @classmethod
    def available(cls):
        return (nltk is not None,
                "ready (punkt ensured in init)" if nltk else "nltk not installed")


class SpaCyAdapter(BaseAdapter):
    """S3: spaCy's rule-based tokenizer (pipeline NOT run — fairness).
    Roundtrip is EXACT because spaCy tokens carry their whitespace."""
    name = "spaCy tokenizer (S3)"
    short = "spacy(S3)"
    family = "word"
    track = "trained"
    deps = "spacy"

    def __init__(self, train):
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.variant = "en_core_web_sm"
        except OSError:
            self.nlp = spacy.blank("en")
            self.variant = "blank(en) fallback"
        self.nlp.max_length = 10 ** 7
        self._vocab = {t.text.lower() for _, txt in train
                       for t in self.nlp.tokenizer(txt)}

    def tokens_of(self, text):
        return [t.text for t in self.nlp.tokenizer(text)]

    def timed_call(self, text):
        return len(self.nlp.tokenizer(text))          # native path: Doc build

    def vocab_size(self):
        return len(self._vocab)

    def is_oov(self, t):
        return t.lower() not in self._vocab

    def roundtrip(self, text):
        return self.nlp.tokenizer(text).text          # exact: ws preserved

    @classmethod
    def available(cls):
        return (spacy is not None,
                "ready" if spacy else "spacy not installed")


class HFWordPieceAdapter(BaseAdapter):
    """S4: pretrained bert-base-uncased WordPiece. Demonstrates the two
    classic failure modes: [UNK] and lossy lowercase normalization."""
    name = "HF WordPiece bert-base-uncased (S4)"
    short = "bert-wp(S4)"
    family = "subword"
    track = "pretrained"
    deps = "transformers"
    MODEL = "bert-base-uncased"

    def __init__(self, train):
        self.tk = transformers.AutoTokenizer.from_pretrained(self.MODEL)

    def tokens_of(self, text):
        return self.tk.tokenize(text)

    def vocab_size(self):
        return self.tk.vocab_size

    def is_oov(self, t):
        return t == "[UNK]"

    def roundtrip(self, text):
        return self.tk.decode(self.tk.encode(text, add_special_tokens=False))

    @classmethod
    def available(cls):
        return (transformers is not None,
                "ready" if transformers else "transformers not installed")


class HFBPETrainedAdapter(BaseAdapter):
    """S9: byte-level BPE trained on THIS corpus. Seeding all 256 byte chars
    (initial_alphabet) guarantees losslessness even on unseen scripts."""
    name = "HF byte-level BPE, trained V=1000 (S9)"
    short = "bpe(S9)"
    family = "subword"
    track = "trained"
    deps = "tokenizers"
    VOCAB = 1000

    def __init__(self, train):
        tok = tokenizers.Tokenizer(tokenizers.models.BPE())
        tok.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
            add_prefix_space=False)
        tok.decoder = tokenizers.decoders.ByteLevel()
        kw = dict(vocab_size=self.VOCAB, min_frequency=1, show_progress=False)
        try:                                            # version-guarded flags
            kw["initial_alphabet"] = tokenizers.pre_tokenizers.ByteLevel.alphabet()
        except Exception:
            pass
        tok.train_from_iterator([txt for _, txt in train],
                                trainer=tokenizers.trainers.BpeTrainer(**kw))
        self.tok = tok

    def tokens_of(self, text):
        return self.tok.encode(text).tokens

    def timed_call(self, text):
        return len(self.tok.encode(text).ids)          # native path: encode

    def vocab_size(self):
        return self.tok.get_vocab_size()

    def roundtrip(self, text):
        return self.tok.decode(self.tok.encode(text).ids)

    @classmethod
    def available(cls):
        return (tokenizers is not None,
                "ready" if tokenizers else "tokenizers not installed")


class SentencePieceAdapter(BaseAdapter):
    """S7: Unigram trained on THIS corpus with identity normalization and
    byte_fallback — the 'modern' recipe (Llama-style)."""
    name = "SentencePiece Unigram V=1000, byte_fallback (S7)"
    short = "spm(S7)"
    family = "subword"
    track = "trained"
    deps = "sentencepiece"
    VOCAB = 1000

    def __init__(self, train):
        flags = dict(model_type="unigram", vocab_size=self.VOCAB,
                     character_coverage=1.0,
                     normalization_rule_name="identity",
                     byte_fallback=True, add_dummy_prefix=True,
                     remove_extra_whitespaces=False, hard_vocab_limit=False)
        sents = [txt for _, txt in train]
        try:
            buf = io.BytesIO()
            sentencepiece.SentencePieceTrainer.train(
                sentence_iterator=iter(sents), model_writer=buf, **flags)
            self.sp = sentencepiece.SentencePieceProcessor(
                model_proto=buf.getvalue())
        except TypeError:                               # older versions
            with tempfile.TemporaryDirectory() as tmp:
                src, pfx = os.path.join(tmp, "c.txt"), os.path.join(tmp, "sp")
                with open(src, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(sents))
                sentencepiece.SentencePieceTrainer.train(
                    input=src, model_prefix=pfx, **flags)
                self.sp = sentencepiece.SentencePieceProcessor(
                    model_file=pfx + ".model")

    def tokens_of(self, text):
        return self.sp.encode(text, out_type=str)

    def timed_call(self, text):
        return self.sp.piece_size() and len(self.sp.encode(text))  # native path

    def vocab_size(self):
        return self.sp.piece_size()

    def is_oov(self, t):
        return t == "<unk>"

    def roundtrip(self, text):
        return self.sp.decode(self.sp.encode(text))

    @classmethod
    def available(cls):
        return (sentencepiece is not None,
                "ready" if sentencepiece else "sentencepiece not installed")


class TiktokenAdapter(BaseAdapter):
    """S5: cl100k_base (GPT-3.5/4 family). Byte-level: OOV is structurally
    impossible. Timing uses the raw Rust encode path (no token strings)."""
    name = "Tiktoken cl100k_base (S5)"
    short = "tiktoken(S5)"
    family = "subword"
    track = "pretrained"
    deps = "tiktoken"
    ENCODING = "cl100k_base"

    def __init__(self, train):
        self.enc = tiktoken.get_encoding(self.ENCODING)

    def tokens_of(self, text):
        out = []
        for i in self.enc.encode(text):
            try:
                out.append(self.enc.decode_single_token_bytes(i)
                           .decode("utf-8", "replace"))
            except Exception:
                out.append(f"<id:{i}>")
        return out

    def timed_call(self, text):
        return len(self.enc.encode(text))              # native path: pure Rust

    def vocab_size(self):
        return self.enc.n_vocab

    def roundtrip(self, text):
        return self.enc.decode(self.enc.encode(text))

    @classmethod
    def available(cls):
        return (tiktoken is not None,
                "ready" if tiktoken else "tiktoken not installed")


class CharacterAdapter(BaseAdapter):
    """S8 condensed: one char = one token, with UTF-8 byte fallback so the
    vocabulary is closed and the round-trip exact."""
    name = "Characters + byte fallback (S8)"
    short = "char(S8)"
    family = "character"
    track = "trained"

    def __init__(self, train):
        freq = Counter(c for _, txt in train for c in txt)
        self.c2i = {"<UNK>": 0}
        for c, _ in freq.most_common():
            self.c2i[c] = len(self.c2i)

    def tokens_of(self, text):
        out = []
        for c in text:
            if c in self.c2i:
                out.append(c)
            else:
                out.extend(f"<0x{b:02X}>" for b in c.encode("utf-8"))
        return out

    def vocab_size(self):
        return len(self.c2i)

    def is_oov(self, t):
        return t == "<UNK>"

    def roundtrip(self, text):
        toks, out, i = self.tokens_of(text), [], 0
        while i < len(toks):
            if toks[i].startswith("<0x"):
                buf = bytearray()
                while (i < len(toks) and toks[i].startswith("<0x")
                       and toks[i].endswith(">")):
                    buf.append(int(toks[i][3:-1], 16))
                    i += 1
                out.append(buf.decode("utf-8", "replace"))
            else:
                out.append(toks[i])
                i += 1
        return "".join(out)


# ----------------------------- MEASUREMENT ----------------------------------
_BYTE_PIECE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


class MeasurementEngine:
    """Produces one fully comparable record per adapter."""

    def __init__(self, eval_docs: List[Tuple[str, str]], repeats: int = 5):
        self.eval_docs = eval_docs
        self.repeats = repeats

    def measure(self, adapter: BaseAdapter, init_ms: float) -> Dict:
        per: Dict[str, Dict] = defaultdict(
            lambda: {"chars": 0, "tokens": 0, "words": 0, "oov": 0})
        totals = {"chars": 0, "tokens": 0, "words": 0, "oov": 0,
                  "byte_pieces": 0}
        oov_examples: List[str] = []

        for sec, text in self.eval_docs:
            toks = adapter.tokens_of(text)
            words = max(len(re.findall(r"\S+", text)), 1)   # CJK: ws-words≈1
            oov = sum(adapter.is_oov(t) for t in toks)
            for d in (per[sec], totals):
                d["chars"] += len(text)
                d["tokens"] += len(toks)
                d["words"] += words
                d["oov"] += oov
            totals["byte_pieces"] += sum(bool(_BYTE_PIECE.match(t))
                                         for t in toks)
            oov_examples += [t for t in toks if adapter.is_oov(t)
                             and t not in oov_examples][:0] or \
                            [t for t in toks if adapter.is_oov(t)
                             and t not in oov_examples]

        # round-trip fidelity on every eval doc
        rt_exact = sum(adapter.roundtrip(t) == t for _, t in self.eval_docs)

        # steady-state timing: warmup, then min-of-repeats (GC-robust)
        texts = [t for _, t in self.eval_docs]
        for t in texts:
            adapter.timed_call(t)                        # warmup
        best = float("inf")
        for _ in range(self.repeats):
            t0 = time.perf_counter()
            n = sum(adapter.timed_call(t) for t in texts)
            best = min(best, time.perf_counter() - t0)

        t = totals
        return {
            "short": adapter.short, "name": adapter.name,
            "family": adapter.family, "track": adapter.track,
            "deps": adapter.deps,
            "vocab": adapter.vocab_size(),
            "tokens_per_doc": round(t["tokens"] / len(texts), 1),
            "chars_per_token": round(t["chars"] / t["tokens"], 2),
            "fertility": round(t["tokens"] / t["words"], 2),
            "oov_pct": round(100 * t["oov"] / t["tokens"], 1),
            "oov_examples": oov_examples[:6],
            "byte_pieces": t["byte_pieces"],
            "lossless": f"{rt_exact}/{len(texts)}",
            "init_ms": round(init_ms, 1),
            "ms_per_doc": round(best * 1000 / len(texts), 3),
            "tokens_per_sec": round(t["tokens"] / best),
            "per_section": {s: {"chars_per_token":
                                round(d["chars"] / d["tokens"], 1) if d["tokens"] else None,
                                "fertility": round(d["tokens"] / d["words"], 1)}
                            for s, d in per.items()},
        }


class BenchmarkRunner:
    """Constructs each adapter under a timer, measures it, records skips."""

    ADAPTERS = [RegexWordAdapter, NLTKAdapter, SpaCyAdapter,
                HFWordPieceAdapter, HFBPETrainedAdapter,
                SentencePieceAdapter, TiktokenAdapter, CharacterAdapter]

    def __init__(self, corpus: BenchmarkCorpus, repeats: int = 5):
        self.corpus = corpus
        self.repeats = repeats

    def run(self) -> Tuple[List[Dict], List[Dict]]:
        records, skips = [], []
        engine = MeasurementEngine(self.corpus.eval_docs, self.repeats)
        for cls in self.ADAPTERS:
            ok, note = cls.available()
            if not ok:
                skips.append({"short": cls.short, "reason": note})
                continue
            try:
                t0 = time.perf_counter()
                adapter = cls(self.corpus.train_docs())
                init_ms = (time.perf_counter() - t0) * 1000
                records.append(engine.measure(adapter, init_ms))
            except Exception as exc:                    # e.g. offline model DL
                skips.append({"short": cls.short,
                              "reason": f"{type(exc).__name__}: {str(exc)[:60]}"})
        return records, skips


# ----------------------------- REPORTING ------------------------------------
class ReportBuilder:
    """One report, five tools, zero apples-vs-oranges."""

    SECTIONS = ["clean_en", "social", "code", "cjk", "stress"]

    @staticmethod
    def availability(skips: List[Dict], records: List[Dict]) -> None:
        print("--- A. Adapter availability ---")
        for r in records:
            print(f"  {r['short']:13} {r['deps']:13} ready")
        for s in skips:
            print(f"  {s['short']:13} {'-':13} [skip] {s['reason']}")

    @staticmethod
    def main_table(records: List[Dict], fingerprint: str) -> None:
        print("\n--- B. THE unified report (identical eval corpus, "
              f"fingerprint {fingerprint}) ---")
        print("  fairness: 'trained' tokenizers overfit this corpus (S9-H); "
              "'pretrained' carry open-world vocabs")
        print(f"  {'tokenizer':13} {'family':9} {'track':10} {'vocab':>8} "
              f"{'tok/doc':>8} {'ch/tok':>6} {'fert':>5} {'OOV%':>5} "
              f"{'lossless':>8} {'init ms':>8} {'ms/doc':>7} {'tok/s':>10}")
        for r in records:
            print(f"  {r['short']:13} {r['family']:9} {r['track']:10} "
                  f"{r['vocab']:>8,} {r['tokens_per_doc']:>8} "
                  f"{r['chars_per_token']:>6} {r['fertility']:>5} "
                  f"{r['oov_pct']:>5} {r['lossless']:>8} {r['init_ms']:>8} "
                  f"{r['ms_per_doc']:>7} {r['tokens_per_sec']:>10,}")

    @classmethod
    def section_matrices(cls, records: List[Dict]) -> None:
        for metric, label in (("chars_per_token", "compression (chars/token — higher = denser)"),
                              ("fertility", "fertility (tokens per whitespace word)")):
            print(f"\n--- C{'.1' if metric == 'chars_per_token' else '.2'} "
                  f"Per-section {label} ---")
            hdr = "  " + f"{'tokenizer':13}" + "".join(
                f"{s:>10}" for s in cls.SECTIONS)
            print(hdr)
            for r in records:
                cells = []
                for s in cls.SECTIONS:
                    v = r["per_section"].get(s, {}).get(metric)
                    cells.append(f"{v:>10}" if v is not None else f"{'n/a':>10}")
                print("  " + f"{r['short']:13}" + "".join(cells))
            if metric == "fertility":
                print("  (cjk has ~1 whitespace word per doc: low fertility there")
                print("   means the tokenizer FAILED to segment, not that it's cheap)")

    @staticmethod
    def oov_hall_of_fame(records: List[Dict]) -> None:
        print("\n--- D. OOV hall of fame + byte-fallback coverage ---")
        for r in records:
            line = (f"  {r['short']:13} OOV={r['oov_pct']:>4}%  "
                    f"examples={r['oov_examples']}")
            if r["byte_pieces"]:
                line += (f"\n                 {r['byte_pieces']} byte-fallback "
                         f"pieces fired (coverage, NOT loss)")
            print(line)
        print("  word-level OOV = eval tokens unseen in that tokenizer's train")
        print("  vocab (closed-vocab simulation, S8). [UNK] = irrecoverable.")

    @staticmethod
    def speed_chart(records: List[Dict]) -> None:
        print("\n--- E. Throughput (log-scale bars; steady state) ---")
        fastest = max(r["tokens_per_sec"] for r in records)
        for r in sorted(records, key=lambda r: -r["tokens_per_sec"]):
            n = max(1, round((len(f"{r['tokens_per_sec']:,}") and
                              __import__("math").log10(r["tokens_per_sec"]))
                             / __import__("math").log10(fastest) * 34))
            print(f"  {r['short']:13} │{'█' * n} {r['tokens_per_sec']:,} tok/s")

    @staticmethod
    def leaderboard(records: List[Dict]) -> None:
        print("\n--- F. Leaderboard (with fairness caveats) ---")
        fast = max(records, key=lambda r: r["tokens_per_sec"])
        print(f"  fastest            : {fast['short']}  "
              f"{fast['tokens_per_sec']:,} tok/s")
        for track in ("trained", "pretrained"):
            pool = [r for r in records if r["track"] == track]
            if pool:
                best = max(pool, key=lambda r: r["chars_per_token"])
                print(f"  densest ({track:9}): {best['short']}  "
                      f"{best['chars_per_token']} chars/token"
                      + ("  (overfits this corpus!)" if track == "trained" else ""))
        zero = [r["short"] for r in records if r["oov_pct"] == 0]
        loss = [r["short"] for r in records
                if r["lossless"].startswith(f"{len(records[0]['lossless'][-1:] and '10'[:0] or '10')}")
                or r["lossless"].split("/")[0] == r["lossless"].split("/")[1]]
        std = [r["short"] for r in records if r["deps"] == "stdlib"]
        print(f"  zero OOV           : {', '.join(zero)}")
        print(f"  lossless           : {', '.join(loss)}")
        print(f"  stdlib-only        : {', '.join(std)}")
        print("  bert-wp is lossy: lowercase normalization + [UNK] (cf. S7-G)")

    @staticmethod
    def decision_matrix() -> None:
        print("\n--- G. Capstone decision matrix (need -> tool -> scenario) ---")
        rows = [
            ("exact words, punctuation-aware",      "spaCy / NLTK",              "S2, S3"),
            ("zero-dependency, embedded / IoT",     "regex rules",               "S1, S6"),
            ("domain-specific, OOV-proof vocab",    "trained byte-level BPE",    "S4, S9"),
            ("multilingual / spaceless scripts",    "SentencePiece Unigram",     "S7"),
            ("LLM token budgets, pricing, RAG",     "Tiktoken",                  "S5"),
            ("typo / leetspeak robustness",         "character n-grams",         "S8"),
            ("vocab-size engineering",              "sweep + knee analysis",     "S9"),
            ("apples-to-apples comparison",         "this harness",              "S10"),
        ]
        for need, tool, sc in rows:
            print(f"  {need:38} {tool:28} {sc}")


# ----------------------------- MAIN -----------------------------------------
if __name__ == "__main__":
    corpus = BenchmarkCorpus(TRAIN_SECTIONS, EVAL_DOCS)
    st = corpus.stats()
    print("=== SCENARIO 10: cross-tool tokenizer benchmark ===")
    print(f"corpus: train={st['train_docs']} docs ({st['train_chars']:,} chars) "
          f"| eval={st['eval_docs']} docs ({st['eval_chars']:,} chars, "
          f"{st['eval_ws_words']} ws-words)")
    print(f"sections: {st['sections']} | eval fingerprint: {st['fingerprint']} "
          "(identical input for every adapter)")
    # corpus = BenchmarkCorpus.from_files(["wikitext.txt", "thestack.txt"])  # scale-up

    records, skips = BenchmarkRunner(corpus, repeats=5).run()

    ReportBuilder.availability(skips, records)
    ReportBuilder.main_table(records, st["fingerprint"])
    ReportBuilder.section_matrices(records)
    ReportBuilder.oov_hall_of_fame(records)
    ReportBuilder.speed_chart(records)
    ReportBuilder.leaderboard(records)
    ReportBuilder.decision_matrix()
