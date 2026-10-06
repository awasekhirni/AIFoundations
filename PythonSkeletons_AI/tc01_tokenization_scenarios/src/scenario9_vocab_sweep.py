"""
#  Copyright 2025 β ORI Inc.Canada All Rights Reserved.
#  Author: Awase Khirni Syed
#  AlphaFactory Bootcamps: β ORI Inc. AI Foundations - NLP Techniques Part-I
#  Version: 1.2
#  Date: 2025-01-25
#  technical card 01: Tokenization
# scenario9_vocab_sweep.py

# Tokenizer engineering, end to end:
#   A. CorpusBuilder          - deterministic NL + code + multilingual corpus
#   B. Fragmentation demo     - one code line at V=300 vs V=2400 (BPE)
#   C. VocabSweeper           - BPE / Unigram / WordPiece x 5 vocab budgets,
#                               controlled: byte-level for BPE+WordPiece,
#                               Metaspace (SentencePiece-style) for Unigram
#   D. ASCIIChart + Pareto    - compression curves + quadratic attention cost
#   E. KneeFinder             - chord/max-curvature knee of each curve
#   F. EmbeddingCostModel     - GPT-2 / Llama-2 / Llama-3 param accounting
#   G. TokenizerAdvisor       - a real decision memo with a recommendation
#   H. ProductionReferencePoints - optional tiktoken (cl100k / o200k) benchmarks

# Real-scale data sources to swap in via CorpusBuilder.from_files():
#   code: HF 'bigcode/the-stack' or 'code_search_net'
#   NL:   'wikitext-103', C4, OpenWebText
#   multilingual: FLORES-200, OPUS
# pip3 install tokenizers
# pip3 install tiktoken
# scenario : vocabulary size decision - comparison between 3 approaches
# sweeping bpe (byte pair encoding) vs. unigram vs. wordpiece across vocab budgets
# context windows are measured in tokens - token efficiency is context efficiency
# domain: natural language (multilingual)
"""
import math
import random
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
except ImportError:
    sys.exit("Missing dependency -> pip install tokenizers")


# ----------------------------- SAMPLE DATA ---------------------------------
NL_TEMPLATES = [
    "The {model} model returns NaN losses when trained with lr={lr} on the {dataset} dataset.",
    "Why does {fn} raise a ValueError when the input tensor has shape {shape}?",
    "After upgrading to {version}, our CI pipeline fails on the {test} tests.",
    "The {model} checkpoint is {size}GB — can we shard it across {n} GPUs?",
    "How do I serialize a {type} without pickling the entire module?",
    "Benchmarks: {model} reaches {score} on {benchmark}, but inference is slow on CPU.",
    "Docs for {fn} are unclear about the {arg} argument — what does it do?",
    "Our {dataset} loader leaks memory: RSS grows by {mb}MB per epoch.",
]
SLOTS = {
    "model": ["resnet", "bert-base", "gpt2-medium", "vit", "llama-7b", "t5-small",
              "mistral", "deberta"],
    "fn": ["model.forward", "DataLoader.__iter__", "optimizer.step", "collate_fn",
           "trainer.evaluate", "tokenizer.encode", "model.generate"],
    "dataset": ["cifar10", "imdb", "squad", "wikitext-103", "c4", "openwebtext",
                "commoncrawl", "glue"],
    "lr": ["1e-3", "3e-4", "1e-4", "5e-5"],
    "version": ["2.0.1", "2.1.0", "3.8", "4.31.0"],
    "test": ["unit", "integration", "regression", "e2e"],
    "shape": ["(None, 128)", "(B, 3, 224, 224)", "(N, 512)", "(batch, seq, d)"],
    "type": ["DataLoader", "optimizer state", "tokenizer", "model config", "dataset"],
    "n": ["2", "4", "8"], "size": ["0.7", "1.4", "4.9", "13"],
    "benchmark": ["glue", "mmlu", "squad", "hellaswag"],
    "score": ["82.4", "91.2", "67.3"], "mb": ["120", "310", "512"],
    "arg": ["num_workers", "pin_memory", "grad_accum", "fp16"],
}
CODE_SNIPPETS = [
    "def train_one_epoch(model, loader, optimizer):\n    model.train()\n    for batch in loader:\n        optimizer.zero_grad()\n        loss = model(batch)\n        loss.backward()\n        optimizer.step()\n    return loss.item()",
    "logits = torch.nn.functional.softmax(x / math.sqrt(d), dim=-1)",
    "df = pd.read_csv(path, parse_dates=['timestamp'])",
    "x = np.asarray(x, dtype=np.float32).reshape(-1, seq_len, d_model)",
    "embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)",
    "grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)",
    "with torch.no_grad():\n    out = model.generate(**inputs, max_new_tokens=128, do_sample=True)",
    "mask = (scores > threshold) & (~df.duplicated(subset=['id']))",
    "return F.cross_entropy(logits.view(-1, vocab), labels.view(-1), ignore_index=-100)",
    "attn = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)",
    "assert x.shape == (batch, seq, d), f'got {x.shape}'",
    "optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)",
    "for i, (xb, yb) in enumerate(loader, start=1):\n    step_loss = loss_fn(model(xb), yb)",
    "preds = np.where(probs > 0.5, 1, 0).astype(np.int64)",
    "state = {'epoch': epoch, 'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]}",
    "total_loss += loss.item() * grad_accum_steps",
]

# add more examples of multilingual context here for arabic, urdu, farsi, german
MULTILINGUAL = [
    "El modelo devuelve NaN durante el entrenamiento.",
    "Le modèle renvoie NaN pendant l'entraînement.",
    "モデルがNaNを返します。学習率を下げてください。",
    "模型在训练过程中返回NaN。",
    "Модель возвращает NaN во время обучения.",
]

# Held-out eval texts: same domain, fresh surface forms (2 multilingual lines
# are reused verbatim so the tiny Unigram vocab covers every eval character).
EVAL_TEXTS = [
    "The vit model diverges when trained with lr=1e-2 on the cifar100 dataset.",
    "Why does trainer.save raise a RuntimeError when the output dir is read-only?",
    "logits = torch.nn.functional.log_softmax(x / math.sqrt(d), dim=-1)",
    "optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)",
    "df = pd.read_parquet(path, columns=['user_id', 'ts'])",
    "attn = (q @ k.transpose(-2, -1)) / math.sqrt(d_k) + mask",
    "El modelo devuelve NaN durante el entrenamiento.",
    "モデルがNaNを返します。学習率を下げてください。",
]
DEMO_LINE = "logits = torch.nn.functional.softmax(x / math.sqrt(d), dim=-1)"


class _SafeDict(dict):
    def __missing__(self, key):        # unfilled template slots -> ""
        return ""


class CorpusBuilder:
    """Deterministic pretraining-style mix: NL templates, Python snippets,
    multilingual lines, and chat-style NL+code documents."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.counts = {"nl": 0, "code": 0, "multilingual": 0, "mixed": 0}

    def _nl(self) -> str:
        template = self.rng.choice(NL_TEMPLATES)
        vals = {k: self.rng.choice(v) for k, v in SLOTS.items()}
        return template.format_map(_SafeDict(vals))

    def build(self, n: int = 150) -> List[str]:
        docs = []
        for _ in range(n):
            r = self.rng.random()
            if r < 0.55:
                docs.append(self._nl()); self.counts["nl"] += 1
            elif r < 0.85:
                docs.append(self.rng.choice(CODE_SNIPPETS)); self.counts["code"] += 1
            elif r < 0.93:
                docs.append(self.rng.choice(MULTILINGUAL)); self.counts["multilingual"] += 1
            else:
                docs.append(self._nl() + "\n```python\n"
                            + self.rng.choice(CODE_SNIPPETS) + "\n```")
                self.counts["mixed"] += 1
        return docs

    @staticmethod
    def stats(docs: List[str]) -> Dict:
        chars = sum(map(len, docs))
        return {"docs": len(docs), "chars": chars,
                "avg_chars": round(chars / len(docs), 1),
                "distinct_words": len({w for d in docs for w in re.findall(r"\w+", d)}),
                "distinct_chars": len({c for d in docs for c in d})}

    @staticmethod
    def from_files(paths: List[str]) -> List[str]:
        """Real-scale: point at the-stack / wikitext / OPUS dumps."""
        docs = []
        for p in paths:
            with open(p, encoding="utf-8", errors="ignore") as fh:
                docs += [ln for ln in (l.rstrip("\n") for l in fh) if ln]
        return docs


class ByteLevelManual:
    """GPT-2's byte-level mapping implemented from scratch — used to decode
    WordPiece+ByteLevel pieces manually (ties back to S5/S8 byte tricks)."""

    def __init__(self):
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs = bs[:]
        extra = 0
        for b in range(256):
            if b not in bs:
                bs.append(b); cs.append(256 + extra); extra += 1
        self._c2b = {chr(c): b for b, c in zip(bs, cs)}

    def decode_pieces(self, pieces: Iterable[str]) -> str:
        joined = "".join(pieces)
        data = bytes(self._c2b.get(ch, ord(ch) & 0xFF) for ch in joined)
        return data.decode("utf-8", errors="replace")


class TrainedTokenizer:
    """Uniform interface over the three trained algorithms."""

    def __init__(self, tok: Tokenizer, algorithm: str, requested: int):
        self.tok, self.algorithm, self.requested = tok, algorithm, requested
        self.actual = tok.get_vocab_size(with_added_tokens=True)
        self._is_wp = algorithm == "wordpiece"
        self._bl = ByteLevelManual()

    def tokens(self, text: str) -> List[str]:
        return self.tok.encode(text).tokens

    def decode_ids(self, ids: List[int]) -> str:
        if not self._is_wp:
            return self.tok.decode(ids)
        pieces = [self.tok.id_to_token(i) for i in ids]        # strip WordPiece ##
        return self._bl.decode_pieces(p[2:] if p.startswith("##") else p
                                      for p in pieces)

    def display(self, piece: str) -> str:                       # human-readable
        if self._is_wp and piece.startswith("##"):
            piece = piece[2:]
        return piece.replace("Ġ", " ").replace("▁", " ")

    def analyze(self, texts: List[str]) -> Dict:
        n_tok = n_char = frag = 0
        ids, lossless = set(), True
        for t in texts:
            enc = self.tok.encode(t)
            n_tok += len(enc.ids); n_char += len(t)
            ids.update(enc.ids)
            frag += sum(len(self.display(p)) <= 1 for p in enc.tokens)
            if self.decode_ids(list(enc.ids)) != t:
                lossless = False
        return {"algorithm": self.algorithm, "requested": self.requested,
                "actual": self.actual, "ok": True,
                "tokens_per_doc": round(n_tok / len(texts), 1),
                "chars_per_token": round(n_char / n_tok, 2),
                "frag_rate": round(frag / n_tok, 2),
                "utilization": round(len(ids) / self.actual, 2),
                "lossless": lossless}


class TokenizerFactory:
    """Controlled training: byte-level pre-tokenizer for BPE/WordPiece
    (GPT-2 family), Metaspace for Unigram (SentencePiece family — S7).
    NOTE: '#' kept out of the corpus — WordPiece's ## marker would collide."""

    ALGORITHMS = ("bpe", "unigram", "wordpiece")

    def train(self, algorithm: str, vocab_size: int,
              corpus: List[str]) -> TrainedTokenizer:
        model = {"bpe": lambda: models.BPE(),
                 "wordpiece": lambda: models.WordPiece(unk_token="[UNK]"),
                 "unigram": lambda: models.Unigram()}[algorithm]()
        tok = Tokenizer(model)
        if algorithm == "unigram":
            tok.pre_tokenizer = pre_tokenizers.Metaspace()
            tok.decoder = decoders.Metaspace()
        else:
            # add_prefix_space=False -> EXACT round-trip (GPT-2 setting)
            tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tok.decoder = decoders.ByteLevel()
        tok.train_from_iterator(iter(corpus), trainer=self._trainer(algorithm,
                                                                    vocab_size))
        tok.add_special_tokens(["<|endoftext|>"])               # post-training
        return TrainedTokenizer(tok, algorithm, vocab_size)

    def _trainer(self, algorithm: str, vocab_size: int):
        alphabet = self._byte_alphabet()                        # all 256 bytes
        specs = {
            "bpe": [(trainers.BpeTrainer, {"vocab_size": vocab_size, "min_frequency": 1,
                                            "show_progress": False,
                                            "initial_alphabet": alphabet}),
                    (trainers.BpeTrainer, {"vocab_size": vocab_size, "min_frequency": 1,
                                            "show_progress": False})],
            "wordpiece": [(trainers.WordPieceTrainer, {"vocab_size": vocab_size,
                                                       "min_frequency": 1,
                                                       "show_progress": False,
                                                       "initial_alphabet": alphabet}),
                          (trainers.WordPieceTrainer, {"vocab_size": vocab_size,
                                                       "min_frequency": 1,
                                                       "show_progress": False})],
            "unigram": [(trainers.UnigramTrainer, {"vocab_size": vocab_size,
                                                   "show_progress": False}),
                        (trainers.UnigramTrainer, {"vocab_size": vocab_size})],
        }[algorithm]
        for cls, kwargs in specs:                               # version fallbacks
            try:
                return cls(**kwargs)
            except TypeError:
                continue
        raise RuntimeError(f"cannot construct trainer for {algorithm}")

    @staticmethod
    def _byte_alphabet() -> List[str]:
        try:
            return pre_tokenizers.ByteLevel.alphabet()
        except Exception:
            return []                                           # corpus chars only


class VocabSweeper:
    """The experiment: algorithms x vocab budgets -> metrics on held-out text.
    Trainer failures (e.g. vocab beyond corpus support) are CAPTURED, not
    crashed : they are findings (the data ceiling)."""

    def __init__(self, corpus: List[str], eval_texts: List[str],
                 factory: Optional[TokenizerFactory] = None):
        self.corpus, self.eval_texts = corpus, eval_texts
        self.factory = factory or TokenizerFactory()

    def run(self, algorithms: Tuple[str, ...], vocab_sizes: Tuple[int, ...]
            ) -> List[Dict]:
        results = []
        for alg in algorithms:
            for v in vocab_sizes:
                try:
                    tk = self.factory.train(alg, v, self.corpus)
                    results.append(tk.analyze(self.eval_texts))
                except Exception as exc:                        # data ceiling!
                    results.append({"algorithm": alg, "requested": v, "ok": False,
                                    "reason": str(exc)[:70]})
        return results


class ASCIIChart:
    def __init__(self, width: int = 40):
        self.width = width

    def bar_chart(self, title: str, rows: List[Tuple[str, float]]) -> None:
        print(f"  {title}")
        vmax = max(v for _, v in rows) or 1.0
        for label, val in rows:
            n = max(1, round(val / vmax * self.width))
            print(f"    {label:>8} │{'█' * n} {val:,.1f}")


class KneeFinder:
    @staticmethod
    def chord(points: List[Tuple[float, float]]) -> Optional[int]:
        """Knee = max perpendicular distance to the chord joining the ends
        (classic curve-of-max-curvature heuristic)."""
        if len(points) < 3:
            return None
        (x0, y0), (x1, y1) = points[0], points[-1]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy) or 1.0
        best_i, best_d = 0, -1.0
        for i, (x, y) in enumerate(points):
            d = abs(dy * (x - x0) - dx * (y - y0)) / length
            if d > best_d:
                best_d, best_i = d, i
        return int(points[best_i][0])


class SweepAnalytics:
    def __init__(self, results: List[Dict],
                 algorithms: Tuple[str, ...] = ("bpe", "unigram", "wordpiece")):
        self.results, self.algorithms = results, algorithms

    def ok(self) -> List[Dict]:
        return [r for r in self.results if r.get("ok")]

    def _alg_rows(self, alg: str) -> List[Dict]:
        return sorted([r for r in self.ok() if r["algorithm"] == alg],
                      key=lambda r: r["requested"])

    def knees(self) -> Dict[str, int]:
        out = {}
        for alg in self.algorithms:
            pts = [(r["requested"], r["tokens_per_doc"])
                   for r in self._alg_rows(alg)]
            k = KneeFinder.chord(pts)
            if k:
                out[alg] = k
        return out

    def ceilings(self) -> Dict[str, Dict]:
        """Largest vocab each algorithm could actually reach on this corpus."""
        out = {}
        for alg in self.algorithms:
            rows = self._alg_rows(alg)
            if rows:
                top = max(rows, key=lambda r: r["actual"])
                out[alg] = {"actual": top["actual"], "requested": top["requested"],
                            "tokens": top["tokens_per_doc"],
                            "utilization": top["utilization"]}
        return out

    def print_pareto(self) -> None:
        sizes = sorted({r["requested"] for r in self.ok()})
        worst = max(r["tokens_per_doc"] for r in self.ok())

        def cell(v):
            return f"{v:>6.1f}" if isinstance(v, (int, float)) else "  --  "

        print("  vocab │    bpe │ unigram │ wordpiece │   best │ attn-cost*")
        for v in sizes:
            row = {a: next((r["tokens_per_doc"] for r in self._alg_rows(a)
                            if r["requested"] == v), None) for a in self.algorithms}
            best = min(x for x in row.values() if x is not None)
            attn = f"{100 * (best / worst) ** 2:>4.0f}%"
            print(f"  {v:>5,} │ {cell(row['bpe'])} │ {cell(row['unigram'])} │"
                  f" {cell(row['wordpiece'])} │ {cell(best)} │ {attn}")
        print("  *attention proxy: (tokens/worst)² — quadratic in sequence length")


class EmbeddingCostModel:
    """Params vs. vocab: why 32k -> 128k was a real engineering tradeoff."""

    REFERENCE = [  # (name, vocab, hidden, tied?, total params)
        ("GPT-2 (124M)", 50_257, 768, True, 0.124e9),
        ("Llama-2 (7B)", 32_000, 4_096, False, 6.74e9),
        ("Llama-3 (8B)", 128_256, 4_096, False, 8.03e9),
    ]

    @staticmethod
    def _fmt(n: float) -> str:
        return f"{n / 1e6:,.0f}M" if n < 1e9 else f"{n / 1e9:,.2f}B"

    def reference_table(self) -> None:
        print("  model            vocab   hidden tied  emb-params  %params"
              "  head-FLOPs/token")
        for name, v, h, tied, total in self.REFERENCE:
            emb = v * h * (1 if tied else 2)
            print(f"  {name:16} {v:>7,} {h:>7,} {'yes' if tied else ' no':>4}"
                  f" {self._fmt(emb):>10} {emb / total:>7.1%}"
                  f" {v * h / (2 * total):>13.1%}")
        print("  (o200k / GPT-4o: vocab 200,019 — hidden size not public)")
        delta = (128_256 - 32_000) * 4_096 * 2
        print(f"  Llama-2 -> Llama-3: +96,256 pieces = +{delta / 1e6:,.0f}M embedding"
              f" params (~{delta / 8.03e9:.0%} of an 8B model), exchanged for")
        print("  fewer tokens/doc on multilingual+code — amortized over 15T+ train tokens.")


class TokenizerAdvisor:
    """Decision-support: score = normalized(tokens/doc) + lam*normalized(vocab),
    penalizing under-utilized vocabularies. Then snap to a 'standard' size."""

    STD_SIZES = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    def __init__(self, analytics: SweepAnalytics, corpus_chars: int, lam: float = 0.5):
        self.analytics, self.corpus_chars, self.lam = analytics, corpus_chars, lam

    def recommend(self) -> Dict:
        ok = self.analytics.ok()
        cands = [r for r in ok if r["lossless"]] or ok
        toks = [r["tokens_per_doc"] for r in cands]
        vocs = [r["actual"] for r in cands]
        t_lo, t_hi = min(toks), max(toks)
        v_lo, v_hi = min(vocs), max(vocs)
        best = None
        for r in cands:
            tn = (r["tokens_per_doc"] - t_lo) / (t_hi - t_lo or 1)
            vn = (r["actual"] - v_lo) / (v_hi - v_lo or 1)
            score = tn + self.lam * vn
            if r["utilization"] < 0.40:
                score += 0.10                                # dead-row penalty # need to increase and decrease penalty to see affects on efficiency
            if best is None or score < best[0]:
                best = (score, r)
        r = best[1]
        suggested = next((s for s in self.STD_SIZES if s >= r["actual"] * 1.5),
                         self.STD_SIZES[-1])
        return {"algorithm": r["algorithm"], "vocab": r["actual"],
                "tokens_per_doc": r["tokens_per_doc"],
                "suggested_standard": suggested}

    def memo(self) -> None:
        ok = self.analytics.ok()
        ceil, knees = self.analytics.ceilings(), self.analytics.knees()
        print("=== TOKENIZER-ENGINEERING DECISION MEMO ===")
        print(f"  corpus: {self.corpus_chars:,} chars | {len(ok)} successful sweep points")
        print("  1) data ceiling : " + " | ".join(
            f"{a} {c['actual']:,}" for a, c in ceil.items()))
        print("  2) knee points  : " + " | ".join(
            f"{a} {k:,}" for a, k in knees.items()) + "  (diminishing returns beyond)")
        rec = self.recommend(); alg = rec["algorithm"]
        rows = self.analytics._alg_rows(alg)
        small = rows[0]
        knee_row = next((r for r in rows if r["requested"] == knees.get(alg)), None)
        top = max(rows, key=lambda r: r["actual"])
        if knee_row:
            g1 = 100 * (small["tokens_per_doc"] - knee_row["tokens_per_doc"]) \
                 / small["tokens_per_doc"]
            g2 = 100 * (knee_row["tokens_per_doc"] - top["tokens_per_doc"]) \
                 / knee_row["tokens_per_doc"]
            print(f"  3) compression  : {alg} V{small['requested']:,}->"
                  f"{knee_row['requested']:,}: -{g1:.0f}% tokens; "
                  f"V{knee_row['requested']:,}->{top['actual']:,}: -{g2:.0f}% more"
                  f" for {top['actual'] / knee_row['actual']:.1f}x the vocab")
        if alg in ceil:
            print(f"  4) utilization  : at ceiling V={ceil[alg]['actual']:,} only "
                  f"{ceil[alg]['utilization']:.0%} of pieces fire on eval"
                  f" ({1 - ceil[alg]['utilization']:.0%} dead embedding rows)")
        print(f"  5) RECOMMENDATION: {alg} @ V≈{rec['vocab']:,} "
              f"(snap to standard {rec['suggested_standard']:,})")
        print("     (algorithms tie within noise here; production favors byte-level")
        print("      BPE for exact lossless round-trips and proven scaling)")
        print(f"  6) CAVEAT: this corpus is {self.corpus_chars:,} chars. Real vocab")
        print("     decisions use 10^12+ chars — which is why Llama-2 chose 32k and")
        print("     Llama-3 moved to 128k. Small data -> small vocab.")


class ProductionReferencePoints:
    """Optional tiktoken benchmarks on the SAME eval texts (cl100k / o200k).
    Downloads encoding files on first use — needs network."""

    NAMES = ("cl100k_base", "o200k_base")

    def evaluate(self, texts: List[str]) -> List[Dict]:
        try:
            import tiktoken
        except ImportError:
            print("  [skip] pip install tiktoken for this section")
            return []
        out = []
        total_chars = sum(map(len, texts))
        for name in self.NAMES:
            try:
                enc = tiktoken.get_encoding(name)
                n = sum(len(enc.encode(t)) for t in texts)
                out.append({"name": name, "tokens_per_doc": round(n / len(texts), 1),
                            "chars_per_token": round(total_chars / n, 2)})
            except Exception as exc:
                print(f"  {name:12} [skip] {type(exc).__name__} (offline?)")
        return out


if __name__ == "__main__":
    print("--- A. Corpus (pretraining-style mix) ---")
    builder = CorpusBuilder(seed=42)
    train_docs = builder.build(n=150)
    st = CorpusBuilder.stats(train_docs)
    print(f"  {st}  |  composition: {builder.counts}")
    print(f"  eval: {len(EVAL_TEXTS)} held-out docs")

    factory = TokenizerFactory()

    print("\n--- B. Fragmentation: one code line at two vocab budgets (BPE) ---")
    print(f"  text: {DEMO_LINE}")
    for v in (300, 2400):
        tk = factory.train("bpe", v, train_docs)
        pieces = tk.tokens(DEMO_LINE)
        print(f"  V={v:>4}: {len(pieces):2d} tokens | chars/token="
              f"{len(DEMO_LINE) / len(pieces):.2f}")
        print(f"          {' '.join(repr(p) for p in pieces[:15])}...")
    print("  (Ġ = a space in byte-level encoding; small vocab shreds 'functional')")

    print("\n--- C. Sweep: 3 algorithms x 5 vocab budgets ---")
    sweeper = VocabSweeper(train_docs, EVAL_TEXTS, factory)
    results = sweeper.run(("bpe", "unigram", "wordpiece"),
                          (300, 600, 1200, 2400, 4800))
    print(f"  {'algorithm':9} {'V-req':>6} {'V-act':>6} {'tok/doc':>8} "
          f"{'ch/tok':>7} {'frag':>5} {'util':>5} {'lossless':>8}")
    for r in results:
        if r["ok"]:
            print(f"  {r['algorithm']:9} {r['requested']:>6,} {r['actual']:>6,} "
                  f"{r['tokens_per_doc']:>8} {r['chars_per_token']:>7} "
                  f"{r['frag_rate']:>5} {r['utilization']:>5} {str(r['lossless']):>8}"
                  + ("  <- corpus ceiling" if r["actual"] < r["requested"] else ""))
        else:
            print(f"  {r['algorithm']:9} {r['requested']:>6,} {'FAILED':>6}"
                  f"  -> {r['reason']}")

    analytics = SweepAnalytics(results)
    chart = ASCIIChart()

    print("\n--- D. Compression curves + attention cost ---")
    for alg in analytics.algorithms:
        rows = [(f"V={r['requested']:>4}", r["tokens_per_doc"])
                for r in analytics._alg_rows(alg)]
        chart.bar_chart(f"tokens per eval doc (lower = better) — {alg}", rows)
    analytics.print_pareto()

    print("\n--- E. Knee points (max curvature of each curve) ---")
    print("  " + " | ".join(f"{a}: {k:,}" for a, k in analytics.knees().items()))
    print("  (below the knee each extra piece buys a lot; above it, little)")

    print("\n--- F. What vocab size costs in a real model ---")
    EmbeddingCostModel().reference_table()

    print("\n--- G. Decision memo ---")
    TokenizerAdvisor(analytics, st["chars"]).memo()

    print("\n--- H. Production reference points (tiktoken, same eval texts) ---")
    for r in ProductionReferencePoints().evaluate(EVAL_TEXTS):
        print(f"  {r['name']:12} tokens/doc={r['tokens_per_doc']:>5} "
              f"chars/token={r['chars_per_token']}")
    print("  NOTE: our V=2,400 tokenizer 'beats' them only because it OVERFITS this")
    print("  narrow template domain; production tokenizers trade domain compression")
    print("  for open-world coverage — which is the whole big-vocab-at-scale story.")
