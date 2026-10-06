"""
#  Copyright 2025 β ORI Inc.Canada All Rights Reserved.
#  Author: Awase Khirni Syed
#  AlphaFactory Bootcamps: β ORI Inc. AI Foundations - NLP Techniques Part-I
#  Version: 1.2
#  Date: 2025-01-25
#  technical card 01: Tokenization
# scenario8_character_spam.py

# Character-level tokenization — the THIRD granularity from the technique
# card's description ("words, subwords, or characters"): words = S1-S3,
# subwords = S4/S5/S7, characters = here.

# Domain: SMS spam filtering over noisy user-generated content.
#   * SMS-speak, leetspeak, typos, ALL-CAPS: word-level OOV is the NORM.
#   * A character vocab is tiny and effectively closed; byte fallback (same
#     trick as SentencePiece S7 / Tiktoken S5) makes even unseen symbols safe.

# End-to-end pipeline, PURE STDLIB (Python 3.8+, zero installs):
#   A. CharacterTokenizer — vocab, ids, UNK vs byte fallback, exact round-trip
#   B. GranularityComparator — char vs subword (mini-BPE) vs word, incl. OOV
#                              rates under a leetspeak attack
#   C. TypoRobustnessBenchmark — word lookup vs char n-gram similarity
#   D. NaiveBayesTextClassifier — word vs char-ngram features (strategy
#      pattern: one classifier class, two featurizers), clean + adversarial
#   E. CharacterLanguageModel — trigram char LM: generate synthetic spam,
#      perplexity as an obfuscation detector
#   F. SpamFilterPipeline — everything wired into one moderate() API

# Sample data: synthetic messages in the style of the UCI SMS Spam Collection
# (5,574 real ham/spam SMS, public domain). Scale-up: SpamDataset.from_uci().



"""
import math
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


# ----------------------------- SAMPLE DATA ---------------------------------
# Synthetic messages written in the style of the UC Irvine ML Repository UCI SMS Spam Collection.
# Real-scale source (public domain, 5,574 msgs):
#   https://archive.ics.uci.edu/dataset/228/sms+spam+collection
SPAM = [
    "WINNER! You have been selected for a £1000 prize. Txt CLAIM to 89001 now!",
    "UR selected 4 a FREE iPod shuffle. Just txt PRIZE to 80085 to claim!",
    "Fr3e ringtones 4 ur mobile!! Txt TONE to 69966",
    "URGENT: Your account has been suspended. Verify now at secure-check.example",
    "Congrats! Your mobile number won £5000. Call 0900 100 100 to claim prize!",
    "FREE entry to our prize draw! Txt WIN to 81118",
    "You have 1 new voicemail from an unknown number. Call 0844 773 0123 to hear it",
    "Hiya! Im home alone feelin naughty. Txt me 4 my pics",
    "Loan approved! £5000 in your account today. No credit checks. Reply YES",
    "UR going 2 luv this! 100% free wallpaper txts. Send GO to 83332",
    "4get ur bills! We clear debt 4 u. Txt DEBT to 66633",
    "Last chance 2 claim ur £50 voucher at retail-rewards.example. Txt YES",
    "Congratulations ur awarded £500 of shopping vouchers. Call 0906 661 2211 now",
    "Awnser this Q to win a top prize: Where is the Eiffel Tower? Txt PARIS",
    "New! Video phone 4 u. Call 0800 111 2222 2day only!",
    "Txt EDIT to 85080 to get ur free ringtone of the week!",
    "UR a winner in our summer draw! Claim ur holiday by txtng HOLIDAY to 85555",
    "Free msg: reply DATE to meet singles in ur area 2nite!",
    "Warning! Your computer is infected. Scan now at fix-now.example",
    "80085 = free txts 4 a month. Send FREE to start",
    "Special offer: 2 bottles of herbal boost for the price of 1. Order 2day!",
    "You are being contacted about your compensation claim. Reply INFO now",
    "Get ur free casino chips! Txt CHIPS to 69988. 18+ only",
    "Someone has a crush on u! Find out who. Txt SECRET to 85544",
    "Free trial of horoscopes 4 a week. Txt STARS to stop txt STOP",
]
HAM = [
    "Sorry I'll b late home, traffic is awful on the ring road",
    "lol ok see u 2morrow at the usual place",
    "Mum the train is delayed, home by 7ish",
    "R u coming 2 the party on sat? bring beer",
    "haha yeah it was gr8 seeing everyone again",
    "Can u pick up milk on ur way home pls?",
    "I'm in a meeting til 5, call u after",
    "No worries, I already ate. save me some tho",
    "Just landed! taxiing now, see u in 20",
    "Thanks 4 the birthday msg! had a great day",
    "ok cool. i'll book the table for 7:30",
    "my phone battery is dying, call the landline",
    "Did u record the match? don't tell me the score!",
    "U around later? fancy a pint",
    "Running 10 mins behind, start without me",
    "yas! the tickets arrived this morning",
    "mom says dinner at 6, dont b late",
    "I got the job!!! start on monday :D",
    "cheers mate, owe u one",
    "ill text u when im outside",
    "gr8, thanks! see u then",
    "can we move the call to 4pm? conflicts with the school run",
    "omw now",
    "It was so good 2 see u guys, lets do it again soon",
    "night night, dont stay up too late",
]
INBOX = [  # unseen messages for the end-to-end demo (Part F)
    "Hey are we still on for Friday? bring the umbrella lol",
    "URGENT: ur account is locked. Verify at secure-login.example now!",
    "W1NN3R! Txt CL41M 2 89001 4 ur £50 pr1z3!!!",
]


# ============================= TOKENIZERS ===================================
class CharacterTokenizer:
    """One character = one token: the degenerate — and maximally robust —
    end of the tokenization spectrum.

    * vocab is tiny and (with byte fallback) CLOSED: nothing is OOV
    * encode/decode are exact inverses (verified in Part A)
    * keeps case for free — ALL-CAPS is itself a spam signal
    * GPT-2 byte-level BPE is this idea + merges (256 base 'characters');
      SentencePiece byte_fallback (Scenario 7) is the same fallback.
    """

    def __init__(self, unk_token: str = "<UNK>", byte_fallback: bool = True):
        self.unk = unk_token
        self.byte_fallback = byte_fallback
        self.char2id: Dict[str, int] = {}
        self.id2char: Dict[int, str] = {}
        self.char_freq: Counter = Counter()

    def fit(self, corpus: List[str]) -> "CharacterTokenizer":
        self.char_freq = Counter(ch for t in corpus for ch in t)
        self.char2id = {self.unk: 0}
        for ch, _ in self.char_freq.most_common():
            self.char2id[ch] = len(self.char2id)
        self.id2char = {i: c for c, i in self.char2id.items()}
        return self

    # ---- token level -------------------------------------------------------
    def tokenize(self, text: str) -> List[str]:
        out = []
        for ch in text:
            if ch in self.char2id:
                out.append(ch)
            elif self.byte_fallback:                 # unseen char -> byte pieces
                out.extend(f"<0x{b:02X}>" for b in ch.encode("utf-8"))
            else:
                out.append(self.unk)
        return out

    def detokenize(self, tokens: List[str]) -> str:
        out, i = [], 0
        while i < len(tokens):
            if tokens[i].startswith("<0x") and tokens[i].endswith(">"):
                buf = bytearray()
                while (i < len(tokens) and tokens[i].startswith("<0x")
                       and tokens[i].endswith(">")):
                    buf.append(int(tokens[i][3:-1], 16))
                    i += 1
                out.append(buf.decode("utf-8", errors="replace"))
            else:
                out.append(tokens[i])
                i += 1
        return "".join(out)

    # ---- id level ------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        return [self.char2id.get(t, self._byte_id(t)) for t in self.tokenize(text)]

    def _byte_id(self, tok: str) -> int:
        if tok == self.unk:
            return 0
        return 0x110000 + int(tok[3:-1], 16)         # ids beyond all code points

    def decode(self, ids: List[int]) -> str:
        toks = []
        for i in ids:
            if i in self.id2char:
                toks.append(self.id2char[i])
            elif i >= 0x110000:
                toks.append(f"<0x{i - 0x110000:02X}>")
            else:
                toks.append(self.unk)
        return self.detokenize(toks)

    def vocab_report(self) -> Dict:
        letters = sum(c.isalpha() for c in self.char_freq)
        digits = sum(c.isdigit() for c in self.char_freq)
        spaces = sum(c.isspace() for c in self.char_freq)
        return {"vocab_size": len(self.char2id), "letters": letters,
                "digits": digits, "whitespace": spaces,
                "punct_other": len(self.char_freq) - letters - digits - spaces,
                "top5": [c for c, _ in self.char_freq.most_common(5)]}


class MiniBPETokenizer:
    """Pure-Python BPE (~30 lines) so the three-granularity comparison needs
    zero installs. Same greedy-merge algorithm as Scenario 4's tokenizers-
    library version — minus the speed. Simplification: \\S+ words keep their
    punctuation attached."""

    def __init__(self, n_merges: int = 120):
        self.n_merges = n_merges
        self.merges: List[Tuple[str, str]] = []
        self.vocab: set = set()

    def fit(self, corpus: List[str]) -> "MiniBPETokenizer":
        word_freq = Counter()
        for text in corpus:
            word_freq.update(re.findall(r"\S+", text))
        base = {ch for w in word_freq for ch in w}
        units = [(list(w) + ["</w>"], f) for w, f in word_freq.items()]
        for _ in range(self.n_merges):
            pairs = Counter()
            for syms, f in units:
                for j in range(len(syms) - 1):
                    pairs[(syms[j], syms[j + 1])] += f
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            units = [(self._merge(syms, best), f) for syms, f in units]
        self.vocab = (base | {"</w>"} | {a + b for a, b in self.merges}
                      | {s for syms, _ in units for s in syms})
        return self

    @staticmethod
    def _merge(syms: List[str], pair: Tuple[str, str]) -> List[str]:
        out, i = [], 0
        while i < len(syms):
            if i < len(syms) - 1 and (syms[i], syms[i + 1]) == pair:
                out.append(syms[i] + syms[i + 1])
                i += 2
            else:
                out.append(syms[i])
                i += 1
        return out

    def tokenize(self, text: str) -> List[str]:
        out = []
        for w in re.findall(r"\S+", text):
            syms = list(w) + ["</w>"]
            for pair in self.merges:                 # apply merges in learned order
                syms = self._merge(syms, pair)
            out.extend(syms)
        return out


# ============================= DATA / ATTACKS ================================
class TypoSimulator:
    """Deterministic typo & obfuscation attacks (no randomness needed)."""

    LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5"}

    @classmethod
    def leetspeak(cls, text: str) -> str:
        return "".join(cls.LEET.get(c.lower(), c) for c in text)

    @staticmethod
    def swap_adjacent(word: str, i: int = 1) -> str:
        if len(word) <= i + 1:
            i = 0
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]

    @staticmethod
    def drop_char(word: str, i: int = 2) -> str:
        i = min(i, len(word) - 1)
        return word[:i] + word[i + 1:]


class SpamDataset:
    """Train/test/adversarial split + a hook to the real UCI corpus."""

    def __init__(self, spam: List[str], ham: List[str], n_test: int = 3):
        self.spam_train, self.spam_test = spam[:-n_test], spam[-n_test:]
        self.ham_train, self.ham_test = ham[:-n_test], ham[-n_test:]
        self.train_texts = self.spam_train + self.ham_train
        self.train_labels = (["spam"] * len(self.spam_train)
                             + ["ham"] * len(self.ham_train))
        self.test_texts = self.spam_test + self.ham_test
        self.test_labels = ["spam"] * n_test + ["ham"] * n_test
        self.adversarial_texts = [TypoSimulator.leetspeak(t)
                                  for t in self.test_texts]
        self.adversarial_labels = list(self.test_labels)

    @staticmethod
    def sample() -> "SpamDataset":
        return SpamDataset(SPAM, HAM)

    @staticmethod
    def from_uci(path: str = "SMSSpamCollection",
                 n_test: int = 300) -> "SpamDataset":
        """Real data: UCI SMS Spam Collection (5,574 messages).
        One message per line: '<ham|spam>\\t<message>'."""
        spam, ham = [], []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                label, _, msg = line.rstrip("\n").partition("\t")
                (spam if label == "spam" else ham).append(msg)
        return SpamDataset(spam, ham, n_test)


# ============================= ANALYSIS ======================================
class GranularityComparator:
    """The card's description field made concrete: the SAME text at word,
    subword, and character granularity — vocab size, token count, and OOV
    behavior under attack for each."""

    WORD_RE = re.compile(r"[a-z0-9]+")

    def __init__(self, train_texts: List[str], n_merges: int = 120):
        self.char_tok = CharacterTokenizer().fit(train_texts)
        self.bpe = MiniBPETokenizer(n_merges).fit(train_texts)
        self.word_vocab = {w for t in train_texts
                           for w in self.WORD_RE.findall(t.lower())}

    def tokenize_words(self, text: str) -> List[str]:
        return [w if w in self.word_vocab else "<UNK>"
                for w in self.WORD_RE.findall(text.lower())]

    def example(self, text: str) -> Dict:
        return {"text": text,
                "word": self.tokenize_words(text),
                "subword": self.bpe.tokenize(text),
                "char": self.char_tok.tokenize(text)}

    def report(self, texts: List[str]) -> List[Dict]:
        rows = []
        for name, tokens, vocab in (
                ("word", [t for x in texts for t in self.tokenize_words(x)],
                 self.word_vocab | {"<UNK>"}),
                ("subword", [t for x in texts for t in self.bpe.tokenize(x)],
                 self.bpe.vocab),
                ("character", [t for x in texts for t in self.char_tok.tokenize(x)],
                 set(self.char_tok.char2id))):
            n = len(tokens) or 1
            rows.append({"granularity": name,
                         "tokens_per_msg": round(n / len(texts), 1),
                         "vocab_size": len(vocab),
                         "oov_rate": round(sum(t not in vocab for t in tokens) / n, 3)})
        return rows


class TypoRobustnessBenchmark:
    """Recover attacked words two ways: exact word lookup (fails) vs nearest
    neighbor by char-bigram Jaccard similarity (works)."""

    CLEAN_WORDS = ["free", "winner", "claim", "prize", "mobile",
                   "account", "urgent", "congrats", "txt", "selected"]

    @staticmethod
    def _ngrams(s: str, n: int = 2) -> set:
        s = f"^{s}$"
        return {s[i:i + n] for i in range(len(s) - n + 1)}

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        return len(a & b) / len(a | b) if (a | b) else 0.0

    def run(self) -> List[Dict]:
        self.rows = []
        for w in self.CLEAN_WORDS:
            variants = [("leetspeak", TypoSimulator.leetspeak(w)),
                        ("swap", TypoSimulator.swap_adjacent(w)),
                        ("drop", TypoSimulator.drop_char(w))]
            for attack, v in variants:
                best, best_sim = None, -1.0
                for cand in self.CLEAN_WORDS:
                    sim = self._jaccard(self._ngrams(v), self._ngrams(cand))
                    if sim > best_sim:
                        best, best_sim = cand, sim
                self.rows.append({"clean": w, "attack": attack, "variant": v,
                                  "word_match": v if v in self.CLEAN_WORDS else None,
                                  "ngram_match": best, "sim": round(best_sim, 2)})
        return self.rows

    def recall(self) -> Dict:
        n = len(self.rows)
        return {"word_level_exact":
                    round(sum(r["word_match"] == r["clean"] for r in self.rows) / n, 2),
                "char_bigram_nn":
                    round(sum(r["ngram_match"] == r["clean"] for r in self.rows) / n, 2)}


# ============================= CLASSIFIERS ===================================
class WordFeaturizer:
    """Lowercased words; unseen words collapse to <UNK>.
    NOTE: '<UNK>' never occurs in training, so on attacked text nearly every
    feature contributes the SAME smoothed score to both classes — the word
    model effectively 'goes deaf' and falls back to the prior."""

    TOKEN_RE = re.compile(r"[a-z0-9]+")

    def __init__(self):
        self.vocab = set()

    def fit(self, texts: List[str]) -> "WordFeaturizer":
        self.vocab = {w for t in texts for w in self.TOKEN_RE.findall(t.lower())}
        return self

    def transform(self, text: str) -> List[str]:
        return [w if w in self.vocab else "<UNK>"
                for w in self.TOKEN_RE.findall(text.lower())]


class CharNgramFeaturizer:
    """Character n-grams over the RAW text (case preserved — ALL-CAPS is a
    spam signal a char tokenizer captures for free), ^/$ boundary marks.
    Cross-word n-grams also catch phrase patterns like 'Txt ' and ' to 8'."""

    def __init__(self, n: int = 3):
        self.n = n

    def fit(self, texts: List[str]) -> "CharNgramFeaturizer":
        return self                                  # stateless

    def transform(self, text: str) -> List[str]:
        s = "^" * (self.n - 1) + text + "$" * (self.n - 1)
        return [s[i:i + self.n] for i in range(len(s) - self.n + 1)]


class NaiveBayesTextClassifier:
    """Multinomial NB over whatever features a featurizer emits.
    Strategy pattern: ONE classifier, TWO featurizers (word vs char n-gram)."""

    def __init__(self, featurizer, alpha: float = 0.5, name: str = "nb"):
        self.featurizer = featurizer
        self.alpha = alpha
        self.name = name

    def fit(self, texts: List[str], labels: List[str]) -> "NaiveBayesTextClassifier":
        self.featurizer.fit(texts)
        self.classes = sorted(set(labels))
        self.feature_counts = {c: Counter() for c in self.classes}
        self.class_totals = {c: 0 for c in self.classes}
        self.class_docs = Counter(labels)
        self.n_docs = len(labels)
        for text, label in zip(texts, labels):
            feats = self.featurizer.transform(text)
            self.feature_counts[label].update(feats)
            self.class_totals[label] += len(feats)
        self.vocab = set().union(*self.feature_counts.values())
        self.V = max(len(self.vocab), 1)
        return self

    def _log_score(self, text: str, c: str) -> float:
        logp = math.log(self.class_docs[c] / self.n_docs)
        denom = self.class_totals[c] + self.alpha * self.V
        for f in self.featurizer.transform(text):
            logp += math.log(
                (self.feature_counts[c].get(f, 0) + self.alpha) / denom)
        return logp

    def predict(self, text: str) -> Tuple[str, Dict[str, float]]:
        scores = {c: self._log_score(text, c) for c in self.classes}
        return max(scores, key=scores.get), scores

    def evaluate(self, texts: List[str], labels: List[str]) -> Dict:
        rows, correct = [], 0
        for t, gold in zip(texts, labels):
            pred, scores = self.predict(t)
            ok = pred == gold
            correct += ok
            a, b = self.classes
            rows.append({"gold": gold, "pred": pred, "ok": ok,
                         "margin": round(scores[b] - scores[a], 1)})
        return {"accuracy": round(correct / len(labels), 2), "rows": rows}

    def top_features(self, k: int = 8) -> List[str]:
        """Most discriminative features (binary case)."""
        def lp(f, c):
            return math.log((self.feature_counts[c].get(f, 0) + self.alpha)
                            / (self.class_totals[c] + self.alpha * self.V))
        a, b = self.classes
        return sorted(self.vocab,
                      key=lambda f: abs(lp(f, a) - lp(f, b)), reverse=True)[:k]


# ============================= CHARACTER LM ==================================
class CharacterLanguageModel:
    """Character trigram LM with Laplace smoothing — a genuine (tiny)
    language model whose TOKENS are characters. The pure-Python ancestor of
    byte-level GPT: same token granularity, minus a few billion parameters."""

    def __init__(self, order: int = 3, alpha: float = 0.5):
        self.k = order - 1                           # context length
        self.alpha = alpha

    def fit(self, texts: List[str]) -> "CharacterLanguageModel":
        self.counts: Dict[str, Counter] = defaultdict(Counter)
        self.context_total = Counter()
        self.vocab = set()
        for text in texts:
            s = "^" * self.k + text + "$"
            for i in range(len(s) - self.k):
                ctx, nxt = s[i:i + self.k], s[i + self.k]
                self.counts[ctx][nxt] += 1
                self.context_total[ctx] += 1
                self.vocab.add(nxt)
        self.V = max(len(self.vocab), 1)
        return self

    def perplexity(self, text: str) -> float:
        """exp(mean NLL) over character tokens."""
        s = "^" * self.k + text + "$"
        n, logp = 0, 0.0
        for i in range(len(s) - self.k):
            ctx, nxt = s[i:i + self.k], s[i + self.k]
            cnt = self.counts.get(ctx, {}).get(nxt, 0)
            p = (cnt + self.alpha) / (self.context_total[ctx]
                                      + self.alpha * self.V)
            logp += math.log(p)
            n += 1
        return math.exp(-logp / n)

    def generate(self, seed: str, n_chars: int = 100,
                 temperature: float = 0.8, rng: random.Random = None) -> str:
        rng = rng or random.Random(0)
        out = list(seed)
        ctx = ("^" * self.k + seed)[-self.k:]
        for _ in range(n_chars):
            counts = self.counts.get(ctx)
            if not counts:
                ctx = "^" * self.k                    # restart on unseen context
                continue
            chars, weights = zip(*counts.items())
            if temperature != 1.0:
                weights = tuple(w ** (1.0 / temperature) for w in weights)
            ch = rng.choices(chars, weights=weights, k=1)[0]
            if ch == "$":                             # message boundary
                out.append("\n")
                ctx = "^" * self.k
                continue
            out.append(ch)
            ctx = (ctx + ch)[-self.k:]
        return "".join(out)


# ============================= END-TO-END =====================================
class SpamFilterPipeline:
    """Caps the scenario: char tokenizer -> char n-gram NB -> char-LM
    perplexity as a secondary signal, behind one moderate() API."""

    def __init__(self, classifier: NaiveBayesTextClassifier,
                 lm: CharacterLanguageModel):
        self.clf, self.lm = classifier, lm

    def moderate(self, message: str) -> Dict:
        label, scores = self.clf.predict(message)
        a, b = self.clf.classes
        return {"message": message[:48], "label": label,
                "log_odds_spam": round(scores[b] - scores[a], 1),
                "char_lm_pp": round(self.lm.perplexity(message), 1),
                "action": "BLOCK" if label == b else "DELIVER"}


if __name__ == "__main__":
    ds = SpamDataset.sample()
    print(f"Dataset: train={len(ds.train_texts)} "
          f"(spam={len(ds.spam_train)}, ham={len(ds.ham_train)}) "
          f"test={len(ds.test_texts)} adversarial={len(ds.adversarial_texts)}")
    # ds = SpamDataset.from_uci()                    # <- real 5,574-msg corpus

    print("\n--- A. CharacterTokenizer: closed vocab + byte fallback ---")
    ct = CharacterTokenizer(byte_fallback=True).fit(ds.train_texts)
    ct_unk = CharacterTokenizer(byte_fallback=False).fit(ds.train_texts)
    print("  vocab report:", ct.vocab_report())
    probe = "WINNER! ☕"                              # emoji never seen in training
    print(f"  tokens for {probe!r}: {ct.tokenize(probe)}")
    print(f"  byte-fallback round trip exact: {ct.decode(ct.encode(probe)) == probe}")
    print(f"  UNK-mode round trip:            "
          f"{ct_unk.detokenize(ct_unk.tokenize(probe))!r}")

    print("\n--- B. One text, three granularities "
          "(card: 'words, subwords, or characters') ---")
    gc = GranularityComparator(ds.train_texts)
    ex = gc.example(ds.test_texts[0])
    print(f"  text: {ex['text']!r}")
    print(f"  word     ({len(ex['word']):2d}): {ex['word'][:11]}")
    print(f"  subword  ({len(ex['subword']):2d}): {ex['subword'][:14]}...")
    print(f"  char     ({len(ex['char']):2d}): {ex['char'][:16]}...")
    for tag, texts in (("clean test", ds.test_texts),
                       ("leetspeak attack", ds.adversarial_texts)):
        print(f"  [{tag}]")
        for row in gc.report(texts):
            print(f"    {row['granularity']:10} tokens/msg={row['tokens_per_msg']:5} "
                  f"vocab={row['vocab_size']:4}  oov_rate={row['oov_rate']}")

    print("\n--- C. Typo recovery: word lookup vs char n-gram similarity ---")
    bench = TypoRobustnessBenchmark()
    for r in bench.run()[:6]:
        print(f"  {r['variant']:10} ({r['attack']:9}) word_lookup="
              f"{str(r['word_match']):6} ngram_match={r['ngram_match']:8} "
              f"sim={r['sim']}")
    print("  recall:", bench.recall())

    print("\n--- D. Spam classifier: word features vs char n-gram features ---")
    word_nb = NaiveBayesTextClassifier(WordFeaturizer(), name="word-NB")
    char_nb = NaiveBayesTextClassifier(CharNgramFeaturizer(n=3),
                                       name="char-trigram-NB")
    for model in (word_nb, char_nb):
        model.fit(ds.train_texts, ds.train_labels)
    print("  top discriminative features:")
    print(f"    word : {word_nb.top_features(8)}")
    print(f"    char : {char_nb.top_features(8)}")
    print(f"  attack example: {ds.adversarial_texts[0]!r}")
    for tag, texts, labels in (("clean", ds.test_texts, ds.test_labels),
                               ("leetspeak attack", ds.adversarial_texts,
                                ds.adversarial_labels)):
        print(f"  [{tag}]")
        for model in (word_nb, char_nb):
            r = model.evaluate(texts, labels)
            print(f"    {model.name:16} accuracy={r['accuracy']:.2f}")
            for row in r["rows"]:
                if not row["ok"]:
                    print(f"      miss: gold={row['gold']} pred={row['pred']} "
                          f"log_odds={row['margin']}")

    print("\n--- E. Character trigram LM trained on spam only ---")
    lm = CharacterLanguageModel(order=3, alpha=0.5).fit(ds.spam_train)
    print("  generated (seed='WINNER', temp=0.8):")
    print("   ", lm.generate("WINNER", n_chars=110,
                             rng=random.Random(7)).replace("\n", "\n    "))
    for tag, text in (("spam test", ds.test_texts[0]),
                      ("ham test", ds.test_texts[3]),
                      ("leetspeak spam", ds.adversarial_texts[0])):
        print(f"  perplexity[{tag:15}] = {lm.perplexity(text):6.1f}")

    print("\n--- F. End-to-end: SpamFilterPipeline.moderate() ---")
    pipe = SpamFilterPipeline(char_nb, lm)
    for msg in INBOX:
        r = pipe.moderate(msg)
        print(f"  {r['action']:8} label={r['label']:4} "
              f"log_odds={r['log_odds_spam']:>7} lm_pp={r['char_lm_pp']:>6} "
              f" {r['message']!r}")
