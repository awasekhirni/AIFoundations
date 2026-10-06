"""
#  Copyright 2025 β ORI Inc.Canada All Rights Reserved.
#  Author: Awase Khirni Syed
#  AlphaFactory Bootcamps: β ORI Inc. AI Foundations - NLP Techniques Part-I
#  Version: 1.2
#  Date: 2025-01-25
#  technical card 01: Tokenization
# scenario1_regex_reviews.py
# Rule-based tokenization of messy customer reviews.
"""
import re
from collections import Counter
from typing import Dict, List


class RegexTokenizer:
    """
    Pure rule-based tokenizer. Pattern ORDER matters: high-information tokens
    (URLs, emails, hashtags, contractions, prices) are matched BEFORE plain
    words so they aren't shredded into meaningless fragments.
    """

    KEEP_TYPES = {"URL", "EMAIL", "HASHTAG", "MENTION", "CONTRACTION",
                  "CURRENCY", "NUMBER", "WORD", "EMOJI"}

    _PATTERN = re.compile(
        r"""
          (?P<URL>          https?://\S+ | www\.\S+ )
        | (?P<EMAIL>        [\w.+-]+@[\w-]+\.[\w.]+ )
        | (?P<HASHTAG>      \#\w+ )
        | (?P<MENTION>      @\w+ )
        | (?P<CONTRACTION>  \w+['’](?:t|re|ve|ll|s|d|m) | \w+n['’]t )
        | (?P<CURRENCY>     \$\d+(?:[.,]\d+)* )
        | (?P<NUMBER>       \d+(?:[.,]\d+)*%? )
        | (?P<WORD>         [A-Za-z]+ )
        | (?P<PUNCT>        [!?.,;:()\-'"] )
        | (?P<EMOJI>        [\U0001F300-\U0001FAFF\u2600-\u27BF] )
        """,
        re.VERBOSE,
    )

    def tokenize(self, text: str) -> List[str]:
        return [m.group() for m in self._PATTERN.finditer(text)]

    def tokenize_annotated(self, text: str) -> List[Dict]:
        """Every token emitted WITH its type and character span."""
        return [
            {"token": m.group(), "type": m.lastgroup, "span": (m.start(), m.end())}
            for m in self._PATTERN.finditer(text)
        ]


class ReviewAnalysisPipeline:
    """End-to-end: raw reviews --> typed tokens --> corpus statistics. etc..."""

    def __init__(self, tokenizer: RegexTokenizer):
        self.tokenizer = tokenizer
        self.reviews: List[Dict] = []

    def load(self, reviews: List[Dict]) -> "ReviewAnalysisPipeline":
        self.reviews = reviews
        return self

    def process(self) -> List[Dict]:
        results = []
        for rev in self.reviews:
            annotated = self.tokenizer.tokenize_annotated(rev["text"])
            tokens = [a["token"] for a in annotated
                      if a["type"] in RegexTokenizer.KEEP_TYPES]
            results.append({
                "review_id": rev["review_id"],
                "rating": rev["rating"],
                "tokens": tokens,
                "n_tokens": len(tokens),
                "n_urls": sum(a["type"] == "URL" for a in annotated),
                "n_hashtags": sum(a["type"] == "HASHTAG" for a in annotated),
                "avg_token_len": round(sum(map(len, tokens)) / len(tokens), 2),
            })
        return results

    def vocabulary_report(self, processed: List[Dict], top_n: int = 6) -> Dict:
        all_tokens = [t for r in processed for t in r["tokens"]]
        freq = Counter(t.lower() for t in all_tokens)
        return {
            "total_tokens": len(all_tokens),
            "vocabulary_size": len(freq),
            "type_token_ratio": round(len(freq) / len(all_tokens), 3),
            "most_common": freq.most_common(top_n),
        }


# ----------------------------- SAMPLE INPUT DATA ---------------------------------
SAMPLE_REVIEWS = [
    {"review_id": "R001", "rating": 5,
     "text": "Absolutely LOVE this blender! Bought it for $49.99 and it's worth "
             "every penny. Visit https://example.com/deals for more. 10/10 "
             "would recommend!!!"},
    {"review_id": "R002", "rating": 2,
     "text": "Arrived broken (cracked lid). Customer service didn't respond to "
             "my email me@example.com for 3 weeks. #disappointed"},
    {"review_id": "R003", "rating": 4,
     "text": "Solid quality for the price. It blends frozen fruit in ~30 "
             "seconds. 4 stars, would've been 5 if the lid fit tighter."},
    {"review_id": "R004", "rating": 1,
     "text": "Stopped working after 2 days 😡 total waste of $49.99. "
             "@manufacturer please fix your QC!"},
]

#-----------------------------ASSINGMENT - READ THE REVIEWS ON REDDIT ABOUT R/REVIEW AND PROCESS THE DATA ----------------
# ----------------------------ASSIGNMENT - READ THE REVIEWS ON SHEIN AND PROCESS THE DATA -------------------------

if __name__ == "__main__":
    tok = RegexTokenizer()

    print("Annotated tokens — review R004:")
    for a in tok.tokenize_annotated(SAMPLE_REVIEWS[3]["text"]):
        print(f"  {a['token']!r:16} {a['type']:<12} span={a['span']}")

    pipeline = ReviewAnalysisPipeline(tok).load(SAMPLE_REVIEWS)
    processed = pipeline.process()

    print("\nPer-review summary:")
    for r in processed:
        print(f"  {r['review_id']} rating={r['rating']} tokens={r['n_tokens']:2d} "
              f"urls={r['n_urls']} hashtags={r['n_hashtags']} "
              f"avg_len={r['avg_token_len']}")

    print("\nFirst 12 tokens of R001:", processed[0]["tokens"][:12])

    print("\nVocabulary report:")
    for k, v in pipeline.vocabulary_report(processed).items():
        print(f"  {k:18}: {v}")
