"""
#  Copyright 2025 β ORI Inc.Canada All Rights Reserved.
#  Author: Awase Khirni Syed
#  AlphaFactory Bootcamps: β ORI Inc. AI Foundations - NLP Techniques Part-I
#  Version: 1.2
#  Date: 2025-01-25
#  technical card 01: Tokenization
# scenario2_nltk_news.py
# Sentence- and word-level tokenization for a news pipeline.
# this focus on word and sentence tokenization
# nltk.corpus.gutenbuerg
# pip3 install nltk
# reuters news corpus - in 2009 my master's student Dr.Imaduddin Humayan worked on Reuters News Corpus data @IIIT-Bangalore (NER/GNER) named entity recognition and geographic named entity recognition
# #lets take a basic example of how we analyze news articles
"""
import nltk
from nltk.tokenize import (RegexpTokenizer, TreebankWordTokenizer,
                           sent_tokenize, word_tokenize)
from typing import Dict, List


class NLTKMultiTokenizer:
    """One wrapper, several NLTK tokenization strategies."""

    def __init__(self):
        self._ensure_resources()
        self.treebank = TreebankWordTokenizer()
        self.regex_tok = RegexpTokenizer(r"\w+(?:[-']\w+)*|[^\w\s]")

    @staticmethod
    def _ensure_resources():
        for path, pkg in [("tokenizers/punkt", "punkt"),
                          ("tokenizers/punkt_tab", "punkt_tab")]:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(pkg, quiet=True)

    def sentences(self, text: str) -> List[str]:
        return sent_tokenize(text)

    def words(self, text: str) -> List[str]:
        return word_tokenize(text)

    def words_treebank(self, text: str) -> List[str]:
        return self.treebank.tokenize(text)

    def words_alnum(self, text: str) -> List[str]:
        return self.regex_tok.tokenize(text)

    def word_spans(self, text: str) -> List[tuple]:
        """Tokens with character offsets — needed when downstream tools
        (NER, highlighters) must map tokens back to original text."""
        return list(self.regex_tok.span_tokenize(text))


class NewsArticleAnalyzer:
    """Sentence + word tokenization for a news ingestion pipeline."""

    def __init__(self):
        self.tk = NLTKMultiTokenizer()

    def analyze(self, article: Dict[str, str]) -> Dict:
        sentences = self.tk.sentences(article["body"])
        rows = [{"sent_id": i, "text": s, "n_tokens": len(self.tk.words(s))}
                for i, s in enumerate(sentences, 1)]
        words = [t for t in self.tk.words(article["body"])
                 if any(c.isalnum() for c in t)]
        return {
            "article_id": article["id"],
            "n_sentences": len(sentences),
            "n_words": len(words),
            "avg_words_per_sentence": round(len(words) / len(sentences), 1),
            "sentences": rows,
        }

    def strategy_comparison(self, sentence: str) -> Dict[str, List[str]]:
        return {"punkt": self.tk.words(sentence),
                "treebank": self.tk.words_treebank(sentence),
                "regex_alnum": self.tk.words_alnum(sentence)}


# ----------------------------- SAMPLE DATA ---------------------------------
ARTICLE = {
    "id": "NW-2024-0613-01",
    "headline": "Tech shares rally as inflation cools",
    "body": (
        "Technology shares rallied on Friday after new data showed U.S. "
        "inflation slowing to 3.2%, its lowest level since early 2024. The "
        "Nasdaq Composite climbed 1.8% to 16,842.55, while the S&P 500 gained "
        "0.9%. Dr. Elena Marquez, chief economist at Hartwell Capital, called "
        "the report 'a clear signal that rate cuts are back on the table.' "
        "Investors now expect the Federal Reserve to lower rates by September. "
        "Trading volume reached 11.2 million shares, up from 9.8 million last "
        "week. Analysts remain cautious. Mr. Tanaka of Pacific Securities "
        "noted that wage growth is still running hot at 4.1% year over year."
    ),
}
# Real public-domain alternative (uncomment):
# nltk.download("gutenberg"); from nltk.corpus import gutenberg
# ARTICLE["body"] = gutenberg.raw("austen-emma.txt")[:2000]

TRICKY = ("Dr. O'Neill bought 1,000 shares at $42.50 — a 3.2% discount — "
          "before Q3 earnings.")

if __name__ == "__main__":
    analyzer = NewsArticleAnalyzer()
    result = analyzer.analyze(ARTICLE)

    print(f"Article {result['article_id']}: sentences={result['n_sentences']} "
          f"words={result['n_words']} avg_len={result['avg_words_per_sentence']}")
    for s in result["sentences"]:
        print(f"  S{s['sent_id']:02d} [{s['n_tokens']:2d} tk] {s['text'][:62]}...")

    print("\nStrategy comparison on:", TRICKY)
    for name, toks in analyzer.strategy_comparison(TRICKY).items():
        print(f"  {name:11} -> {toks}")

    print("\nToken spans:", analyzer.tk.word_spans("GPT-4 costs $20.00 per 1M tokens.")[:8])
