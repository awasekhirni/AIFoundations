"""
#  Copyright 2025 β ORI Inc.Canada All Rights Reserved.
#  Author: Awase Khirni Syed
#  AlphaFactory Bootcamps: β ORI Inc. AI Foundations - NLP Techniques Part-I
#  Version: 1.2
#  Date: 2025-01-25
#  technical card 01: Tokenization
# scenario5_tiktoken_llm.py
# Count tokens the way the model does, price prompts, and chunk documents
# into token-budgeted pieces for RAG ingestion.
# to compute the cost
# well i gave this example with gpt, i prefer chineese models, they are pioneering and open source deepseek, qwen,z, kimi,
"""
import re
from typing import Dict, List
import tiktoken


class LLMBudgetManager:
    """Counts tokens exactly like the target model, and prices them."""

    PRICING = {  # USD per 1M tokens (illustrative — check current prices)
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o":      {"input": 2.50, "output": 10.00},
    }

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        try:
            self.enc = tiktoken.encoding_for_model(model)
        except KeyError:
            self.enc = tiktoken.get_encoding("o200k_base")

    def n_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    def stats(self, text: str) -> Dict:
        toks, words = self.n_tokens(text), len(text.split())
        return {"model": self.model, "words": words, "tokens": toks,
                "tokens_per_word": round(toks / max(words, 1), 2),
                "est_input_cost_usd": round(toks / 1e6 * self.PRICING[self.model]["input"], 6)}


class TokenAwareChunker:
    """Splits a document into chunks that each fit a token budget —
    the standard pre-processing step before RAG indexing or LLM calls."""

    def __init__(self, encoder, max_tokens: int = 50):
        self.enc = encoder
        self.max_tokens = max_tokens

    def _n(self, text: str) -> int:
        return len(self.enc.encode(text))

    def chunk(self, document: str) -> List[Dict]:
        sentences = re.split(r"(?<=[.!?])\s+", document.strip())
        chunks, current, cur_tok = [], [], 0

        def flush():
            nonlocal current, cur_tok
            if current:
                chunks.append(" ".join(current))
                current, cur_tok = [], 0

        for sent in sentences:
            stoks = self._n(sent)
            if cur_tok + stoks <= self.max_tokens:
                current.append(sent); cur_tok += stoks; continue
            flush()
            if stoks <= self.max_tokens:
                current.append(sent); cur_tok = stoks; continue
            for word in sent.split():               # oversized sentence: word split
                wtoks = self._n(word) + 1           # +1 ≈ space token
                if cur_tok + wtoks > self.max_tokens:
                    flush()
                current.append(word); cur_tok += wtoks
        flush()
        return [{"chunk_id": i, "n_tokens": self._n(c), "text": c}
                for i, c in enumerate(chunks, 1)]


class RAGPromptBuilder:
    """Assembles a retrieval prompt and GUARANTEES it fits the budget."""

    TEMPLATE = ("You are an assistant. Answer ONLY from the context.\n\n"
                "Context:\n{context}\n\nQuestion: {question}\nAnswer:")

    def __init__(self, budget: LLMBudgetManager, max_prompt_tokens: int = 300):
        self.budget = budget
        self.max_prompt_tokens = max_prompt_tokens

    def build(self, question: str, chunks: List[Dict]) -> Dict:
        reserved = self.budget.n_tokens(
            self.TEMPLATE.format(context="", question=question))
        room = self.max_prompt_tokens - reserved
        parts, used = [], 0
        for c in chunks:
            if used + c["n_tokens"] > room:
                break
            parts.append(c["text"]); used += c["n_tokens"]
        prompt = self.TEMPLATE.format(
            context="\n---\n".join(parts) or "(no context)", question=question)
        n = self.budget.n_tokens(prompt)
        return {"prompt": prompt, "n_tokens": n, "fits": n <= self.max_prompt_tokens,
                "chunks_used": len(parts)}


# ----------------------------- SAMPLE DATA ---------------------------------
KNOWLEDGE_DOC = (
    "Remote Work Policy. Employees may work remotely up to three days per week. "
    "Remote work requests must be approved by the direct manager. Core hours "
    "are 10:00 to 16:00 in the employee's local time zone. Team meetings "
    "should be scheduled during core hours whenever possible. Equipment: the "
    "company provides a laptop and a monitor for home use. Security: "
    "employees must use the corporate VPN when accessing internal systems. "
    "Data protection rules apply to printed documents at home. Reimbursement: "
    "home internet costs are reimbursed up to 40 dollars per month with a "
    "valid receipt. Travel to the office is required once per quarter for "
    "team on-site weeks. Managers review remote work arrangements annually."
)
MULTILINGUAL_SAMPLES = [
    "The café serves 100 cups of coffee daily.",
    "Le café sert 100 tasses de café par jour.",
    "咖啡店每天卖100杯咖啡。",
    "Coffee ☕ costs $5.00 — cheap!",
]



#-------------------Assignment --- please try with urdu/arabic/farsi ----------
# ------------------Assignment ----please try with hindi/marathi/telugu/malyalam/tamil ----------------
# ----------------- it would be interesting to see for malyalam-----------------------

if __name__ == "__main__":
    budget = LLMBudgetManager("gpt-4o-mini")

    print("--- words vs tokens (same meaning, different cost!) ---")
    for s in MULTILINGUAL_SAMPLES:
        st = budget.stats(s)
        print(f"  {st['words']:3d} words -> {st['tokens']:3d} tokens | {s}")

    print("\n--- token-aware chunking (budget = 50 tokens/chunk) ---")
    chunker = TokenAwareChunker(budget.enc, max_tokens=50)
    for c in chunker.chunk(KNOWLEDGE_DOC):
        print(f"  chunk {c['chunk_id']}: {c['n_tokens']:3d} tokens | {c['text'][:58]}...")

    print("\n--- RAG prompt within budget ---")
    built = RAGPromptBuilder(budget, 300).build(
        "How many remote days are allowed per week?", chunker.chunk(KNOWLEDGE_DOC))
    print(f"  chunks_used={built['chunks_used']} n_tokens={built['n_tokens']} "
          f"fits={built['fits']}")
    print("  preview:", built["prompt"][:100].replace("\n", " ") + "...")
