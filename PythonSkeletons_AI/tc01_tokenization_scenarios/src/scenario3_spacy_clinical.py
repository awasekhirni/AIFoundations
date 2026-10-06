"""
#  Copyright 2025 β ORI Inc.Canada All Rights Reserved.
#  Author: Awase Khirni Syed
#  AlphaFactory Bootcamps: β ORI Inc. AI Foundations - NLP Techniques Part-I
#  Version: 1.2
#  Date: 2025-01-25
#  technical card 01: Tokenization
# scenario3_spacy_clinical.py
# Token attributes, custom special cases (dosages), and mapping regex PHI
# spans back onto spaCy tokens for de-identification.
# health care data.  Medical Discharge Summary Processing  - Processing medical records - I did some prototype work on automating medical record recognition
"""
import re
import spacy
from spacy.symbols import ORTH, NORM
from typing import Dict, List


class ClinicalNoteTokenizer:
    """spaCy tokenizer + domain special cases + PHI span detection."""

    SPECIAL_CASES = {  # keep clinical shorthand as single, normalized tokens
        "500mg":  [{ORTH: "500mg", NORM: "500 milligrams"}],
        "1000mg": [{ORTH: "1000mg", NORM: "1000 milligrams"}],
        "10mg":   [{ORTH: "10mg", NORM: "10 milligrams"}],
        "q12h":   [{ORTH: "q12h", NORM: "every 12 hours"}],
        "BID":    [{ORTH: "BID", NORM: "twice a day"}],
        "TID":    [{ORTH: "TID", NORM: "three times a day"}],
    }

    PHI_PATTERNS = {
        "MRN":  r"\bMRN[:\s]*\d{6,10}\b",
        "DATE": r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        "TIME": r"\b\d{1,2}:\d{2}\b",
    }

    def __init__(self, model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"[warn] '{model}' not installed — using blank 'en' (no POS).")
            self.nlp = spacy.blank("en")
        for text, spec in self.SPECIAL_CASES.items():
            self.nlp.tokenizer.add_special_case(text, spec)

    def token_table(self, text: str) -> List[Dict]:
        return [{
            "text": t.text, "norm": t.norm_, "pos": t.pos_,
            "shape": t.shape_, "like_num": t.like_num, "is_stop": t.is_stop,
        } for t in self.nlp(text)]

    def detect_phi(self, text: str) -> List[Dict]:
        """Regex hits on raw text, mapped onto token spans via char_span()."""
        doc = self.nlp(text)
        hits = []
        for label, pattern in self.PHI_PATTERNS.items():
            for m in re.finditer(pattern, text):
                span = doc.char_span(m.start(), m.end(), label=label)
                if span is not None:
                    hits.append({"label": label, "text": span.text,
                                 "token_texts": [t.text for t in span]})
        return hits

    def redact(self, text: str) -> str:
        for label, pattern in self.PHI_PATTERNS.items():
            text = re.sub(pattern, f"[{label}]", text)
        return text


# ----------------------------- SAMPLE DATA (synthetic — never real PHI) ----
CLINICAL_NOTE = (
    "Discharge Summary — Patient: J. Doe, MRN 8837291, DOB 03/15/1962. "
    "Admitted 12/03/2023 with hyperglycemia (HbA1c 9.2%). Meds: metformin "
    "500mg BID, lisinopril 10mg once daily. Patient reported nausea and "
    "dizziness at 14:30 on 12/04/2023. Fasting glucose 238 mg/dL. Plan: "
    "continue metformin 500mg q12h, follow up in 2 weeks. Dx: Type 2 "
    "Diabetes Mellitus (E11.9)."
)

if __name__ == "__main__":
    clin = ClinicalNoteTokenizer()

    print("--- Token table (first 16) ---")
    for row in clin.token_table(CLINICAL_NOTE)[:16]:
        print(f"  {row['text']!r:12} norm={row['norm']!r:18} "
              f"pos={row['pos']:6} like_num={row['like_num']}")

    print("\n--- PHI detection (regex spans -> token spans) ---")
    for hit in clin.detect_phi(CLINICAL_NOTE):
        print(f"  {hit['label']:5} {hit['text']:12} tokens={hit['token_texts']}")

    print("\n--- Redacted note ---\n ", clin.redact(CLINICAL_NOTE))
