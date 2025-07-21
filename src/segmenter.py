"""Document segmentation utilities."""
from __future__ import annotations

from typing import Dict, List
import nltk

from .utils import download_nltk_data

# Sections containing references or metadata that should not be ranked
EXCLUDED_SECTIONS = {
    "references",
    "external links",
    "see also",
    "notes",
    "bibliography",
    "further reading",
}

class Segmenter:
    """Base class that yields text segments from a KB document."""

    def get_segments(self, doc: Dict) -> List[Dict]:
        raise NotImplementedError


class SectionSegmenter(Segmenter):
    """Use existing section texts as segments."""

    def get_segments(self, doc: Dict) -> List[Dict]:
        segments: List[Dict] = []
        title = doc.get("title", "N/A")
        for idx, (sec_title, sec_text) in enumerate(
            zip(doc.get("section_titles", []), doc.get("section_texts", []))
        ):
            if sec_title.lower().strip() in EXCLUDED_SECTIONS:
                continue
            if sec_text and not sec_text.isspace():
                segments.append(
                    {
                        "section_text": sec_text,
                        "source_title": title,
                        "section_title": sec_title,
                        "section_idx": idx,
                    }
                )
        return segments


class SentenceSegmenter(Segmenter):
    """Split section texts into individual sentences."""

    def get_segments(self, doc: Dict) -> List[Dict]:
        download_nltk_data()
        segments: List[Dict] = []
        title = doc.get("title", "N/A")
        for sec_idx, (sec_title, sec_text) in enumerate(
            zip(doc.get("section_titles", []), doc.get("section_texts", []))
        ):
            if sec_title.lower().strip() in EXCLUDED_SECTIONS:
                continue
            if not sec_text or sec_text.isspace():
                continue
            for sent_idx, sent in enumerate(nltk.sent_tokenize(sec_text)):
                segments.append(
                    {
                        "section_text": sent,
                        "source_title": title,
                        "section_title": f"{sec_title} [sent {sent_idx}]",
                        "section_idx": sec_idx,
                        "sentence_idx": sent_idx,
                    }
                )
        return segments
    
class ParagraphSegmenter(Segmenter):
    """Split section texts into balanced-length paragraphs."""

    def __init__(self, max_length: int = 1024) -> None:
        self.max_length = max_length

    def _balanced_chunks(self, sentences: List[str]) -> List[str]:
        """Return balanced chunks from a list of sentences."""
        import math

        lengths = [len(s) + 1 for s in sentences]
        total_len = sum(lengths)
        num_chunks = max(1, math.ceil(total_len / self.max_length))
        chunks: List[str] = []
        cur, cur_len = [], 0
        remaining_len = total_len
        remaining_sentences = len(sentences)
        for idx, (sent, slen) in enumerate(zip(sentences, lengths)):
            avg_len = remaining_len / max(1, num_chunks - len(chunks))
            if cur and ((cur_len + slen > self.max_length) or (cur_len >= avg_len and remaining_sentences)):
                chunks.append(" ".join(cur).strip())
                remaining_len -= cur_len
                cur, cur_len = [], 0
            cur.append(sent)
            cur_len += slen
            remaining_sentences -= 1
        if cur:
            chunks.append(" ".join(cur).strip())
        return chunks

    def get_segments(self, doc: Dict) -> List[Dict]:
        download_nltk_data()
        segments: List[Dict] = []
        title = doc.get("title", "N/A")
        for sec_idx, (sec_title, sec_text) in enumerate(
            zip(doc.get("section_titles", []), doc.get("section_texts", []))
        ):
            if sec_title.lower().strip() in EXCLUDED_SECTIONS:
                continue
            if not sec_text or sec_text.isspace():
                continue
            sentences = nltk.sent_tokenize(sec_text)
            paragraphs = self._balanced_chunks(sentences)
            for para_idx, para in enumerate(paragraphs):
                segments.append(
                    {
                        "section_text": para,
                        "source_title": title,
                        "section_title": f"{sec_title} [para {para_idx}]",
                        "section_idx": sec_idx,
                        "paragraph_idx": para_idx,
                    }
                )
        return segments