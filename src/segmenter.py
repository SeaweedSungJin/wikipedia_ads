"""Document segmentation utilities."""
from __future__ import annotations

from typing import Dict, List
import nltk

from .pipeline import EXCLUDED_SECTIONS
from .utils import download_nltk_data


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
    """Split section texts into individual paragraphs."""

    def get_segments(self, doc: Dict) -> List[Dict]:
        segments: List[Dict] = []
        title = doc.get("title", "N/A")
        for sec_idx, (sec_title, sec_text) in enumerate(
            zip(doc.get("section_titles", []), doc.get("section_texts", []))
        ):
            if sec_title.lower().strip() in EXCLUDED_SECTIONS:
                continue
            if not sec_text or sec_text.isspace():
                continue
            # Split on blank lines to obtain paragraphs
            paragraphs = [p.strip() for p in sec_text.split("\n\n") if p.strip()]
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