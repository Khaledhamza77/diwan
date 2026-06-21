# -*- coding: utf-8 -*-
"""
Unit tests for chainlit_handlers pure helpers.
No Chainlit context required — tests _parse_citations and _promote_first_sentence only.
"""
import pytest
from diwan.chainlit_handlers import _parse_citations, _promote_first_sentence

# ── Fixtures — realistic synthesizer output ────────────────────────────────────

ANSWER_TWO_PAGES = (
    "تُعرَّف الأداة المالية [1] وفقاً للمعيار رقم 25 بأنها أي عقد [2].\n\n"
    "---\n"
    "[1] معايير_المحاسبة_المصرية_2020، ص. 14\n"
    "[2] معايير_المحاسبة_المصرية_2020، ص. 7\n"
)

ANSWER_MULTI_DOC = (
    "صافي الربح [1] بلغ 5.2 مليار جنيه [2].\n\n"
    "---\n"
    "[1] معايير_المحاسبة_المصرية_2020، ص. 3\n"
    "[2] qawaim، ص. 12\n"
    "[3] معايير_المحاسبة_المصرية_2020، ص. 3\n"   # duplicate — should be deduped
)

ANSWER_NO_FOOTER = "مرحباً! يسعدني مساعدتك."

ANSWER_NO_PAGE = (
    "نص الإجابة.\n\n"
    "---\n"
    "[1] doc_without_page\n"
)


# ── _parse_citations ───────────────────────────────────────────────────────────

class TestParseCitations:
    def test_two_distinct_pages(self):
        parsed = _parse_citations(ANSWER_TWO_PAGES)
        assert len(parsed) == 2
        labels = {p[0] for p in parsed}
        assert "معايير_المحاسبة_المصرية_2020 | ص. 14" in labels
        assert "معايير_المحاسبة_المصرية_2020 | ص. 7" in labels

    def test_page_number_captured(self):
        parsed = _parse_citations(ANSWER_TWO_PAGES)
        pages = {p[2] for p in parsed}
        assert "14" in pages
        assert "7" in pages

    def test_dedup_same_doc_same_page(self):
        parsed = _parse_citations(ANSWER_MULTI_DOC)
        # lines 1 and 3 are the same doc+page — only 2 unique entries
        assert len(parsed) == 2
        labels = {p[0] for p in parsed}
        assert "معايير_المحاسبة_المصرية_2020 | ص. 3" in labels
        assert "qawaim | ص. 12" in labels

    def test_no_footer_returns_empty(self):
        assert _parse_citations(ANSWER_NO_FOOTER) == []

    def test_no_page_label_is_doc_name_only(self):
        parsed = _parse_citations(ANSWER_NO_PAGE)
        assert len(parsed) == 1
        label, content, page = parsed[0]
        assert label == "doc_without_page"
        assert page is None

    def test_content_contains_doc_and_page(self):
        parsed = _parse_citations(ANSWER_TWO_PAGES)
        for label, content, page in parsed:
            assert "معايير_المحاسبة_المصرية_2020" in content
            assert page in content


# ── _promote_first_sentence ────────────────────────────────────────────────────

class TestPromoteFirstSentence:
    def test_promotes_arabic_prose(self):
        text = "تُعرَّف الأداة المالية بأنها أي عقد. ويشمل ذلك الأسهم."
        result = _promote_first_sentence(text)
        assert result.startswith("### ")
        assert "ويشمل" in result

    def test_existing_heading_untouched(self):
        text = "### عنوان\n\nتفاصيل."
        assert _promote_first_sentence(text) == text

    def test_code_fence_untouched(self):
        text = "```python\ncode\n```"
        assert _promote_first_sentence(text) == text

    def test_numbered_list_untouched(self):
        text = "1. الخطوة الأولى\n2. الخطوة الثانية"
        assert _promote_first_sentence(text) == text

    def test_citation_markers_dont_break_detection(self):
        text = "صافي الربح [1] بلغ 5.2 مليار. وقد ارتفع [2]."
        result = _promote_first_sentence(text)
        assert result.startswith("### ")

    def test_footer_preserved_after_promotion(self):
        result = _promote_first_sentence(ANSWER_TWO_PAGES)
        assert result.startswith("### ")
        assert "---" in result
        assert "ص. 14" in result

    def test_arabic_question_mark_as_boundary(self):
        text = "هل يشمل المعيار الأصول غير الملموسة؟ نعم، يشملها."
        result = _promote_first_sentence(text)
        assert result.startswith("### ")
