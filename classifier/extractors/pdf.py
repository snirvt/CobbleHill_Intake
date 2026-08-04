import logging
import os
from pathlib import Path

from liteparse import LiteParse
from liteparse.types import ParseError, ParseResult

from classifier.models import ExtractedText
from classifier.naming import parse_note_name
from config.settings import PageTargetRule, settings

logger = logging.getLogger(__name__)


class PdfExtractor:
    """ContentExtractor for PDF files using the LiteParse npm CLI.

    Most PDFs return their full text. A PDF whose filename parses to a note
    matching one of ``page_target_rules`` is narrowed to the single page
    holding that rule's section header (see :class:`PageTargetRule`).
    """

    def __init__(
        self,
        node_bin_path: str = settings.node_bin_path,
        dpi: int = settings.ocr_dpi,
        page_target_rules: list[PageTargetRule] = settings.page_target_rules,
    ) -> None:
        if node_bin_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = node_bin_path + ":" + os.environ["PATH"]
        self._parser = LiteParse()
        self._dpi = dpi
        self._page_target_rules = page_target_rules

    async def extract(self, file_path: Path) -> ExtractedText:
        """Parse a PDF, narrowing to a section page when a rule matches."""
        rule = self._match_rule(file_path.name)
        if rule is not None:
            return await self._extract_target_page(file_path, rule)
        logger.debug("Parsing PDF: %s (dpi=%d)", file_path, self._dpi)
        result = await self._parse(file_path)
        logger.debug("Parsed %s — %d page(s)", file_path.name, result.num_pages)
        return ExtractedText(
            file_path=file_path, text=result.text, num_pages=result.num_pages
        )

    async def _parse(
        self, file_path: Path, target_pages: str | None = None
    ) -> ParseResult:
        """Run LiteParse, translating ParseError into RuntimeError."""
        try:
            return await self._parser.parse_async(
                file_path, dpi=self._dpi, target_pages=target_pages
            )
        except ParseError as exc:
            raise RuntimeError(f"LiteParse failed on {file_path}: {exc}") from exc

    def _match_rule(self, name: str) -> PageTargetRule | None:
        """Return the page-target rule matching this filename's note, if any."""
        parsed = parse_note_name(name)
        if parsed is None:
            return None
        for rule in self._page_target_rules:
            if rule.role == parsed.role and rule.category == parsed.category:
                return rule
        return None

    async def _extract_target_page(
        self, file_path: Path, rule: PageTargetRule
    ) -> ExtractedText:
        """Return only the page holding ``rule.section_header``.

        Tries ``rule.page_hint`` first, then scans every page, then falls back
        to the full document text if the header is never found.
        """
        needle = self._normalize(rule.section_header)

        hinted = await self._parse(file_path, target_pages=str(rule.page_hint))
        for page in hinted.pages:
            if needle in self._normalize(page.text):
                logger.debug(
                    "Found %r on hinted page %d of %s",
                    rule.section_header,
                    page.pageNum,
                    file_path.name,
                )
                return ExtractedText(
                    file_path=file_path, text=page.text, num_pages=hinted.num_pages
                )

        full = await self._parse(file_path)
        for page in full.pages:
            if needle in self._normalize(page.text):
                logger.debug(
                    "Found %r on page %d of %s (hint %d missed)",
                    rule.section_header,
                    page.pageNum,
                    file_path.name,
                    rule.page_hint,
                )
                return ExtractedText(
                    file_path=file_path, text=page.text, num_pages=full.num_pages
                )

        logger.warning(
            "Section %r not found in %s; falling back to full text",
            rule.section_header,
            file_path.name,
        )
        return ExtractedText(
            file_path=file_path, text=full.text, num_pages=full.num_pages
        )

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase and collapse whitespace for tolerant header matching."""
        return " ".join(text.lower().split())
