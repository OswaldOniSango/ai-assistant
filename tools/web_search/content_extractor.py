"""Extracts readable text from retrieved URLs."""

from __future__ import annotations

import logging
import ssl
from urllib.error import URLError
from urllib.request import Request, urlopen

from lxml import html

from .models import RetrievedDocument, SearchResult

logger = logging.getLogger(__name__)


class WebContentExtractor:
    """Fetches pages and extracts article-like text content."""

    def __init__(self, timeout: int = 10, max_chars: int = 1500) -> None:
        self.timeout = timeout
        self.max_chars = max_chars

    def extract_documents(
        self,
        results: list[SearchResult],
        limit: int = 3,
    ) -> list[RetrievedDocument]:
        documents: list[RetrievedDocument] = []

        for result in results[:limit]:
            document = self.extract_document(result)
            if document is not None:
                documents.append(document)

        return documents

    def extract_document(self, result: SearchResult) -> RetrievedDocument | None:
        request = Request(
            result.url,
            headers={
                "Accept-Language": "en-US,en;q=0.9",
                "User-Agent": "Mozilla/5.0",
            },
        )

        try:
            page = self._load_page(request)
        except Exception as exc:
            logger.info("Content extraction failed for '%s': %s", result.url, exc)
            return None

        try:
            document = html.fromstring(page)
        except Exception as exc:
            logger.info("HTML parsing failed for '%s': %s", result.url, exc)
            return None

        for bad_node in document.xpath("//script|//style|//noscript|//svg"):
            bad_node.drop_tree()

        text_parts = [
            text.strip()
            for text in document.xpath(
                "//main//p//text() | //article//p//text() | //body//p//text()"
            )
            if text.strip()
        ]
        content = " ".join(text_parts)
        content = " ".join(content.split())

        if not content:
            content = result.snippet.strip()

        if not content:
            return None

        return RetrievedDocument(
            title=result.title.strip() or "Untitled",
            url=result.url.strip(),
            content=content[: self.max_chars],
        )

    def _load_page(self, request: Request) -> bytes:
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except ssl.SSLCertVerificationError:
            return self._load_page_without_ssl_verification(request)
        except URLError as exc:
            if isinstance(exc.reason, ssl.SSLCertVerificationError):
                return self._load_page_without_ssl_verification(request)
            raise

    def _load_page_without_ssl_verification(self, request: Request) -> bytes:
        logger.info(
            "Content extraction SSL verification failed, retrying without verification."
        )
        insecure_context = ssl._create_unverified_context()
        with urlopen(
            request,
            timeout=self.timeout,
            context=insecure_context,
        ) as response:
            return response.read()
