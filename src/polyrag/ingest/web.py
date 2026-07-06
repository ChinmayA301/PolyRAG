"""Web page ingestion: Crawl4AI primary, plain-httpx fallback.

Crawl4AI's HTTP crawler strategy fetches and converts pages to markdown without
needing a Playwright browser; pass js=True for JS-heavy pages (requires
`crawl4ai-setup` to have installed a browser). If crawl4ai isn't installed the
fallback fetches with httpx and extracts readable text with trafilatura, or as
a last resort strips tags. The record notes which path produced the text.
"""

from __future__ import annotations

import asyncio
import re


async def _crawl4ai_fetch(url: str, js: bool) -> str | None:
    try:
        from crawl4ai import AsyncWebCrawler
    except ImportError:
        return None
    kwargs = {}
    if not js:
        try:
            from crawl4ai.async_crawler_strategy import AsyncHTTPCrawlerStrategy

            kwargs["crawler_strategy"] = AsyncHTTPCrawlerStrategy()
        except ImportError:
            pass  # older crawl4ai: browser strategy only
    async with AsyncWebCrawler(**kwargs) as crawler:
        result = await crawler.arun(url=url)
        if not getattr(result, "success", False):
            return None
        md = getattr(result, "markdown", None)
        text = getattr(md, "raw_markdown", None) or (md if isinstance(md, str) else None)
        return text


def _httpx_fetch(url: str) -> tuple[str, str]:
    import httpx

    html = httpx.get(url, follow_redirects=True, timeout=30,
                     headers={"User-Agent": "polyrag/0.1 (+https://github.com)"}).text
    try:
        import trafilatura

        text = trafilatura.extract(html) or ""
        if text:
            return text, "httpx+trafilatura"
    except ImportError:
        pass
    # last resort: crude tag strip
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(), "httpx+strip"


def fetch_page(url: str, source_id: str, title: str, js: bool = False) -> dict:
    text = None
    extraction = "crawl4ai"
    try:
        text = asyncio.run(_crawl4ai_fetch(url, js))
    except Exception:
        text = None
    if not text:
        text, extraction = _httpx_fetch(url)
    return {
        "source_id": source_id,
        "title": title,
        "url": url,
        "page": None,
        "extraction": extraction,
        "text": text or "",
    }
