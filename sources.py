from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from paper import JournalPaper


CATALOG_PATH = Path(__file__).with_name("journal_sources.json")
CROSSREF_API = "https://api.crossref.org"


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    value = re.sub(r"<[^>]+>", " ", str(value))
    return " ".join(value.split())


def _crossref_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": user_agent})
    return session


def load_journal_catalog(groups: str | Iterable[str]) -> list[dict]:
    """Load selected journal groups plus optional EXTRA_JOURNALS_JSON entries."""
    if isinstance(groups, str):
        selected = [g.strip().lower() for g in groups.split(",") if g.strip()]
    else:
        selected = [str(g).strip().lower() for g in groups if str(g).strip()]

    with CATALOG_PATH.open("r", encoding="utf-8") as f:
        catalog = json.load(f)

    journals: list[dict] = []
    for group in selected:
        if group not in catalog:
            logger.warning("Unknown journal group '{}'; skipping.", group)
            continue
        for item in catalog[group]:
            journals.append({"group": group, "journal": item["journal"], "issn": item["issn"]})

    extra = os.environ.get("EXTRA_JOURNALS_JSON", "").strip()
    if extra:
        try:
            parsed = json.loads(extra)
            if not isinstance(parsed, list):
                raise ValueError("EXTRA_JOURNALS_JSON must be a JSON list")
            for item in parsed:
                journals.append({
                    "group": str(item.get("group", "extra")),
                    "journal": str(item["journal"]),
                    "issn": str(item["issn"]),
                })
        except Exception as exc:
            logger.warning("Ignoring invalid EXTRA_JOURNALS_JSON: {}", exc)

    seen: set[str] = set()
    unique: list[dict] = []
    for item in journals:
        issn = item["issn"].upper()
        if issn in seen:
            continue
        seen.add(issn)
        unique.append(item)
    return unique


def _date_parts(item: dict) -> str | None:
    for key in ("published-online", "published-print", "published", "issued"):
        parts = item.get(key, {}).get("date-parts")
        if parts and parts[0]:
            values = list(parts[0])
            while len(values) < 3:
                values.append(1)
            try:
                return f"{int(values[0]):04d}-{int(values[1]):02d}-{int(values[2]):02d}"
            except (TypeError, ValueError):
                pass
    return None


def _authors_and_affiliations(item: dict) -> tuple[list[str], list[str]]:
    authors: list[str] = []
    affiliations: list[str] = []
    for author in item.get("author", []) or []:
        given = _clean_text(author.get("given"))
        family = _clean_text(author.get("family"))
        name = " ".join(x for x in (given, family) if x).strip()
        if name:
            authors.append(name)
        for affiliation in author.get("affiliation", []) or []:
            aff_name = _clean_text(affiliation.get("name"))
            if aff_name and aff_name not in affiliations:
                affiliations.append(aff_name)
    return authors, affiliations


def _best_pdf_url(item: dict) -> str | None:
    for link in item.get("link", []) or []:
        content_type = str(link.get("content-type", "")).lower()
        url = link.get("URL")
        if url and "pdf" in content_type:
            return url
    return None


def _paper_from_crossref(item: dict, configured_journal: str) -> JournalPaper | None:
    titles = item.get("title") or []
    title = _clean_text(titles[0] if titles else "")
    if not title:
        return None

    container = item.get("container-title") or []
    journal = _clean_text(container[0] if container else configured_journal) or configured_journal
    doi = _clean_text(item.get("DOI")) or None
    article_url = _clean_text(item.get("URL")) or (f"https://doi.org/{doi}" if doi else "")
    abstract = _clean_text(item.get("abstract"))
    authors, affiliations = _authors_and_affiliations(item)

    return JournalPaper(
        title=title,
        summary=abstract,
        authors=authors,
        journal=journal,
        source="Crossref",
        doi=doi,
        paper_url=article_url,
        pdf_url=_best_pdf_url(item),
        affiliations=affiliations,
        published_at=_date_parts(item),
    )


def fetch_crossref_papers(
    groups: str = "nature,science,acs,materials",
    lookback_days: int = 3,
    rows_per_journal: int = 100,
    mailto: str | None = None,
) -> list[JournalPaper]:
    """Retrieve newly deposited journal articles from configured journal families.

    Uses Crossref's created-date window, so a daily run picks up a paper when
    its Crossref metadata first appears even if the publisher publication date
    is earlier. The default covers today plus two preceding UTC dates,
    providing overlap for early/moving daily runs; this is not a rolling 24h window.
    """
    journals = load_journal_catalog(groups)
    if not journals:
        return []

    user_agent = "zotero-arxiv-daily/0.3.5 (https://github.com/Andyuch/zotero-arxiv-daily)"
    session = _crossref_session(user_agent)

    today = datetime.now(timezone.utc).date()
    lookback_days = max(1, int(lookback_days))
    start = today - timedelta(days=lookback_days - 1)
    date_filter = f"from-created-date:{start.isoformat()},until-created-date:{today.isoformat()}"

    papers: list[JournalPaper] = []
    for index, spec in enumerate(journals):
        params = {
            "filter": date_filter,
            "rows": max(1, min(int(rows_per_journal), 1000)),
            "sort": "created",
            "order": "desc",
        }
        if mailto:
            params["mailto"] = mailto

        url = f"{CROSSREF_API}/journals/{spec['issn']}/works"
        try:
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            message = response.json().get("message", {})
            items = message.get("items", [])
            if message.get("total-results", len(items)) > len(items):
                logger.warning(
                    "Crossref window for {} contains more records than the configured cap {}; "
                    "increase CROSSREF_ROWS_PER_JOURNAL if needed.",
                    spec["journal"], params["rows"],
                )
        except Exception as exc:
            logger.warning(
                "Crossref retrieval failed for {} ({}): {}",
                spec["journal"], spec["issn"], exc,
            )
            continue

        count = 0
        for item in items:
            if item.get("type") not in (None, "journal-article"):
                continue
            paper = _paper_from_crossref(item, spec["journal"])
            if paper is not None:
                papers.append(paper)
                count += 1

        logger.info("Crossref: {} new records from {} ({})", count, spec["journal"], spec["issn"])
        if index < len(journals) - 1:
            time.sleep(0.15)

    logger.info(
        "Crossref: built {} candidate records from {} configured journals.",
        len(papers), len(journals),
    )
    return papers


def _normalized_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def _priority(paper) -> int:
    journal = str(getattr(paper, "journal", "") or "").strip().lower()
    doi = getattr(paper, "doi", None)
    if doi:
        return 3
    if journal and journal != "arxiv":
        return 2
    return 1


def deduplicate_papers(papers: list) -> list:
    """Deduplicate DOI/title collisions, preferring the journal version."""
    result: list = []
    title_to_index: dict[str, int] = {}
    doi_to_index: dict[str, int] = {}

    for paper in papers:
        title_key = _normalized_title(getattr(paper, "title", ""))
        doi = str(getattr(paper, "doi", "") or "").lower().strip()

        existing_index = None
        if doi and doi in doi_to_index:
            existing_index = doi_to_index[doi]
        elif title_key and title_key in title_to_index:
            existing_index = title_to_index[title_key]

        if existing_index is None:
            index = len(result)
            result.append(paper)
            if title_key:
                title_to_index[title_key] = index
            if doi:
                doi_to_index[doi] = index
            continue

        existing = result[existing_index]
        if _priority(paper) > _priority(existing):
            old_title = _normalized_title(getattr(existing, "title", ""))
            old_doi = str(getattr(existing, "doi", "") or "").lower().strip()
            result[existing_index] = paper
            if old_title:
                title_to_index.pop(old_title, None)
            if old_doi:
                doi_to_index.pop(old_doi, None)
            if title_key:
                title_to_index[title_key] = existing_index
            if doi:
                doi_to_index[doi] = existing_index

    return result
