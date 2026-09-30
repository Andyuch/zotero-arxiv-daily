"""Successful recommendation history, independent of ranking and enrichment."""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path


def paper_keys(paper) -> set[str]:
    """Match DOI spellings, arXiv revisions and preprint/journal title aliases."""
    get = paper.get if isinstance(paper, dict) else lambda k, default=None: getattr(paper, k, default)
    keys = set()
    doi = str(get("doi") or "").strip().lower()
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi)
    if doi:
        keys.add("doi:" + doi)
    arxiv = str(get("arxiv_id") or "").strip().lower()
    arxiv = re.sub(r"^(?:https?://arxiv\.org/(?:abs|pdf)/|arxiv:|oai:arxiv\.org:)", "", arxiv)
    arxiv = re.sub(r"\.pdf$", "", arxiv)
    arxiv = re.sub(r"v\d+$", "", arxiv)
    if arxiv:
        keys.add("arxiv:" + arxiv)
    title = "".join(c for c in str(get("title") or "").casefold() if c.isalnum())
    if title:
        keys.add("title:" + title)
    return keys


def _valid_date(value: str) -> str:
    # Require canonical ISO dates, so lexicographic comparisons are safe.
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("Invalid recommendation history date")
    return value


def load_history(output_dir: str = "site") -> dict[str, str]:
    """Bootstrap legacy selected archives; overlay durable successful-send ledger.

    Existing legacy archives are treated as recommendations, not proof of SMTP
    delivery. New archives and ledger writes happen only after successful send.
    """
    root = Path(output_dir) / "data"
    history: dict[str, str] = {}
    for path in sorted((root / "daily").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            day = _valid_date(payload.get("date") or path.stem)
            for record in payload.get("papers", []):
                for key in paper_keys(record):
                    history[key] = max(day, history.get(key, day))
        except (ValueError, TypeError, AttributeError, OSError) as exc:
            logging.warning("Skipping invalid legacy recommendation archive %s: %s", path, exc)

    ledger = root / "recommendation-history.json"
    if ledger.exists():
        # Do not silently erase corrupted durable history and resend everything.
        payload = json.loads(ledger.read_text(encoding="utf-8"))
        if payload.get("version") != 1 or not isinstance(payload.get("entries"), dict):
            raise ValueError("Unsupported recommendation history format")
        for key, value in payload["entries"].items():
            day = _valid_date(value)
            history[key] = max(day, history.get(key, day))
    return history


def select_recommendations(ranked: list, history: dict[str, str], max_papers: int,
                           cooldown_days: int = 7, today: date | None = None) -> list:
    """Keep rank within each tier; fill only a shortfall with labeled repeats."""
    today = today or datetime.now(timezone.utc).date()
    if max_papers < -1 or cooldown_days < 0:
        raise ValueError("max_papers must be -1 or nonnegative; cooldown_days must be nonnegative")
    fresh, recent = [], []
    for paper in ranked:
        days = [history[key] for key in paper_keys(paper) if key in history]
        previous = max(days) if days else None
        # A future entry (clock skew) is conservatively kept in the cooldown tier.
        cooling = previous is not None and cooldown_days > 0 and (today - date.fromisoformat(previous)).days < cooldown_days
        paper.previous_recommended_at = previous
        paper.recommendation_status = "repeat_highlight" if cooling else "revisit" if previous else "new"
        (recent if cooling else fresh).append(paper)
    ordered = fresh + recent
    return ordered if max_papers == -1 else ordered[:max_papers]


def save_history(papers: list, output_dir: str = "site", today: date | None = None) -> Path:
    """Record only successfully sent selected papers; preserve same-day reruns."""
    day = (today or datetime.now(timezone.utc).date()).isoformat()
    history = load_history(output_dir)
    for paper in papers:
        for key in paper_keys(paper):
            history[key] = max(day, history.get(key, day))
    destination = Path(output_dir) / "data" / "recommendation-history.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".history-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "entries": history}, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination
