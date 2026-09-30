"""Persistent Crossref ingestion history for overlap-safe daily retrieval."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from recommendation_history import paper_keys


def crossref_keys(paper) -> set[str]:
    """Stable identities for journal records: DOI first, normalized title fallback."""
    return {
        key for key in paper_keys(paper)
        if key.startswith("doi:") or key.startswith("title:")
    }


def _timestamp(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return current.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def load_ingestion_history(output_dir: str = "site") -> dict[str, str]:
    """Load all previously observed Crossref identities.

    Existing daily recommendation archives are used as a conservative bootstrap
    so journal papers already shown before this ledger existed are not treated as
    completely unseen on the first overlap-enabled run.
    """
    root = Path(output_dir) / "data"
    history: dict[str, str] = {}

    for path in sorted((root / "daily").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            day = str(payload.get("date") or path.stem)
            for record in payload.get("papers", []):
                if str(record.get("source") or "").casefold() != "crossref":
                    continue
                for key in crossref_keys(record):
                    history.setdefault(key, day)
        except (OSError, ValueError, TypeError, AttributeError):
            # Recommendation history owns strict archive validation. This ledger
            # bootstrap is best-effort and should not make an otherwise healthy
            # run fail because of one legacy archive.
            continue

    ledger = root / "crossref-ingestion-history.json"
    if not ledger.exists():
        return history

    payload = json.loads(ledger.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not isinstance(payload.get("entries"), dict):
        raise ValueError("Unsupported Crossref ingestion history format")

    for key, value in payload["entries"].items():
        if not isinstance(key, str) or not isinstance(value, str) or not value:
            raise ValueError("Invalid Crossref ingestion history entry")
        history.setdefault(key, value)

    return history


def partition_unseen_crossref(
    papers: list,
    history: dict[str, str],
) -> tuple[list, list]:
    """Split retrieved records into unseen candidates and overlap repeats."""
    unseen, seen = [], []
    for paper in papers:
        keys = crossref_keys(paper)
        if keys and any(key in history for key in keys):
            seen.append(paper)
        else:
            unseen.append(paper)
    return unseen, seen


def save_ingestion_history(
    papers: list,
    output_dir: str = "site",
    now: datetime | None = None,
) -> Path:
    """Persist every successfully processed Crossref record atomically.

    This is intentionally called only after successful email delivery. A failed
    run therefore does not consume unseen journal records.
    """
    seen_at = _timestamp(now)
    history = load_ingestion_history(output_dir)

    for paper in papers:
        for key in crossref_keys(paper):
            history.setdefault(key, seen_at)

    destination = Path(output_dir) / "data" / "crossref-ingestion-history.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=".crossref-ingestion-",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(
                {"version": 1, "entries": history},
                f,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    return destination
