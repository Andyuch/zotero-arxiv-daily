from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


TOPIC_RULES = {
    "Battery": (
        "battery", "lithium", "sodium", "magnesium", "cathode", "anode",
        "intercalation", "electrode", "electrolyte", "solid-state",
    ),
    "Electrochemistry": (
        "electrochem", "redox", "ion transport", "double layer", "charging",
        "nucleation", "phase front",
    ),
    "Microscopy": (
        "microscopy", "imaging", "iscat", "reflectance", "reflection",
        "operando imaging", "single-particle",
    ),
    "Spectroscopy": (
        "spectroscopy", "infrared", "raman", "photoelectron", "transient",
        "pump-probe", "2d infrared", "2dir",
    ),
    "Semiconductors": (
        "semiconductor", "perovskite", "gan", "ga2o3", "gallium oxide",
        "carrier", "photocatalyst", "heterojunction",
    ),
    "Materials": (
        "materials", "nanomaterial", "graphite", "graphene", "oxide",
        "framework", "crystal", "thin film",
    ),
    "AI & Computation": (
        "machine learning", "artificial intelligence", "neural", "simulation",
        "first-principles", "density functional", "molecular dynamics",
    ),
}


def _clean(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _author_names(paper) -> list[str]:
    names = []
    for author in getattr(paper, "authors", []) or []:
        name = getattr(author, "name", author)
        name = _clean(name)
        if name:
            names.append(name)
    return names


def _topics(title: str, summary: str) -> list[str]:
    text = f"{title} {summary}".lower()
    hits = []
    for topic, terms in TOPIC_RULES.items():
        if any(term in text for term in terms):
            hits.append(topic)
    return hits[:4] or ["Other"]


def _paper_key(record: dict) -> str:
    if record.get("doi"):
        return f"doi:{record['doi'].lower()}"
    if record.get("arxiv_id"):
        return f"arxiv:{record['arxiv_id'].lower()}"
    normalized = re.sub(r"[^a-z0-9]+", "", record.get("title", "").lower())
    return f"title:{normalized}"


def _record_from_paper(paper, seen_date: str) -> dict:
    title = _clean(getattr(paper, "title", ""))
    summary = _clean(getattr(paper, "summary", ""))
    affiliations = getattr(paper, "affiliations", None) or []
    score_components = getattr(paper, "score_components", {}) or {}

    record = {
        "title": title,
        "authors": _author_names(paper),
        "journal": _clean(getattr(paper, "journal", "")) or "arXiv",
        "source": _clean(getattr(paper, "source", "")) or "arXiv",
        "doi": _clean(getattr(paper, "doi", "")) or None,
        "arxiv_id": _clean(getattr(paper, "arxiv_id", "")) or None,
        "published_at": _clean(getattr(paper, "published_at", "")) or None,
        "article_url": _clean(getattr(paper, "paper_url", "")) or None,
        "pdf_url": _clean(getattr(paper, "pdf_url", "")) or None,
        "code_url": _clean(getattr(paper, "code_url", "")) or None,
        "abstract": summary,
        "tldr": _clean(getattr(paper, "tldr", "")),
        "affiliations": [_clean(x) for x in affiliations if _clean(x)],
        "score": round(float(getattr(paper, "score", 0.0) or 0.0), 4),
        "relevance_percentile": round(
            float(getattr(paper, "relevance_percentile", 0.0) or 0.0), 2
        ),
        "score_components": {
            key: round(float(value), 4)
            for key, value in score_components.items()
        },
        "topics": _topics(title, summary),
        "seen_date": seen_date,
    }
    record["key"] = _paper_key(record)
    return record


def _rebuild_index(data_dir: Path) -> dict:
    daily_dir = data_dir / "daily"
    daily_files = sorted(daily_dir.glob("*.json"), reverse=True)

    by_key: dict[str, dict] = {}
    day_counts: list[dict] = []

    for path in daily_files:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        date = payload.get("date") or path.stem
        records = payload.get("papers", [])
        day_counts.append({"date": date, "count": len(records)})

        for record in records:
            key = record.get("key") or _paper_key(record)
            if key not in by_key:
                merged = dict(record)
                merged["first_seen"] = date
                merged["last_seen"] = date
                merged["seen_dates"] = [date]
                by_key[key] = merged
            else:
                existing = by_key[key]
                existing["first_seen"] = min(existing["first_seen"], date)
                existing["last_seen"] = max(existing["last_seen"], date)
                if date not in existing["seen_dates"]:
                    existing["seen_dates"].append(date)

    papers = sorted(
        by_key.values(),
        key=lambda p: (
            p.get("last_seen", ""),
            p.get("relevance_percentile", 0),
            p.get("score", 0),
        ),
        reverse=True,
    )

    journals = sorted({p.get("journal", "") for p in papers if p.get("journal")})
    sources = sorted({p.get("source", "") for p in papers if p.get("source")})
    topics = sorted({t for p in papers for t in p.get("topics", []) if t})

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_count": len(papers),
        "days": day_counts,
        "journals": journals,
        "sources": sources,
        "topics": topics,
        "papers": papers,
    }


def update_site_archive(papers: list, output_dir: str = "site") -> Path:
    """Persist today's selected recommendations and rebuild the site index."""
    root = Path(output_dir)
    data_dir = root / "data"
    daily_dir = data_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now(timezone.utc).date().isoformat()
    records = [_record_from_paper(paper, today) for paper in papers]

    daily_payload = {
        "date": today,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(records),
        "papers": records,
    }

    daily_path = daily_dir / f"{today}.json"
    daily_path.write_text(
        json.dumps(daily_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    index = _rebuild_index(data_dir)
    index_path = data_dir / "index.json"
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return index_path
