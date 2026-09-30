import argparse
import os
import random
import re
import shutil
import sys
import tarfile
import time
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import feedparser
from dotenv import load_dotenv
from gitignore_parser import parse_gitignore
from loguru import logger
from pyzotero import zotero
from tempfile import mkstemp
from tqdm import tqdm

from construct_email import render_email, send_email
from llm import set_global_llm
from paper import ArxivPaper
from recommender import rerank_paper
from sources import deduplicate_papers, fetch_crossref_papers
from site_builder import update_site_archive
from recommendation_history import load_history, select_recommendations, save_history

load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ARXIV_USER_AGENT = "zotero-arxiv-daily/0.3.5 (+https://github.com/Andyuch/zotero-arxiv-daily)"


def get_zotero_corpus(id: str, key: str) -> list[dict]:
    zot = zotero.Zotero(id, "user", key)
    collections = zot.everything(zot.collections())
    collections = {c["key"]: c for c in collections}
    corpus = zot.everything(zot.items(itemType="conferencePaper || journalArticle || preprint"))
    corpus = [c for c in corpus if c["data"]["abstractNote"] != ""]

    def get_collection_path(col_key: str) -> str:
        if p := collections[col_key]["data"]["parentCollection"]:
            return get_collection_path(p) + "/" + collections[col_key]["data"]["name"]
        return collections[col_key]["data"]["name"]

    for c in corpus:
        paths = [get_collection_path(col) for col in c["data"]["collections"]]
        c["paths"] = paths
    return corpus


def filter_corpus(corpus: list[dict], pattern: str) -> list[dict]:
    _, filename = mkstemp()
    with open(filename, "w") as file:
        file.write(pattern)
    matcher = parse_gitignore(filename, base_dir="./")
    new_corpus = []
    for c in corpus:
        match_results = [matcher(p) for p in c["paths"]]
        if not any(match_results):
            new_corpus.append(c)
    os.remove(filename)
    return new_corpus


def _retry_after_seconds(exc: HTTPError) -> float | None:
    value = exc.headers.get("Retry-After") if exc.headers is not None else None
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _fetch_arxiv_feed(query: str):
    """Fetch the daily arXiv Atom feed with a bounded retry budget."""
    url = f"https://rss.arxiv.org/atom/{query}"
    max_attempts = 4

    for attempt in range(max_attempts):
        try:
            request = Request(url, headers={"User-Agent": ARXIV_USER_AGENT})
            with urlopen(request, timeout=60) as response:
                feed = feedparser.parse(response.read())

            if getattr(feed, "bozo", 0):
                raise RuntimeError(
                    f"Failed to parse arXiv RSS/ATOM feed for query={query}: "
                    f"{feed.bozo_exception}"
                )

            title = getattr(getattr(feed, "feed", None), "title", "")
            if "Feed error for query" in title:
                raise ValueError(f"Invalid ARXIV_QUERY: {query}.")

            return feed

        except HTTPError as exc:
            if exc.code not in (429, 503) or attempt == max_attempts - 1:
                raise

            retry_after = _retry_after_seconds(exc)
            if retry_after is None:
                retry_after = min(180.0, 30.0 * (2 ** attempt))
            wait = retry_after + random.uniform(0.0, 3.0)
            logger.warning(
                "arXiv Atom feed HTTP {} (attempt {}/{}); retrying in {:.1f}s.",
                exc.code,
                attempt + 1,
                max_attempts,
                wait,
            )
            time.sleep(wait)

    raise RuntimeError("Failed to retrieve arXiv Atom feed.")


def _clean_feed_abstract(summary: str) -> str:
    """The arXiv Atom feed prefixes the abstract with ID/announce metadata."""
    if "Abstract:" in summary:
        summary = summary.split("Abstract:", 1)[1]
    return " ".join(summary.split())


class _RSSArxivResult:
    """Small adapter exposing the subset of arxiv.Result used by ArxivPaper.

    The daily Atom feed already contains ID, title, abstract and authors, so we
    can rank papers without re-querying export.arxiv.org/api/query for every
    batch. This avoids the GitHub-hosted-runner 429/503 failure mode.
    """

    def __init__(self, entry):
        self._short_id = entry.id.removeprefix("oai:arXiv.org:")
        self.title = " ".join(entry.get("title", "").split())
        self.summary = _clean_feed_abstract(entry.get("summary", ""))

        author_names = []
        if entry.get("authors"):
            author_names = [a.get("name", "").strip() for a in entry.authors]
        else:
            creator = (
                entry.get("author")
                or entry.get("dc_creator")
                or entry.get("creator")
                or ""
            )
            author_names = [name.strip() for name in creator.split(",")]
        self.authors = [SimpleNamespace(name=name) for name in author_names if name]

        base_id = re.sub(r"v\d+$", "", self._short_id)
        abstract_url = entry.get("link") or f"https://arxiv.org/abs/{base_id}"
        self.pdf_url = f"https://arxiv.org/pdf/{base_id}.pdf"
        self.links = [SimpleNamespace(href=abstract_url)]

    def get_short_id(self) -> str:
        return self._short_id

    def download_source(self, dirpath: str) -> str:
        """Download and validate source for a selected paper.

        Source is optional enrichment. Use export.arxiv.org for source
        tarballs, while keeping completeness validation, retry, and graceful
        fallback for transient or malformed responses.
        """
        base_id = re.sub(r"v\d+$", "", self._short_id)
        safe_name = base_id.replace("/", "_")
        destination = os.path.join(dirpath, f"{safe_name}.tar")
        partial = destination + ".part"
        source_url = f"https://export.arxiv.org/src/{base_id}"
        max_attempts = 2

        for attempt in range(max_attempts):
            try:
                if os.path.exists(partial):
                    os.remove(partial)

                request = Request(
                    source_url,
                    headers={"User-Agent": ARXIV_USER_AGENT},
                )
                with urlopen(request, timeout=90) as response, open(partial, "wb") as file:
                    expected_length = response.headers.get("Content-Length")
                    shutil.copyfileobj(response, file)

                if expected_length is not None:
                    try:
                        expected_bytes = int(expected_length)
                    except ValueError:
                        expected_bytes = None
                    if expected_bytes is not None:
                        actual_bytes = os.path.getsize(partial)
                        if actual_bytes != expected_bytes:
                            raise OSError(
                                f"incomplete source download: expected {expected_bytes} bytes, "
                                f"received {actual_bytes} bytes"
                            )

                # Force a complete archive scan here. tarfile.open() alone can
                # succeed on a truncated gzip and only raise EOFError later in
                # getnames()/getmembers(), which used to abort the whole digest.
                with tarfile.open(partial, mode="r:*") as archive:
                    archive.getmembers()

                os.replace(partial, destination)
                return destination

            except HTTPError as exc:
                if os.path.exists(partial):
                    os.remove(partial)
                if exc.code in (429, 503) and attempt < max_attempts - 1:
                    retry_after = _retry_after_seconds(exc)
                    wait = (
                        retry_after
                        if retry_after is not None
                        else 10.0 + random.uniform(0.0, 3.0)
                    )
                    logger.warning(
                        "arXiv source HTTP {} for {} (attempt {}/{}); "
                        "retrying in {:.1f}s.",
                        exc.code,
                        base_id,
                        attempt + 1,
                        max_attempts,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                raise

            except (EOFError, tarfile.ReadError, OSError) as exc:
                if os.path.exists(partial):
                    os.remove(partial)
                if attempt < max_attempts - 1:
                    wait = 3.0 + random.uniform(0.0, 2.0)
                    logger.warning(
                        "Incomplete/corrupt arXiv source for {} (attempt {}/{}): {}. "
                        "Retrying in {:.1f}s.",
                        base_id,
                        attempt + 1,
                        max_attempts,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                raise OSError(
                    f"arXiv source archive for {base_id} remained invalid after "
                    f"{max_attempts} attempts"
                ) from exc

        raise RuntimeError(f"Failed to download arXiv source for {base_id}")


def get_arxiv_paper(query: str, debug: bool = False) -> list["ArxivPaper"]:
    """Build daily candidates directly from the official arXiv Atom feed.

    The previous implementation fetched the feed and then re-fetched metadata
    through the legacy arXiv search API in batches. On shared GitHub Actions
    egress IPs, a single 429 could leave the job throttled for hours. The Atom
    feed already includes all metadata required for recommendation, so the
    normal path now makes zero legacy search-API metadata calls.
    """
    feed = _fetch_arxiv_feed(query)
    entries = list(feed.entries)

    if debug:
        logger.debug("Debug mode: use the first 5 papers from the arXiv Atom feed.")
        entries = entries[:5]
    else:
        entries = [
            entry
            for entry in entries
            if getattr(entry, "arxiv_announce_type", None) == "new"
        ]

    papers = [
        ArxivPaper(_RSSArxivResult(entry))
        for entry in tqdm(entries, desc="Reading arXiv Atom metadata")
    ]
    logger.info(
        "Built {} arXiv paper records from the Atom feed; legacy metadata API calls: 0.",
        len(papers),
    )
    return papers


parser = argparse.ArgumentParser(description="Recommender system for academic papers")


def add_argument(*args, **kwargs):
    def get_env(key: str, default=None):
        # Unset workflow environment variables are passed as ''. Treat them as None.
        value = os.environ.get(key)
        if value == "" or value is None:
            return default
        return value

    parser.add_argument(*args, **kwargs)
    arg_full_name = kwargs.get("dest", args[-1][2:])
    env_name = arg_full_name.upper()
    env_value = get_env(env_name)
    if env_value is not None:
        if kwargs.get("type") == bool:
            env_value = env_value.lower() in ["true", "1"]
        else:
            env_value = kwargs.get("type")(env_value)
        parser.set_defaults(**{arg_full_name: env_value})


if __name__ == "__main__":
    add_argument("--zotero_id", type=str, help="Zotero user ID")
    add_argument("--zotero_key", type=str, help="Zotero API key")
    add_argument(
        "--zotero_ignore",
        type=str,
        help="Zotero collection to ignore, using gitignore-style pattern.",
    )
    add_argument(
        "--send_empty",
        type=bool,
        help="If get no arxiv paper, send empty email",
        default=False,
    )
    add_argument(
        "--max_paper_num",
        type=int,
        help="Maximum number of papers to recommend",
        default=100,
    )
    add_argument("--arxiv_query", type=str, help="Arxiv search query")
    add_argument(
        "--enable_crossref",
        type=bool,
        help="Retrieve journal papers from Crossref in addition to arXiv",
        default=True,
    )
    add_argument(
        "--journal_groups",
        type=str,
        help="Comma-separated journal groups from journal_sources.json",
        default="nature,science,acs,materials",
    )
    add_argument(
        "--crossref_lookback_days",
        type=int,
        help="Crossref UTC calendar-date window including today (default: 3 days)",
        default=3,
    )
    add_argument(
        "--recommendation_cooldown_days",
        type=int,
        help="Prefer papers not recommended in the last N UTC days; 0 disables cooldown",
        default=7,
    )
    add_argument(
        "--crossref_rows_per_journal",
        type=int,
        help="Maximum Crossref records retrieved per configured journal",
        default=100,
    )
    add_argument(
        "--crossref_mailto",
        type=str,
        help="Optional contact email for Crossref polite-pool requests",
        default=None,
    )
    add_argument("--smtp_server", type=str, help="SMTP server")
    add_argument("--smtp_port", type=int, help="SMTP port")
    add_argument("--sender", type=str, help="Sender email address")
    add_argument("--receiver", type=str, help="Receiver email address")
    add_argument("--sender_password", type=str, help="Sender email password")
    add_argument(
        "--use_llm_api",
        type=bool,
        help="Use OpenAI API to generate TLDR",
        default=False,
    )
    add_argument(
        "--openai_api_key",
        type=str,
        help="OpenAI API key",
        default=None,
    )
    add_argument(
        "--openai_api_base",
        type=str,
        help="OpenAI API base URL",
        default="https://api.openai.com/v1",
    )
    add_argument(
        "--model_name",
        type=str,
        help="LLM Model Name",
        default="gpt-4o",
    )
    add_argument(
        "--language",
        type=str,
        help="Language of TLDR",
        default="English",
    )
    add_argument(
        "--publish_site",
        type=bool,
        help="Write the selected recommendations into the static research archive",
        default=True,
    )
    add_argument(
        "--site_output_dir",
        type=str,
        help="Static research archive output directory",
        default="site",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    if args.recommendation_cooldown_days < 0 or args.max_paper_num < -1:
        parser.error("cooldown must be nonnegative; max_paper_num must be -1 or nonnegative")
    history = load_history(args.site_output_dir)

    assert not args.use_llm_api or args.openai_api_key is not None

    # Remove Loguru's default stderr sink before adding exactly one stdout sink.
    logger.remove()
    logger.add(sys.stdout, level="DEBUG" if args.debug else "INFO")
    if args.debug:
        logger.debug("Debug mode is on.")

    logger.info("Retrieving Zotero corpus...")
    corpus = get_zotero_corpus(args.zotero_id, args.zotero_key)
    logger.info(f"Retrieved {len(corpus)} papers from Zotero.")
    if args.zotero_ignore:
        logger.info(f"Ignoring papers in:\n {args.zotero_ignore}...")
        corpus = filter_corpus(corpus, args.zotero_ignore)
        logger.info(f"Remaining {len(corpus)} papers after filtering.")

    logger.info("Retrieving arXiv papers...")
    papers = get_arxiv_paper(args.arxiv_query, args.debug)
    logger.info("arXiv candidates: {}", len(papers))

    if args.enable_crossref:
        logger.info(
            "Retrieving journal papers from Crossref groups: {}...",
            args.journal_groups,
        )
        try:
            journal_papers = fetch_crossref_papers(
                groups=args.journal_groups,
                lookback_days=args.crossref_lookback_days,
                rows_per_journal=args.crossref_rows_per_journal,
                mailto=args.crossref_mailto,
            )
            logger.info("Journal candidates from Crossref: {}", len(journal_papers))
            papers.extend(journal_papers)
        except Exception as exc:
            logger.warning(
                "Crossref retrieval failed; continuing with arXiv candidates only: {}",
                exc,
            )

    before_dedup = len(papers)
    papers = deduplicate_papers(papers)
    logger.info(
        "Unified candidate pool: {} papers after deduplication (removed {}).",
        len(papers),
        before_dedup - len(papers),
    )

    if len(papers) == 0:
        logger.info(
            "No new papers found from the configured sources. "
            "Check ARXIV_QUERY and journal source configuration."
        )
        if not args.send_empty:
            exit(0)
    else:
        logger.info("Reranking unified paper pool...")
        papers = rerank_paper(papers, corpus)
        papers = select_recommendations(
            papers, history, args.max_paper_num, args.recommendation_cooldown_days
        )
        logger.info("Selected {} papers ({} repeat highlights).", len(papers),
                    sum(p.recommendation_status == "repeat_highlight" for p in papers))
        if args.use_llm_api:
            logger.info("Using OpenAI-compatible API as global LLM.")
            set_global_llm(
                api_key=args.openai_api_key,
                base_url=args.openai_api_base,
                model=args.model_name,
                lang=args.language,
            )
        else:
            logger.info("Using Local LLM as global LLM.")
            set_global_llm(lang=args.language)

    html = render_email(papers)

    logger.info("Sending email...")
    send_email(
        args.sender,
        args.receiver,
        args.sender_password,
        args.smtp_server,
        args.smtp_port,
        html,
    )
    logger.success(
        "Email sent successfully! If you don't receive the email, "
        "please check the configuration and the junk box."
    )

    # Only successful delivery consumes cooldown; never record fetched/unselected papers.
    save_history(papers, args.site_output_dir)
    if args.publish_site:
        try:
            archive_path = update_site_archive(papers, args.site_output_dir)
            logger.info("Updated research archive at {}.", archive_path)
        except Exception as exc:
            logger.warning(
                "Failed to update the research archive after email delivery; history was saved: {}",
                exc,
            )
