import time, random, inspect, math
from http import HTTPStatus
from email.utils import parsedate_to_datetime
import feedparser


import arxiv
import argparse
import os
import sys
from dotenv import load_dotenv
load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pyzotero import zotero
from recommender import rerank_paper
from construct_email import render_email, send_email
from tqdm import trange,tqdm
from loguru import logger
from gitignore_parser import parse_gitignore
from tempfile import mkstemp
from paper import ArxivPaper
from llm import set_global_llm
import feedparser

def get_zotero_corpus(id:str,key:str) -> list[dict]:
    zot = zotero.Zotero(id, 'user', key)
    collections = zot.everything(zot.collections())
    collections = {c['key']:c for c in collections}
    corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
    corpus = [c for c in corpus if c['data']['abstractNote'] != '']
    def get_collection_path(col_key:str) -> str:
        if p := collections[col_key]['data']['parentCollection']:
            return get_collection_path(p) + '/' + collections[col_key]['data']['name']
        else:
            return collections[col_key]['data']['name']
    for c in corpus:
        paths = [get_collection_path(col) for col in c['data']['collections']]
        c['paths'] = paths
    return corpus

def filter_corpus(corpus:list[dict], pattern:str) -> list[dict]:
    _,filename = mkstemp()
    with open(filename,'w') as file:
        file.write(pattern)
    matcher = parse_gitignore(filename,base_dir='./')
    new_corpus = []
    for c in corpus:
        match_results = [matcher(p) for p in c['paths']]
        if not any(match_results):
            new_corpus.append(c)
    os.remove(filename)
    return new_corpus

"""
def get_arxiv_paper(query:str, debug:bool=False) -> list[ArxivPaper]:
    client = arxiv.Client(num_retries=10,delay_seconds=10)
    feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
    if 'Feed error for query' in feed.feed.title:
        raise Exception(f"Invalid ARXIV_QUERY: {query}.")
    if not debug:
        papers = []
        all_paper_ids = [i.id.removeprefix("oai:arXiv.org:") for i in feed.entries if i.arxiv_announce_type == 'new']
        bar = tqdm(total=len(all_paper_ids),desc="Retrieving Arxiv papers")
        for i in range(0,len(all_paper_ids),50):
            search = arxiv.Search(id_list=all_paper_ids[i:i+50])
            batch = [ArxivPaper(p) for p in client.results(search)]
            bar.update(len(batch))
            papers.extend(batch)
        bar.close()

    else:
        logger.debug("Retrieve 5 arxiv papers regardless of the date.")
        search = arxiv.Search(query='cat:cs.AI', sort_by=arxiv.SortCriterion.SubmittedDate)
        papers = []
        for i in client.results(search):
            papers.append(ArxivPaper(i))
            if len(papers) == 5:
                break

    return papers
"""

def _parse_retry_after(value):
    """Return seconds to wait from a Retry-After header (int or HTTP-date)."""
    if not value:
        return None
    try:
        return int(value)
    except Exception:
        try:
            dt = parsedate_to_datetime(value)
            secs = (dt - parsedate_to_datetime(
                time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime())
            )).total_seconds()
            return max(0, int(secs))
        except Exception:
            return None

def _safe_client():
    """Create an arxiv.Client compatible with multiple versions (no hard timeout arg)."""
    kwargs = {"num_retries": 0, "delay_seconds": 0}
    try:
        sig = inspect.signature(arxiv.Client)
        if "page_size" in sig.parameters:
            kwargs["page_size"] = 50  # conservative; lib may clamp anyway
        # If your installed version supports 'timeout', uncomment these two lines:
        # if "timeout" in sig.parameters:
        #     kwargs["timeout"] = 30
    except Exception:
        pass
    client = arxiv.Client(**kwargs)
    try:
        if hasattr(client, "session") and hasattr(client.session, "headers"):
            client.session.headers.update({
                "User-Agent": "zotero-arxiv-daily/1.0 (mailto:your-email@example.com)"
            })
    except Exception:
        pass
    return client

def _fetch_batch_with_retries(client, ids, *, max_attempts=6, base_backoff=2.0, allow_split=True):
    """
    Try to fetch a list of IDs as one batch.
    On persistent server errors, optionally split the batch and fetch halves.
    Returns a list of arxiv.Result.
    """
    attempts = 0
    while True:
        try:
            search = arxiv.Search(id_list=ids)
            return list(client.results(search))
        except Exception as e:
            attempts += 1
            status = getattr(e, "status_code", None)
            # Try to respect Retry-After if the exception exposes a response
            retry_after = None
            try:
                resp = getattr(e, "response", None)
                if resp is not None and hasattr(resp, "headers"):
                    retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
            except Exception:
                pass

            # For 503/429 or any 5xx, exponential backoff (+ jitter)
            if status in (HTTPStatus.SERVICE_UNAVAILABLE, HTTPStatus.TOO_MANY_REQUESTS) or (
                isinstance(status, int) and 500 <= status < 600
            ):
                if attempts >= max_attempts:
                    # Last resort: split the batch if allowed
                    if allow_split and len(ids) > 1:
                        mid = len(ids) // 2
                        left  = _fetch_batch_with_retries(client, ids[:mid], max_attempts=max_attempts,
                                                          base_backoff=base_backoff, allow_split=True)
                        right = _fetch_batch_with_retries(client, ids[mid:], max_attempts=max_attempts,
                                                          base_backoff=base_backoff, allow_split=True)
                        return left + right
                    # otherwise, give up
                    raise

                wait = retry_after if retry_after is not None else base_backoff * (2 ** (attempts - 1))
                wait += random.uniform(0, 1.0)  # jitter
                logger.warning(
                    f"arXiv {status} for {len(ids)} ids. "
                    f"Attempt {attempts}/{max_attempts}. Sleeping {wait:.1f}s and retrying…"
                )
                time.sleep(wait)
            else:
                # Non-5xx errors: small backoff, then optionally split quickly
                if attempts >= min(3, max_attempts):
                    if allow_split and len(ids) > 1:
                        mid = len(ids) // 2
                        left  = _fetch_batch_with_retries(client, ids[:mid], max_attempts=max_attempts,
                                                          base_backoff=base_backoff, allow_split=True)
                        right = _fetch_batch_with_retries(client, ids[mid:], max_attempts=max_attempts,
                                                          base_backoff=base_backoff, allow_split=True)
                        return left + right
                    raise
                wait = base_backoff + random.uniform(0, 1.0)
                logger.warning(
                    f"arXiv error (status={status}) for {len(ids)} ids. "
                    f"Attempt {attempts}/{max_attempts}. Sleeping {wait:.1f}s and retrying…"
                )
                time.sleep(wait)

def get_arxiv_paper(query: str, debug: bool = False) -> list[ArxivPaper]:
    client = _safe_client()

    # Parse the RSS feed safely
    feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
    title = getattr(feed, "feed", {}).get("title", "") if isinstance(getattr(feed, "feed", {}), dict) \
            else getattr(getattr(feed, "feed", None), "title", "")  # robust access
    if title and 'Feed error for query' in title:
        raise Exception(f"Invalid ARXIV_QUERY: {query}.")

    papers: list[ArxivPaper] = []

    if not debug:
        # collect today's "new" IDs
        entries = getattr(feed, "entries", []) or []
        all_paper_ids = [
            getattr(i, "id", "").removeprefix("oai:arXiv.org:")
            for i in entries
            if getattr(i, "arxiv_announce_type", "new") == "new" and getattr(i, "id", "")
        ]

        # Tune these safely for arXiv export API
        batch_size = 20                 # small to avoid 503s
        polite_pause = 2.0              # seconds between successful batches
        max_attempts_per_batch = 6
        base_backoff = 2.0

        bar = tqdm(total=len(all_paper_ids), desc="Retrieving Arxiv papers")

        for start in range(0, len(all_paper_ids), batch_size):
            ids = all_paper_ids[start:start + batch_size]
            try:
                results = _fetch_batch_with_retries(
                    client, ids,
                    max_attempts=max_attempts_per_batch,
                    base_backoff=base_backoff,
                    allow_split=True
                )
            except Exception as e:
                # Log and continue with the rest instead of crashing the whole run
                logger.error(f"Failed to fetch batch starting at {start}: {e}")
                results = []

            batch = [ArxivPaper(r) for r in results]
            papers.extend(batch)
            bar.update(len(batch))
            # polite pacing between batches
            time.sleep(polite_pause)

        bar.close()

    else:
        logger.debug("Retrieve 5 arxiv papers regardless of the date.")
        search = arxiv.Search(query='cat:cs.AI', sort_by=arxiv.SortCriterion.SubmittedDate)
        for r in client.results(search):
            papers.append(ArxivPaper(r))
            if len(papers) == 5:
                break

    return papers


parser = argparse.ArgumentParser(description='Recommender system for academic papers')

def add_argument(*args, **kwargs):
    def get_env(key:str,default=None):
        # handle environment variables generated at Workflow runtime
        # Unset environment variables are passed as '', we should treat them as None
        v = os.environ.get(key)
        if v == '' or v is None:
            return default
        return v
    parser.add_argument(*args, **kwargs)
    arg_full_name = kwargs.get('dest',args[-1][2:])
    env_name = arg_full_name.upper()
    env_value = get_env(env_name)
    if env_value is not None:
        #convert env_value to the specified type
        if kwargs.get('type') == bool:
            env_value = env_value.lower() in ['true','1']
        else:
            env_value = kwargs.get('type')(env_value)
        parser.set_defaults(**{arg_full_name:env_value})


if __name__ == '__main__':
    
    add_argument('--zotero_id', type=str, help='Zotero user ID')
    add_argument('--zotero_key', type=str, help='Zotero API key')
    add_argument('--zotero_ignore',type=str,help='Zotero collection to ignore, using gitignore-style pattern.')
    add_argument('--send_empty', type=bool, help='If get no arxiv paper, send empty email',default=False)
    add_argument('--max_paper_num', type=int, help='Maximum number of papers to recommend',default=100)
    add_argument('--arxiv_query', type=str, help='Arxiv search query')
    add_argument('--smtp_server', type=str, help='SMTP server')
    add_argument('--smtp_port', type=int, help='SMTP port')
    add_argument('--sender', type=str, help='Sender email address')
    add_argument('--receiver', type=str, help='Receiver email address')
    add_argument('--sender_password', type=str, help='Sender email password')
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
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    assert (
        not args.use_llm_api or args.openai_api_key is not None
    )  # If use_llm_api is True, openai_api_key must be provided
    if args.debug:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
        logger.debug("Debug mode is on.")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    logger.info("Retrieving Zotero corpus...")
    corpus = get_zotero_corpus(args.zotero_id, args.zotero_key)
    logger.info(f"Retrieved {len(corpus)} papers from Zotero.")
    if args.zotero_ignore:
        logger.info(f"Ignoring papers in:\n {args.zotero_ignore}...")
        corpus = filter_corpus(corpus, args.zotero_ignore)
        logger.info(f"Remaining {len(corpus)} papers after filtering.")
    logger.info("Retrieving Arxiv papers...")
    papers = get_arxiv_paper(args.arxiv_query, args.debug)
    if len(papers) == 0:
        logger.info("No new papers found. Yesterday maybe a holiday and no one submit their work :). If this is not the case, please check the ARXIV_QUERY.")
        if not args.send_empty:
          exit(0)
    else:
        logger.info("Reranking papers...")
        papers = rerank_paper(papers, corpus)
        if args.max_paper_num != -1:
            papers = papers[:args.max_paper_num]
        if args.use_llm_api:
            logger.info("Using OpenAI API as global LLM.")
            set_global_llm(api_key=args.openai_api_key, base_url=args.openai_api_base, model=args.model_name, lang=args.language)
        else:
            logger.info("Using Local LLM as global LLM.")
            set_global_llm(lang=args.language)

    html = render_email(papers)
    logger.info("Sending email...")
    send_email(args.sender, args.receiver, args.sender_password, args.smtp_server, args.smtp_port, html)
    logger.success("Email sent successfully! If you don't receive the email, please check the configuration and the junk box.")

