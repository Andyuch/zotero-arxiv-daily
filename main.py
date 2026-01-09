import arxiv

def _get_pdf_url_patch(links) -> str:
    """
    Finds the PDF link among a result's links and returns its URL.
    Should only be called once for a given `Result`, in its constructor.
    After construction, the URL should be available in `Result.pdf_url`.
    """
    pdf_urls = [link.href for link in links if "pdf" in link.href]
    if len(pdf_urls) == 0:
        return None
    return pdf_urls[0]

arxiv.Result._get_pdf_url = _get_pdf_url_patch

import argparse
import time
import random
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

def get_arxiv_paper(query: str, debug: bool = False) -> list["ArxivPaper"]:
    # arXiv legacy API etiquette: ~1 request / 3 seconds, one connection.
    MIN_INTERVAL_S = 3.2   # margin above 3s
    MAX_BACKOFF_S = 120.0

    # We control backoff ourselves (avoid arxiv.py retry loops that can amplify 429).
    client = arxiv.Client(num_retries=0, delay_seconds=0, page_size=100)

    feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
    if getattr(feed, "bozo", 0):
        raise Exception(
            f"Failed to parse arXiv RSS/ATOM feed for query={query}: {feed.bozo_exception}"
        )
    if hasattr(feed, "feed") and hasattr(feed.feed, "title") and "Feed error for query" in feed.feed.title:
        raise Exception(f"Invalid ARXIV_QUERY: {query}.")

    if debug:
        logger.debug("Retrieve 5 arxiv papers regardless of the date.")
        search = arxiv.Search(
            query="cat:cs.AI",
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        papers: list["ArxivPaper"] = []
        for r in client.results(search):
            papers.append(ArxivPaper(r))
            if len(papers) == 5:
                break
        return papers

    all_paper_ids = [
        e.id.removeprefix("oai:arXiv.org:")
        for e in feed.entries
        if getattr(e, "arxiv_announce_type", None) == "new"
    ]

    papers: list["ArxivPaper"] = []

    # Track IDs requested, not results returned.
    bar = tqdm(total=len(all_paper_ids), desc="Retrieving Arxiv papers")

    last_request_t = 0.0

    def _sleep_to_respect_rate_limit() -> None:
        nonlocal last_request_t
        now = time.monotonic()
        dt = now - last_request_t
        if dt < MIN_INTERVAL_S:
            time.sleep((MIN_INTERVAL_S - dt) + random.uniform(0.0, 0.25))
        last_request_t = time.monotonic()

    def _fetch_batch(id_batch: list[str]) -> list["ArxivPaper"]:
        backoff = MIN_INTERVAL_S
        while True:
            _sleep_to_respect_rate_limit()
            try:
                search = arxiv.Search(id_list=id_batch)
                # client.results(search) is a generator; force evaluation so exceptions happen here.
                results = list(client.results(search))
                return [ArxivPaper(r) for r in results]

            except arxiv.HTTPError as e:
                # arxiv.py stores status code inside the exception; attribute name can differ by version.
                code = getattr(e, "status", None) or getattr(e, "status_code", None)

                if code in (429, 503):
                    sleep_s = min(MAX_BACKOFF_S, backoff) + random.uniform(0.0, 1.0)
                    logger.warning(
                        "arXiv API HTTP {} for batch size={}; sleeping {:.1f}s then retrying.",
                        code, len(id_batch), sleep_s
                    )
                    time.sleep(sleep_s)
                    backoff = min(MAX_BACKOFF_S, backoff * 2.0)
                    continue

                # fail fast on other codes
                logger.exception("Unexpected arXiv HTTPError (code={})", code)
                raise

            except Exception:
                # Anything else (network hiccup, parse failure, etc.)
                logger.exception("Unexpected error while fetching arXiv batch of size={}", len(id_batch))
                raise

    CHUNK = 10  # 10 is gentler; 20 also ok if you’re disciplined with 3s gaps

    for i in range(0, len(all_paper_ids), CHUNK):
        batch_ids = all_paper_ids[i:i + CHUNK]
        batch = _fetch_batch(batch_ids)
        papers.extend(batch)
        bar.update(len(batch_ids))

    bar.close()
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
        # logger.remove()
        logger.add(sys.stdout, level="DEBUG")
        logger.debug("Debug mode is on.")
    else:
        # logger.remove()
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
