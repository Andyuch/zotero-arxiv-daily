import math
import html
from tqdm import tqdm
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
import smtplib
import datetime
import time
from loguru import logger

framework = """
<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em; /* 调整星星大小 */
      line-height: 1; /* 确保垂直对齐 */
      display: inline-flex;
      align-items: center; /* 保持对齐 */
    }
    .half-star {
      display: inline-block;
      width: 0.5em; /* 半颗星的宽度 */
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
    .journal-badge {
      display: inline-block;
      font-size: 13px;
      font-weight: bold;
      color: #17365d;
      background: #e8f1fb;
      border: 1px solid #c5d9ef;
      border-radius: 12px;
      padding: 3px 9px;
      margin: 6px 0 2px 0;
    }
    .source-note {
      font-size: 12px;
      color: #888;
      margin-left: 6px;
    }
  </style>
</head>
<body>

<div>
    __CONTENT__
</div>

<br><br>
<div>
To unsubscribe, remove your email in your Github Action setting.
</div>

</body>
</html>
"""

def get_empty_html():
  block_template = """
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        No Papers Today. Take a Rest!
    </td>
  </tr>
  </table>
  """
  return block_template

def get_block_html(
    *,
    title: str,
    authors: str,
    rate: str,
    journal: str,
    source: str,
    abstract: str,
    paper_url: str,
    pdf_url: str | None = None,
    code_url: str | None = None,
    affiliations: str | None = None,
    arxiv_id: str | None = None,
    doi: str | None = None,
    published_at: str | None = None,
):
    title = html.escape(title or "")
    authors = html.escape(authors or "Unknown Authors")
    journal = html.escape(journal or source or "Unknown Journal")
    source = html.escape(source or "")
    abstract = html.escape(abstract or "")
    affiliations = html.escape(affiliations or "Unknown Affiliation")

    meta_rows = []
    if published_at:
        meta_rows.append(f"<strong>Published:</strong> {html.escape(published_at)}")
    if doi:
        safe_doi = html.escape(doi)
        meta_rows.append(
            f'<strong>DOI:</strong> <a href="https://doi.org/{safe_doi}" target="_blank">{safe_doi}</a>'
        )
    if arxiv_id:
        safe_id = html.escape(arxiv_id)
        meta_rows.append(
            f'<strong>arXiv ID:</strong> <a href="https://arxiv.org/abs/{safe_id}" target="_blank">{safe_id}</a>'
        )

    metadata_row = ""
    if meta_rows:
        metadata_row = (
            '<tr><td style="font-size: 13px; color: #555; padding: 5px 0;">'
            + "<br>".join(meta_rows)
            + "</td></tr>"
        )

    buttons = []
    if paper_url:
        buttons.append(
            f'<a href="{html.escape(paper_url, quote=True)}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #337ab7; padding: 8px 16px; border-radius: 4px;">Article</a>'
        )
    if pdf_url:
        buttons.append(
            f'<a href="{html.escape(pdf_url, quote=True)}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 8px 16px; border-radius: 4px; margin-left: 8px;">PDF</a>'
        )
    if code_url:
        buttons.append(
            f'<a href="{html.escape(code_url, quote=True)}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #5bc0de; padding: 8px 16px; border-radius: 4px; margin-left: 8px;">Code</a>'
        )

    return f"""
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
      <tr><td style="font-size: 20px; font-weight: bold; color: #333;">{title}</td></tr>
      <tr><td><span class="journal-badge">{journal}</span><span class="source-note">via {source}</span></td></tr>
      <tr><td style="font-size: 14px; color: #666; padding: 8px 0;">{authors}<br><i>{affiliations}</i></td></tr>
      <tr><td style="font-size: 14px; color: #333; padding: 6px 0;"><strong>Relevance:</strong> {rate}</td></tr>
      {metadata_row}
      <tr><td style="font-size: 14px; color: #333; padding: 8px 0;"><strong>TLDR:</strong> {abstract}</td></tr>
      <tr><td style="padding: 8px 0;">{''.join(buttons)}</td></tr>
    </table>
    """

def get_stars(percentile: float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    p = float(percentile or 0.0)
    if p >= 95:
        stars = 5.0
    elif p >= 85:
        stars = 4.5
    elif p >= 70:
        stars = 4.0
    elif p >= 50:
        stars = 3.5
    elif p >= 30:
        stars = 3.0
    elif p >= 15:
        stars = 2.5
    else:
        stars = 2.0

    full_star_num = int(stars)
    half_star_num = 1 if stars % 1 else 0
    return (
        '<div class="star-wrapper">'
        + full_star * full_star_num
        + half_star * half_star_num
        + '</div>'
    )


def get_relevance_html(score: float, percentile: float) -> str:
    top_pct = max(1, round(100.0 - float(percentile or 0.0)))
    return (
        f"{get_stars(percentile)}"
        f'<span style="margin-left:8px;color:#777;font-size:12px;">'
        f"{float(score or 0.0):.2f} · top {top_pct}%"
        f"</span>"
    )

def render_email(papers:list):
    parts = []
    if len(papers) == 0 :
        return framework.replace('__CONTENT__', get_empty_html())

    for p in tqdm(papers,desc='Rendering Email'):
        rate = get_relevance_html(
            p.score,
            getattr(p, "relevance_percentile", 0.0),
        )
        author_list = [a.name for a in p.authors]
        num_authors = len(author_list)

        if num_authors <= 5:
            authors = ', '.join(author_list)
        else:
            authors = ', '.join(author_list[:3] + ['...'] + author_list[-2:])
        if p.affiliations is not None:
            affiliations = p.affiliations[:5]
            affiliations = ', '.join(affiliations)
            if len(p.affiliations) > 5:
                affiliations += ', ...'
        else:
            affiliations = 'Unknown Affiliation'

        parts.append(
            get_block_html(
                title=p.title,
                authors=authors,
                rate=rate,
                journal=getattr(p, "journal", "arXiv"),
                source=getattr(p, "source", "arXiv"),
                abstract=p.tldr,
                paper_url=getattr(p, "paper_url", ""),
                pdf_url=p.pdf_url,
                code_url=p.code_url,
                affiliations=affiliations,
                arxiv_id=getattr(p, "arxiv_id", None),
                doi=getattr(p, "doi", None),
                published_at=getattr(p, "published_at", None),
            )
        )
        time.sleep(10)

    content = '<br>' + '</br><br>'.join(parts) + '</br>'
    return framework.replace('__CONTENT__', content)

def send_email(sender:str, receiver:str, password:str,smtp_server:str,smtp_port:int, html:str,):
    def _format_addr(s):
        name, addr = parseaddr(s)
        return formataddr((Header(name, 'utf-8').encode(), addr))

    msg = MIMEText(html, 'html', 'utf-8')
    msg['From'] = _format_addr('Github Action <%s>' % sender)
    msg['To'] = _format_addr('You <%s>' % receiver)
    today = datetime.datetime.now().strftime('%Y/%m/%d')
    msg['Subject'] = Header(f'Daily Research Papers {today}', 'utf-8').encode()

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
    except Exception as e:
        logger.warning(f"Failed to use TLS. {e}")
        logger.warning(f"Try to use SSL.")
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)

    server.login(sender, password)
    server.sendmail(sender, [receiver], msg.as_string())
    server.quit()
