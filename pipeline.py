"""
Agentic news-monitoring pipeline — multi-feed version with Claude API analysis.

Feed list (names + output folders) lives in feeds.json, which is plain
config and safe to commit. Each feed's actual RSS URL is supplied via
its own environment variable (named in feeds.json), set as a GitHub
Actions secret. No URL is ever hardcoded or committed.

For each new article:
1. Fetch and extract text from the URL
2. Send to Claude API for analysis:
   - French summary
   - Sentiment toward LFI/Mélenchon (including propaganda talking points)
   - Author background if available
3. Write comprehensive markdown file with metadata, analysis, and original Norwegian text

Secrets required (set as GitHub Actions secrets, injected as env vars):
  - one per feed, named whatever feeds.json says (e.g. RSS_URL_LFI_MELENCHON)
  - ANTHROPIC_API_KEY  : Claude API key for article analysis

Run locally for testing a single feed:
  RSS_URL_LFI_MELENCHON="https://..." ANTHROPIC_API_KEY="sk-..." python pipeline.py
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import feedparser
import trafilatura
import requests

CONFIG_PATH = Path(__file__).parent / "feeds.json"


def load_feed_config() -> list[dict]:
    if not CONFIG_PATH.exists():
        print(f"ERROR: {CONFIG_PATH} not found.", file=sys.stderr)
        sys.exit(1)
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)
    feeds = config.get("feeds", [])
    if not feeds:
        print("ERROR: feeds.json has no entries under 'feeds'.", file=sys.stderr)
        sys.exit(1)
    return feeds


def get_feed_url(env_var: str) -> str | None:
    """Read a feed's RSS URL from its designated env var. Returns None
    (rather than exiting) if missing, so one missing secret doesn't
    block other feeds from running."""
    url = os.environ.get(env_var)
    if not url:
        print(
            f"WARNING: env var {env_var} is not set — skipping this feed. "
            "Set it as a GitHub Actions secret, or export it locally for testing.",
            file=sys.stderr,
        )
    return url


def slugify_url(url: str) -> str:
    """Short, stable, filesystem-safe identifier for a URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def already_seen(url: str, output_dir: Path) -> bool:
    return (output_dir / f"{slugify_url(url)}.md").exists()


def fetch_feed_entries(rss_url: str):
    feed = feedparser.parse(rss_url)
    if feed.bozo:
        print(f"WARNING: feed parsing issue: {feed.bozo_exception}", file=sys.stderr)
    return feed.entries


def extract_article_text(url: str) -> str | None:
    """Best-effort fetch + extraction. Returns None on failure rather
    than raising, so one bad/paywalled URL doesn't kill the whole run."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        return trafilatura.extract(downloaded)
    except Exception as e:
        print(f"WARNING: extraction failed for {url}: {e}", file=sys.stderr)
        return None


def analyze_article_with_claude(article_text: str, url: str) -> dict | None:
    """Send article to Claude for analysis: French summary, sentiment toward
    LFI/Mélenchon (including propaganda talking points), author CV if available.
    Returns a dict with analysis, or None on API failure."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("WARNING: ANTHROPIC_API_KEY not set, skipping Claude analysis", file=sys.stderr)
        return None

    domain = urlparse(url).netloc

    prompt = f"""Analyze this Norwegian news article about French politics, LFI, or Mélenchon.

**Article URL:** {url}
**Source domain:** {domain}

---

{article_text}

---

Return ONLY a JSON object (no markdown, no preamble) with these fields:

{{
  "summary_fr": "2-3 sentence summary in French of the article's main points",
  "sentiment_fr": "French analysis: Is the tone positive, negative, or neutral toward LFI/Mélenchon? Note any common propaganda talking points (antisemitic tropes, 'brutal,' 'Islamist,' etc.) that appear and deserve correction. 2-3 paragraphs.",
  "author_name": "Author name if found in the article, or null",
  "author_cv": "2-3 sentence background/credentials of the author if available, in French, or null",
}}

Return ONLY the JSON object, with no additional text or markdown formatting."""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        data = response.json()

        # Extract text from Claude's response
        content = data.get("content", [])
        if not content or content[0].get("type") != "text":
            return None

        text = content[0]["text"]
        # Claude might wrap JSON in markdown backticks; strip them
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)

    except Exception as e:
        print(f"WARNING: Claude API call failed: {e}", file=sys.stderr)
        return None


def write_article_markdown(entry, extracted_text: str | None, output_dir: Path, analysis: dict | None = None) -> Path:
    slug = slugify_url(entry.link)
    path = output_dir / f"{slug}.md"

    title = getattr(entry, "title", "(no title)")
    link = entry.link
    published = getattr(entry, "published", "unknown")
    fetched_at = datetime.now(timezone.utc).isoformat()

    # Metadata section
    domain = urlparse(link).netloc
    author_name = analysis.get("author_name") if analysis else None
    author_cv = analysis.get("author_cv") if analysis else None

    metadata = f"""# {title}

## Metadata

- **URL:** {link}
- **Site:** {domain}
- **Published:** {published}
- **Fetched:** {fetched_at}
- **Author:** {author_name if author_name else "Unknown"}
"""

    # Analysis section (if available)
    analysis_section = ""
    if analysis:
        summary_fr = analysis.get("summary_fr", "")
        sentiment_fr = analysis.get("sentiment_fr", "")

        analysis_section = f"""
## Analyse (Français)

### Résumé
{summary_fr}

### Sentiment & Analyse critique
{sentiment_fr}
"""
        if author_cv:
            analysis_section += f"""
### Auteur (Background)
{author_cv}
"""

    # Original text section
    text_section = ""
    if extracted_text:
        text_section = f"""
## Texte original (Norvégien)

{extracted_text}
"""
    else:
        text_section = "\n## Texte original (Norvégien)\n\n_Extraction failed or content unavailable._"

    content = metadata + analysis_section + text_section

    path.write_text(content, encoding="utf-8")
    return path


def process_feed(feed_config: dict) -> int:
    name = feed_config["name"]
    env_var = feed_config["env_var"]
    output_dir = Path(__file__).parent / feed_config["output_dir"]

    print(f"--- Feed: {name} ---")

    rss_url = get_feed_url(env_var)
    if not rss_url:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    entries = fetch_feed_entries(rss_url)
    print(f"Feed '{name}' returned {len(entries)} entries.")

    new_count = 0
    for entry in entries:
        link = getattr(entry, "link", None)
        if not link:
            continue
        if already_seen(link, output_dir):
            continue

        print(f"  New article: {link}")
        text = extract_article_text(link)
        
        # Call Claude to analyze the article
        analysis = None
        if text:
            print(f"    Analyzing with Claude...")
            analysis = analyze_article_with_claude(text, link)
        
        path = write_article_markdown(entry, text, output_dir, analysis)
        print(f"    -> wrote {path}")
        new_count += 1

    print(f"Feed '{name}' done: {new_count} new article(s).")
    return new_count


def main():
    feeds = load_feed_config()
    total_new = 0
    for feed_config in feeds:
        total_new += process_feed(feed_config)
    print(f"\nAll feeds done. {total_new} new article(s) total.")


if __name__ == "__main__":
    main()