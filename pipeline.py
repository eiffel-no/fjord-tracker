"""
Agentic news-monitoring pipeline — multi-feed version.

Feed list (names + output folders) lives in feeds.json, which is plain
config and safe to commit. Each feed's actual RSS URL is supplied via
its own environment variable (named in feeds.json), set as a GitHub
Actions secret. No URL is ever hardcoded or committed.

Secrets required (set as GitHub Actions secrets, injected as env vars):
  - one per feed, named whatever feeds.json says (e.g. RSS_URL_LFI_MELENCHON)
  - ANTHROPIC_API_KEY      : (used in a later step, not yet wired in here)

Run locally for testing a single feed:
  RSS_URL_LFI_MELENCHON="https://..." python pipeline.py
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import trafilatura

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


def write_article_markdown(entry, extracted_text: str | None, output_dir: Path) -> Path:
    slug = slugify_url(entry.link)
    path = output_dir / f"{slug}.md"

    title = getattr(entry, "title", "(no title)")
    link = entry.link
    published = getattr(entry, "published", "unknown")
    fetched_at = datetime.now(timezone.utc).isoformat()

    body = extracted_text if extracted_text else "_Extraction failed or content unavailable._"

    content = f"""# {title}

- **Source URL:** {link}
- **Published (per feed):** {published}
- **Fetched:** {fetched_at}
- **Extraction status:** {"ok" if extracted_text else "failed"}

## Raw extracted text

{body}

## LLM summary (Norwegian → English/French, sentiment, themes)

_TODO: Claude API call goes here. Not yet wired in._
"""
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
        path = write_article_markdown(entry, text, output_dir)
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
