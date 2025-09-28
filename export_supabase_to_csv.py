"""
Export articles from Supabase database to CSV for WorldVue training
"""

import os
import csv
import sys
from datetime import datetime
from dotenv import load_dotenv

def ensure_httpx_proxy_support() -> None:
    """Ensure httpx.Client accepts the legacy proxy argument used by supabase."""
    try:
        import httpx
        import inspect
    except Exception:
        return

    client = httpx.Client
    if getattr(client, '_legacy_proxy_patch', False):
        return

    signature = inspect.signature(client.__init__)
    if 'proxy' in signature.parameters:
        client._legacy_proxy_patch = True
        return

    original_init = client.__init__

    def patched_init(self, *args, proxy=None, **kwargs):
        if proxy is not None and 'proxies' not in kwargs:
            kwargs['proxies'] = proxy
        return original_init(self, *args, **kwargs)

    client.__init__ = patched_init

    if hasattr(httpx, 'AsyncClient'):
        async_client = httpx.AsyncClient
        async_signature = inspect.signature(async_client.__init__)
        if 'proxy' not in async_signature.parameters:
            original_async_init = async_client.__init__

            def async_patched_init(self, *args, proxy=None, **kwargs):
                if proxy is not None and 'proxies' not in kwargs:
                    kwargs['proxies'] = proxy
                return original_async_init(self, *args, **kwargs)

            async_client.__init__ = async_patched_init

    client._legacy_proxy_patch = True

ensure_httpx_proxy_support()

from supabase import create_client

load_dotenv()

# Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_ANON_KEY')
OUTPUT_FILE = 'articles_for_training.csv'

def export_articles_to_csv(limit=None, min_word_count=50, output_file='articles_for_training.csv'):
    """Export articles from Supabase to CSV format compatible with WorldVue"""

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: Supabase credentials not found!")
        print("Set SUPABASE_URL and SUPABASE_KEY in .env file")
        sys.exit(1)

    print("Connecting to Supabase...")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    try:
        # Build query
        query = supabase.table('articles').select('*')

        # Filter for articles with substantial content
        if min_word_count:
            print(f"Filtering for articles with at least {min_word_count} words...")

        # Add limit if specified
        if limit:
            query = query.limit(limit)
            print(f"Limiting to {limit} articles...")

        # Execute query
        print("Fetching articles from database...")

        # Fetch all articles in batches if no limit specified
        if not limit:
            print("Fetching all articles (this may take a moment)...")
            articles = []
            batch_size = 1000
            offset = 0

            while True:
                batch_query = supabase.table('articles').select('*').range(offset, offset + batch_size - 1)
                batch_result = batch_query.execute()
                batch_articles = batch_result.data

                if not batch_articles:
                    break

                articles.extend(batch_articles)
                offset += batch_size
                print(f"Fetched {len(articles)} articles so far...")

                if len(batch_articles) < batch_size:
                    break
        else:
            result = query.execute()
            articles = result.data

        if not articles:
            print("No articles found in database!")
            return

        print(f"Found {len(articles)} articles")

        # Filter by word count if specified
        if min_word_count:
            filtered_articles = []
            for article in articles:
                content = (article.get('content') or '') + ' ' + (article.get('summary') or '')
                word_count = len(content.split()) if content else 0
                if word_count >= min_word_count:
                    filtered_articles.append(article)

            print(f"After filtering: {len(filtered_articles)} articles with >={min_word_count} words")
            articles = filtered_articles

        if not articles:
            print("No articles meet the word count criteria!")
            return

        # Write to CSV
        print(f"Writing to {output_file}...")
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'id', 'title', 'text', 'source_name', 'source_country',
                'language', 'published_at', 'url', 'tilt', 'frames',
                'keywords', 'word_count', 'has_full_text'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for article in articles:
                # Combine content and summary for text field
                content = article.get('content') or ''
                summary = article.get('summary') or ''
                combined_text = f"{summary} {content}".strip() if summary else content

                # Handle missing fields with defaults
                row = {
                    'id': article.get('id', ''),
                    'title': article.get('title', ''),
                    'text': combined_text,
                    'source_name': article.get('source_name', ''),
                    'source_country': article.get('source_country', 'UN'),
                    'language': 'en',  # Default to English for now
                    'published_at': article.get('published_at', ''),
                    'url': article.get('url', ''),
                    'tilt': article.get('tilt', 0.0),
                    'frames': str(article.get('frames', {})),
                    'keywords': ', '.join(article.get('keywords', [])),
                    'word_count': article.get('word_count', 0),
                    'has_full_text': article.get('has_full_text', False)
                }
                writer.writerow(row)

        print(f"Successfully exported {len(articles)} articles to {output_file}")
        print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")

        # Show sample statistics
        word_counts = [article.get('word_count', 0) for article in articles]
        if word_counts:
            avg_words = sum(word_counts) / len(word_counts)
            print(f"Average word count: {avg_words:.0f}")

        sources = {}
        for article in articles:
            source = article.get('source_name', 'Unknown')
            sources[source] = sources.get(source, 0) + 1

        print(f"Top sources:")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {source}: {count} articles")

    except Exception as e:
        print(f"Error exporting articles: {e}")
        sys.exit(1)

def main():
    """Main function with command line arguments"""
    import argparse

    parser = argparse.ArgumentParser(description='Export Supabase articles to CSV')
    parser.add_argument('--limit', type=int, help='Limit number of articles')
    parser.add_argument('--min-words', type=int, default=50,
                       help='Minimum word count (default: 50)')
    parser.add_argument('--output', default='articles_for_training.csv',
                       help='Output CSV file (default: articles_for_training.csv)')

    args = parser.parse_args()

    output_file = args.output

    print("="*60)
    print(" Supabase to CSV Exporter for WorldVue Training")
    print("="*60)

    export_articles_to_csv(limit=args.limit, min_word_count=args.min_words, output_file=output_file)

if __name__ == '__main__':
    main()