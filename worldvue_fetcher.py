"""
WorldVue AI - Parallel High-Performance Fetcher
Processes multiple feeds concurrently for 10x faster updates
"""

import os
import sys
import hashlib
import re
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Set, Dict, Any, Tuple
from collections import Counter, deque
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore
import queue

import feedparser
import requests
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

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

# Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_ANON_KEY')
MAX_ARTICLES_PER_FEED = int(os.getenv('MAX_ARTICLES_PER_FEED', '30'))
USER_AGENT = os.getenv('USER_AGENT', 'WorldVueBot/2.0')

# Parallel processing settings
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '10'))  # Number of parallel threads
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '50'))    # Articles per database batch
RATE_LIMIT_PER_DOMAIN = 2  # Max requests per second per domain
FETCH_TIMEOUT = 12
RETRY_ATTEMPTS = 2

# Initialize Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None

# Setup logging with thread info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(threadName)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Thread-safe components
insert_lock = Lock()
stats_lock = Lock()
domain_limiters = {}
domain_limiter_lock = Lock()

# Global statistics
global_stats = {
    'feeds_processed': 0,
    'feeds_failed': 0,
    'articles_new': 0,
    'articles_duplicate': 0,
    'articles_failed': 0,
    'start_time': None,
    'errors': []
}

# Enhanced bias lexicons
LEXICONS = {
    'left': {
        'climate crisis': 3, 'climate emergency': 3, 'social justice': 3,
        'systemic racism': 3, 'wealth inequality': 3, 'universal healthcare': 3,
        'inequality': 2, 'diversity': 2, 'inclusion': 2, 'equity': 2,
        'progressive': 2, 'regulation': 2, 'workers rights': 2,
        'environment': 1, 'sustainable': 1, 'renewable': 1, 'green': 1
    },
    'right': {
        'radical left': 3, 'law and order': 3, 'illegal immigration': 3,
        'traditional values': 3, 'free market': 3, 'deregulation': 3,
        'freedom': 2, 'liberty': 2, 'patriotism': 2, 'sovereignty': 2,
        'border security': 2, 'gun rights': 2, 'small government': 2,
        'business': 1, 'military': 1, 'defense': 1, 'security': 1
    }
}

FRAMES = {
    'economic': ['economy', 'jobs', 'market', 'gdp', 'inflation', 'trade', 'unemployment', 'recession', 'growth', 'finance'],
    'political': ['election', 'vote', 'campaign', 'government', 'president', 'parliament', 'policy', 'legislation', 'democracy'],
    'humanitarian': ['refugee', 'crisis', 'aid', 'rights', 'humanitarian', 'charity', 'poverty', 'hunger', 'disaster'],
    'security': ['military', 'defense', 'security', 'threat', 'terrorism', 'war', 'conflict', 'attack', 'cyber'],
    'scientific': ['research', 'study', 'data', 'science', 'evidence', 'analysis', 'technology', 'innovation', 'discovery'],
    'scandal': ['scandal', 'corruption', 'investigation', 'allegation', 'fraud', 'controversy', 'crime', 'lawsuit']
}

DROP_PARAMS = {
    'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
    'utm_id', 'gclid', 'fbclid', 'mc_cid', 'mc_eid', 's', 'ref', 'source'
}

class RateLimiter:
    """Thread-safe rate limiter per domain"""
    def __init__(self, max_per_second=2):
        self.max_per_second = max_per_second
        self.requests = deque()
        self.lock = Lock()
    
    def wait_if_needed(self):
        """Wait if we've exceeded rate limit"""
        with self.lock:
            now = time.time()
            # Remove old requests
            while self.requests and self.requests[0] < now - 1:
                self.requests.popleft()
            
            # Check if we need to wait
            if len(self.requests) >= self.max_per_second:
                sleep_time = 1 - (now - self.requests[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.requests.append(time.time())

def get_domain_limiter(url: str) -> RateLimiter:
    """Get or create rate limiter for domain"""
    domain = urlparse(url).netloc
    
    with domain_limiter_lock:
        if domain not in domain_limiters:
            domain_limiters[domain] = RateLimiter(RATE_LIMIT_PER_DOMAIN)
        return domain_limiters[domain]

def canonicalize_url(url: str) -> str:
    """Normalize URL by removing tracking params"""
    try:
        p = urlparse(url)
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) 
             if k not in DROP_PARAMS]
        new_q = urlencode(q)
        netloc = p.netloc.lower()
        scheme = p.scheme.lower() if p.scheme else 'http'
        if netloc.endswith(':80') and scheme == 'http':
            netloc = netloc[:-3]
        if netloc.endswith(':443') and scheme == 'https':
            netloc = netloc[:-4]
        cleaned = p._replace(scheme=scheme, netloc=netloc, query=new_q, fragment='')
        return urlunparse(cleaned)
    except:
        return url

def normalize_text(text: str) -> str:
    """Normalize text for analysis with proper encoding"""
    if not text:
        return ""

    # Handle encoding issues
    if isinstance(text, bytes):
        try:
            text = text.decode('utf-8', errors='replace')
        except UnicodeDecodeError:
            text = text.decode('latin-1', errors='replace')

    # Normalize Unicode characters
    import unicodedata
    text = unicodedata.normalize('NFKD', text)

    # Clean up HTML and URLs
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    text = re.sub(r'http[s]?://\S+', '', text)  # Remove URLs
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)  # Remove HTML entities

    # Normalize whitespace and case
    text = re.sub(r'\s+', ' ', text.lower().strip())

    # Ensure ASCII compatibility for database storage
    try:
        text = text.encode('utf-8', errors='replace').decode('utf-8')
    except UnicodeEncodeError:
        # Fallback: keep only ASCII characters
        text = ''.join(char for char in text if ord(char) < 128)

    return text

# Import the advanced bias detector
try:
    from advanced_bias_detector import calculate_tilt as advanced_calculate_tilt
    USE_ADVANCED_BIAS = True
except ImportError:
    USE_ADVANCED_BIAS = False
    logger.warning("Advanced bias detector not found, using basic method")

def calculate_tilt(text: str, source_name: str = None) -> float:
    """Calculate political bias with advanced detection"""
    if USE_ADVANCED_BIAS:
        # Use the sophisticated multi-layer analysis
        return advanced_calculate_tilt(text, source_name)
    
    # Fallback to basic method if advanced not available
    text_normalized = normalize_text(text)
    
    left_score = 0
    right_score = 0
    
    for term, weight in LEXICONS['left'].items():
        count = text_normalized.count(term)
        if count > 0:
            left_score += weight * count * (1.2 if len(term.split()) > 1 else 1.0)
    
    for term, weight in LEXICONS['right'].items():
        count = text_normalized.count(term)
        if count > 0:
            right_score += weight * count * (1.2 if len(term.split()) > 1 else 1.0)
    
    text_length = max(1, len(text_normalized.split()))
    normalization_factor = min(1.0, 100.0 / text_length)
    
    left_score *= normalization_factor
    right_score *= normalization_factor
    
    total = max(1, left_score + right_score)
    raw_tilt = (right_score - left_score) / total
    
    import math
    smoothed_tilt = 2 / (1 + math.exp(-2 * raw_tilt)) - 1
    
    return max(-1.0, min(1.0, smoothed_tilt))

def calculate_frames(text: str) -> dict:
    """Calculate coverage frames"""
    text_normalized = normalize_text(text)
    words = text_normalized.split()
    word_set = set(words)
    
    frame_scores = {}
    for frame, keywords in FRAMES.items():
        score = 0
        for keyword in keywords:
            if keyword in word_set:
                score += len(keyword) / 5
        frame_scores[frame] = score
    
    total = max(1, sum(frame_scores.values()))
    return {k: round(v/total, 3) for k, v in frame_scores.items()}

def extract_keywords(text: str, limit: int = 15) -> List[str]:
    """Extract keywords efficiently"""
    text_normalized = normalize_text(text)
    words = re.findall(r'\b[a-z]+\b', text_normalized)
    
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'as', 'is', 'was', 'are', 'were', 'been', 'be',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could'
    }
    
    words = [w for w in words if w not in stopwords and len(w) > 3]
    word_counts = Counter(words)
    
    keywords = []
    for word, count in word_counts.most_common(limit * 2):
        if count > 1 and count < len(words) / 10:
            keywords.append(word)
            if len(keywords) >= limit:
                break
    
    return keywords

def fetch_article_fulltext(url: str) -> str:
    """Simple but effective text extraction"""
    try:
        headers = {'User-Agent': USER_AGENT}
        r = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        if r.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.text, 'html.parser')

            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'form']):
                element.decompose()

            # Look for main content areas
            content = None
            for selector in ['article', '[class*="content"]', '[class*="article"]', '[class*="story"]', 'main', '[id*="content"]']:
                found = soup.select(selector)
                if found:
                    content = found[0]
                    break

            if not content:
                content = soup.find('body') or soup

            # Extract text
            text = content.get_text(separator=' ', strip=True)

            # Clean up
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()

            if len(text) > 500:  # Ensure substantial content
                return text[:10000]

    except Exception as e:
        # Fallback to simple extraction
        try:
            headers = {'User-Agent': USER_AGENT}
            r = requests.get(url, headers=headers, timeout=5, allow_redirects=True)
            if r.status_code == 200:
                html = r.text
                # Remove scripts and styles only
                html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
                html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
                # Extract text from body
                body_match = re.search(r'<body[^>]*>(.*?)</body>', html, re.DOTALL)
                if body_match:
                    text = re.sub(r'<[^>]+>', ' ', body_match.group(1))
                    text = re.sub(r'\s+', ' ', text).strip()
                    return text[:5000] if len(text) > 300 else ""
        except:
            pass
    return ""

def process_feed_entry(entry: dict, source_name: str, country: str, 
                       fetch_fulltext: bool = False) -> Dict[str, Any]:
    """Process a single feed entry into article data"""
    url = entry.get('link', '')
    if not url:
        return None
    
    title = entry.get('title', '')
    if not title:
        return None
    
    summary = entry.get('summary', '')
    
    # Generate article ID
    canonical = canonicalize_url(url)
    article_id = hashlib.sha256(f"{canonical}:{source_name}".encode()).hexdigest()[:16]
    
    # Parse date
    published_at = datetime.now(timezone.utc)
    for date_field in ['published_parsed', 'updated_parsed', 'created_parsed']:
        if entry.get(date_field):
            try:
                published_at = datetime(*entry[date_field][:6], tzinfo=timezone.utc)
                break
            except:
                pass
    
    # Fetch full text if enabled
    full_text = ""
    if fetch_fulltext:
        full_text = fetch_article_fulltext(url)
    
    # Content Quality Check - Skip articles without sufficient content
    # Require either substantial full text OR a meaningful summary
    content_quality_check = (
        len(full_text) > 100 or  # Has substantial full text content
        (len(summary) > 80 and len(title.split()) > 3)  # OR has meaningful summary + title
    )

    if not content_quality_check:
        logger.debug(f"Skipping article with insufficient content: {title[:50]}...")
        return None  # Skip articles without sufficient content

    # Analyze with advanced bias detection
    analysis_text = f"{title} {summary} {full_text}"

    # Additional quality check - ensure we have enough text for analysis
    if len(analysis_text.split()) < 10:
        logger.debug(f"Skipping article with insufficient analysis text: {title[:50]}...")
        return None

    return {
        'id': article_id,
        'url': url,
        'source_name': source_name,
        'source_country': country.upper(),
        'title': title[:500],
        'summary': summary[:1000] if summary else None,
        'content': full_text[:10000] if full_text else None,
        'published_at': published_at.isoformat(),
        'fetched_at': datetime.now(timezone.utc).isoformat(),
        'tilt': calculate_tilt(analysis_text, source_name),  # Pass source for bias adjustment
        'frames': calculate_frames(analysis_text),
        'keywords': extract_keywords(analysis_text),
        'scoring_version': 'v3.0-advanced-filtered',  # Updated version to indicate content filtering
        'word_count': len(analysis_text.split()),
        'has_full_text': len(full_text) > 100
    }

def bulk_check_duplicates(urls: List[str]) -> Set[str]:
    """Check for duplicate URLs in database"""
    if not urls or not supabase:
        return set()
    
    try:
        # Check in batches
        existing = set()
        batch_size = 100
        
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i+batch_size]
            result = supabase.table('articles').select('url').in_('url', batch).execute()
            existing.update(a['url'] for a in result.data)
        
        return existing
    except Exception as e:
        logger.error(f"Duplicate check error: {e}")
        return set()

def insert_articles_batch(articles: List[Dict[str, Any]]) -> int:
    """Thread-safe batch insert to BOTH Supabase and local backup"""
    if not articles:
        return 0
    
    inserted_cloud = 0
    inserted_local = 0
    
    # Insert to Supabase (primary)
    if supabase:
        with insert_lock:
            try:
                result = supabase.table('articles').upsert(
                    articles,
                    on_conflict='url'
                ).execute()
                inserted_cloud = len(articles)
            except Exception as e:
                logger.error(f"Batch insert error: {e}")
                # Try individual inserts
                for article in articles:
                    try:
                        supabase.table('articles').upsert(article, on_conflict='url').execute()
                        inserted_cloud += 1
                    except:
                        pass
    
    # ALWAYS backup locally (even if Supabase fails)
    try:
        from local_backup import backup_articles_locally
        inserted_local = backup_articles_locally(articles)
        if inserted_local > 0:
            logger.debug(f"Backed up {inserted_local} articles locally")
    except ImportError:
        logger.debug("Local backup not available")
    except Exception as e:
        logger.error(f"Local backup error: {e}")
    
    return inserted_cloud

def process_single_feed(feed_data: Tuple[str, str, str]) -> Dict[str, Any]:
    """Process a single feed - runs in thread pool"""
    source_name, country, feed_url = feed_data
    
    result = {
        'source': source_name,
        'success': False,
        'new_articles': 0,
        'duplicates': 0,
        'errors': []
    }
    
    try:
        # Rate limit per domain
        limiter = get_domain_limiter(feed_url)
        limiter.wait_if_needed()
        
        logger.info(f"Fetching {source_name} ({country})")
        
        # Fetch and parse feed
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(feed_url, headers=headers, timeout=FETCH_TIMEOUT)
        feed = feedparser.parse(response.content)
        
        if feed.bozo and feed.bozo_exception:
            logger.warning(f"Feed parse warning for {source_name}: {feed.bozo_exception}")
        
        # Process entries
        entries_to_process = []
        urls_to_check = []
        
        for entry in feed.entries[:MAX_ARTICLES_PER_FEED]:
            if entry.get('link') and entry.get('title'):
                urls_to_check.append(entry['link'])
                entries_to_process.append(entry)
        
        if not urls_to_check:
            result['errors'].append("No valid entries found")
            return result
        
        # Check for duplicates
        existing_urls = bulk_check_duplicates(urls_to_check)
        result['duplicates'] = len(existing_urls)
        
        # Process new articles
        new_articles = []
        for entry in entries_to_process:
            if entry['link'] not in existing_urls:
                article = process_feed_entry(entry, source_name, country, fetch_fulltext=True)
                if article:
                    new_articles.append(article)
        
        # Insert new articles
        if new_articles:
            inserted = insert_articles_batch(new_articles)
            result['new_articles'] = inserted
            logger.info(f"✓ {source_name}: {inserted} new, {result['duplicates']} duplicates")
        else:
            logger.info(f"✓ {source_name}: All {result['duplicates']} articles exist")
        
        result['success'] = True
        
    except requests.RequestException as e:
        result['errors'].append(f"Network error: {str(e)}")
        logger.error(f"✗ {source_name}: Network error")
    except Exception as e:
        result['errors'].append(str(e))
        logger.error(f"✗ {source_name}: {e}")
    
    # Update global stats
    with stats_lock:
        if result['success']:
            global_stats['feeds_processed'] += 1
        else:
            global_stats['feeds_failed'] += 1
        global_stats['articles_new'] += result['new_articles']
        global_stats['articles_duplicate'] += result['duplicates']
        if result['errors']:
            global_stats['errors'].extend(result['errors'])
    
    return result

def process_feeds_parallel(feeds: List[Tuple[str, str, str]], max_workers: int = MAX_WORKERS):
    """Process multiple feeds in parallel"""
    global_stats['start_time'] = time.time()
    
    logger.info(f"Starting parallel processing with {max_workers} workers for {len(feeds)} feeds")
    
    results = []
    failed_feeds = []
    
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='Feed') as executor:
        # Submit all tasks
        future_to_feed = {
            executor.submit(process_single_feed, feed): feed 
            for feed in feeds
        }
        
        # Process completed tasks with progress bar
        completed = 0
        total = len(feeds)
        
        for future in as_completed(future_to_feed):
            feed = future_to_feed[future]
            completed += 1
            
            try:
                result = future.result(timeout=30)
                results.append(result)
                
                if not result['success']:
                    failed_feeds.append(feed[0])
                
                # Progress update
                if completed % 10 == 0 or completed == total:
                    elapsed = time.time() - global_stats['start_time']
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    logger.info(f"Progress: {completed}/{total} feeds ({completed*100//total}%) - "
                              f"Rate: {rate:.1f} feeds/sec - ETA: {eta:.0f}s")
                    
            except Exception as e:
                logger.error(f"Failed to process {feed[0]}: {e}")
                failed_feeds.append(feed[0])
    
    return results, failed_feeds

def load_feeds(feeds_file: str = 'feeds.csv') -> List[Tuple[str, str, str]]:
    """Load feeds from CSV file"""
    feeds = []
    
    if not os.path.exists(feeds_file):
        logger.error(f"Feeds file {feeds_file} not found!")
        return feeds
    
    with open(feeds_file, 'r', encoding='utf-8') as f:
        # Skip header
        header = f.readline().strip()
        if not header.startswith('source_name'):
            logger.warning("Unexpected CSV header, trying anyway...")
        
        for line_num, line in enumerate(f, start=2):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split(',', 2)
            if len(parts) != 3:
                logger.warning(f"Line {line_num}: Invalid format, skipping")
                continue
            
            source_name, country, feed_url = [p.strip() for p in parts]
            feeds.append((source_name, country, feed_url))
    
    logger.info(f"Loaded {len(feeds)} feeds from {feeds_file}")
    return feeds

def print_statistics():
    """Print final statistics"""
    elapsed = time.time() - global_stats['start_time']
    
    print("\n" + "="*60)
    print(" FETCHING COMPLETE - STATISTICS")
    print("="*60)
    print(f" Duration: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f" Feeds processed: {global_stats['feeds_processed']}")
    print(f" Feeds failed: {global_stats['feeds_failed']}")
    print(f" Articles added: {global_stats['articles_new']}")
    print(f" Duplicates skipped: {global_stats['articles_duplicate']}")
    
    if global_stats['feeds_processed'] > 0:
        print(f" Average time per feed: {elapsed/global_stats['feeds_processed']:.2f}s")
        print(f" Processing rate: {global_stats['feeds_processed']/elapsed:.2f} feeds/sec")
    
    if global_stats['errors']:
        print(f"\n Errors encountered: {len(set(global_stats['errors']))}")
    
    print("="*60)

def get_database_stats():
    """Get current database statistics"""
    try:
        if not supabase:
            return
        
        # Total articles
        total = supabase.table('articles').select('id', count='exact').execute()
        
        # Recent articles (last 24h)
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        recent = supabase.table('articles').select('id', count='exact').gte(
            'published_at', yesterday
        ).execute()
        
        print("\nDatabase Status:")
        print(f"   Total articles: {total.count:,}")
        print(f"   Last 24 hours: {recent.count:,}")
        
    except Exception as e:
        logger.error(f"Stats error: {e}")

def main():
    """Main execution"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: Supabase not configured!")
        print("Set SUPABASE_URL and SUPABASE_KEY in .env")
        sys.exit(1)
    
    print("="*60)
    print(" WorldVue AI - Parallel Fetcher v2.0")
    print("="*60)
    print(f" Workers: {MAX_WORKERS}")
    print(f" Batch size: {BATCH_SIZE}")
    print(f" Articles per feed: {MAX_ARTICLES_PER_FEED}")
    print("="*60)
    
    # Show initial stats
    get_database_stats()
    
    # Load feeds
    feeds = load_feeds('feeds.csv')
    if not feeds:
        print("No feeds to process!")
        return
    
    # Process feeds in parallel
    print(f"\nProcessing {len(feeds)} feeds...")
    results, failed_feeds = process_feeds_parallel(feeds, max_workers=MAX_WORKERS)
    
    # Print statistics
    print_statistics()
    
    # Show final stats
    get_database_stats()
    
    # Report failed feeds
    if failed_feeds:
        print(f"\n⚠️  Failed feeds ({len(failed_feeds)}):")
        for feed in failed_feeds[:10]:  # Show first 10
            print(f"   - {feed}")
        if len(failed_feeds) > 10:
            print(f"   ... and {len(failed_feeds) - 10} more")
    
    print("\n✅ All done! Use 'python worldvue_server.py' to view the data.")

if __name__ == '__main__':
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='WorldVue Parallel Fetcher')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS,
                       help=f'Number of parallel workers (default: {MAX_WORKERS})')
    parser.add_argument('--feeds', type=str, default='feeds.csv',
                       help='Path to feeds CSV file')
    parser.add_argument('--limit', type=int, default=MAX_ARTICLES_PER_FEED,
                       help=f'Max articles per feed (default: {MAX_ARTICLES_PER_FEED})')
    
    args = parser.parse_args()
    
    # Override settings
    MAX_WORKERS = args.workers
    MAX_ARTICLES_PER_FEED = args.limit
    
    # Run with custom settings
    main()