"""
WorldVue AI - Correct Fix Web Server
Maps ISO2 (in database) to ISO3 (for map) properly
"""

import json
import logging
import os
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from supabase import create_client

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

try:
    from sklearn.cluster import DBSCAN
except ImportError:  # pragma: no cover
    DBSCAN = None  # type: ignore

load_dotenv()


def ensure_httpx_proxy_support() -> None:
    """Ensure httpx.Client accepts the legacy proxy argument used by supabase."""
    try:
        import httpx  # type: ignore
        import inspect
    except Exception:
        return

    client = httpx.Client
    if getattr(client, '_legacy_proxy_patch', False):
        return

    signature = inspect.signature(client.__init__)
    if 'proxy' in signature.parameters:
        client._legacy_proxy_patch = True  # type: ignore[attr-defined]
        return

    original_init = client.__init__

    def patched_init(self, *args, proxy=None, **kwargs):
        if proxy is not None and 'proxies' not in kwargs:
            kwargs['proxies'] = proxy
        return original_init(self, *args, **kwargs)

    client.__init__ = patched_init  # type: ignore[assignment]

    if hasattr(httpx, 'AsyncClient'):
        async_client = httpx.AsyncClient
        async_signature = inspect.signature(async_client.__init__)
        if 'proxy' not in async_signature.parameters:
            original_async_init = async_client.__init__

            def async_patched_init(self, *args, proxy=None, **kwargs):
                if proxy is not None and 'proxies' not in kwargs:
                    kwargs['proxies'] = proxy
                return original_async_init(self, *args, **kwargs)

            async_client.__init__ = async_patched_init  # type: ignore[assignment]

    client._legacy_proxy_patch = True  # type: ignore[attr-defined]


ensure_httpx_proxy_support()

ENGLISH_LANG_CODES = {"en", "en-us", "en-gb", "en-au", "en-ca", "en-nz", "en-ie", "en-sg", "en-za", "en-in"}
ENGLISH_COUNTRY_CODES = {"US", "GB", "UK", "IE", "CA", "AU", "NZ", "SG", "ZA", "NG", "KE", "PH"}

def is_probably_english(article: Dict[str, Any]) -> bool:
    """Heuristic check to determine if an article is likely in English."""
    lang = (article.get('language') or '').strip().lower()
    if lang in ENGLISH_LANG_CODES or lang.startswith('en'):
        return True
    country = (article.get('source_country') or '').strip().upper()
    if country in ENGLISH_COUNTRY_CODES:
        return True
    return False


def normalize_topic_title(title: str) -> str:
    """Return a concise, English-friendly topic label derived from a headline."""
    if not title:
        return 'Untitled'

    translation_map = {
        0x201c: '"',
        0x201d: '"',
        0x2014: ' - ',
        0x2013: ' - '
    }
    original = title.strip().translate(translation_map)

    for separator in (':', ' - '):
        if separator in original:
            head, tail = original.split(separator, 1)
            if len(head) >= 32:
                original = head.strip()
                break

    if len(original) > 120:
        cut_points = ['.', ';', '?', '!']
        for symbol in cut_points:
            pos = original.find(symbol)
            if 40 <= pos <= 110:
                original = original[:pos]
                break
        if len(original) > 120:
            original = original[:117].rstrip() + '...'

    ascii_candidate = original.encode('ascii', 'ignore').decode().strip()
    if ascii_candidate:
        original = ascii_candidate

    words = original.split()
    cleaned_words = []
    for word in words:
        cleaned = ''.join(ch for ch in word if ch.isalnum() or ch in {"'", '-'})
        if cleaned:
            cleaned_words.append(cleaned)
    if cleaned_words:
        original = ' '.join(cleaned_words)

    if original and original[0].islower():
        original = original[0].upper() + original[1:]

    return original or 'Untitled'


def build_search_query(headline: str) -> str:
    """Return a compact search query derived from a headline."""
    if not headline:
        return 'news'
    words = [word for word in headline.split() if len(word) > 2]
    if not words:
        words = headline.split()[:6]
    else:
        words = words[:6]
    query = ' '.join(words).strip()
    return query or headline


# Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_ANON_KEY')
TIME_WINDOW_DAYS = int(os.getenv('TIME_WINDOW_DAYS', '14'))
CACHE_TTL = int(os.getenv('CACHE_TTL', '300'))  # 5 minutes
PORT = int(os.getenv('PORT', '5000'))
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2')
EMBEDDING_DEFAULT_THRESHOLD = float(os.getenv('EMBEDDING_DEFAULT_THRESHOLD', '0.5'))
EMBEDDING_CANDIDATE_POOL = int(os.getenv('EMBEDDING_CANDIDATE_POOL', '800'))
TRENDING_WINDOW_HOURS = int(os.getenv('TRENDING_WINDOW_HOURS', '24'))
TRENDING_POOL_SIZE = int(os.getenv('TRENDING_POOL_SIZE', '1200'))
TRENDING_MAX_TOPICS = int(os.getenv('TRENDING_MAX_TOPICS', '20'))
TRENDING_CLUSTER_EPS = float(os.getenv('TRENDING_CLUSTER_EPS', '0.3'))
TRENDING_CLUSTER_MIN = int(os.getenv('TRENDING_CLUSTER_MIN', '5'))
TRENDING_HOT_THRESHOLD = int(os.getenv('TRENDING_HOT_THRESHOLD', '10'))
TRENDING_ENGLISH_SIM_THRESHOLD = float(os.getenv('TRENDING_ENGLISH_SIM_THRESHOLD', '0.55'))

# Initialize Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_embedding_model: Optional[Any] = None
_embedding_lock = threading.Lock()

# Initialize Flask app
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})

# Simple in-memory cache
cache = {}
cache_timestamps = {}

# COMPREHENSIVE ISO2 to ISO3 mapping
ISO2_TO_ISO3 = {
    'AF': 'AFG', 'AL': 'ALB', 'DZ': 'DZA', 'AD': 'AND', 'AO': 'AGO', 'AG': 'ATG',
    'AR': 'ARG', 'AM': 'ARM', 'AU': 'AUS', 'AT': 'AUT', 'AZ': 'AZE', 'BS': 'BHS',
    'BH': 'BHR', 'BD': 'BGD', 'BB': 'BRB', 'BY': 'BLR', 'BE': 'BEL', 'BZ': 'BLZ',
    'BJ': 'BEN', 'BT': 'BTN', 'BO': 'BOL', 'BA': 'BIH', 'BW': 'BWA', 'BR': 'BRA',
    'BN': 'BRN', 'BG': 'BGR', 'BF': 'BFA', 'BI': 'BDI', 'KH': 'KHM', 'CM': 'CMR',
    'CA': 'CAN', 'CV': 'CPV', 'CF': 'CAF', 'TD': 'TCD', 'CL': 'CHL', 'CN': 'CHN',
    'CO': 'COL', 'KM': 'COM', 'CG': 'COG', 'CD': 'COD', 'CR': 'CRI', 'CI': 'CIV',
    'HR': 'HRV', 'CU': 'CUB', 'CY': 'CYP', 'CZ': 'CZE', 'DK': 'DNK', 'DJ': 'DJI',
    'DM': 'DMA', 'DO': 'DOM', 'EC': 'ECU', 'EG': 'EGY', 'SV': 'SLV', 'GQ': 'GNQ',
    'ER': 'ERI', 'EE': 'EST', 'ET': 'ETH', 'FJ': 'FJI', 'FI': 'FIN', 'FR': 'FRA',
    'GA': 'GAB', 'GM': 'GMB', 'GE': 'GEO', 'DE': 'DEU', 'GH': 'GHA', 'GR': 'GRC',
    'GD': 'GRD', 'GT': 'GTM', 'GN': 'GIN', 'GW': 'GNB', 'GY': 'GUY', 'HT': 'HTI',
    'HN': 'HND', 'HU': 'HUN', 'IS': 'ISL', 'IN': 'IND', 'ID': 'IDN', 'IR': 'IRN',
    'IQ': 'IRQ', 'IE': 'IRL', 'IL': 'ISR', 'IT': 'ITA', 'JM': 'JAM', 'JP': 'JPN',
    'JO': 'JOR', 'KZ': 'KAZ', 'KE': 'KEN', 'KI': 'KIR', 'KP': 'PRK', 'KR': 'KOR',
    'KW': 'KWT', 'KG': 'KGZ', 'LA': 'LAO', 'LV': 'LVA', 'LB': 'LBN', 'LS': 'LSO',
    'LR': 'LBR', 'LY': 'LBY', 'LI': 'LIE', 'LT': 'LTU', 'LU': 'LUX', 'MK': 'MKD',
    'MG': 'MDG', 'MW': 'MWI', 'MY': 'MYS', 'MV': 'MDV', 'ML': 'MLI', 'MT': 'MLT',
    'MH': 'MHL', 'MR': 'MRT', 'MU': 'MUS', 'MX': 'MEX', 'FM': 'FSM', 'MD': 'MDA',
    'MC': 'MCO', 'MN': 'MNG', 'ME': 'MNE', 'MA': 'MAR', 'MZ': 'MOZ', 'MM': 'MMR',
    'NA': 'NAM', 'NR': 'NRU', 'NP': 'NPL', 'NL': 'NLD', 'NZ': 'NZL', 'NI': 'NIC',
    'NE': 'NER', 'NG': 'NGA', 'NO': 'NOR', 'OM': 'OMN', 'PK': 'PAK', 'PW': 'PLW',
    'PS': 'PSE', 'PA': 'PAN', 'PG': 'PNG', 'PY': 'PRY', 'PE': 'PER', 'PH': 'PHL',
    'PL': 'POL', 'PT': 'PRT', 'QA': 'QAT', 'RO': 'ROU', 'RU': 'RUS', 'RW': 'RWA',
    'KN': 'KNA', 'LC': 'LCA', 'VC': 'VCT', 'WS': 'WSM', 'SM': 'SMR', 'ST': 'STP',
    'SA': 'SAU', 'SN': 'SEN', 'RS': 'SRB', 'SC': 'SYC', 'SL': 'SLE', 'SG': 'SGP',
    'SK': 'SVK', 'SI': 'SVN', 'SB': 'SLB', 'SO': 'SOM', 'ZA': 'ZAF', 'SS': 'SSD',
    'ES': 'ESP', 'LK': 'LKA', 'SD': 'SDN', 'SR': 'SUR', 'SZ': 'SWZ', 'SE': 'SWE',
    'CH': 'CHE', 'SY': 'SYR', 'TW': 'TWN', 'TJ': 'TJK', 'TZ': 'TZA', 'TH': 'THA',
    'TL': 'TLS', 'TG': 'TGO', 'TO': 'TON', 'TT': 'TTO', 'TN': 'TUN', 'TR': 'TUR',
    'TM': 'TKM', 'TV': 'TUV', 'UG': 'UGA', 'UA': 'UKR', 'AE': 'ARE', 'GB': 'GBR',
    'UK': 'GBR', 'US': 'USA', 'UY': 'URY', 'UZ': 'UZB', 'VU': 'VUT', 'VE': 'VEN',
    'VN': 'VNM', 'YE': 'YEM', 'ZM': 'ZMB', 'ZW': 'ZWE'
}

# Extended stopwords list
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'as', 'is', 'was', 'are', 'were', 'been', 'be',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'that',
    'this', 'these', 'those', 'i', 'you', 'we', 'they', 'he', 'she', 'it',
    'said', 'says', 'saying', 'tell', 'told', 'telling', 'which', 'where',
    'when', 'what', 'who', 'whom', 'whose', 'why', 'how', 'there', 'here',
    'then', 'than', 'more', 'most', 'less', 'very', 'just', 'only', 'also',
    'back', 'after', 'before', 'their', 'them', 'from', 'into', 'about',
    'other', 'another', 'some', 'any', 'all', 'each', 'every', 'either',
    'neither', 'both', 'few', 'many', 'much', 'several', 'such', 'own',
    'same', 'different', 'new', 'old', 'good', 'bad', 'great', 'small',
    'large', 'big', 'high', 'low', 'early', 'late', 'long', 'short',
    'right', 'left', 'best', 'better', 'worst', 'worse', 'next', 'last',
    'first', 'second', 'third', 'able', 'like', 'need', 'must', 'going'
}


def load_embedding_model():
    '''Load and cache the multilingual embedding model.'''
    if SentenceTransformer is None:
        raise RuntimeError('sentence-transformers is required for multilingual embeddings.')
    global _embedding_model
    if _embedding_model is None:
        with _embedding_lock:
            if _embedding_model is None:
                logger.info("Loading multilingual embedding model '%s'", EMBEDDING_MODEL_NAME)
                _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def to_numpy_embedding(raw: Any) -> Optional["np.ndarray"]:
    '''Convert stored embedding payloads into numpy arrays.'''
    if np is None:
        return None
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug('Skipping malformed embedding payload')
            return None
    array = np.asarray(raw, dtype=np.float32)
    if array.ndim == 0 or array.size == 0:
        return None
    if array.ndim > 1:
        array = array.reshape(-1)
    return array


def cosine_similarity(vec1: "np.ndarray", vec2: "np.ndarray") -> float:
    '''Return cosine similarity between two vectors.'''
    denominator = float(np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / denominator)


def run_multilingual_search(
    query: str,
    *,
    days: int,
    limit: int,
    threshold: float,
    pool_size: int
) -> Dict[str, Any]:
    '''Search articles using multilingual embeddings.'''
    if not supabase:
        raise RuntimeError('Supabase client is not configured.')
    if np is None or SentenceTransformer is None:
        raise RuntimeError('Install numpy and sentence-transformers to use multilingual search.')

    days = max(1, min(days, 90))
    limit = max(1, min(limit, 200))
    pool_size = max(limit, min(pool_size, 2000))
    threshold = max(0.0, min(threshold, 1.0))

    model = load_embedding_model()
    query_vector = model.encode(query)
    query_embedding = to_numpy_embedding(query_vector)
    if query_embedding is None:
        return {
            'query': query,
            'results': [],
            'total': 0,
            'threshold': threshold,
            'pool_size': 0,
            'limit': limit,
            'days': days
        }

    date_threshold = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    select_fields = 'id, url, title, summary, source_name, source_country, published_at, embedding, language'
    try:
        response = (
            supabase.table('articles')
            .select(select_fields)
            .gte('published_at', date_threshold)
            .order('published_at', desc=True)
            .limit(pool_size)
            .execute()
        )
    except Exception as exc:
        logger.debug('Falling back to query without language column: %s', exc)
        response = (
            supabase.table('articles')
            .select('id, url, title, summary, source_name, source_country, published_at, embedding')
            .gte('published_at', date_threshold)
            .order('published_at', desc=True)
            .limit(pool_size)
            .execute()
        )

    rows = response.data or []
    scored: List[Dict[str, Any]] = []

    for article in rows:
        vector = to_numpy_embedding(article.get('embedding'))
        if vector is None:
            continue
        similarity = cosine_similarity(query_embedding, vector)
        if similarity < threshold:
            continue
        scored.append({
            'id': article.get('id'),
            'title': article.get('title'),
            'summary': (article.get('summary') or '')[:500],
            'url': article.get('url'),
            'source': article.get('source_name'),
            'country': article.get('source_country'),
            'language': article.get('language') or article.get('source_country'),
            'published_at': article.get('published_at'),
            'similarity': round(float(similarity), 4)
        })

    scored.sort(key=lambda item: item['similarity'], reverse=True)

    return {
        'query': query,
        'results': scored[:limit],
        'total': len(scored),
        'threshold': threshold,
        'pool_size': len(rows),
        'limit': limit,
        'days': days
    }


def get_multilingual_trending(max_topics: int = TRENDING_MAX_TOPICS) -> List[Dict[str, Any]]:
    '''Cluster recent articles to surface multilingual trending topics.'''
    if not supabase:
        raise RuntimeError('Supabase client is not configured.')
    if np is None or DBSCAN is None:
        raise RuntimeError('Install numpy and scikit-learn to use multilingual trending.')

    since = (datetime.now(timezone.utc) - timedelta(hours=TRENDING_WINDOW_HOURS)).isoformat()

    select_fields = 'id, title, summary, source_name, source_country, published_at, embedding, language'
    try:
        response = (
            supabase.table('articles')
            .select(select_fields)
            .gte('published_at', since)
            .order('published_at', desc=True)
            .limit(TRENDING_POOL_SIZE)
            .execute()
        )
    except Exception as exc:
        logger.debug('Falling back to trending query without language column: %s', exc)
        response = (
            supabase.table('articles')
            .select('id, title, summary, source_name, source_country, published_at, embedding')
            .gte('published_at', since)
            .order('published_at', desc=True)
            .limit(TRENDING_POOL_SIZE)
            .execute()
        )

    rows = response.data or []
    embeddings: List["np.ndarray"] = []
    articles: List[Dict[str, Any]] = []

    for row in rows:
        vector = to_numpy_embedding(row.get('embedding'))
        if vector is None:
            continue
        if float(np.linalg.norm(vector)) == 0.0:
            continue
        embeddings.append(vector)
        articles.append(row)

    if len(embeddings) < TRENDING_CLUSTER_MIN:
        return []

    matrix = np.vstack(embeddings)
    clustering = DBSCAN(
        eps=TRENDING_CLUSTER_EPS,
        min_samples=TRENDING_CLUSTER_MIN,
        metric='cosine'
    ).fit(matrix)

    clusters: Dict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(clustering.labels_):
        if label == -1:
            continue
        clusters[int(label)].append(index)

    if not clusters:
        return []

    english_pool: List[Tuple[int, "np.ndarray"]] = []
    for idx, article in enumerate(articles):
        if is_probably_english(article):
            vector = matrix[idx]
            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                continue
            english_pool.append((idx, vector / norm))

    topics: List[Dict[str, Any]] = []
    for cluster_id, indices in sorted(clusters.items(), key=lambda item: len(item[1]), reverse=True):
        cluster_vectors = matrix[indices]
        center = cluster_vectors.mean(axis=0)
        distances = np.linalg.norm(cluster_vectors - center, axis=1)
        rep_index = indices[int(np.argmin(distances))]
        representative_idx = rep_index
        label_idx = representative_idx

        best_global_idx: Optional[int] = None
        best_global_sim = -1.0

        cluster_english = [
            idx for idx in indices
            if is_probably_english(articles[idx])
        ]
        if cluster_english:
            representative_idx = min(
                cluster_english,
                key=lambda idx: float(np.linalg.norm(matrix[idx] - center))
            )
            label_idx = representative_idx
        elif english_pool:
            center_norm = float(np.linalg.norm(center))
            if center_norm > 0.0:
                center_unit = center / center_norm
                for pool_idx, pool_vector in english_pool:
                    sim = float(np.dot(pool_vector, center_unit))
                    if sim > best_global_sim:
                        best_global_sim = sim
                        best_global_idx = pool_idx
                if best_global_idx is not None:
                    if best_global_sim >= TRENDING_ENGLISH_SIM_THRESHOLD:
                        representative_idx = best_global_idx
                    label_idx = best_global_idx

        representative = articles[representative_idx]
        label_article = articles[label_idx]

        cluster_countries = sorted({
            articles[i].get('source_country')
            for i in indices
            if articles[i].get('source_country')
        })
        cluster_languages = sorted({
            articles[i].get('language') or articles[i].get('source_country')
            for i in indices
            if articles[i].get('language') or articles[i].get('source_country')
        })
        sample_ids = [
            articles[i].get('id')
            for i in indices
            if articles[i].get('id')
        ]

        label_id = label_article.get('id')
        if label_id and label_id not in sample_ids:
            sample_ids = [label_id] + sample_ids

        raw_title = label_article.get('title', 'Untitled')
        headline = normalize_topic_title(raw_title)
        search_query = build_search_query(headline)

        topics.append({
            'name': headline,
            'topic': headline,
            'search_query': search_query,
            'count': len(indices),
            'cluster_id': cluster_id,
            'countries': cluster_countries,
            'languages': cluster_languages,
            'representative_id': label_id,
            'article_ids': sample_ids[:10],
            'hot': len(indices) >= max(TRENDING_HOT_THRESHOLD, TRENDING_CLUSTER_MIN * 2)
        })

        if len(topics) >= max_topics:
            break

    return topics


def clear_old_cache():
    """Clear expired cache entries"""
    current_time = time.time()
    keys_to_delete = [
        key for key, timestamp in cache_timestamps.items() 
        if current_time - timestamp > CACHE_TTL
    ]
    for key in keys_to_delete:
        del cache[key]
        del cache_timestamps[key]

@app.route('/api/topic')
def api_topic():
    """Topic search with proper ISO2 to ISO3 conversion"""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'error': 'Query parameter q is required'}), 400
    
    if not supabase:
        return jsonify({'error': 'Database not configured'}), 500
    
    # Parameters
    days = int(request.args.get('days', TIME_WINDOW_DAYS))
    limit = int(request.args.get('limit', 100))
    offset = int(request.args.get('offset', 0))
    country = request.args.get('country', '').strip().upper()
    
    # Cache key
    cache_key = f"topic:{query}:{days}:{limit}:{offset}:{country}"
    
    # Check cache
    if cache_key in cache and cache_key in cache_timestamps:
        if time.time() - cache_timestamps[cache_key] < CACHE_TTL:
            cached = cache[cache_key]
            cached['cache_hit'] = True
            return jsonify(cached)
    
    try:
        # Build base query
        date_threshold = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        
        # Build query - search in title
        query_builder = supabase.table('articles').select(
            'id, url, title, summary, source_name, source_country, published_at, tilt, frames, keywords'
        ).gte('published_at', date_threshold)
        
        # Add country filter if specified (convert ISO3 to ISO2 if needed)
        if country:
            # Check if country is ISO3 and convert to ISO2 for database query
            iso2_country = country
            for iso2, iso3 in ISO2_TO_ISO3.items():
                if iso3 == country:
                    iso2_country = iso2
                    break
            query_builder = query_builder.eq('source_country', iso2_country)
        
        # Search in title
        title_query = query_builder.ilike('title', f'%{query}%')
        title_results = title_query.limit(limit).offset(offset).order('published_at', desc=True).execute()
        
        # Also search in summary
        summary_query = supabase.table('articles').select(
            'id, url, title, summary, source_name, source_country, published_at, tilt, frames, keywords'
        ).gte('published_at', date_threshold)
        
        if country:
            summary_query = summary_query.eq('source_country', iso2_country)
        
        summary_query = summary_query.ilike('summary', f'%{query}%')
        summary_results = summary_query.limit(limit).offset(offset).order('published_at', desc=True).execute()
        
        # Combine and deduplicate results
        seen_ids = set()
        articles = []
        
        for article in title_results.data + summary_results.data:
            if article['id'] not in seen_ids:
                seen_ids.add(article['id'])
                articles.append(article)
        
        # Sort by date
        articles.sort(key=lambda x: x['published_at'], reverse=True)
        articles = articles[:limit]
        
        # Process results and convert ISO2 to ISO3 for map
        countries_data = defaultdict(lambda: {
            'tilts': [],
            'frames_list': [],
            'headlines': [],
            'sources': set(),
            'keywords': Counter()
        })
        
        unmapped_countries = set()
        
        for article in articles:
            # Convert ISO2 to ISO3
            iso2_code = article['source_country'].strip().upper()
            iso3_code = ISO2_TO_ISO3.get(iso2_code)
            
            if not iso3_code:
                unmapped_countries.add(iso2_code)
                continue  # Skip unmapped countries
            
            if article['tilt'] is not None:
                countries_data[iso3_code]['tilts'].append(article['tilt'])
            
            if article['frames']:
                countries_data[iso3_code]['frames_list'].append(article['frames'])
            
            countries_data[iso3_code]['sources'].add(article['source_name'])
            
            # Aggregate keywords
            if article.get('keywords'):
                for keyword in article['keywords'][:5]:
                    countries_data[iso3_code]['keywords'][keyword] += 1
            
            # Collect headlines
            if len(countries_data[iso3_code]['headlines']) < 3:
                countries_data[iso3_code]['headlines'].append({
                    't': article['title'],
                    'u': article['url'],
                    's': article['source_name']
                })
        
        # Log unmapped countries
        if unmapped_countries:
            logger.warning(f"Unmapped ISO2 codes: {unmapped_countries}")
        
        # Calculate aggregates
        result_countries = {}
        
        for iso3_code, data in countries_data.items():
            if not data['tilts']:
                continue
            
            # Calculate statistics
            tilts = data['tilts']
            avg_tilt = sum(tilts) / len(tilts)
            
            # Average frames
            avg_frames = defaultdict(float)
            for frames in data['frames_list']:
                if frames:
                    for frame, value in frames.items():
                        avg_frames[frame] += value
            
            num_articles = len(tilts)
            if avg_frames:
                for frame in avg_frames:
                    avg_frames[frame] = round(avg_frames[frame] / num_articles, 3)
            else:
                # Default frames if none found
                avg_frames = {
                    'economic': 0.2, 'political': 0.2, 'humanitarian': 0.1,
                    'security': 0.1, 'scientific': 0.1, 'scandal': 0.1
                }
            
            result_countries[iso3_code] = {
                'tilt': round(avg_tilt, 2),
                'frames': dict(avg_frames),
                'headlines': data['headlines'],
                'article_count': num_articles,
                'source_count': len(data['sources'])
            }
        
        response = {
            'topic': query,
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'countries': result_countries,  # Now properly using ISO3 codes
            'total_matches': len(articles),
            'cache_hit': False,
            'search_params': {
                'days': days,
                'limit': limit,
                'offset': offset,
                'country': country if country else None
            }
        }
        
        # Log for debugging
        logger.info(f"Query '{query}' returned {len(result_countries)} countries")
        if result_countries:
            sample = list(result_countries.keys())[:5]
            logger.info(f"Sample ISO3 codes: {sample}")
        
        # Cache the response
        cache[cache_key] = response
        cache_timestamps[cache_key] = time.time()
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Database query failed', 'details': str(e)}), 500


@app.route('/api/search_multilingual')
def api_search_multilingual():
    '''Search using multilingual embeddings.'''
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'error': 'Query parameter q is required'}), 400

    if not supabase:
        return jsonify({'error': 'Database not configured'}), 500

    try:
        days = int(request.args.get('days', TIME_WINDOW_DAYS))
    except ValueError:
        days = TIME_WINDOW_DAYS

    try:
        limit = int(request.args.get('limit', 40))
    except ValueError:
        limit = 40

    try:
        threshold = float(request.args.get('threshold', EMBEDDING_DEFAULT_THRESHOLD))
    except ValueError:
        threshold = EMBEDDING_DEFAULT_THRESHOLD

    try:
        pool_size = int(request.args.get('pool', EMBEDDING_CANDIDATE_POOL))
    except ValueError:
        pool_size = EMBEDDING_CANDIDATE_POOL

    cache_key = f"mlsearch:{query}:{days}:{limit}:{threshold}:{pool_size}"

    if cache_key in cache and cache_key in cache_timestamps:
        if time.time() - cache_timestamps[cache_key] < CACHE_TTL:
            cached = dict(cache[cache_key])
            cached['cache_hit'] = True
            return jsonify(cached)

    try:
        result = run_multilingual_search(
            query,
            days=days,
            limit=limit,
            threshold=threshold,
            pool_size=pool_size
        )
    except RuntimeError as runtime_error:
        return jsonify({'error': str(runtime_error)}), 500
    except Exception as exc:
        logger.error(f"Multilingual search error: {exc}")
        return jsonify({'error': 'Search failed', 'details': str(exc)}), 500

    response = {
        'query': result['query'],
        'results': result['results'],
        'total': result['total'],
        'threshold': result['threshold'],
        'pool_size': result['pool_size'],
        'limit': result['limit'],
        'days': result['days'],
        'cache_hit': False
    }

    cache[cache_key] = dict(response)
    cache_timestamps[cache_key] = time.time()

    return jsonify(response)


@app.route('/api/country')
def api_country():
    """Get detailed country analysis - handles both ISO2 and ISO3"""
    code = request.args.get('code', '').strip().upper()
    query = request.args.get('q', '')
    
    if not code or not supabase:
        return jsonify({'error': 'Invalid request'}), 400
    
    # Convert ISO3 to ISO2 for database query if needed
    iso2_code = code
    if len(code) == 3:
        # It's ISO3, convert to ISO2
        for iso2, iso3 in ISO2_TO_ISO3.items():
            if iso3 == code:
                iso2_code = iso2
                break
    
    cache_key = f"country:{code}:{query}"
    
    # Check cache
    if cache_key in cache and cache_key in cache_timestamps:
        if time.time() - cache_timestamps[cache_key] < CACHE_TTL:
            return jsonify(cache[cache_key])
    
    try:
        # Build query with ISO2 code
        query_builder = supabase.table('articles').select(
            'title, url, tilt, source_name, published_at, summary, frames, keywords'
        ).eq('source_country', iso2_code)
        
        # Add search filter if query provided
        if query:
            query_builder = query_builder.ilike('title', f'%{query}%')
        
        # Get recent articles
        result = query_builder.order('published_at', desc=True).limit(100).execute()
        
        if not result.data:
            logger.warning(f"No data found for country {code} (ISO2: {iso2_code})")
            return jsonify({
                'error': 'No data found',
                'country': code,
                'iso2_searched': iso2_code
            }), 404
        
        # Calculate statistics
        articles = result.data
        tilts = [a['tilt'] for a in articles if a['tilt'] is not None]
        sources = Counter(a['source_name'] for a in articles)
        
        # Aggregate frames
        all_frames = defaultdict(float)
        frame_count = 0
        for article in articles:
            if article.get('frames'):
                frame_count += 1
                for frame, value in article['frames'].items():
                    all_frames[frame] += value
        
        if frame_count > 0:
            for frame in all_frames:
                all_frames[frame] = round(all_frames[frame] / frame_count, 3)
        
        # Get top keywords with filtering
        all_keywords = Counter()
        for article in articles:
            if article.get('keywords'):
                for keyword in article['keywords'][:5]:
                    keyword_clean = str(keyword).lower().strip()
                    if keyword_clean not in STOPWORDS and len(keyword_clean) > 3:
                        all_keywords[keyword_clean] += 1
        
        # Prepare headlines
        headlines = []
        for article in articles[:10]:
            headlines.append({
                'title': article['title'],
                'url': article['url'],
                'source': article['source_name'],
                'tilt': round(article['tilt'], 2) if article['tilt'] else 0
            })
        
        # Calculate bias distribution
        left = sum(1 for t in tilts if t < -0.2)
        neutral = sum(1 for t in tilts if -0.2 <= t <= 0.2)
        right = sum(1 for t in tilts if t > 0.2)
        
        response = {
            'country': code,  # Return original code requested
            'query': query if query else 'all',
            'statistics': {
                'avg_tilt': round(sum(tilts) / len(tilts), 2) if tilts else 0,
                'article_count': len(articles),
                'source_count': len(sources)
            },
            'bias_distribution': {
                'left': left,
                'neutral': neutral,
                'right': right
            },
            'top_outlets': dict(sources.most_common(5)),
            'top_keywords': dict(all_keywords.most_common(10)),
            'frames': dict(all_frames),
            'headlines': headlines
        }
        
        # Cache the response
        cache[cache_key] = response
        cache_timestamps[cache_key] = time.time()
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Country query error: {e}")
        return jsonify({'error': 'Query failed', 'details': str(e)}), 500


@app.route('/api/trending')
def api_trending():
    '''Get trending topics with multilingual support.'''
    if not supabase:
        return jsonify({'error': 'Database not configured'}), 500

    cache_key = 'trending'

    if cache_key in cache and cache_key in cache_timestamps:
        if time.time() - cache_timestamps[cache_key] < CACHE_TTL:
            cached = dict(cache[cache_key])
            cached['cache_hit'] = True
            return jsonify(cached)

    topics: List[Dict[str, Any]] = []
    mode = 'embeddings'

    try:
        topics = get_multilingual_trending()
    except RuntimeError as dependency_error:
        logger.info(f"Multilingual trending unavailable: {dependency_error}")
    except Exception as exc:
        logger.warning(f"Multilingual trending failed: {exc}")

    try:
        if not topics:
            mode = 'materialized-view'
            result = supabase.table('mv_trending_topics').select('*').limit(25).execute()
            for row in result.data:
                keyword = row['keyword']
                if keyword.lower() not in STOPWORDS and len(keyword) > 3:
                    topics.append({
                        'name': keyword.capitalize(),
                        'topic': keyword.capitalize(),
                        'count': row['article_count'],
                        'countries': [],
                        'languages': [],
                        'country_count': row.get('country_count', 0),
                        'hot': row['article_count'] > TRENDING_HOT_THRESHOLD
                    })
    except Exception as mv_error:
        logger.info(f"Materialized view not available, calculating dynamically: {mv_error}")

    if not topics:
        mode = 'keywords'
        try:
            since = (datetime.now(timezone.utc) - timedelta(hours=TRENDING_WINDOW_HOURS)).isoformat()
            result = supabase.table('articles').select('keywords, title').gte(
                'published_at', since
            ).limit(500).execute()

            keyword_counter = Counter()
            for article in result.data:
                if article.get('keywords'):
                    for keyword in article['keywords'][:10]:
                        keyword_clean = str(keyword).lower().strip()
                        if (
                            len(keyword_clean) > 3
                            and keyword_clean not in STOPWORDS
                            and not keyword_clean.isdigit()
                        ):
                            keyword_counter[keyword_clean] += 1

            for keyword, count in keyword_counter.most_common(25):
                if count > 2:
                    topics.append({
                        'name': keyword.capitalize(),
                        'topic': keyword.capitalize(),
                        'count': count,
                        'countries': [],
                        'languages': [],
                        'hot': count > TRENDING_HOT_THRESHOLD
                    })
        except Exception as keyword_error:
            logger.error(f"Keyword trending fallback failed: {keyword_error}")

    if not topics:
        mode = 'fallback'
        topics = [
            {'name': 'Ukraine', 'topic': 'Ukraine', 'count': 15, 'countries': [], 'languages': [], 'hot': True},
            {'name': 'Israel', 'topic': 'Israel', 'count': 12, 'countries': [], 'languages': [], 'hot': True},
            {'name': 'Climate', 'topic': 'Climate', 'count': 10, 'countries': [], 'languages': [], 'hot': False},
            {'name': 'Economy', 'topic': 'Economy', 'count': 7, 'countries': [], 'languages': [], 'hot': False},
        ]

    # Fix null names in topics
    for i, topic in enumerate(topics):
        if not topic.get('name') or not topic.get('topic'):
            fallback_name = f"Topic {i+1}"
            if topic.get('countries'):
                country_sample = topic['countries'][:3]  # Take first 3 countries
                if country_sample:
                    fallback_name = f"News from {', '.join(country_sample)}"
            topic['name'] = fallback_name
            topic['topic'] = fallback_name

    response = {
        'topics': topics,
        'mode': mode,
        'cache_hit': False
    }

    cache[cache_key] = dict(response)
    cache_timestamps[cache_key] = time.time()

    return jsonify(response)

@app.route('/api/status')
def api_status():
    """System status endpoint"""
    if not supabase:
        return jsonify({'error': 'Database not configured'}), 500
    
    cache_key = 'status'
    
    # Check cache
    if cache_key in cache and cache_key in cache_timestamps:
        if time.time() - cache_timestamps[cache_key] < 60:
            return jsonify(cache[cache_key])
    
    try:
        now = datetime.now(timezone.utc)
        
        # Get statistics
        total_result = supabase.table('articles').select('id', count='exact').execute()
        total_articles = total_result.count
        
        # Recent articles
        day_ago = (now - timedelta(days=1)).isoformat()
        week_ago = (now - timedelta(days=7)).isoformat()
        
        daily_result = supabase.table('articles').select('id', count='exact').gte(
            'published_at', day_ago
        ).execute()
        
        weekly_result = supabase.table('articles').select('id', count='exact').gte(
            'published_at', week_ago
        ).execute()
        
        # Get country distribution with ISO2 to ISO3 conversion
        recent_articles = supabase.table('articles').select(
            'source_country'
        ).gte('published_at', week_ago).limit(1000).execute()
        
        country_counts = Counter()
        for article in recent_articles.data:
            iso2 = article['source_country'].strip().upper()
            iso3 = ISO2_TO_ISO3.get(iso2, iso2)
            country_counts[iso3] += 1
        
        countries_data = [
            {'country': c, 'article_count': count} 
            for c, count in country_counts.most_common(20)
        ]
        
        response = {
            'status': 'online',
            'database': {
                'type': 'supabase',
                'total_articles': total_articles,
                'indexed': True
            },
            'recent_activity': {
                'last_24h': daily_result.count,
                'last_week': weekly_result.count
            },
            'top_countries': countries_data,
            'cache': {
                'entries': len(cache),
                'ttl_seconds': CACHE_TTL
            },
            'server': {
                'version': '2.0-iso-correct',
                'timestamp': now.isoformat()
            }
        }
        
        # Cache the response
        cache[cache_key] = response
        cache_timestamps[cache_key] = time.time()
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 500

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear the cache"""
    cache.clear()
    cache_timestamps.clear()
    return jsonify({'message': 'Cache cleared successfully'})

@app.route('/countries.geojson')
def serve_geojson():
    """Serve the GeoJSON file"""
    if os.path.exists('countries.geojson'):
        return send_from_directory('.', 'countries.geojson')
    else:
        return jsonify({'error': 'countries.geojson not found'}), 404

@app.route('/')
def index():
    """Serve the main page"""
    if os.path.exists('index.html'):
        return send_from_directory('.', 'index.html')
    else:
        return jsonify({
            'name': 'WorldVue AI API',
            'version': '2.0-iso-correct',
            'status': 'online',
            'endpoints': [
                '/api/status',
                '/api/topic?q=<query>',
                '/api/search_multilingual?q=<query>',
                '/api/country?code=<ISO3>',
                '/api/trending'
            ]
        })

@app.before_request
def cleanup_cache():
    """Periodically clean up expired cache entries"""
    if time.time() % 100 < 1:
        clear_old_cache()

if __name__ == '__main__':
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("="*60)
        print("ERROR: Supabase not configured!")
        print("Please set SUPABASE_URL and SUPABASE_ANON_KEY in .env")
        print("="*60)
        exit(1)
    
    print("="*60)
    print(" WorldVue AI - Web Server v2.0 CORRECT ISO MAPPING")
    print("="*60)
    print(" Database has ISO2 codes (US, DE, FR)")
    print(" Converting to ISO3 for map (USA, DEU, FRA)")
    print("="*60)
    print(f"\n Starting server on port {PORT}...")
    print(f" Access at: http://localhost:{PORT}")
    print("="*60 + "\n")
    
    app.run(host='127.0.0.1', port=PORT, debug=False)
