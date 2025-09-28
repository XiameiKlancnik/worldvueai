"""
WorldVue AI - Local Database Backup System
Saves articles to both Supabase and local SQLite
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class LocalBackupDB:
    """
    SQLite backup for all articles
    Lightweight, fast, and reliable local storage
    """
    
    def __init__(self, db_path: str = "worldvue_backup.db"):
        self.db_path = db_path
        self.conn = None
        self.init_database()
    
    def init_database(self):
        """Create tables if they don't exist"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Create articles table matching Supabase schema
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY,
                url TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                summary TEXT,
                content TEXT,
                source_name TEXT,
                source_country TEXT,
                published_at TIMESTAMP,
                fetched_at TIMESTAMP,
                tilt REAL,
                frames TEXT,  -- JSON stored as text
                keywords TEXT,  -- JSON stored as text
                scoring_version TEXT,
                word_count INTEGER,
                has_full_text BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for faster queries
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_published_at ON articles(published_at DESC)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_source_country ON articles(source_country)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_tilt ON articles(tilt)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_source_name ON articles(source_name)')
        
        # Create fetch history table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS fetch_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fetch_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                feeds_processed INTEGER,
                articles_added INTEGER,
                articles_duplicate INTEGER,
                errors INTEGER,
                duration_seconds REAL
            )
        ''')
        
        self.conn.commit()
        logger.info(f"Local backup database initialized at {self.db_path}")
    
    def insert_articles(self, articles: List[Dict[str, Any]]) -> int:
        """Insert articles into local database"""
        if not articles:
            return 0
        
        inserted = 0
        
        for article in articles:
            try:
                # Convert JSON fields to strings
                frames_json = json.dumps(article.get('frames', {}))
                keywords_json = json.dumps(article.get('keywords', []))
                
                self.conn.execute('''
                    INSERT OR REPLACE INTO articles (
                        id, url, title, summary, content, source_name, source_country,
                        published_at, fetched_at, tilt, frames, keywords,
                        scoring_version, word_count, has_full_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article['id'],
                    article['url'],
                    article['title'],
                    article.get('summary'),
                    article.get('content'),
                    article['source_name'],
                    article['source_country'],
                    article['published_at'],
                    article['fetched_at'],
                    article.get('tilt', 0),
                    frames_json,
                    keywords_json,
                    article.get('scoring_version', 'v1'),
                    article.get('word_count', 0),
                    article.get('has_full_text', False)
                ))
                inserted += 1
            except sqlite3.IntegrityError:
                # Duplicate URL, skip
                pass
            except Exception as e:
                logger.error(f"Error inserting article: {e}")
        
        self.conn.commit()
        return inserted
    
    def check_duplicates(self, urls: List[str]) -> set:
        """Check which URLs already exist"""
        if not urls:
            return set()
        
        placeholders = ','.join('?' * len(urls))
        cursor = self.conn.execute(
            f'SELECT url FROM articles WHERE url IN ({placeholders})',
            urls
        )
        
        return {row['url'] for row in cursor}
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        stats = {}
        
        # Total articles
        cursor = self.conn.execute('SELECT COUNT(*) as count FROM articles')
        stats['total_articles'] = cursor.fetchone()['count']
        
        # Articles by country
        cursor = self.conn.execute('''
            SELECT source_country, COUNT(*) as count 
            FROM articles 
            GROUP BY source_country 
            ORDER BY count DESC 
            LIMIT 20
        ''')
        stats['top_countries'] = [dict(row) for row in cursor]
        
        # Bias distribution
        cursor = self.conn.execute('''
            SELECT 
                SUM(CASE WHEN tilt < -0.3 THEN 1 ELSE 0 END) as left,
                SUM(CASE WHEN tilt >= -0.3 AND tilt <= 0.3 THEN 1 ELSE 0 END) as center,
                SUM(CASE WHEN tilt > 0.3 THEN 1 ELSE 0 END) as right
            FROM articles
        ''')
        stats['bias_distribution'] = dict(cursor.fetchone())
        
        # Recent activity
        cursor = self.conn.execute('''
            SELECT COUNT(*) as count 
            FROM articles 
            WHERE datetime(published_at) > datetime('now', '-24 hours')
        ''')
        stats['last_24h'] = cursor.fetchone()['count']
        
        # Database size
        stats['db_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)
        
        return stats
    
    def export_to_json(self, output_file: str = "worldvue_backup.json", 
                       limit: int = None) -> int:
        """Export database to JSON file"""
        query = 'SELECT * FROM articles ORDER BY published_at DESC'
        if limit:
            query += f' LIMIT {limit}'
        
        cursor = self.conn.execute(query)
        
        articles = []
        for row in cursor:
            article = dict(row)
            # Parse JSON fields
            article['frames'] = json.loads(article['frames']) if article['frames'] else {}
            article['keywords'] = json.loads(article['keywords']) if article['keywords'] else []
            articles.append(article)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'export_date': datetime.now().isoformat(),
                'article_count': len(articles),
                'articles': articles
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(articles)} articles to {output_file}")
        return len(articles)
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


# Create global instance
local_db = LocalBackupDB()


def backup_articles_locally(articles: List[Dict]) -> int:
    """
    Backup articles to local SQLite database
    This is called by the fetcher after Supabase insert
    """
    return local_db.insert_articles(articles)


def get_local_stats() -> Dict:
    """Get statistics from local backup"""
    return local_db.get_statistics()


if __name__ == "__main__":
    # Test and show stats
    stats = get_local_stats()
    
    print("Local Backup Database Statistics")
    print("=" * 50)
    print(f"Total Articles: {stats['total_articles']:,}")
    print(f"Database Size: {stats['db_size_mb']:.2f} MB")
    print(f"Last 24h: {stats['last_24h']:,}")
    
    if stats['bias_distribution']:
        print("\nBias Distribution:")
        for bias, count in stats['bias_distribution'].items():
            print(f"  {bias}: {count:,}")
    
    if stats['top_countries']:
        print("\nTop Countries:")
        for country in stats['top_countries'][:10]:
            print(f"  {country['source_country']}: {country['count']:,}")
    
    # Export option
    response = input("\nExport to JSON? (yes/no): ")
    if response.lower() == 'yes':
        count = local_db.export_to_json()
        print(f"Exported {count} articles to worldvue_backup.json")
