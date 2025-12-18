"""
NewsAPI Client for OSINT AI System
Fetches high-quality English news articles about geopolitical events
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from Data.models import GeopoliticalEvent, EventSource
from Data.settings import Settings

logger = logging.getLogger(__name__)


class NewsAPIClient:
    """Client for NewsAPI.org - High-quality English news"""
    
    BASE_URL = "https://newsapi.org/v2"
    
    # Keywords for geopolitical threat detection
    THREAT_KEYWORDS = [
        "military",
        "conflict",
        "war",
        "attack",
        "drone strike",
        "sanctions",
        "border conflict",
        "naval incident",
        "airspace violation",
        "terrorism",
        "cyber attack",
        "nuclear",
        "missile",
        "deployment",
        "coup",
        "protest",
        "unrest"
    ]
    
    # High-quality news sources
    PRIORITY_SOURCES = [
        "reuters",
        "bbc-news",
        "the-washington-post",
        "the-wall-street-journal",
        "cnn",
        "associated-press",
        "bloomberg",
        "al-jazeera-english",
        "the-guardian-uk",
        "financial-times"
    ]
    
    def __init__(self):
        self.settings = Settings()
        self.api_key = self.settings.NEWSAPI_KEY
        
        if not self.api_key:
            logger.warning("NewsAPI key not configured. NewsAPI features disabled.")
            self.enabled = False
            return
        
        self.enabled = True
        self.session = self._create_session()
        logger.info("NewsAPI client initialized")
    
    def _create_session(self) -> requests.Session:
        """Create requests session with retry logic"""
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def fetch_everything(
        self,
        query: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch articles using the /everything endpoint
        
        Args:
            query: Search query
            from_date: Start date
            to_date: End date
            language: Language code (default: en)
            sort_by: Sort order (publishedAt, relevancy, popularity)
            page_size: Results per page (max 100)
        
        Returns:
            List of article dictionaries
        """
        if not self.enabled:
            logger.warning("NewsAPI is disabled")
            return []
        
        # Default date range: last 24 hours
        if not from_date:
            from_date = datetime.utcnow() - timedelta(days=1)
        if not to_date:
            to_date = datetime.utcnow()
        
        try:
            params = {
                "q": query,
                "from": from_date.strftime("%Y-%m-%dT%H:%M:%S"),
                "to": to_date.strftime("%Y-%m-%dT%H:%M:%S"),
                "language": language,
                "sortBy": sort_by,
                "pageSize": page_size,
                "apiKey": self.api_key
            }
            
            response = self.session.get(
                f"{self.BASE_URL}/everything",
                params=params,
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "ok":
                articles = data.get("articles", [])
                logger.info(f"Fetched {len(articles)} articles from NewsAPI")
                return articles
            else:
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []
        
        except Exception as e:
            logger.error(f"Error fetching from NewsAPI: {e}")
            return []
    
    def fetch_threat_news(
        self,
        lookback_hours: int = 24,
        max_articles: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch threat-related news using predefined keywords
        
        Args:
            lookback_hours: How many hours to look back
            max_articles: Maximum articles to return
        
        Returns:
            List of article dictionaries
        """
        if not self.enabled:
            return []
        
        # Build query from threat keywords
        query = " OR ".join([f'"{keyword}"' for keyword in self.THREAT_KEYWORDS[:5]])
        
        from_date = datetime.utcnow() - timedelta(hours=lookback_hours)
        
        return self.fetch_everything(
            query=query,
            from_date=from_date,
            sort_by="publishedAt",
            page_size=max_articles
        )
    
    def fetch_by_region(
        self,
        region: str,
        lookback_hours: int = 24,
        max_articles: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Fetch news for a specific region
        
        Args:
            region: Region name (e.g., "Ukraine", "Taiwan", "Middle East")
            lookback_hours: Hours to look back
            max_articles: Max articles
        
        Returns:
            List of articles
        """
        if not self.enabled:
            return []
        
        # Combine region with threat keywords
        threat_terms = ["military", "conflict", "attack", "crisis"]
        query = f'{region} AND ({" OR ".join(threat_terms)})'
        
        from_date = datetime.utcnow() - timedelta(hours=lookback_hours)
        
        return self.fetch_everything(
            query=query,
            from_date=from_date,
            page_size=max_articles
        )
    
    def fetch_top_headlines(
        self,
        category: str = "general",
        country: Optional[str] = None,
        sources: Optional[str] = None,
        page_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch top headlines using /top-headlines endpoint
        
        Args:
            category: News category (business, entertainment, general, health, science, sports, technology)
            country: Country code (us, gb, etc.)
            sources: Comma-separated source IDs
            page_size: Results per page
        
        Returns:
            List of articles
        """
        if not self.enabled:
            return []
        
        try:
            params = {
                "pageSize": page_size,
                "apiKey": self.api_key
            }
            
            if category:
                params["category"] = category
            if country:
                params["country"] = country
            if sources:
                params["sources"] = sources
            
            response = self.session.get(
                f"{self.BASE_URL}/top-headlines",
                params=params,
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "ok":
                articles = data.get("articles", [])
                logger.info(f"Fetched {len(articles)} top headlines")
                return articles
            else:
                logger.error(f"NewsAPI error: {data.get('message')}")
                return []
        
        except Exception as e:
            logger.error(f"Error fetching top headlines: {e}")
            return []
    
    def parse_article_to_event(
        self,
        article: Dict[str, Any]
    ) -> Optional[GeopoliticalEvent]:
        """
        Convert NewsAPI article to GeopoliticalEvent
        
        Args:
            article: NewsAPI article dictionary
        
        Returns:
            GeopoliticalEvent or None
        """
        try:
            # Extract timestamp
            published_at = article.get("publishedAt", "")
            if published_at:
                timestamp = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            else:
                timestamp = datetime.utcnow()
            
            # Extract content
            title = article.get("title", "")
            description = article.get("description", "") or ""
            url = article.get("url", "")
            content = article.get("content", "")
            
            # Extract source
            source_info = article.get("source", {})
            source_name = source_info.get("name", "Unknown")
            
            # Create event
            event = GeopoliticalEvent(
                timestamp=timestamp,
                source=EventSource.NEWSAPI,
                source_url=url,
                title=title,
                description=description,
                full_text=content,
                language="en"  # NewsAPI provides English articles
            )
            
            return event
        
        except Exception as e:
            logger.error(f"Error parsing NewsAPI article: {e}")
            return None
    
    def fetch_and_parse(
        self,
        query: Optional[str] = None,
        lookback_hours: int = 24,
        max_articles: int = 100
    ) -> List[GeopoliticalEvent]:
        """
        Fetch and parse articles into GeopoliticalEvents
        
        Args:
            query: Search query (uses threat keywords if None)
            lookback_hours: Hours to look back
            max_articles: Max articles
        
        Returns:
            List of GeopoliticalEvents
        """
        # Fetch articles
        if query:
            from_date = datetime.utcnow() - timedelta(hours=lookback_hours)
            articles = self.fetch_everything(
                query=query,
                from_date=from_date,
                page_size=max_articles
            )
        else:
            articles = self.fetch_threat_news(
                lookback_hours=lookback_hours,
                max_articles=max_articles
            )
        
        # Parse to events
        events = []
        for article in articles:
            event = self.parse_article_to_event(article)
            if event:
                events.append(event)
        
        logger.info(f"Parsed {len(events)} NewsAPI articles into events")
        return events
    
    def get_sources(self) -> List[Dict[str, Any]]:
        """Get available news sources"""
        if not self.enabled:
            return []
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/sources",
                params={"apiKey": self.api_key},
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "ok":
                sources = data.get("sources", [])
                logger.info(f"Retrieved {len(sources)} news sources")
                return sources
            
            return []
        
        except Exception as e:
            logger.error(f"Error getting sources: {e}")
            return []


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize client
    newsapi = NewsAPIClient()
    
    if newsapi.enabled:
        print("✅ NewsAPI client initialized")
        
        # Fetch threat-related news
        print("\n📰 Fetching threat-related news...")
        articles = newsapi.fetch_threat_news(lookback_hours=24, max_articles=10)
        print(f"Found {len(articles)} articles")
        
        if articles:
            print("\nTop 3 headlines:")
            for i, article in enumerate(articles[:3], 1):
                print(f"{i}. {article.get('title', 'No title')}")
                print(f"   Source: {article.get('source', {}).get('name', 'Unknown')}")
                print(f"   URL: {article.get('url', 'No URL')}")
        
        # Fetch region-specific news
        print("\n🌍 Fetching Ukraine news...")
        ukraine_articles = newsapi.fetch_by_region("Ukraine", lookback_hours=24, max_articles=5)
        print(f"Found {len(ukraine_articles)} Ukraine-related articles")
        
        # Parse to events
        print("\n🔄 Parsing articles to events...")
        events = newsapi.fetch_and_parse(lookback_hours=24, max_articles=5)
        print(f"Parsed {len(events)} events")
        
        if events:
            print("\nSample event:")
            event = events[0]
            print(f"  Title: {event.title}")
            print(f"  Timestamp: {event.timestamp}")
            print(f"  Source: {event.source.value}")
    
    else:
        print("❌ NewsAPI not configured (missing API key)")
        print("Get your free API key at: https://newsapi.org")
