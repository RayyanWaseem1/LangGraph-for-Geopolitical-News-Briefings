"""
Aggregated Ingestion - Multi-Source Event Collection
Combines GDELT, NewsAPI, and EventRegistry for comprehensive coverage
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict

from Data.Gdelt_client import GDELTIngestions
from Data.newsapi_client import NewsAPIClient
from Data.eventregistry_client import EventRegistryClient
from Data.models import GeopoliticalEvent, EventSource
from Storage.vector_store import VectorStore
from Storage.cache_manager import CacheManager
from Data.settings import Settings

logger = logging.getLogger(__name__)


class AggregatedIngestion:
    """
    Multi-source event ingestion with deduplication and prioritization
    
    Combines:
    - GDELT: Real-time global events (100+ languages)
    - NewsAPI: High-quality English news
    - EventRegistry: Comprehensive event-based coverage
    """
    
    def __init__(self):
        self.settings = Settings()
        
        # Initialize clients
        logger.info("Initializing data source clients...")
        self.gdelt = GDELTIngestions()
        self.newsapi = NewsAPIClient()
        self.eventregistry = EventRegistryClient()
        
        # Initialize utilities
        self.vector_store = VectorStore(use_pinecone=False)  # Local for dedup
        self.cache = CacheManager()
        
        # Track statistics
        self.stats = {
            "gdelt": {"fetched": 0, "parsed": 0, "duplicates": 0},
            "newsapi": {"fetched": 0, "parsed": 0, "duplicates": 0},
            "eventregistry": {"fetched": 0, "parsed": 0, "duplicates": 0},
            "total_unique": 0
        }
        
        logger.info("Aggregated ingestion initialized")
    
    def fetch_from_gdelt(
        self,
        lookback_minutes: int = 60,
        max_records: int = 250
    ) -> List[GeopoliticalEvent]:
        """Fetch events from GDELT"""
        try:
            logger.info(f"Fetching from GDELT (lookback: {lookback_minutes}m)...")
            
            # Fetch articles
            articles = self.gdelt.fetch_recent_events(
                lookback_minutes=lookback_minutes,
                max_records=max_records
            )
            
            self.stats["gdelt"]["fetched"] = len(articles)
            
            # Parse to events
            events = []
            for article in articles:
                event = self.gdelt.parse_article_to_event(article)
                if event:
                    events.append(event)
            
            self.stats["gdelt"]["parsed"] = len(events)
            logger.info(f"GDELT: Fetched {len(articles)}, parsed {len(events)}")
            
            return events
        
        except Exception as e:
            logger.error(f"Error fetching from GDELT: {e}")
            return []
    
    def fetch_from_newsapi(
        self,
        lookback_hours: int = 24,
        max_articles: int = 100
    ) -> List[GeopoliticalEvent]:
        """Fetch events from NewsAPI"""
        if not self.newsapi.enabled:
            logger.info("NewsAPI disabled, skipping")
            return []
        
        try:
            logger.info(f"Fetching from NewsAPI (lookback: {lookback_hours}h)...")
            
            # Fetch and parse
            events = self.newsapi.fetch_and_parse(
                lookback_hours=lookback_hours,
                max_articles=max_articles
            )
            
            self.stats["newsapi"]["fetched"] = len(events)
            self.stats["newsapi"]["parsed"] = len(events)
            
            logger.info(f"NewsAPI: Fetched {len(events)} events")
            return events
        
        except Exception as e:
            logger.error(f"Error fetching from NewsAPI: {e}")
            return []
    
    def fetch_from_eventregistry(
        self,
        lookback_days: int = 1,
        max_items: int = 100
    ) -> List[GeopoliticalEvent]:
        """Fetch events from EventRegistry"""
        if not self.eventregistry.enabled:
            logger.info("EventRegistry disabled, skipping")
            return []
        
        try:
            logger.info(f"Fetching from EventRegistry (lookback: {lookback_days}d)...")
            
            # Fetch and parse
            events = self.eventregistry.fetch_and_parse_events(
                lookback_days=lookback_days,
                max_items=max_items
            )
            
            self.stats["eventregistry"]["fetched"] = len(events)
            self.stats["eventregistry"]["parsed"] = len(events)
            
            logger.info(f"EventRegistry: Fetched {len(events)} events")
            return events
        
        except Exception as e:
            logger.error(f"Error fetching from EventRegistry: {e}")
            return []
    
    def deduplicate_events(
        self,
        events: List[GeopoliticalEvent],
        similarity_threshold: float = 0.85
    ) -> List[GeopoliticalEvent]:
        """
        Deduplicate events using multiple strategies
        
        1. URL deduplication (exact match)
        2. Vector similarity (embedding-based)
        3. Title/time proximity
        """
        if not events:
            return []
        
        logger.info(f"Deduplicating {len(events)} events...")
        
        unique_events = []
        seen_urls = set()
        
        # Strategy 1: URL-based deduplication
        for event in events:
            # Skip if URL already seen
            if self.cache.is_url_seen(event.source_url):
                event.is_duplicate = True
                source = event.source.value
                self.stats[source]["duplicates"] += 1
                continue
            
            if event.source_url in seen_urls:
                event.is_duplicate = True
                source = event.source.value
                self.stats[source]["duplicates"] += 1
                continue
            
            seen_urls.add(event.source_url)
            unique_events.append(event)
            
            # Mark URL as seen in cache
            self.cache.mark_url_seen(event.source_url, ttl=86400)
        
        logger.info(f"After URL dedup: {len(unique_events)} events")
        
        # Strategy 2: Vector similarity deduplication
        try:
            final_unique = self.vector_store.deduplicate_events(
                unique_events,
                similarity_threshold=similarity_threshold
            )
            
            # Count duplicates
            duplicates = len(unique_events) - len(final_unique)
            for event in unique_events:
                if event.is_duplicate:
                    source = event.source.value
                    self.stats[source]["duplicates"] += 1
            
            logger.info(f"After similarity dedup: {len(final_unique)} events")
            
            self.stats["total_unique"] = len(final_unique)
            return final_unique
        
        except Exception as e:
            logger.error(f"Error in vector deduplication: {e}")
            # Fallback to URL-only dedup
            self.stats["total_unique"] = len(unique_events)
            return unique_events
    
    def prioritize_events(
        self,
        events: List[GeopoliticalEvent]
    ) -> List[GeopoliticalEvent]:
        """
        Prioritize events by source quality and importance
        
        Priority order:
        1. NewsAPI (highest quality, curated sources)
        2. EventRegistry (event-based aggregation)
        3. GDELT (volume, multi-language)
        """
        # Source weights
        source_priority = {
            EventSource.NEWSAPI: 3,
            EventSource.EVENTREGISTRY: 2,
            EventSource.GDELT: 1
        }
        
        # Calculate priority score
        for event in events:
            base_score = source_priority.get(event.source, 1)
            
            # Boost score based on event characteristics
            if event.locations:
                base_score += 0.5
            if event.actors:
                base_score += 0.5
            if event.sentiment_score and event.sentiment_score < -0.5:
                base_score += 0.3  # Negative sentiment = potential threat
            
            event.importance_score = min(base_score / 5.0, 1.0)  # Normalize to 0-1
        
        # Sort by importance
        events.sort(key=lambda e: e.importance_score, reverse=True)
        
        return events
    
    async def fetch_all_sources(
        self,
        gdelt_lookback_minutes: int = 60,
        newsapi_lookback_hours: int = 24,
        eventregistry_lookback_days: int = 1,
        max_events_per_source: int = 100
    ) -> List[GeopoliticalEvent]:
        """
        Fetch from all sources concurrently
        
        Args:
            gdelt_lookback_minutes: GDELT lookback period
            newsapi_lookback_hours: NewsAPI lookback period
            eventregistry_lookback_days: EventRegistry lookback period
            max_events_per_source: Max events per source
        
        Returns:
            Deduplicated and prioritized events
        """
        logger.info("="*80)
        logger.info("AGGREGATED INGESTION - Multi-Source Fetch")
        logger.info("="*80)
        
        # Fetch concurrently using asyncio
        tasks = []
        
        # GDELT (always enabled)
        async def fetch_gdelt():
            return self.fetch_from_gdelt(
                lookback_minutes=gdelt_lookback_minutes,
                max_records=max_events_per_source
            )
        
        tasks.append(fetch_gdelt())
        
        # NewsAPI (if enabled)
        if self.newsapi.enabled:
            async def fetch_newsapi():
                return self.fetch_from_newsapi(
                    lookback_hours=newsapi_lookback_hours,
                    max_articles=max_events_per_source
                )
            tasks.append(fetch_newsapi())
        
        # EventRegistry (if enabled)
        if self.eventregistry.enabled:
            async def fetch_eventregistry():
                return self.fetch_from_eventregistry(
                    lookback_days=eventregistry_lookback_days,
                    max_items=max_events_per_source
                )
            tasks.append(fetch_eventregistry())
        
        # Execute all fetches concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        all_events = []
        for result in results:
            if isinstance(result, list):
                all_events.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Error in concurrent fetch: {result}")
        
        logger.info(f"\nTotal events fetched: {len(all_events)}")
        
        # Deduplicate
        unique_events = self.deduplicate_events(all_events)
        
        # Prioritize
        prioritized_events = self.prioritize_events(unique_events)
        
        # Log statistics
        self.log_statistics()
        
        return prioritized_events
    
    def fetch_all_sources_sync(
        self,
        gdelt_lookback_minutes: int = 60,
        newsapi_lookback_hours: int = 24,
        eventregistry_lookback_days: int = 1,
        max_events_per_source: int = 100
    ) -> List[GeopoliticalEvent]:
        """Synchronous version of fetch_all_sources"""
        return asyncio.run(self.fetch_all_sources(
            gdelt_lookback_minutes=gdelt_lookback_minutes,
            newsapi_lookback_hours=newsapi_lookback_hours,
            eventregistry_lookback_days=eventregistry_lookback_days,
            max_events_per_source=max_events_per_source
        ))
    
    def fetch_by_region(
        self,
        region: str,
        lookback_hours: int = 24
    ) -> List[GeopoliticalEvent]:
        """
        Fetch events for a specific region from all sources
        
        Args:
            region: Region name (e.g., "Ukraine", "Taiwan")
            lookback_hours: Hours to look back
        
        Returns:
            Deduplicated events for the region
        """
        logger.info(f"Fetching region-specific events: {region}")
        
        all_events = []
        
        # GDELT
        gdelt_events = []
        articles = self.gdelt.fetch_by_location(region, lookback_days=lookback_hours//24)
        for article in articles:
            event = self.gdelt.parse_article_to_event(article)
            if event:
                gdelt_events.append(event)
        all_events.extend(gdelt_events)
        logger.info(f"GDELT: {len(gdelt_events)} events for {region}")
        
        # NewsAPI
        if self.newsapi.enabled:
            newsapi_events = []
            articles = self.newsapi.fetch_by_region(region, lookback_hours=lookback_hours)
            for article in articles:
                event = self.newsapi.parse_article_to_event(article)
                if event:
                    newsapi_events.append(event)
            all_events.extend(newsapi_events)
            logger.info(f"NewsAPI: {len(newsapi_events)} events for {region}")
        
        # EventRegistry
        if self.eventregistry.enabled:
            er_events = self.eventregistry.fetch_by_location(region, lookback_days=lookback_hours//24)
            parsed_events = []
            for event in er_events:
                parsed = self.eventregistry.parse_event_to_geopolitical_event(event)
                if parsed:
                    parsed_events.append(parsed)
            all_events.extend(parsed_events)
            logger.info(f"EventRegistry: {len(parsed_events)} events for {region}")
        
        # Deduplicate and prioritize
        unique_events = self.deduplicate_events(all_events)
        prioritized = self.prioritize_events(unique_events)
        
        logger.info(f"Total unique events for {region}: {len(prioritized)}")
        return prioritized
    
    def log_statistics(self):
        """Log ingestion statistics"""
        logger.info("\n" + "="*80)
        logger.info("INGESTION STATISTICS")
        logger.info("="*80)
        
        for source in ["gdelt", "newsapi", "eventregistry"]:
            stats = self.stats[source]
            if stats["fetched"] > 0:
                logger.info(f"\n{source.upper()}:")
                logger.info(f"  Fetched: {stats['fetched']}")
                logger.info(f"  Parsed: {stats['parsed']}")
                logger.info(f"  Duplicates: {stats['duplicates']}")
                logger.info(f"  Unique: {stats['parsed'] - stats['duplicates']}")
        
        logger.info(f"\nTOTAL UNIQUE EVENTS: {self.stats['total_unique']}")
        logger.info("="*80 + "\n")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics dictionary"""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset statistics"""
        self.stats = {
            "gdelt": {"fetched": 0, "parsed": 0, "duplicates": 0},
            "newsapi": {"fetched": 0, "parsed": 0, "duplicates": 0},
            "eventregistry": {"fetched": 0, "parsed": 0, "duplicates": 0},
            "total_unique": 0
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize aggregated ingestion
    aggregator = AggregatedIngestion()
    
    print("\n🌍 Testing Multi-Source Ingestion...")
    
    # Fetch from all sources
    events = aggregator.fetch_all_sources_sync(
        gdelt_lookback_minutes=60,
        newsapi_lookback_hours=24,
        eventregistry_lookback_days=1,
        max_events_per_source=50
    )
    
    print(f"\n✅ Fetched {len(events)} unique events")
    
    if events:
        print("\nTop 5 events by importance:")
        for i, event in enumerate(events[:5], 1):
            print(f"\n{i}. {event.title[:70]}...")
            print(f"   Source: {event.source.value}")
            print(f"   Importance: {event.importance_score:.2f}")
            print(f"   Locations: {', '.join(event.locations[:3])}")
    
    # Test region-specific fetch
    print("\n\n🇺🇦 Fetching Ukraine-specific events...")
    ukraine_events = aggregator.fetch_by_region("Ukraine", lookback_hours=24)
    print(f"Found {len(ukraine_events)} Ukraine events")
