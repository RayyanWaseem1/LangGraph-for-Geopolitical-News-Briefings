"""
EventRegistry Client for OSINT AI System
Comprehensive global event coverage with entity extraction
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from Data.models import GeopoliticalEvent, EventSource
from Data.settings import Settings

if TYPE_CHECKING:
    from eventregistry import EventRegistry, QueryArticlesIter, QueryEvents, RequestEventsInfo

logger = logging.getLogger(__name__)


class EventRegistryClient:
    """Client for EventRegistry - Comprehensive event-based news"""
    
    # Threat-related concepts in EventRegistry
    THREAT_CONCEPTS = [
        "military operation",
        "armed conflict",
        "terrorism",
        "border dispute",
        "cyber attack",
        "sanctions",
        "naval incident",
        "airspace violation",
        "drone strike",
        "nuclear threat",
        "military deployment"
    ]
    
    # Priority regions for monitoring
    PRIORITY_REGIONS = [
        "Ukraine",
        "Taiwan",
        "Middle East",
        "South China Sea",
        "Korean Peninsula",
        "Kashmir",
        "Syria",
        "Yemen",
        "Afghanistan"
    ]
    
    def __init__(self):
        self.settings = Settings()
        self.api_key = self.settings.EVENTREGISTRY_KEY
        
        if not self.api_key:
            logger.warning("EventRegistry API key not configured. Features disabled.")
            self.enabled = False
            return
        
        try:
            from eventregistry import EventRegistry, QueryArticlesIter, QueryEvents  # type: ignore[import]
            self.er = EventRegistry(apiKey=self.api_key, allowUseOfArchive=False)
            self.QueryArticlesIter = QueryArticlesIter
            self.QueryEvents = QueryEvents
            self.enabled = True
            logger.info("EventRegistry client initialized")
        except ImportError:
            logger.error("eventregistry package not installed. Run: pip install eventregistry")
            self.enabled = False
        except Exception as e:
            logger.error(f"Error initializing EventRegistry: {e}")
            self.enabled = False
    
    def fetch_recent_events(
        self,
        keywords: Optional[List[str]] = None,
        concepts: Optional[List[str]] = None,
        lookback_days: int = 1,
        max_items: int = 100,
        language: str = "eng"
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent events from EventRegistry
        
        Args:
            keywords: List of keywords to search
            concepts: List of concept URIs or names
            lookback_days: Days to look back
            max_items: Maximum items to return
            language: Language code (eng, spa, etc.)
        
        Returns:
            List of event dictionaries
        """
        if not self.enabled:
            logger.warning("EventRegistry is disabled")
            return []
        
        try:
            from eventregistry import QueryEvents, RequestEventsInfo  # type: ignore[import]
            
            # Build query
            query_params = {
                "lang": language,
                "minArticlesInEvent": 2  # At least 2 articles per event
            }
            
            # Add keywords
            if keywords:
                query_params["keywords"] = " OR ".join(keywords)
            elif concepts:
                query_params["conceptUri"] = concepts
            else:
                # Use default threat keywords
                query_params["keywords"] = " OR ".join(self.THREAT_CONCEPTS[:5])
            
            # Date range
            date_start = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            date_end = datetime.utcnow().strftime("%Y-%m-%d")
            query_params["dateStart"] = date_start
            query_params["dateEnd"] = date_end
            
            # Create query
            q = QueryEvents(**query_params)
            
            # Request event info
            q.setRequestedResult(RequestEventsInfo(
                page=1,
                count=max_items,
                sortBy="date",
                sortByAsc=False,
                returnInfo=["title", "summary", "concepts", "location", "date", "uri", "articles"]
            ))
            
            # Execute query
            res = self.er.execQuery(q)
            
            events = res.get("events", {}).get("results", [])
            logger.info(f"Fetched {len(events)} events from EventRegistry")
            
            return events
        
        except Exception as e:
            logger.error(f"Error fetching EventRegistry events: {e}")
            return []
    
    def fetch_articles_by_query(
        self,
        query: str,
        lookback_days: int = 1,
        max_items: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch articles matching a query
        
        Args:
            query: Search query
            lookback_days: Days to look back
            max_items: Maximum articles
        
        Returns:
            List of article dictionaries
        """
        if not self.enabled:
            return []
        
        try:
            from eventregistry import QueryArticlesIter
            
            date_start = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            
            # Create iterator query
            q = QueryArticlesIter(
                keywords=query,
                dateStart=date_start,
                lang="eng"
            )
            
            # Fetch articles
            articles = []
            for article in q.execQuery(self.er, maxItems=max_items):
                articles.append(article)
            
            logger.info(f"Fetched {len(articles)} articles from EventRegistry")
            return articles
        
        except Exception as e:
            logger.error(f"Error fetching EventRegistry articles: {e}")
            return []
    
    def fetch_by_location(
        self,
        location: str,
        lookback_days: int = 7,
        max_items: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Fetch events for a specific location
        
        Args:
            location: Location name
            lookback_days: Days to look back
            max_items: Max items
        
        Returns:
            List of events
        """
        if not self.enabled:
            return []
        
        try:
            from eventregistry import QueryEvents, RequestEventsInfo
            
            date_start = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            
            q = QueryEvents(
                locationUri=self.er.getLocationUri(location),
                dateStart=date_start,
                lang="eng"
            )
            
            q.setRequestedResult(RequestEventsInfo(
                count=max_items,
                sortBy="date",
                sortByAsc=False
            ))
            
            res = self.er.execQuery(q)
            events = res.get("events", {}).get("results", [])
            
            logger.info(f"Fetched {len(events)} events for location: {location}")
            return events
        
        except Exception as e:
            logger.error(f"Error fetching events for location {location}: {e}")
            return []
    
    def parse_event_to_geopolitical_event(
        self,
        event: Dict[str, Any]
    ) -> Optional[GeopoliticalEvent]:
        """
        Convert EventRegistry event to GeopoliticalEvent
        
        Args:
            event: EventRegistry event dictionary
        
        Returns:
            GeopoliticalEvent or None
        """
        try:
            # Extract timestamp
            event_date = event.get("eventDate", "")
            if event_date:
                timestamp = datetime.strptime(event_date, "%Y-%m-%d")
            else:
                timestamp = datetime.utcnow()
            
            # Extract title and summary
            title = event.get("title", {}).get("eng", "Unknown Event")
            summary = event.get("summary", {}).get("eng", "")
            
            # Extract URI for source URL
            uri = event.get("uri", "")
            source_url = f"https://eventregistry.org/event/{uri}" if uri else ""
            
            # Extract locations
            locations = []
            location_data = event.get("location", {})
            if location_data:
                if isinstance(location_data, dict):
                    label = location_data.get("label", {})
                    if isinstance(label, dict):
                        locations.append(label.get("eng", ""))
                    else:
                        locations.append(str(label))
            
            # Extract concepts (actors, themes)
            concepts = event.get("concepts", [])
            actors = []
            keywords = []
            
            for concept in concepts[:10]:  # Limit to top 10
                concept_label = concept.get("label", {})
                if isinstance(concept_label, dict):
                    label = concept_label.get("eng", "")
                else:
                    label = str(concept_label)
                
                concept_type = concept.get("type", "")
                if concept_type in ["person", "org"]:
                    actors.append(label)
                else:
                    keywords.append(label)
            
            # Create event
            geopolitical_event = GeopoliticalEvent(
                timestamp=timestamp,
                source=EventSource.EVENTREGISTRY,
                source_url=source_url,
                title=title,
                description=summary,
                locations=locations,
                actors=actors,
                keywords=keywords,
                language="en"
            )
            
            return geopolitical_event
        
        except Exception as e:
            logger.error(f"Error parsing EventRegistry event: {e}")
            return None
    
    def parse_article_to_geopolitical_event(
        self,
        article: Dict[str, Any]
    ) -> Optional[GeopoliticalEvent]:
        """
        Convert EventRegistry article to GeopoliticalEvent
        
        Args:
            article: EventRegistry article dictionary
        
        Returns:
            GeopoliticalEvent or None
        """
        try:
            # Extract timestamp
            date = article.get("date", "")
            if date:
                timestamp = datetime.strptime(date, "%Y-%m-%d")
            else:
                timestamp = datetime.utcnow()
            
            # Extract content
            title = article.get("title", "Unknown")
            body = article.get("body", "")
            url = article.get("url", "")
            
            # Extract location
            location = article.get("location", {})
            locations = []
            if location:
                label = location.get("label", {})
                if isinstance(label, dict):
                    locations.append(label.get("eng", ""))
            
            # Create event
            event = GeopoliticalEvent(
                timestamp=timestamp,
                source=EventSource.EVENTREGISTRY,
                source_url=url,
                title=title,
                description=body[:500] if body else "",
                full_text=body,
                locations=locations,
                language="en"
            )
            
            return event
        
        except Exception as e:
            logger.error(f"Error parsing EventRegistry article: {e}")
            return None
    
    def fetch_and_parse_events(
        self,
        keywords: Optional[List[str]] = None,
        lookback_days: int = 1,
        max_items: int = 100
    ) -> List[GeopoliticalEvent]:
        """
        Fetch and parse events into GeopoliticalEvents
        
        Args:
            keywords: Keywords to search
            lookback_days: Days to look back
            max_items: Max items
        
        Returns:
            List of GeopoliticalEvents
        """
        # Fetch events
        events = self.fetch_recent_events(
            keywords=keywords or self.THREAT_CONCEPTS[:5],
            lookback_days=lookback_days,
            max_items=max_items
        )
        
        # Parse to GeopoliticalEvents
        parsed_events = []
        for event in events:
            parsed = self.parse_event_to_geopolitical_event(event)
            if parsed:
                parsed_events.append(parsed)
        
        logger.info(f"Parsed {len(parsed_events)} EventRegistry events")
        return parsed_events
    
    def search_concepts(self, query: str, max_concepts: int = 10) -> List[Dict[str, Any]]:
        """
        Search for concept URIs
        
        Args:
            query: Concept to search for
            max_concepts: Max concepts to return
        
        Returns:
            List of concept dictionaries
        """
        if not self.enabled:
            return []
        
        try:
            concepts = self.er.suggestConcepts(query, lang="eng")[:max_concepts]
            return concepts
        except Exception as e:
            logger.error(f"Error searching concepts: {e}")
            return []


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize client
    er_client = EventRegistryClient()
    
    if er_client.enabled:
        print("✅ EventRegistry client initialized")
        
        # Fetch recent conflict events
        print("\n📰 Fetching conflict-related events...")
        events = er_client.fetch_recent_events(
            keywords=["military", "conflict", "attack"],
            lookback_days=1,
            max_items=5
        )
        
        print(f"Found {len(events)} events")
        
        if events:
            print("\nTop event:")
            event = events[0]
            title = event.get("title", {})
            if isinstance(title, dict):
                print(f"  Title: {title.get('eng', 'No title')}")
            else:
                print(f"  Title: {title}")
            
            print(f"  Date: {event.get('eventDate', 'Unknown')}")
            print(f"  URI: {event.get('uri', 'No URI')}")
        
        # Fetch Ukraine events
        print("\n🌍 Fetching Ukraine events...")
        ukraine_events = er_client.fetch_by_location("Ukraine", lookback_days=3, max_items=3)
        print(f"Found {len(ukraine_events)} Ukraine events")
        
        # Parse to GeopoliticalEvents
        print("\n🔄 Parsing events...")
        parsed = er_client.fetch_and_parse_events(
            keywords=["military", "conflict"],
            lookback_days=1,
            max_items=3
        )
        
        print(f"Parsed {len(parsed)} GeopoliticalEvents")
        if parsed:
            print("\nSample GeopoliticalEvent:")
            print(f"  Title: {parsed[0].title}")
            print(f"  Timestamp: {parsed[0].timestamp}")
            print(f"  Locations: {parsed[0].locations}")
    
    else:
        print("❌ EventRegistry not configured")
        print("Get your API key at: https://eventregistry.org")
