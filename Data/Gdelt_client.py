"""
GDELT Event Ingestion
Fetches geopolitical events from the GDELT API
"""

import asyncio
import logging 
from datetime import datetime, timedelta 
from typing import List, Dict, Any, Optional 
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry 
import pandas as pd 
from .models import GeopoliticalEvent, EventSource
from .settings import Settings 

logger = logging.getLogger(__name__)

class GDELTIngestions:
    #GDELT (Global Database of Events, Language, and Tone) ingestion client 
    #GDELT updates every 15 minute with events from global news source in 100+ languages.
    #This cleint focuses on conflict-related events 

    BASE_URL = "https://api.gdeltproject.org/api/v2"

    #GDELT cameo event codes for geopolitical events
    RELEVANT_CAMEO_CODES = [
        "14", #Protest
        "15", #Exhibit military posture 
        "16", #Reduce relations
        "17", #Coerce
        "18", #Assault
        "19", #Fight
        "20", #Unconventional violence
    ]

    #Theme codes for specific threat types
    RELEVANT_THEMES = [
        "TERROR",
        "MILITARY_ACTION",
        "ARMED_CONFLICT",
        "MILITARY_DEPLOYMENT",
        "BORDER_CONFLICT",
        "CYBERATTACK",
        "SANCTION",
        "REFUGEE",
        "ENERGY_SECURITY",
    ]

    def __init__(self):
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        #Create requests session with retry logic 
        session = requests.Session()
        retries = Retry(
            total = 3,
            backoff_factor = 1,
            status_forcelist = [429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(max_retries = retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session 
    
    def fetch_recent_events(
        self,
        lookback_minutes: int = 15,
        max_records: int = 250
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent GDELT events

        Args:
            lookback_minutes: How far back to look 
            max_records: Maximum number of records to return

        Return: 
            List of raw event dictionaries
        """

        try:
            # Calculate absolute time range (more reliable than timespan)
            now = datetime.utcnow()
            start_time = now - timedelta(minutes=lookback_minutes)
            
            # Format dates as YYYYMMDDHHMMSS
            start_str = start_time.strftime("%Y%m%d%H%M%S")
            end_str = now.strftime("%Y%m%d%H%M%S")
            
            logger.info(f"GDELT query: {start_str} to {end_str}")
            
            # GDELT GEO 2.0 API query
            # Filter for conflict-related themes
            themes_query = "(" + " OR ".join(f"theme:{theme}" for theme in self.RELEVANT_THEMES) + ")"

            params = {
                "query": themes_query,
                "mode": "artlist",
                "format": "json", 
                "maxrecords": max_records,
                "startdatetime": start_str,  # ← CHANGED: Use absolute dates
                "enddatetime": end_str,       # ← CHANGED: Use absolute dates
                "sort": "datedesc"
            }

            response = self.session.get(
                f"{self.BASE_URL}/doc/doc",
                params=params,
                timeout=30,
            )
            
            try:
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                body = getattr(response, "text", "")[:200]
                logger.error("Error fetching GDELT events: %s; body=%r", e, body)
                return []

            articles = data.get("articles", [])

            logger.info(f"Fetched {len(articles)} GDELT articles from last {lookback_minutes} minutes")

            return articles 
    
        except Exception as e:
            logger.error(f"Error fetching GDELT events: {e}")
            return []
        
    def fetch_by_location(
        self,
        location: str, 
        lookback_days: int = 7,
        max_records: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch events for a specific location
        Args:
            location: Country, region, or place name
            lookback_days: Days to look back
            max_records: maximum records
        Returns:
            list of event dictionaries
        """

        try:
            # Calculate absolute time range
            now = datetime.utcnow()
            start_time = now - timedelta(days=lookback_days)
            
            # Format dates
            start_str = start_time.strftime("%Y%m%d%H%M%S")
            end_str = now.strftime("%Y%m%d%H%M%S")
            
            themes_query = "(" + " OR ".join(f"theme:{theme}" for theme in self.RELEVANT_THEMES) + ")"
            query = f"{themes_query} AND \"{location}\""

            params = {
                "query": query,
                "mode": "artlist",
                "format": "json",
                "maxrecords": max_records,
                "startdatetime": start_str,  # ← CHANGED
                "enddatetime": end_str,       # ← CHANGED
                "sort": "datedesc"
            }

            response = self.session.get(
                f"{self.BASE_URL}/doc/doc",
                params=params,
                timeout=30,
            )
            
            try:
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                body = getattr(response, "text", "")[:200]
                logger.error("Error fetching GDELT event for %s: %s; body=%r", location, e, body)
                return []

            articles = data.get("articles", [])
            logger.info(f"Fetched {len(articles)} GDELT articles for location: {location}")
            return articles 
    
        except Exception as e:
            logger.error(f"Error fetching GDELT event for {location}: {e}")
            return []
        
    def fetch_event_database(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch GDELT Event Database (structured events)

        This is the core GDELT 2.0 event database with CAMEO codes
        Args:
            start_date: Start datetime
            end_date: End datetime
        Returns:
            DataFrame of structured events
        """

        try:
            #GDELT 2.0 Event Database uses different endpoints
            #Format: YYYYMMDDHHMMSS
            start_str = start_date.strftime("%Y%m%d%H%M%S")
            end_str = end_date.strftime("%Y%m%d%H%M%S")

            #Build URL for event export 
            url = f"http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

            #For production, you'd want to 
                #1. Download the .CSV.zip files
                #2. Filter by CAMEO codes
                #3. Parse and return as DataFrame

            logger.info(f"GDELT Event Database query from {start_date} to {end_date}")
            logger.warning("Full event database parsing not implemented in this example")

            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error fetching GDELT event database: {e}")
            return pd.DataFrame()
        
    def parse_article_to_event(self, article: Dict[str, Any]) -> Optional[GeopoliticalEvent]:
        """
        Convert GDELT article to GeopoliticalEvent model
        Args:
            article: Raw GDELT article dict

        Returns:
            GeopoliticalEvent or None if parsing fails
        """

        try:
            #Extract timestamp
            seendate = article.get("seendate", "")
            timestamp = datetime.strptime(seendate, "%Y%m%dT%H%M%SZ") if seendate else datetime.utcnow()

            #Extract content
            title = article.get("title", "")
            description = article.get("description", "") or article.get("socialimage", "")
            url = article.get("url", "")

            #Extract locations
            locations = []
            if "locations" in article:
                for loc in article["locations"]:
                    locations.append(loc.get("name", ""))

            #Extract countries
            countries = []
            if "countries" in article:
                for country in article["countries"]:
                    countries.append(country.get("name", ""))

            #Tone (negative tone suggests conflict)
            tone = article.get("tone", 0)
            sentiment_score = float(tone) / 100.0 if tone else None 

            #Create the event 
            event = GeopoliticalEvent(
                timestamp = timestamp,
                source = EventSource.GDELT,
                source_url = url,
                title = title, 
                description = description,
                language = article.get("language", "en"),
                locations = locations,
                countries = countries,
                sentiment_score = sentiment_score
            )

            return event 
        
        except Exception as e:
            logger.error(f"Error parsing GDELT article: {e}")
            return None 
        
    async def stream_events(self, interval_seconds: int = 900):
        """
        Continously stream GDELT events
        Args:
            interval_seconds: polling interval
        """
        logger.info(f"Starting GDELT event stream (interval: {interval_seconds}s)")

        while True:
            try:
                #Fetch recent events
                articles = self.fetch_recent_events(lookback_minutes = interval_seconds // 60)

                #Parse to events
                events = []
                for article in articles:
                    event = self.parse_article_to_event(article)
                    if event:
                        events.append(event)

                #Yield events (in production would push to Kafka or Kinesis)
                logger.info(f"Processed {len(events)} GDELT events")

                #Waiting for the next poll
                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error in GDELT event stream: {e}")
                await asyncio.sleep(60) #short backoff on error 

class GDELTEventDatabase:
    """
    Interface to GDELT 2.0 Event Database for historical queries
    This provides structured event data with CAMEO codes, actors, locations
    """

    def __init__(self):
        pass 

    def query_events(
        self,
        start_date: datetime,
        end_date: datetime, 
        cameo_codes: Optional[List[str]] = None,
        countries: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Query historical GDELT events
        Args:
            start_date: Start date
            end_date: End date
            cameo_codes: Filter by CAMEO event codes
            countries: Filter by country codes (ISO 3166-1 alpha-3)
        Returns:
            DataFrame of events
        """

        #In production, this would 
            #1. query GDELT BigQuery public dataset
            #2. Or download and parse daily export files 
            #3. Filter by criteria 
            #4. Return structured data

        logger.info(f"Querying GDELT events from {start_date}")
        logger.warning("Historical event query not fully implemented")

        return pd.DataFrame()
    

#Example usage
if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)

    gdelt = GDELTIngestions()

    #Fetch recent events
    events = gdelt.fetch_recent_events(lookback_minutes=60)
    print(f"Fetched {len(events)} events")

    #Fetch for specific location
    ukraine_events = gdelt.fetch_by_location("Ukraine", lookback_days=1)
    print(f"Fetched {len(ukraine_events)} Ukraine events")
