"""
Database Manager for OSINT AI System
Handles all PostgreSQL/TimescaleDB operations
"""

import logging
from typing import List, Optional, Dict, Any, Iterator, Generator
from datetime import datetime, timedelta
from uuid import UUID
from sqlalchemy import create_engine, text, and_, or_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager

from Data.models import (
    GeopoliticalEvent,
    EventCluster,
    EscalationPrediction,
    ThreatAlert,
    IntelligenceBrief,
    AlertLevel,
    ThreatCategory
)
from Data.settings import Settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages all database operations for OSINT AI system"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        
        # Create engine with connection pooling
        self.engine = create_engine(
            self.settings.postgres_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=self.settings.DEBUG
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info(f"Database manager initialized: {self.settings.POSTGRES_HOST}")
    
    @contextmanager
    def get_session(self) -> Iterator[Session]:
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    # ==================== Event Operations ====================
    
    def save_event(self, event: GeopoliticalEvent) -> bool:
        """Save a geopolitical event to database"""
        try:
            with self.get_session() as session:
                # Convert Pydantic model to dict
                event_data = {
                    "event_id": event.event_id,
                    "timestamp": event.timestamp,
                    "source": event.source.value,
                    "source_url": event.source_url,
                    "title": event.title,
                    "description": event.description,
                    "full_text": event.full_text,
                    "language": event.language,
                    "threat_category": event.threat_category.value if event.threat_category else None,
                    "threat_confidence": event.threat_confidence,
                    "locations": event.locations,
                    "countries": event.countries,
                    "actors": event.actors,
                    "keywords": event.keywords,
                    "sentiment_score": event.sentiment_score,
                    "importance_score": event.importance_score,
                    "cluster_id": event.cluster_id,
                    "is_duplicate": event.is_duplicate,
                    "duplicate_of": event.duplicate_of,
                    "processed_at": event.processed_at,
                }
                
                # Insert event
                query = text("""
                    INSERT INTO events (
                        event_id, timestamp, source, source_url, title, description,
                        full_text, language, threat_category, threat_confidence,
                        locations, countries, actors, keywords, sentiment_score,
                        importance_score, cluster_id, is_duplicate, duplicate_of, processed_at
                    ) VALUES (
                        :event_id, :timestamp, :source, :source_url, :title, :description,
                        :full_text, :language, :threat_category, :threat_confidence,
                        :locations, :countries, :actors, :keywords, :sentiment_score,
                        :importance_score, :cluster_id, :is_duplicate, :duplicate_of, :processed_at
                    )
                    ON CONFLICT (event_id) DO UPDATE SET
                        threat_category = EXCLUDED.threat_category,
                        threat_confidence = EXCLUDED.threat_confidence,
                        actors = EXCLUDED.actors,
                        importance_score = EXCLUDED.importance_score,
                        cluster_id = EXCLUDED.cluster_id,
                        updated_at = NOW()
                """)
                
                session.execute(query, event_data)
                logger.debug(f"Saved event: {event.event_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving event {event.event_id}: {e}")
            return False
    
    def save_events_bulk(self, events: List[GeopoliticalEvent]) -> int:
        """Bulk save events (more efficient)"""
        saved_count = 0
        
        try:
            with self.get_session() as session:
                for event in events:
                    event_data = {
                        "event_id": event.event_id,
                        "timestamp": event.timestamp,
                        "source": event.source.value,
                        "source_url": event.source_url,
                        "title": event.title,
                        "description": event.description,
                        "full_text": event.full_text,
                        "language": event.language,
                        "threat_category": event.threat_category.value if event.threat_category else None,
                        "threat_confidence": event.threat_confidence,
                        "locations": event.locations,
                        "countries": event.countries,
                        "actors": event.actors,
                        "keywords": event.keywords,
                        "sentiment_score": event.sentiment_score,
                        "importance_score": event.importance_score,
                        "cluster_id": event.cluster_id,
                        "is_duplicate": event.is_duplicate,
                        "duplicate_of": event.duplicate_of,
                        "processed_at": event.processed_at,
                    }
                    
                    query = text("""
                        INSERT INTO events (
                            event_id, timestamp, source, source_url, title, description,
                            full_text, language, threat_category, threat_confidence,
                            locations, countries, actors, keywords, sentiment_score,
                            importance_score, cluster_id, is_duplicate, duplicate_of, processed_at
                        ) VALUES (
                            :event_id, :timestamp, :source, :source_url, :title, :description,
                            :full_text, :language, :threat_category, :threat_confidence,
                            :locations, :countries, :actors, :keywords, :sentiment_score,
                            :importance_score, :cluster_id, :is_duplicate, :duplicate_of, :processed_at
                        )
                        ON CONFLICT (event_id) DO NOTHING
                    """)
                    
                    result = session.execute(query, event_data)
                    if getattr(result, "rowcount", 0) > 0:
                        saved_count += 1
                
                logger.info(f"Bulk saved {saved_count}/{len(events)} events")
                return saved_count
                
        except Exception as e:
            logger.error(f"Error in bulk save: {e}")
            return saved_count
    
    def get_recent_events(
        self,
        hours: int = 24,
        limit: int = 100,
        threat_category: Optional[ThreatCategory] = None
    ) -> List[Dict[str, Any]]:
        """Get recent events from database"""
        try:
            with self.get_session() as session:
                query = text("""
                    SELECT *
                    FROM events
                    WHERE timestamp > NOW() - INTERVAL ':hours hours'
                    AND (:threat_category IS NULL OR threat_category = :threat_category)
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """)
                
                result = session.execute(
                    query,
                    {
                        "hours": hours,
                        "threat_category": threat_category.value if threat_category else None,
                        "limit": limit
                    }
                )
                
                return [dict(row._mapping) for row in result]
                
        except Exception as e:
            logger.error(f"Error fetching recent events: {e}")
            return []
    
    # ==================== Alert Operations ====================
    
    def save_alert(self, alert: ThreatAlert) -> bool:
        """Save a threat alert"""
        try:
            with self.get_session() as session:
                query = text("""
                    INSERT INTO threat_alerts (
                        alert_id, cluster_id, alert_level, title, summary,
                        detailed_analysis, threat_category, region,
                        escalation_probability, supporting_events, source_urls,
                        created_at, expires_at, sent_to, acknowledged
                    ) VALUES (
                        :alert_id, :cluster_id, :alert_level, :title, :summary,
                        :detailed_analysis, :threat_category, :region,
                        :escalation_probability, :supporting_events, :source_urls,
                        :created_at, :expires_at, :sent_to, :acknowledged
                    )
                """)
                
                session.execute(query, {
                    "alert_id": alert.alert_id,
                    "cluster_id": alert.cluster_id,
                    "alert_level": alert.alert_level.value,
                    "title": alert.title,
                    "summary": alert.summary,
                    "detailed_analysis": alert.detailed_analysis,
                    "threat_category": alert.threat_category.value,
                    "region": alert.region,
                    "escalation_probability": alert.escalation_probability,
                    "supporting_events": [str(e) for e in alert.supporting_events],
                    "source_urls": alert.source_urls,
                    "created_at": alert.created_at,
                    "expires_at": alert.expires_at,
                    "sent_to": alert.sent_to,
                    "acknowledged": alert.acknowledged
                })
                
                logger.info(f"Saved alert: {alert.alert_id} - {alert.alert_level.value}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving alert: {e}")
            return False
    
    def get_active_alerts(
        self,
        alert_level: Optional[AlertLevel] = None
    ) -> List[Dict[str, Any]]:
        """Get active (non-acknowledged) alerts"""
        try:
            with self.get_session() as session:
                query = text("""
                    SELECT *
                    FROM active_alerts
                    WHERE (:alert_level IS NULL OR alert_level = :alert_level)
                    ORDER BY 
                        CASE alert_level
                            WHEN 'critical' THEN 1
                            WHEN 'high' THEN 2
                            WHEN 'medium' THEN 3
                            ELSE 4
                        END,
                        created_at DESC
                """)
                
                result = session.execute(
                    query,
                    {"alert_level": alert_level.value if alert_level else None}
                )
                
                return [dict(row._mapping) for row in result]
                
        except Exception as e:
            logger.error(f"Error fetching active alerts: {e}")
            return []
    
    # ==================== Intelligence Brief Operations ====================
    
    def save_brief(self, brief: IntelligenceBrief) -> bool:
        """Save intelligence brief"""
        try:
            with self.get_session() as session:
                # Convert alerts to JSON
                import json
                
                query = text("""
                    INSERT INTO intelligence_briefs (
                        brief_id, date, executive_summary,
                        critical_alerts_json, watch_list_json,
                        declining_tensions_json, regional_summaries_json,
                        total_events_processed, new_alerts, ongoing_situations,
                        generated_at, generated_by, version
                    ) VALUES (
                        :brief_id, :date, :executive_summary,
                        :critical_alerts_json, :watch_list_json,
                        :declining_tensions_json, :regional_summaries_json,
                        :total_events_processed, :new_alerts, :ongoing_situations,
                        :generated_at, :generated_by, :version
                    )
                    ON CONFLICT (date) DO UPDATE SET
                        executive_summary = EXCLUDED.executive_summary,
                        critical_alerts_json = EXCLUDED.critical_alerts_json,
                        new_alerts = EXCLUDED.new_alerts,
                        updated_at = NOW()
                """)
                
                session.execute(query, {
                    "brief_id": brief.brief_id,
                    "date": brief.date.date(),
                    "executive_summary": brief.executive_summary,
                    "critical_alerts_json": json.dumps([a.dict() for a in brief.critical_alerts]),
                    "watch_list_json": json.dumps(brief.watch_list),
                    "declining_tensions_json": json.dumps(brief.declining_tensions),
                    "regional_summaries_json": json.dumps(brief.regional_summaries),
                    "total_events_processed": brief.total_events_processed,
                    "new_alerts": brief.new_alerts,
                    "ongoing_situations": brief.ongoing_situations,
                    "generated_at": brief.generated_at,
                    "generated_by": brief.generated_by,
                    "version": brief.version
                })
                
                logger.info(f"Saved intelligence brief for {brief.date.date()}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving brief: {e}")
            return False
    
    def get_latest_brief(self) -> Optional[Dict[str, Any]]:
        """Get the most recent intelligence brief"""
        try:
            with self.get_session() as session:
                query = text("""
                    SELECT *
                    FROM intelligence_briefs
                    ORDER BY date DESC
                    LIMIT 1
                """)
                
                result = session.execute(query)
                row = result.fetchone()
                
                return dict(row._mapping) if row else None
                
        except Exception as e:
            logger.error(f"Error fetching latest brief: {e}")
            return None
    
    # ==================== Statistics ====================
    
    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            with self.get_session() as session:
                query = text("""
                    SELECT
                        COUNT(*) as total_events,
                        COUNT(DISTINCT threat_category) as unique_categories,
                        COUNT(DISTINCT source) as unique_sources,
                        AVG(threat_confidence) as avg_confidence,
                        AVG(importance_score) as avg_importance
                    FROM events
                    WHERE timestamp > NOW() - INTERVAL ':hours hours'
                """)
                
                result = session.execute(query, {"hours": hours})
                row = result.fetchone()
                
                stats = dict(row._mapping) if row else {}
                
                # Get alert counts
                alert_query = text("""
                    SELECT
                        alert_level,
                        COUNT(*) as count
                    FROM threat_alerts
                    WHERE created_at > NOW() - INTERVAL ':hours hours'
                    GROUP BY alert_level
                """)
                
                alert_result = session.execute(alert_query, {"hours": hours})
                stats["alerts"] = {row.alert_level: row.count for row in alert_result}
                
                return stats
                
        except Exception as e:
            logger.error(f"Error fetching statistics: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Check database connectivity"""
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    db = DatabaseManager()
    
    # Health check
    if db.health_check():
        print("✅ Database connection successful")
        
        # Get statistics
        stats = db.get_statistics(hours=24)
        print(f"\n📊 Statistics (last 24h):")
        print(f"  Total Events: {stats.get('total_events', 0)}")
        print(f"  Avg Confidence: {stats.get('avg_confidence', 0):.2f}")
        print(f"  Active Alerts: {sum(stats.get('alerts', {}).values())}")
    else:
        print("❌ Database connection failed")
