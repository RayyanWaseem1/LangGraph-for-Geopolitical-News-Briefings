"""
Main Pipeline Execution Script for OSINT AI
Fetches events from GDELT and runs through intelligence workflow
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path

# Ensure project root is on sys.path so Data/ and Storage/ resolve when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Data.Gdelt_client import GDELTIngestions
from Pipeline.Intelligence_workflow import run_intelligence_pipeline
from Pipeline.comprehensive_brief_generator import generate_brief_for_alert
from Storage.database_manager import DatabaseManager
from Data.models import IntelligenceBrief
from Data.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)

logger = logging.getLogger(__name__)

class OSINTPipeline:
    """Main OSINT AI Pipeline Orchestrator"""
    
    def __init__(self):
        self.settings = Settings()
        self.gdelt = GDELTIngestions()
        self.db = DatabaseManager()
        
        logger.info("OSINT AI Pipeline initialized")
    
    def fetch_events_from_gdelt(
        self,
        lookback_minutes: int = 60,
        max_records: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetch and parse events from GDELT"""
        
        logger.info(f"Fetching GDELT events (lookback: {lookback_minutes}m, max: {max_records})")
        
        try:
            # Fetch raw articles
            raw_articles = self.gdelt.fetch_recent_events(
                lookback_minutes=lookback_minutes,
                max_records=max_records
            )
            
            logger.info(f"Fetched {len(raw_articles)} raw articles from GDELT")
            
            # Parse articles to events
            raw_events = []
            for article in raw_articles:
                event = self.gdelt.parse_article_to_event(article)
                if event:
                    raw_events.append({
                        "timestamp": event.timestamp.isoformat(),
                        "source": "gdelt",
                        "url": event.source_url,
                        "title": event.title,
                        "description": event.description,
                        "locations": event.locations,
                        "countries": event.countries,
                        "sentiment_score": event.sentiment_score
                    })
            
            logger.info(f"Parsed {len(raw_events)} valid events")
            return raw_events
            
        except Exception as e:
            logger.error(f"Error fetching GDELT events: {e}", exc_info=True)
            return []
    
    async def run_pipeline(
        self,
        lookback_minutes: int = 60,
        max_records: int = 100
    ) -> Optional[IntelligenceBrief]:
        """Run the complete intelligence pipeline"""
        
        logger.info("="*80)
        logger.info("Starting OSINT AI Intelligence Pipeline")
        logger.info("="*80)
        
        # Step 1: Fetch events
        logger.info("\n[1/4] Fetching events from GDELT...")
        raw_events = self.fetch_events_from_gdelt(
            lookback_minutes=lookback_minutes,
            max_records=max_records
        )
        
        if not raw_events:
            logger.warning("No events fetched. Exiting pipeline.")
            return None
        
        # Step 2: Run through LangGraph workflow
        logger.info(f"\n[2/4] Processing {len(raw_events)} events through LangGraph workflow...")
        brief = await run_intelligence_pipeline(raw_events)
        
        if not brief:
            logger.error("Pipeline failed to generate intelligence brief")
            return None
        
        # Step 3: Save to database
        logger.info("\n[3/4] Saving results to database...")
        
        # Save the brief
        if self.db.save_brief(brief):
            logger.info("✓ Intelligence brief saved")
        
        # Save alerts
        saved_alerts = 0
        for alert in brief.critical_alerts:
            if self.db.save_alert(alert):
                saved_alerts += 1
        
        logger.info(f"✓ Saved {saved_alerts}/{len(brief.critical_alerts)} alerts")
        
        # Step 3.5: Generate comprehensive briefs for each alert
        logger.info("\n[3.5/4] Generating comprehensive intelligence briefs for each alert...")
        brief_files = await self.generate_and_save_alert_briefs(
            brief=brief,
            all_events=raw_events,
            predictions=[]  # Predictions embedded in alerts already
        )
        
        if brief_files:
            logger.info(f"✅ Generated {len(brief_files)} comprehensive briefs:")
            for filename in brief_files:
                logger.info(f"   📄 {filename}")
        else:
            logger.info("ℹ️  No critical/high alerts - skipping comprehensive brief generation")
        
        # Step 4: Display results
        logger.info("\n[4/4] Pipeline complete!")
        self.display_brief(brief)
        
        return brief
    
    async def generate_and_save_alert_briefs(
        self,
        brief: IntelligenceBrief,
        all_events: List[Dict[str, Any]],
        predictions: List[Any]
    ) -> List[str]:
        """
        Generate comprehensive intelligence briefs for each critical/high alert
        Matches Streamlit format with detailed analysis
        
        Returns:
            List of filenames for generated briefs
        """
        from Data.models import GeopoliticalEvent, EscalationPrediction, UUID
        
        logger.info("\n[GENERATING COMPREHENSIVE BRIEFS]")
        logger.info("="*80)
        
        brief_files = []
        
        # Filter for critical and high alerts only
        priority_alerts = [
            alert for alert in brief.critical_alerts
            if alert.alert_level.value in ['critical', 'high']
        ]
        
        if not priority_alerts:
            logger.info("No critical/high alerts to generate briefs for")
            return brief_files
        
        logger.info(f"Generating {len(priority_alerts)} comprehensive intelligence briefs...")
        
        # Convert all_events back to GeopoliticalEvent objects for matching
        event_map = {}
        for event_dict in all_events:
            try:
                event = GeopoliticalEvent(
                    timestamp=datetime.fromisoformat(event_dict.get("timestamp", datetime.now(timezone.utc).isoformat())),
                    source=event_dict.get("source", "unknown"),
                    source_url=event_dict["url"],
                    title=event_dict["title"],
                    description=event_dict.get("description", ""),
                    full_text=event_dict.get("full_text"),
                    countries=event_dict.get("countries", []),
                    locations=event_dict.get("locations", []),
                )
                event_map[str(event.event_id)] = event
            except:
                continue
        
        # Generate a brief for each priority alert
        for i, alert in enumerate(priority_alerts, 1):
            try:
                logger.info(f"\n[{i}/{len(priority_alerts)}] Generating brief for: {alert.title[:60]}...")
                
                # Find related events for this alert
                related_events = []
                for event_id in alert.supporting_events[:10]:  # Top 10 supporting events
                    event_id_str = str(event_id)
                    if event_id_str in event_map:
                        related_events.append(event_map[event_id_str])
                
                # Find matching prediction
                matching_prediction = None
                for pred in predictions:
                    if hasattr(pred, 'alert_level') and pred.alert_level == alert.alert_level:
                        matching_prediction = pred
                        break
                
                # Generate comprehensive brief
                comprehensive_brief = await generate_brief_for_alert(
                    alert=alert,
                    related_events=related_events,
                    prediction=matching_prediction
                )
                
                # Save to file
                safe_title = "".join(c for c in alert.title[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
                filename = f"alert_brief_{alert.alert_level.value}_{safe_title}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(comprehensive_brief)
                
                brief_files.append(filename)
                logger.info(f"✅ Saved: {filename}")
                
            except Exception as e:
                logger.error(f"Error generating brief for alert {i}: {e}")
                continue
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ Generated {len(brief_files)} comprehensive intelligence briefs")
        logger.info(f"{'='*80}\n")
        
        return brief_files
    
    def display_brief(self, brief: IntelligenceBrief):
        """Display intelligence brief summary in console"""
        
        print("\n" + "="*80)
        print(" INTELLIGENCE BRIEF SUMMARY")
        print("="*80)
        print(f"\n📅 Date: {brief.date.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"🏭 Generated by: {brief.generated_by} v{brief.version}")
        
        print(f"\n📊 STATISTICS")
        print(f"  • Total Events Processed: {brief.total_events_processed}")
        print(f"  • New Alerts: {brief.new_alerts}")
        print(f"  • Ongoing Situations: {brief.ongoing_situations}")
        
        print(f"\n📝 EXECUTIVE SUMMARY")
        print("-" * 80)
        print(f"{brief.executive_summary[:300]}...")
        
        if brief.critical_alerts:
            print(f"\n🚨 CRITICAL/HIGH ALERTS ({len(brief.critical_alerts)})")
            print("-" * 80)
            
            for i, alert in enumerate(brief.critical_alerts[:5], 1):
                print(f"\n{i}. [{alert.alert_level.value.upper()}] {alert.title[:100]}")
                print(f"   Region: {alert.region}")
                print(f"   Category: {alert.threat_category.value}")
                print(f"   Escalation Probability: {alert.escalation_probability:.0%}")
                print(f"   Summary: {alert.summary[:150]}...")
        else:
            print("\n✅ No critical/high alerts at this time")
        
        print("\n" + "="*80)
        print(" COMPREHENSIVE BRIEFS GENERATED SEPARATELY")
        print(" (See alert_brief_*.txt files for detailed analysis)")
        print("="*80 + "\n")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "database": self.db.health_check(),
            "gdelt": False,
            "statistics": {}
        }
        
        # Test GDELT connectivity (GDELT requires minimum 60 minute timespan)
        try:
            test_events = self.gdelt.fetch_recent_events(lookback_minutes=60, max_records=1)
            status["gdelt"] = len(test_events) > 0
        except Exception as e:
            logger.error(f"GDELT health check failed: {e}")
        
        # Get statistics
        if status["database"]:
            status["statistics"] = self.db.get_statistics(hours=24)
        
        return status


async def main():
    """Main execution function"""
    
    # Initialize pipeline
    pipeline = OSINTPipeline()
    
    # Check system status
    logger.info("Checking system status...")
    status = pipeline.get_system_status()
    
    print("\n🔍 System Status Check")
    print(f"  Database: {'✅' if status['database'] else '❌'}")
    print(f"  GDELT API: {'✅' if status['gdelt'] else '❌'}")
    
    if not status['database']:
        logger.error("Database connection failed. Please check your PostgreSQL setup.")
        return
    
    if not status['gdelt']:
        logger.warning("GDELT API connectivity issue. Continuing anyway...")
    
    # Display recent statistics
    stats = status.get('statistics', {})
    if stats:
        print(f"\n📊 Last 24h Statistics:")
        print(f"  Events: {stats.get('total_events', 0)}")
        print(f"  Alerts: {sum(stats.get('alerts', {}).values())}")
    
    # Run pipeline
    print("\n🚀 Starting intelligence pipeline...\n")
    
    try:
        brief = await pipeline.run_pipeline(
            lookback_minutes=120,  # Last 2 hour
            max_records=200       # Max 200 events
        )
        
        if brief:
            logger.info("✅ Pipeline execution successful!")
            
            # Save brief to file
            brief_filename = f"brief_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(brief_filename, 'w') as f:
                f.write(f"Intelligence Brief - {brief.date}\n")
                f.write("="*80 + "\n\n")
                f.write(f"Executive Summary:\n{brief.executive_summary}\n\n")
                f.write(f"Statistics:\n")
                f.write(f"  Total Events: {brief.total_events_processed}\n")
                f.write(f"  New Alerts: {brief.new_alerts}\n\n")
                
                if brief.critical_alerts:
                    f.write("Critical Alerts:\n")
                    for i, alert in enumerate(brief.critical_alerts, 1):
                        f.write(f"\n{i}. [{alert.alert_level.value}] {alert.title}\n")
                        f.write(f"   {alert.summary}\n")
            
            logger.info(f"📄 Brief saved to: {brief_filename}")
        else:
            logger.error("❌ Pipeline failed to generate brief")
    
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}", exc_info=True)


if __name__ == "__main__":
    # Check Python version
    import sys
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ required")
        sys.exit(1)
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  OSINT AI - Geopolitical Early-Warning System             ║
    ║  Intelligence Pipeline Execution                           ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Run async main
    asyncio.run(main())