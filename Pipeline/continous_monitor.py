"""
Continuous Monitoring Service for OSINT AI
Runs 24/7, polling GDELT every 15 minutes for new events
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Optional
import time

from run_pipeline import OSINTPipeline
from Data.models import IntelligenceBrief, AlertLevel
from Data.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('continuous_monitor.log')
    ]
)

logger = logging.getLogger(__name__)


class ContinuousMonitor:
    """24/7 continuous monitoring service"""
    
    def __init__(self, interval_minutes: int = 15):
        self.settings = Settings()
        self.pipeline = OSINTPipeline()
        self.interval_minutes = interval_minutes
        self.running = False
        self.cycle_count = 0
        self.last_brief: Optional[IntelligenceBrief] = None
        
        # Statistics
        self.stats = {
            "cycles_completed": 0,
            "total_events_processed": 0,
            "total_alerts_generated": 0,
            "errors": 0,
            "start_time": None
        }
        
        logger.info(f"Continuous monitor initialized (interval: {interval_minutes}m)")
    
    def handle_shutdown(self, signum, frame):
        """Graceful shutdown handler"""
        logger.info("\n⚠️  Shutdown signal received. Stopping gracefully...")
        self.running = False
    
    async def run_cycle(self) -> bool:
        """Run a single monitoring cycle"""
        
        cycle_start = time.time()
        
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"CYCLE #{self.cycle_count + 1} - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            logger.info(f"{'='*80}")
            
            # Run pipeline
            brief = await self.pipeline.run_pipeline(
                lookback_minutes=self.interval_minutes,
                max_records=250
            )
            
            if brief:
                self.last_brief = brief
                
                # Update statistics
                self.stats["cycles_completed"] += 1
                self.stats["total_events_processed"] += brief.total_events_processed
                self.stats["total_alerts_generated"] += brief.new_alerts
                
                # Log critical alerts
                critical_alerts = [a for a in brief.critical_alerts if a.alert_level == AlertLevel.CRITICAL]
                if critical_alerts:
                    logger.warning(f"🚨 {len(critical_alerts)} CRITICAL ALERTS in this cycle!")
                    for alert in critical_alerts:
                        logger.warning(f"  • {alert.title} (Region: {alert.region})")
                
                # Display summary
                cycle_duration = time.time() - cycle_start
                logger.info(f"\n✅ Cycle completed in {cycle_duration:.2f}s")
                logger.info(f"   Events: {brief.total_events_processed}")
                logger.info(f"   Alerts: {brief.new_alerts}")
                
                self.cycle_count += 1
                return True
            else:
                logger.error("Cycle failed to generate brief")
                self.stats["errors"] += 1
                return False
        
        except Exception as e:
            logger.error(f"Error in cycle: {e}", exc_info=True)
            self.stats["errors"] += 1
            return False
    
    def display_statistics(self):
        """Display running statistics"""
        
        uptime = datetime.utcnow() - self.stats["start_time"] if self.stats["start_time"] else timedelta(0)
        
        print("\n" + "="*80)
        print(" MONITORING STATISTICS")
        print("="*80)
        print(f"  Uptime: {uptime}")
        print(f"  Cycles Completed: {self.stats['cycles_completed']}")
        print(f"  Total Events: {self.stats['total_events_processed']}")
        print(f"  Total Alerts: {self.stats['total_alerts_generated']}")
        print(f"  Errors: {self.stats['errors']}")
        
        if self.stats['cycles_completed'] > 0:
            avg_events = self.stats['total_events_processed'] / self.stats['cycles_completed']
            avg_alerts = self.stats['total_alerts_generated'] / self.stats['cycles_completed']
            print(f"  Avg Events/Cycle: {avg_events:.1f}")
            print(f"  Avg Alerts/Cycle: {avg_alerts:.1f}")
        
        print("="*80 + "\n")
    
    async def monitor(self):
        """Main monitoring loop"""
        
        logger.info("🚀 Starting continuous monitoring service...")
        logger.info(f"   Interval: {self.interval_minutes} minutes")
        logger.info(f"   Press Ctrl+C to stop\n")
        
        self.running = True
        self.stats["start_time"] = datetime.utcnow()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
        while self.running:
            try:
                # Run monitoring cycle
                success = await self.run_cycle()
                
                if not self.running:
                    break
                
                # Display statistics every 4 cycles (1 hour if 15min interval)
                if self.cycle_count % 4 == 0 and self.cycle_count > 0:
                    self.display_statistics()
                
                # Calculate wait time
                next_cycle = datetime.utcnow() + timedelta(minutes=self.interval_minutes)
                wait_seconds = self.interval_minutes * 60
                
                logger.info(f"\n⏳ Next cycle at {next_cycle.strftime('%H:%M:%S UTC')}")
                logger.info(f"   Waiting {self.interval_minutes} minutes...")
                
                # Wait with periodic status checks
                for i in range(wait_seconds // 10):
                    if not self.running:
                        break
                    await asyncio.sleep(10)
                
            except asyncio.CancelledError:
                logger.info("Monitor cancelled")
                break
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                logger.info("Waiting 60s before retry...")
                await asyncio.sleep(60)
        
        # Shutdown
        logger.info("\n" + "="*80)
        logger.info(" MONITORING STOPPED")
        logger.info("="*80)
        self.display_statistics()
        logger.info("Goodbye! 👋\n")


class ScheduledTasks:
    """Additional scheduled tasks (daily briefs, cleanup, etc.)"""
    
    def __init__(self, monitor: ContinuousMonitor):
        self.monitor = monitor
        self.last_daily_brief = None
    
    async def generate_daily_brief(self):
        """Generate comprehensive daily brief"""
        
        logger.info("\n📰 Generating daily intelligence brief...")
        
        try:
            # Fetch events from last 24 hours
            brief = await self.monitor.pipeline.run_pipeline(
                lookback_minutes=1440,  # 24 hours
                max_records=500
            )
            
            if brief:
                # Save to special daily brief file
                filename = f"daily_brief_{datetime.utcnow().strftime('%Y%m%d')}.txt"
                
                with open(filename, 'w') as f:
                    f.write("="*80 + "\n")
                    f.write(f" DAILY INTELLIGENCE BRIEF - {datetime.utcnow().strftime('%Y-%m-%d')}\n")
                    f.write("="*80 + "\n\n")
                    f.write(f"Executive Summary:\n{brief.executive_summary}\n\n")
                    f.write(f"Statistics:\n")
                    f.write(f"  Total Events Processed: {brief.total_events_processed}\n")
                    f.write(f"  New Alerts: {brief.new_alerts}\n")
                    f.write(f"  Ongoing Situations: {brief.ongoing_situations}\n\n")
                    
                    if brief.critical_alerts:
                        f.write(f"Critical Alerts ({len(brief.critical_alerts)}):\n")
                        f.write("-"*80 + "\n")
                        for i, alert in enumerate(brief.critical_alerts, 1):
                            f.write(f"\n{i}. [{alert.alert_level.value.upper()}] {alert.title}\n")
                            f.write(f"   Region: {alert.region}\n")
                            f.write(f"   Category: {alert.threat_category.value}\n")
                            f.write(f"   Probability: {alert.escalation_probability:.0%}\n")
                            f.write(f"   {alert.summary}\n")
                
                logger.info(f"✅ Daily brief saved to: {filename}")
                self.last_daily_brief = datetime.utcnow()
                
                return True
        
        except Exception as e:
            logger.error(f"Error generating daily brief: {e}", exc_info=True)
            return False
    
    async def cleanup_old_data(self):
        """Cleanup old data (optional)"""
        logger.info("🧹 Running data cleanup...")
        # Implementation depends on retention policy
        pass
    
    async def schedule_tasks(self):
        """Run scheduled tasks"""
        
        while self.monitor.running:
            now = datetime.utcnow()
            
            # Daily brief at 00:00 UTC
            if now.hour == 0 and now.minute < 15:
                if not self.last_daily_brief or \
                   (now - self.last_daily_brief) > timedelta(hours=23):
                    await self.generate_daily_brief()
            
            # Weekly cleanup on Sunday at 02:00 UTC
            if now.weekday() == 6 and now.hour == 2 and now.minute < 15:
                await self.cleanup_old_data()
            
            # Check every 15 minutes
            await asyncio.sleep(900)


async def main():
    """Main entry point"""
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  OSINT AI - Geopolitical Early-Warning System             ║
    ║  Continuous Monitoring Service                             ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Get configuration
    settings = Settings()
    interval = settings.GDELT_POLL_INTERVAL // 60  # Convert seconds to minutes
    
    logger.info(f"Configuration:")
    logger.info(f"  Polling Interval: {interval} minutes")
    logger.info(f"  Primary LLM: {settings.PRIMARY_LLM_MODEL}")
    logger.info(f"  Fast LLM: {settings.FAST_LLM_MODEL}")
    logger.info(f"  Database: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}")
    
    # Initialize monitor
    monitor = ContinuousMonitor(interval_minutes=interval)
    
    # Check system status
    logger.info("\nChecking system status...")
    status = monitor.pipeline.get_system_status()
    
    print(f"\n🔍 System Status:")
    print(f"  Database: {'✅' if status['database'] else '❌'}")
    print(f"  GDELT API: {'✅' if status['gdelt'] else '❌'}")
    
    if not status['database']:
        logger.error("❌ Database connection failed. Cannot start monitoring.")
        return
    
    # Initialize scheduled tasks
    scheduler = ScheduledTasks(monitor)
    
    # Run monitoring and scheduled tasks concurrently
    try:
        await asyncio.gather(
            monitor.monitor(),
            scheduler.schedule_tasks()
        )
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")


if __name__ == "__main__":
    # Verify Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ required")
        sys.exit(1)
    
    # Run
    asyncio.run(main())