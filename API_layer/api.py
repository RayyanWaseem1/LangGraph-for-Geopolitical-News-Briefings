"""
FastAPI REST API for OSINT AI System
Provides endpoints for events, alerts, briefs, and real-time streaming
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import asyncio
import json

from Storage.database_manager import DatabaseManager
from Data.aggregated_ingestion import AggregatedIngestion
from Data.models import AlertLevel, ThreatCategory, GeopoliticalEvent
from Storage.cache_manager import CacheManager
from Data.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OSINT AI - Geopolitical Early-Warning API",
    description="Real-time threat intelligence and geopolitical event monitoring",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
settings = Settings()
db = DatabaseManager()
cache = CacheManager()
aggregator = AggregatedIngestion()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")

manager = ConnectionManager()

# ==================== Pydantic Models ====================

class EventResponse(BaseModel):
    event_id: str
    timestamp: datetime
    source: str
    title: str
    description: str
    threat_category: Optional[str]
    locations: List[str]
    importance_score: float

class AlertResponse(BaseModel):
    alert_id: str
    alert_level: str
    title: str
    summary: str
    threat_category: str
    region: str
    escalation_probability: float
    created_at: datetime

class BriefResponse(BaseModel):
    brief_id: str
    date: datetime
    executive_summary: str
    total_events_processed: int
    new_alerts: int
    critical_alerts_count: int

class StatsResponse(BaseModel):
    total_events: int
    unique_categories: int
    active_alerts: int
    avg_confidence: float

# ==================== Health & Status ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "OSINT AI - Geopolitical Early-Warning System",
        "version": "0.1.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_healthy = db.health_check()
    cache_healthy = cache.health_check()
    
    return {
        "status": "healthy" if (db_healthy and cache_healthy) else "degraded",
        "database": "connected" if db_healthy else "disconnected",
        "cache": "connected" if cache_healthy else "disconnected",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/stats", response_model=StatsResponse)
async def get_statistics(hours: int = Query(24, ge=1, le=168)):
    """Get system statistics"""
    
    # Check cache first
    cache_key = f"stats_{hours}h"
    cached_stats = cache.get_json(cache_key)
    
    if cached_stats:
        return cached_stats
    
    # Get fresh stats
    stats = db.get_statistics(hours=hours)
    
    response = StatsResponse(
        total_events=stats.get("total_events", 0),
        unique_categories=stats.get("unique_categories", 0),
        active_alerts=sum(stats.get("alerts", {}).values()),
        avg_confidence=stats.get("avg_confidence", 0) or 0
    )
    
    # Cache for 5 minutes
    cache.set_json(cache_key, response.dict(), ttl=300)
    
    return response

# ==================== Events ====================

@app.get("/events/recent")
async def get_recent_events(
    hours: int = Query(24, ge=1, le=168),
    limit: int = Query(100, ge=1, le=500),
    threat_category: Optional[str] = None
):
    """Get recent events"""
    
    # Validate threat category
    if threat_category:
        try:
            threat_cat = ThreatCategory(threat_category)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid threat category: {threat_category}")
    else:
        threat_cat = None
    
    # Get events
    events = db.get_recent_events(hours=hours, limit=limit, threat_category=threat_cat)
    
    return {
        "count": len(events),
        "hours": hours,
        "events": events
    }

@app.get("/events/search")
async def search_events(
    query: str = Query(..., min_length=1),
    limit: int = Query(50, ge=1, le=100)
):
    """Search events by text query"""
    
    # This would use vector store for similarity search
    # For now, simple database query
    
    return {
        "query": query,
        "count": 0,
        "events": [],
        "message": "Vector search not yet implemented"
    }

@app.post("/events/ingest")
async def ingest_events(
    gdelt_lookback_minutes: int = Query(60, ge=15, le=1440),
    newsapi_lookback_hours: int = Query(24, ge=1, le=72),
    max_per_source: int = Query(100, ge=10, le=500)
):
    """Trigger manual event ingestion"""
    
    try:
        # Fetch from all sources
        events = aggregator.fetch_all_sources_sync(
            gdelt_lookback_minutes=gdelt_lookback_minutes,
            newsapi_lookback_hours=newsapi_lookback_hours,
            max_events_per_source=max_per_source
        )
        
        # Save to database
        saved = db.save_events_bulk(events)
        
        # Broadcast to WebSocket clients
        await manager.broadcast({
            "type": "ingestion_complete",
            "events_fetched": len(events),
            "events_saved": saved,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "status": "success",
            "events_fetched": len(events),
            "events_saved": saved,
            "statistics": aggregator.get_statistics()
        }
    
    except Exception as e:
        logger.error(f"Error in manual ingestion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Alerts ====================

@app.get("/alerts/active")
async def get_active_alerts(
    alert_level: Optional[str] = None
):
    """Get active (non-acknowledged) alerts"""
    
    # Validate alert level
    if alert_level:
        try:
            level = AlertLevel(alert_level)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid alert level: {alert_level}")
    else:
        level = None
    
    # Get alerts
    alerts = db.get_active_alerts(alert_level=level)
    
    return {
        "count": len(alerts),
        "alerts": alerts
    }

@app.get("/alerts/critical")
async def get_critical_alerts():
    """Get critical alerts"""
    alerts = db.get_active_alerts(alert_level=AlertLevel.CRITICAL)
    
    return {
        "count": len(alerts),
        "alerts": alerts
    }

@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert"""
    
    # This would update the alert in database
    # Not implemented in current database_manager
    
    return {
        "status": "success",
        "alert_id": alert_id,
        "acknowledged_at": datetime.utcnow().isoformat()
    }

# ==================== Intelligence Briefs ====================

@app.get("/brief/latest")
async def get_latest_brief():
    """Get the latest intelligence brief"""
    
    brief = db.get_latest_brief()
    
    if not brief:
        raise HTTPException(status_code=404, detail="No intelligence brief found")
    
    return brief

@app.get("/brief/date/{date}")
async def get_brief_by_date(date: str):
    """Get intelligence brief for a specific date (YYYY-MM-DD)"""
    
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    # This would query database for specific date
    # Not implemented yet
    
    raise HTTPException(status_code=404, detail=f"No brief found for date: {date}")

# ==================== Real-Time Streaming ====================

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time event streaming
    
    Clients receive:
    - New events as they're ingested
    - Alert updates
    - System status updates
    """
    await manager.connect(websocket)
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to OSINT AI event stream",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            # Wait for client messages (ping/pong, subscription updates, etc.)
            data = await websocket.receive_text()
            
            # Echo back for now (could handle subscriptions, filters, etc.)
            await websocket.send_json({
                "type": "echo",
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected from WebSocket")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.get("/stream/sse")
async def event_stream():
    """Server-Sent Events endpoint for real-time updates"""
    
    async def generate():
        """Generate SSE events"""
        while True:
            # Get latest statistics
            stats = db.get_statistics(hours=1)
            
            # Format as SSE
            data = json.dumps({
                "total_events": stats.get("total_events", 0),
                "alerts": stats.get("alerts", {}),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            yield f"data: {data}\n\n"
            
            # Update every 30 seconds
            await asyncio.sleep(30)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# ==================== Regions & Locations ====================

@app.get("/regions")
async def get_monitored_regions():
    """Get list of monitored regions"""
    from Data.settings import REGIONS
    
    return {
        "count": len(REGIONS),
        "regions": REGIONS
    }

@app.get("/regions/{region}/events")
async def get_region_events(
    region: str,
    hours: int = Query(24, ge=1, le=168)
):
    """Get events for a specific region"""
    
    # Fetch region-specific events
    events = aggregator.fetch_by_region(region, lookback_hours=hours)
    
    return {
        "region": region,
        "count": len(events),
        "events": [
            {
                "event_id": str(e.event_id),
                "title": e.title,
                "timestamp": e.timestamp.isoformat(),
                "source": e.source.value,
                "importance_score": e.importance_score
            }
            for e in events
        ]
    }

# ==================== System Management ====================

@app.post("/system/cache/clear")
async def clear_cache():
    """Clear system cache"""
    cache.flush_all()
    
    return {
        "status": "success",
        "message": "Cache cleared",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/system/metrics")
async def get_system_metrics():
    """Get detailed system metrics"""
    
    cache_stats = cache.get_stats()
    db_stats = db.get_statistics(hours=24)
    
    return {
        "cache": cache_stats,
        "database": db_stats,
        "ingestion": aggregator.get_statistics(),
        "timestamp": datetime.utcnow().isoformat()
    }

# ==================== Background Tasks ====================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("OSINT AI API starting up...")
    
    # Test database connection
    if db.health_check():
        logger.info("✅ Database connection established")
    else:
        logger.error("❌ Database connection failed")
    
    # Test cache connection
    if cache.health_check():
        logger.info("✅ Cache connection established")
    else:
        logger.error("❌ Cache connection failed")
    
    logger.info("API ready to serve requests")

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("OSINT AI API shutting down...")
    
    # Close WebSocket connections
    for connection in manager.active_connections[:]:
        await connection.close()
    
    logger.info("Shutdown complete")

# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn  # type: ignore[import]
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  OSINT AI - Geopolitical Early-Warning System API         ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
