-- OSINT AI Database Initialization Script
-- This script sets up the database schema for the OSINT AI system

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For text similarity

-- Create events table with time-series optimization
CREATE TABLE IF NOT EXISTS events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL,
    source VARCHAR(50) NOT NULL,
    source_url TEXT NOT NULL,
    
    -- Content
    title VARCHAR(500) NOT NULL,
    description TEXT,
    full_text TEXT,
    language VARCHAR(10) DEFAULT 'en',
    
    -- Classification
    threat_category VARCHAR(50),
    threat_confidence FLOAT CHECK (threat_confidence >= 0 AND threat_confidence <= 1),
    
    -- Geospatial
    locations TEXT[],
    countries TEXT[],
    coordinates GEOMETRY(Point, 4326),  -- PostGIS point
    region VARCHAR(100),
    
    -- Entities
    entities JSONB DEFAULT '[]'::jsonb,
    actors TEXT[],
    
    -- Metadata
    keywords TEXT[],
    sentiment_score FLOAT CHECK (sentiment_score >= -1 AND sentiment_score <= 1),
    importance_score FLOAT DEFAULT 0.5 CHECK (importance_score >= 0 AND importance_score <= 1),
    
    -- Processing
    embedding VECTOR(1536),  -- For pgvector if available
    cluster_id INTEGER,
    is_duplicate BOOLEAN DEFAULT FALSE,
    duplicate_of UUID REFERENCES events(event_id),
    
    -- Timestamps
    processed_at TIMESTAMPTZ DEFAULT NOW(),
    indexed_at TIMESTAMPTZ,
    
    -- Indexes
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('events', 'timestamp', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_source ON events(source);
CREATE INDEX IF NOT EXISTS idx_events_threat_category ON events(threat_category);
CREATE INDEX IF NOT EXISTS idx_events_countries ON events USING GIN(countries);
CREATE INDEX IF NOT EXISTS idx_events_locations ON events USING GIN(locations);
CREATE INDEX IF NOT EXISTS idx_events_actors ON events USING GIN(actors);
CREATE INDEX IF NOT EXISTS idx_events_keywords ON events USING GIN(keywords);
CREATE INDEX IF NOT EXISTS idx_events_cluster_id ON events(cluster_id);
CREATE INDEX IF NOT EXISTS idx_events_importance ON events(importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_events_title_trgm ON events USING GIN(title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_events_coordinates ON events USING GIST(coordinates);

-- Create event clusters table
CREATE TABLE IF NOT EXISTS event_clusters (
    cluster_id SERIAL PRIMARY KEY,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    
    -- Characteristics
    primary_threat VARCHAR(50) NOT NULL,
    region VARCHAR(100) NOT NULL,
    event_count INTEGER DEFAULT 0,
    
    -- Computed features
    intensity FLOAT CHECK (intensity >= 0 AND intensity <= 1),
    geographic_spread FLOAT DEFAULT 0,
    escalation_trajectory VARCHAR(20) DEFAULT 'stable',
    
    -- Narrative
    summary TEXT,
    key_actors TEXT[],
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_clusters_time ON event_clusters(start_time, end_time);
CREATE INDEX IF NOT EXISTS idx_clusters_region ON event_clusters(region);
CREATE INDEX IF NOT EXISTS idx_clusters_threat ON event_clusters(primary_threat);

-- Create escalation predictions table
CREATE TABLE IF NOT EXISTS escalation_predictions (
    id SERIAL PRIMARY KEY,
    cluster_id INTEGER REFERENCES event_clusters(cluster_id),
    prediction_time TIMESTAMPTZ DEFAULT NOW(),
    
    -- Prediction
    escalation_probability FLOAT NOT NULL CHECK (escalation_probability >= 0 AND escalation_probability <= 1),
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    time_horizon_days INTEGER DEFAULT 30,
    alert_level VARCHAR(20) NOT NULL,
    
    -- Features
    features JSONB DEFAULT '{}'::jsonb,
    
    -- Explanation
    reasoning TEXT,
    historical_parallels TEXT[],
    risk_factors TEXT[],
    mitigating_factors TEXT[],
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_cluster ON escalation_predictions(cluster_id);
CREATE INDEX IF NOT EXISTS idx_predictions_time ON escalation_predictions(prediction_time DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_probability ON escalation_predictions(escalation_probability DESC);

-- Create threat alerts table
CREATE TABLE IF NOT EXISTS threat_alerts (
    alert_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cluster_id INTEGER REFERENCES event_clusters(cluster_id),
    alert_level VARCHAR(20) NOT NULL,
    
    -- Content
    title VARCHAR(500) NOT NULL,
    summary TEXT NOT NULL,
    detailed_analysis TEXT,
    
    -- Context
    threat_category VARCHAR(50) NOT NULL,
    region VARCHAR(100) NOT NULL,
    escalation_probability FLOAT,
    
    -- Evidence
    supporting_events UUID[],
    source_urls TEXT[],
    
    -- Timing
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    
    -- Distribution
    sent_to TEXT[],
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_alerts_level ON threat_alerts(alert_level);
CREATE INDEX IF NOT EXISTS idx_alerts_time ON threat_alerts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_category ON threat_alerts(threat_category);
CREATE INDEX IF NOT EXISTS idx_alerts_region ON threat_alerts(region);
CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON threat_alerts(acknowledged);

-- Create intelligence briefs table
CREATE TABLE IF NOT EXISTS intelligence_briefs (
    brief_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL UNIQUE,
    
    -- Content
    executive_summary TEXT NOT NULL,
    critical_alerts_json JSONB DEFAULT '[]'::jsonb,
    watch_list_json JSONB DEFAULT '[]'::jsonb,
    declining_tensions_json JSONB DEFAULT '[]'::jsonb,
    regional_summaries_json JSONB DEFAULT '{}'::jsonb,
    
    -- Statistics
    total_events_processed INTEGER DEFAULT 0,
    new_alerts INTEGER DEFAULT 0,
    ongoing_situations INTEGER DEFAULT 0,
    
    -- Metadata
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    generated_by VARCHAR(100) DEFAULT 'OSINT-AI-System',
    version VARCHAR(20) DEFAULT '1.0'
);

CREATE INDEX IF NOT EXISTS idx_briefs_date ON intelligence_briefs(date DESC);
CREATE INDEX IF NOT EXISTS idx_briefs_generated ON intelligence_briefs(generated_at DESC);

-- Create historical events table for pattern matching
CREATE TABLE IF NOT EXISTS historical_events (
    id SERIAL PRIMARY KEY,
    event_date DATE NOT NULL,
    title VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    
    -- Context
    location VARCHAR(200),
    countries TEXT[],
    threat_category VARCHAR(50),
    
    -- Outcome
    outcome TEXT NOT NULL,
    escalated BOOLEAN DEFAULT FALSE,
    lessons_learned TEXT,
    
    -- Metadata
    source VARCHAR(200),
    tags TEXT[],
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_historical_date ON historical_events(event_date);
CREATE INDEX IF NOT EXISTS idx_historical_category ON historical_events(threat_category);
CREATE INDEX IF NOT EXISTS idx_historical_countries ON historical_events USING GIN(countries);

-- Create system metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    
    -- Throughput
    events_processed_24h INTEGER DEFAULT 0,
    events_per_minute FLOAT DEFAULT 0,
    
    -- Quality
    classification_accuracy FLOAT,
    false_positive_rate FLOAT,
    false_negative_rate FLOAT,
    
    -- Latency
    avg_processing_time_ms FLOAT,
    p95_processing_time_ms FLOAT,
    p99_processing_time_ms FLOAT,
    
    -- Alerts
    critical_alerts_24h INTEGER DEFAULT 0,
    high_alerts_24h INTEGER DEFAULT 0,
    
    -- Resources
    cpu_usage_percent FLOAT,
    memory_usage_mb FLOAT,
    
    -- LLM
    llm_tokens_used_24h INTEGER DEFAULT 0,
    llm_api_errors_24h INTEGER DEFAULT 0
);

-- Convert to hypertable
SELECT create_hypertable('system_metrics', 'timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp DESC);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add triggers for updated_at
CREATE TRIGGER update_events_updated_at BEFORE UPDATE ON events
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_clusters_updated_at BEFORE UPDATE ON event_clusters
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries
CREATE OR REPLACE VIEW recent_critical_events AS
SELECT 
    e.*,
    c.summary as cluster_summary,
    p.escalation_probability
FROM events e
LEFT JOIN event_clusters c ON e.cluster_id = c.cluster_id
LEFT JOIN escalation_predictions p ON c.cluster_id = p.cluster_id
WHERE e.timestamp > NOW() - INTERVAL '24 hours'
    AND (e.importance_score > 0.7 OR p.escalation_probability > 0.5)
ORDER BY e.timestamp DESC;

CREATE OR REPLACE VIEW active_alerts AS
SELECT 
    a.*,
    c.summary as cluster_summary,
    c.event_count
FROM threat_alerts a
LEFT JOIN event_clusters c ON a.cluster_id = c.cluster_id
WHERE a.expires_at IS NULL OR a.expires_at > NOW()
    AND NOT a.acknowledged
ORDER BY 
    CASE a.alert_level
        WHEN 'critical' THEN 1
        WHEN 'high' THEN 2
        WHEN 'medium' THEN 3
        WHEN 'low' THEN 4
        ELSE 5
    END,
    a.created_at DESC;

-- Grant permissions (adjust as needed for production)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- Insert sample historical events for pattern matching
INSERT INTO historical_events (event_date, title, description, location, countries, threat_category, outcome, escalated, lessons_learned, source) VALUES
('2022-02-24', 'Russian Invasion of Ukraine', 'Russia launched a full-scale invasion of Ukraine', 'Ukraine', ARRAY['Russia', 'Ukraine'], 'military_buildup', 'Ongoing war, sanctions, NATO expansion', TRUE, 'Prewar military buildup was clear indicator', 'Public records'),
('2020-09-27', 'Second Nagorno-Karabakh War', 'Azerbaijan launched offensive against Armenian forces', 'Nagorno-Karabakh', ARRAY['Azerbaijan', 'Armenia'], 'border_conflict', 'Azerbaijani victory, ceasefire agreement', TRUE, 'Drone warfare proved decisive', 'Public records'),
('2014-02-27', 'Russian Annexation of Crimea', 'Russia annexed Crimea from Ukraine', 'Crimea', ARRAY['Russia', 'Ukraine'], 'military_buildup', 'Annexation completed, sanctions imposed', TRUE, 'Little green men tactics used', 'Public records')
ON CONFLICT DO NOTHING;

-- Add compression policy for old data (TimescaleDB)
SELECT add_compression_policy('events', INTERVAL '90 days');
SELECT add_compression_policy('system_metrics', INTERVAL '90 days');

-- Add retention policy (optional - removes data older than 1 year)
-- SELECT add_retention_policy('events', INTERVAL '1 year');
-- SELECT add_retention_policy('system_metrics', INTERVAL '1 year');

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'OSINT AI database initialized successfully!';
    RAISE NOTICE 'Tables created: events, event_clusters, escalation_predictions, threat_alerts, intelligence_briefs, historical_events, system_metrics';
    RAISE NOTICE 'Extensions enabled: timescaledb, postgis, uuid-ossp, pg_trgm';
END $$;