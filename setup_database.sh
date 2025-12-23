#!/bin/bash

# OSINT AI - Automated Setup Script
# Sets up PostgreSQL database and creates required tables

set -e  # Exit on any error

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  OSINT AI - Automated Database Setup                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

echo "✅ Docker is running"

# Check if docker-compose file exists
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml not found in current directory"
    echo "Please run this script from the project root directory"
    exit 1
fi

echo "✅ Found docker-compose.yml"

# Start PostgreSQL and Redis
echo ""
echo "🚀 Starting PostgreSQL and Redis..."
docker-compose up -d

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
sleep 10

# Check if PostgreSQL is running
if ! docker ps | grep -q osint_postgres; then
    echo "❌ PostgreSQL container failed to start"
    echo "Check logs with: docker logs osint_postgres"
    exit 1
fi

echo "✅ PostgreSQL container is running"

# Create database tables
echo ""
echo "📊 Creating database tables..."

docker exec -i osint_postgres psql -U postgres -d osint_ai << 'EOF'
-- Create events table
CREATE TABLE IF NOT EXISTS events (
    event_id UUID PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    source VARCHAR(50) NOT NULL,
    source_url TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    full_text TEXT,
    language VARCHAR(10) DEFAULT 'en',
    threat_category VARCHAR(50),
    threat_confidence FLOAT,
    locations TEXT[],
    countries TEXT[],
    actors TEXT[],
    keywords TEXT[],
    sentiment_score FLOAT,
    importance_score FLOAT DEFAULT 0.5,
    cluster_id INTEGER,
    is_duplicate BOOLEAN DEFAULT FALSE,
    duplicate_of UUID,
    processed_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_threat_category ON events(threat_category);
CREATE INDEX IF NOT EXISTS idx_events_source ON events(source);

-- Create threat_alerts table
CREATE TABLE IF NOT EXISTS threat_alerts (
    alert_id UUID PRIMARY KEY,
    alert_level VARCHAR(20) NOT NULL,
    title TEXT NOT NULL,
    summary TEXT,
    detailed_analysis TEXT,
    threat_category VARCHAR(50) NOT NULL,
    region VARCHAR(100),
    escalation_probability FLOAT,
    supporting_events UUID[],
    source_urls TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    sent_to TEXT[],
    acknowledged BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_alerts_level ON threat_alerts(alert_level);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON threat_alerts(created_at DESC);

-- Create active_alerts view
CREATE OR REPLACE VIEW active_alerts AS
SELECT * FROM threat_alerts
WHERE acknowledged = FALSE
AND (expires_at IS NULL OR expires_at > NOW());

-- Create intelligence_briefs table
CREATE TABLE IF NOT EXISTS intelligence_briefs (
    brief_id UUID PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    executive_summary TEXT NOT NULL,
    critical_alerts_json JSONB,
    watch_list_json JSONB,
    declining_tensions_json JSONB,
    regional_summaries_json JSONB,
    total_events_processed INTEGER DEFAULT 0,
    new_alerts INTEGER DEFAULT 0,
    ongoing_situations INTEGER DEFAULT 0,
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    generated_by VARCHAR(100) DEFAULT 'OSINT-AI-System',
    version VARCHAR(20) DEFAULT '1.0',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_briefs_date ON intelligence_briefs(date DESC);

-- Confirm tables created
\echo ''
\echo '✅ Database tables created successfully!'
\echo ''
EOF

if [ $? -eq 0 ]; then
    echo "✅ All tables created successfully"
else
    echo "❌ Error creating tables"
    exit 1
fi

# Test database connection
echo ""
echo "🔍 Testing database connection..."
docker exec osint_postgres psql -U postgres -d osint_ai -c "SELECT COUNT(*) FROM events;"

if [ $? -eq 0 ]; then
    echo "✅ Database connection test successful"
else
    echo "❌ Database connection test failed"
    exit 1
fi

# Display status
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✅ Setup Complete!                                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Database Configuration:"
echo "  Host: localhost"
echo "  Port: 5432"
echo "  Database: osint_ai"
echo "  User: postgres"
echo "  Password: 650Cloud"
echo ""
echo "Services Running:"
docker ps --filter "name=osint" --format "  • {{.Names}} ({{.Status}})"
echo ""
echo "Next Steps:"
echo "  1. Activate your virtual environment: source .venv/bin/activate"
echo "  2. Run the pipeline: cd Pipeline && python run_pipeline_fixed.py"
echo ""
echo "To stop services: docker-compose down"
echo "To view logs: docker logs osint_postgres"
echo ""