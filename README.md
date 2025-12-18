# LangGraph for Geopolitial Early Warning System

# Overview
An automated intelligence platform that monitors, classifies, and analyzes global geopolitial events in real-time using multi-agent LLM workflow.
The system processes 200+ events/hour from multiple OSINT sources.

# Background and Motivation
Traditional intelligence analysis cannot scale with modern information velocity.
GDELT alone generates tens of thousands of daily event across 100+ languages. 
This system addresses the analytical gap through: 
* Automated Classification: Multi-Agent LangGraph pipeline categorizing threats across 12 categories
* Historical Context: Pattern matching with past conflicts for escalation assessment
* Real-time Processing: Sub-5-minute latency from raw data to actionable intelligence
* Production-Ready: PostgreSQL + Redis + FastAPI infrastructure for operational deployment.

# System Architecture
* Data Ingestion Layer
* LangGraph Multi-Agent Workflow
* Storage and API Layer

# Methodology
## 6-Stage LangGraph Pipeline
* Ingestion Agent: Validates and structures event from GDELT, NewsAPI, EventRegistry
* Classifaction Agent: Claude 3.5 Haiku categorizes threats (military buildups, terrorism, cyber ops, etc.)
* Pattern Matching Agent: Identifies historical parallels (e.g., Ukraine conflict -> Cold War escalation patterns)
* Escalation Predictor: 30-day probability estimates with confidence scoring
* Alert Generator: Triages Critical/High/Medium/Low alerts based on escalation risk
* Brief Generator: Synthesizes daily intelligence reports with executive summaries

## Key Technologies
* LLMs: Claude 3.5 Haiku for reasoning, structured JSON outputs
* Orchestration: LangGraph stateful multi-agents workflows
* Vector Search: Pgvector for semantic deduplications (1536-dim embeddings)
* Caching: Redis for URL dedup, rate limiting, result caching

# Performance & Results
<img width="717" height="362" alt="Screenshot 2025-12-18 at 12 19 47 AM" src="https://github.com/user-attachments/assets/f2aca9e1-f705-4cc2-af29-23f6f4ae56df" />

# Quick Start
## Prerequisites
* Python 3.10+
* PostgreSQL 15+ with pgvector
* Redis 7+
* Docker Compose (recommended)

## Installation
### Start infrastructure
docker-compose up -d

### Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### Configure API keys
cp .env.example .env
### Edit .env with your ANTHROPIC_API_KEY, NEWSAPI_KEY, etc.

### Run pipeline
python -m Pipeline.run_pipeline

## Configuration
ANTHROPIC_API_KEY=sk-ant-...        #Claude API (required)
NEWSAPI_KEY=...                     #NewsAPI.org (optional)
EVENTREGISTRY_KEY=...               #EventRegistry (optional)
POSTGRES_PASSWORD=...               #Database password

# Applications & Impact
## Defense & Intelligence
* Threat Detection: Automated identification of military buildups, terrorism, cyber operations
* Early Warnign: 30-day escalation predictions for strategic planning
* Operational Tempo: Real-time monitoring enabling rapid response

## Humanitarian Aids
* Crisis Detection: Early warning for displacement and humanitarian needs
* Resource Allocation: Predictive intelligence for pre-positioning supplies
* Security Assessment: Operatoinal risk evaluation for field staff
