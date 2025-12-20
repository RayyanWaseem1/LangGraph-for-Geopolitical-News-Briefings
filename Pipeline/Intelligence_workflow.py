"""
LangGraph Multi-Agent Orchestration for Geopolitical Intelligence

Implements the stateful agent workflow for:
1. Event ingestion and validation
2. Threat classification
3. Pattern matching and historical analysis
4. Escalation prediction
5. Intelligence brief generation
6. Alert triage
"""

import logging
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Annotated, TypedDict, cast, Optional
from datetime import datetime
from uuid import UUID 

# Ensure project root is on sys.path so Data/ and Storage/ imports resolve when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, SecretStr

from Data.models import (
    GeopoliticalEvent,
    EventCluster,
    EscalationPrediction,
    ThreatAlert,
    IntelligenceBrief,
    AgentState,
    AlertLevel,
    ThreatCategory
)

from Data.settings import Settings, THREAT_METADATA

logger = logging.getLogger(__name__)

settings = Settings()

### Agent State Definition ###

class WorkflowState(TypedDict):
    #State passed between agents in the workflow

    #Input
    raw_events: List[Dict[str, Any]]

    #Processing stages
    events: List[GeopoliticalEvent]
    clusters: List[EventCluster]
    predictions: List[EscalationPrediction]

    #Outputs
    alerts: List[ThreatAlert]
    brief: Optional[IntelligenceBrief]

    #Metadata
    current_step: str
    errors: List[str]
    messages: Annotated[List[BaseMessage], "agent_messages"]

### LLM Initialization ###
def get_primary_llm():
    #Getting the primary LLM (Claude Sonnet 3.5) for complex reasoning"
    api_key = settings.ANTHROPIC_API_KEY
    return ChatAnthropic(
        model_name = "claude-3-5-haiku-20241022",
        temperature = 0.1,
        api_key = SecretStr(api_key),
        timeout = 60,
        stop = None
    )

def get_fast_llm():
    #Get fast LLM ("claude-sonnet-4-5-20250929") for classification tasks
    api_key = settings.ANTHROPIC_API_KEY
    return ChatAnthropic(
        model_name = "claude-3-5-haiku-20241022",
        temperature = 0.1,
        api_key = SecretStr(api_key),
        timeout = 30,
        stop = None
    )

### Agent 1: Ingestion and Validation ###
async def ingestion_agent(state: WorkflowState) -> WorkflowState:
    """
    Validates and structures raw event data

    -Deduplciates events
    -Extracts the entities
    -initial quality filtering
    """

    logger.info("Starting the ingestion agent")

    try:
        raw_events = state.get("raw_events", [])
        validated_events = []

        for raw_event in raw_events: 
            #basic validation
            if not raw_event.get("title") or not raw_event.get("url"):
                continue

            #Convert to GeopoliticalEvent model
            event = GeopoliticalEvent(
                timestamp = datetime.fromisoformat(raw_event.get("timestamp", datetime.utcnow().isoformat())),
                source = raw_event.get("source", "unknown"),
                source_url = raw_event["url"],
                title = raw_event["title"],
                description = raw_event.get("description", ""),
                full_text = raw_event.get("full_text"),
                countries = raw_event.get("countries", []),
                locations = raw_event.get("locations", []),
            )

            validated_events.append(event)

        logger.info(f"Validated {len(validated_events)}/{len(raw_events)} events")

        state["events"] = validated_events
        state["current_step"] = "classification"

        return state 
    except Exception as e:
        logger.error(f"Error in ingestion agent: {e}")
        state["errors"].append(f"Ingestion error: {str(e)}")
        return state
    
### Agent 2: Threat Classification ###
class ThreatClassification(BaseModel):
    #Structured output for the threat classification
    threat_category: str = Field(description = "Prrimary threat category")
    confidence: float = Field(description = "Confidence score 0-1")
    reasoning: str = Field(description = "Brief explanation")
    severity: float = Field(description = "Severity score 0-1")
    actors: List[str] = Field(description = "Key actors involved")

async def classification_agent(state: WorkflowState) -> WorkflowState:
    """
    Classifies the events into threat categories using fast LLM
    Uses the GPT-4o-mini model for rapid batch classification
    """

    logger.info("Starting the classification agent")

    try:
        events = state.get("events", [])
        llm = get_fast_llm()
        parser = JsonOutputParser(pydantic_object=ThreatClassification)

        #Classification prompt
        classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a geopolitical analyst specializing in threat assessment.messages
             Classify the given event into one of these categories:
             {categories}
             Consider:
             -Event type and nature
             -Actors involved (state/non-state)
             -Geographic location
             -Potential for escalation
             -Historical context
             
             You MUST respond with ONLY a valid JSON object. No Markdown, no explanation, just pure JSON.
             Required format: {{"threat_category": "...", "confidence": 0.0-1.0, "reasoning": "...", "severity": 0.0-1.0, "actors": [...]}} """),
             ("human", "Event Title: {title}\n\nDescription: {description}\n\n Locations: {locations}")
             
        ])

        chain = classification_prompt | llm | parser

        classified_events = []

        #Process in batches for better efficiency
        for event in events[:100]: 
            try:
                result = await chain.ainvoke({
                    "categories": ", ".join([cat.value for cat in ThreatCategory]),
                    "title": event.title,
                    "description": event.description or "No description",
                    "locations": ", ".join(event.locations) if event.locations else "Unknown"
                })

                #Update the event with classification
                event.threat_category = ThreatCategory(result["threat_category"])
                event.threat_confidence = result["confidence"]
                event.actors = result["actors"]
                event.importance_score = result["severity"]

                classified_events.append(event)

                #Rate limiting to help avoid crashing the API
                time.sleep(1.5)

            except Exception as e:
                logger.error(f"Error classifying event {event.event_id}: {e}")
                continue 

        logger.info(f"Classified {len(classified_events)} events")

        state["events"] = classified_events
        state["current_step"] = "pattern_matching"

        return state 
    
    except Exception as e:
        logger.error(f"Error in classification agent: {e}")
        state["errors"].append(f"Classification error: {str(e)}")
        return state 
    
### Agent 3: Pattern Matching and Historial Analysis ###
async def pattern_matching_agent(state: WorkflowState) -> WorkflowState:
    """
    Matches current events to historical patterns
    Uses Claude for complex reasoning about historical parallels
    """
    logger.info("Starting the pattern matching agent")

    try:
        events = state.get("events", [])
        llm = get_primary_llm()

        #historical pattern matching agent
        pattern_prompt = ChatPromptTemplate.from_messages([
            ("system", """ You are a geopolitical historian and strategic analyst.
             Analyze the given event and identify historical parallels or patterns.
             Consider:
             1. Similar past conflicts or incidents
             2. Escalation patterns (what typically happens next)
             3. Diplomatic precedents
             4. Geographic/regional history
             5. Actor behavior patterns
             Provide:
                -Most relevant historical parallel
                -Key similarities and differences
                -What happened next in the historical case
                -Lessons learned
                -Relevance to the current situation"""),
                ("human", """Current Event:
                 Title: {title}
                 Category: {category}
                 Description: {description}
                 Locations: {locations}
                 Actors: {actors}
                 Analyze historical parallels and escalation patterns.""")
        ])

        chain = pattern_prompt | llm 

        #For demonstration, analyze the top 10 most important events
        top_events = sorted(events, key = lambda e: e.importance_score, reverse = True)[:20]

        for event in top_events:
            try:
                result = await chain.ainvoke({
                    "title": event.title,
                    "category": event.threat_category.value if event.threat_category else "unknown",
                    "description": event.description,
                    "locations": ", ".join(event.locations),
                    "actors": ", ".join(event.actors)
                })

                #Store the historical context 
                logger.info(f"Pattern analysis for {event.event_id}: {result.content[:200]}...")

            except Exception as e:
                logger.error(f"Error in pattern matching for event {event.event_id}: {e}")
                continue 

        state["current_step"] = "escalation_prediction"

        return state 
    
    except Exception as e:
        logger.error(f"Error in pattern matching agent: {e}")
        state["errors"].append(f"Pattern matching error: {str(e)}")
        return state
    
### Agent 4: Escalation Prediction ###
class EscalationAnalysis(BaseModel):
    """ Structured output for the escalation prediction"""
    escalation_probability: float = Field(description = "Probability of escalation 0-1")
    confidence: float = Field(description = "Confidence in prediction 0-1")
    alert_level: str = Field(description = "Alert level: critical/high/medium/low")
    reasoning: str = Field(description= "Detailed reasoning")
    risk_factors: List[str] = Field(description = "Factors increasing risk")
    mitigating_factors: List[str] = Field(description = "Factors reducing risk")
    recommended_actions: List[str] = Field(description = "Recommended monitoring actions")


async def escalation_prediction_agent(state: WorkflowState) -> WorkflowState:
    """
    Predicts escalation probability for event clusters
    Uses Claude for nuanced risk assessment
    """
    logger.info("Starting the escalation prediction agent")

    try:
        events = state.get("events", [])
        llm = get_primary_llm()
        parser = JsonOutputParser(pydantic_object=EscalationAnalysis)

        #Grouping events by region/category for the cluster analysis
        #For demonstration, analyze each high-importance event
        high_priority_events = [e for e in events if e.importance_score > 0.6]

        escalation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a strategic intelligence analyst specializing in conflict escalation prediction.
             
             You MUST respond with ONLY a valid JSON object. No narrative, no markdown, no emojis, just pure JSON.
             
             Required JSON format (copy this structure exactly):
             {{{{
               "escalation_probability": 0.65,
               "confidence": 0.85,
               "alert_level": "high",
               "reasoning": "Brief 1-2 sentence explanation",
               "risk_factors": ["factor1", "factor2", "factor3"],
               "mitigating_factors": ["factor1", "factor2"],
               "recommended_actions": ["action1", "action2", "action3"]
             }}}}
             
             Alert levels must be exactly one of: "critical", "high", "medium", "low"
             
             Assess the escalation probability considering:
             1. Historical precedents and patterns
             2. Actor capabilities and intentions
             3. Geographic and strategic context
             4. Recent trends (increasing/decreasing tensions)
             5. Diplomatic channels and constraints
             
             Respond with ONLY the JSON object. No other text."""),
             ("human", """Situation Analysis:
              Title: {title}
              Category: {category}
              Location: {location}
              Actors: {actors}
              Sentiment: {sentiment}
              Importance: {importance}
              Assess escalation probability.""")
        ])

        chain = escalation_prompt | llm | parser 

        predictions = []

        for event in high_priority_events[:30]:
            try:
                result = await chain.ainvoke({
                    "title": event.title,
                    "category": event.threat_category.value if event.threat_category else "unknown",
                    "location": ", ".join(event.locations),
                    "actors": ", ".join(event.actors),
                    "sentiment": event.sentiment_score,
                    "importance": event.importance_score
                })

                #Creating the prediction object
                prediction = EscalationPrediction(
                    cluster_id = 0,
                    escalation_probability = result["escalation_probability"],
                    confidence = result["confidence"],
                    alert_level = AlertLevel(result["alert_level"]),
                    reasoning = result["reasoning"],
                    risk_factors = result["risk_factors"],
                    mitigating_factors=result["mitigating_factors"]
                )

                predictions.append(prediction)

            except Exception as e:
                logger.error(f"Error predicting escalation for event {event.event_id}: {e}")
                continue 

        logger.info(f"Generated {len(predictions)} escalation predictions")

        state["predictions"] = predictions
        state["current_step"] = "alert_generation"

        return state
    
    except Exception as e:
        logger.error(f"Error in escalation prediction agent: {e}")
        state["errors"].append(f"Escalation prediction error: {str(e)}")
        return state 
    
# ==================== Agent 5: Alert Generation ====================

async def alert_generation_agent(state: WorkflowState) -> WorkflowState:
    """
    Generates alerts for high-priority threats
    
    Triages based on escalation probability and creates actionable alerts
    """
    logger.info("Starting alert generation agent")
    
    try:
        events = state.get("events", [])
        predictions = state.get("predictions", [])
        
        alerts = []
        
        # Generate alerts for critical and high-level predictions
        for prediction in predictions:
            if prediction.alert_level in [AlertLevel.CRITICAL, AlertLevel.HIGH]:
                # Find corresponding event(s)
                relevant_events = [e for e in events if e.importance_score > 0.6][:3]
                
                alert = ThreatAlert(
                    cluster_id=prediction.cluster_id,
                    alert_level=prediction.alert_level,
                    title=f"{prediction.alert_level.value.upper()}: {relevant_events[0].title if relevant_events else 'Geopolitical Event'}",
                    summary=prediction.reasoning[:500],
                    detailed_analysis=prediction.reasoning,
                    threat_category=(relevant_events[0].threat_category or ThreatCategory.MILITARY_BUILDUP) if relevant_events else ThreatCategory.MILITARY_BUILDUP,
                    region=relevant_events[0].locations[0] if relevant_events and relevant_events[0].locations else "Unknown",
                    escalation_probability=prediction.escalation_probability,
                    supporting_events=[e.event_id for e in relevant_events],
                    source_urls=[e.source_url for e in relevant_events]
                )
                
                alerts.append(alert)
        
        logger.info(f"Generated {len(alerts)} alerts")
        
        state["alerts"] = alerts
        state["current_step"] = "brief_generation"
        
        return state
        
    except Exception as e:
        logger.error(f"Error in alert generation agent: {e}")
        state["errors"].append(f"Alert generation error: {str(e)}")
        return state


# ==================== Agent 6: Intelligence Brief Generation ====================

async def brief_generation_agent(state: WorkflowState) -> WorkflowState:
    """
    Generates daily intelligence brief
    
    Uses Claude to synthesize all intelligence into coherent brief
    """
    logger.info("Starting brief generation agent")
    
    try:
        events = state.get("events", [])
        alerts = state.get("alerts", [])
        predictions = state.get("predictions", [])
        
        llm = get_primary_llm()
        
        # Brief generation prompt
        brief_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior intelligence officer preparing the daily geopolitical intelligence brief.

Create a professional, concise intelligence brief with:

1. EXECUTIVE SUMMARY (2-3 sentences)
   - Most critical developments
   - Key themes

2. CRITICAL ALERTS
   - For each critical/high alert:
     * Title and location
     * Situation summary
     * Escalation probability
     * Context and significance
     * Sources

3. WATCH LIST
   - Medium-priority situations
   - Emerging patterns

4. DECLINING TENSIONS
   - Positive developments
   - De-escalations

Use clear, professional military/intelligence language. Be precise and factual."""),
            ("human", """Generate daily intelligence brief.

Total Events Processed: {event_count}
Critical Alerts: {critical_count}
High Alerts: {high_count}

Alert Details:
{alert_details}

Create the brief.""")
        ])
        
        chain = brief_prompt | llm
        
        # Prepare alert details
        critical_alerts = [a for a in alerts if a.alert_level == AlertLevel.CRITICAL]
        high_alerts = [a for a in alerts if a.alert_level == AlertLevel.HIGH]
        
        alert_details = "\n\n".join([
            f"[{a.alert_level.value.upper()}] {a.title}\nRegion: {a.region}\nProbability: {a.escalation_probability:.0%}\n{a.summary}"
            for a in alerts[:5]
        ])
        
        result = await chain.ainvoke({
            "event_count": len(events),
            "critical_count": len(critical_alerts),
            "high_count": len(high_alerts),
            "alert_details": alert_details or "No major alerts"
        })
        
        # Convert result content to string
        content_str = result.content if isinstance(result.content, str) else "".join(str(item) for item in result.content) if isinstance(result.content, list) else str(result.content)
        
        # Create intelligence brief
        brief = IntelligenceBrief(
            date=datetime.utcnow(),
            executive_summary=content_str[:1000],
            critical_alerts=critical_alerts,
            watch_list=[{"title": "Watch item"} for _ in high_alerts],
            total_events_processed=len(events),
            new_alerts=len(alerts),
            ongoing_situations=len(predictions)
        )
        
        logger.info("Generated intelligence brief")
        
        state["brief"] = brief
        state["current_step"] = "complete"
        
        return state
        
    except Exception as e:
        logger.error(f"Error in brief generation agent: {e}")
        state["errors"].append(f"Brief generation error: {str(e)}")
        return state
    
def create_intelligence_workflow() -> CompiledStateGraph[WorkflowState]:
    """
    Creates the LangGraph workflow for intelligence processing
    Returns:
        Compiled StateGraph
    """

    workflow: StateGraph = StateGraph(WorkflowState)

    #Adding nodes (agents)
    workflow.add_node("ingestion", ingestion_agent)
    workflow.add_node("classification", classification_agent)
    workflow.add_node("pattern_matching", pattern_matching_agent)
    workflow.add_node("escalation_prediction", escalation_prediction_agent)
    workflow.add_node("alert_generation", alert_generation_agent)
    workflow.add_node("brief_generation", brief_generation_agent)

    #Defining the edges (workflow)
    workflow.set_entry_point("ingestion")
    workflow.add_edge("ingestion", "classification")
    workflow.add_edge("classification", "pattern_matching")
    workflow.add_edge("pattern_matching", "escalation_prediction")
    workflow.add_edge("escalation_prediction", "alert_generation")
    workflow.add_edge("alert_generation", "brief_generation")
    workflow.add_edge("brief_generation", END)

    compiled_workflow: CompiledStateGraph[WorkflowState] = workflow.compile()
    return compiled_workflow 

### Execution ###
async def run_intelligence_pipeline(raw_events: List[Dict[str, Any]]) -> IntelligenceBrief:
    """ 
    Executes the full intelligence pipeline
    Args:
        raw_events: List of raw event dictionaries
    Returns:
        IntelligenceBrief with complete analysis
    """

    logger.info(f"Starting the intelligence pipeline with {len(raw_events)} events")

    #Initializing state
    initial_state = cast(WorkflowState, {
        "raw_events": raw_events,
        "events": [],
        "clusters": [],
        "predictions": [],
        "alerts": [],
        "brief": None,
        "current_step": "ingestion",
        "errors": [],
        "messages": []
    })

    #Create and run the workflow
    workflow = create_intelligence_workflow()

    final_state = await workflow.ainvoke(initial_state)

    logger.info(f"Pipeline complete. Generated brief with {len(final_state['alerts'])} alerts")

    return final_state["brief"]

#Example
if __name__ == "__main__":
    import asyncio 

    logging.basicConfig(level = logging.INFO)

    #Sample events
    sample_events = [
        {
            "timestamp": "2024-12-10T10:00:00Z",
            "source": "gdelt",
            "url": "https://example.com/article1",
            "title": "Chinese warships conduct exercises near Taiwan",
            "description": "PLA Navy vessels spotted in Taiwan Strait",
            "locations": ["Taiwan Strait", "Taiwan"],
            "countries": ["China", "Taiwan"]
        }
    ]
    
    brief = asyncio.run(run_intelligence_pipeline(sample_events))
    print(brief.json(indent=2))
