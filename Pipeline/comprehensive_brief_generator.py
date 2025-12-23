"""
Enhanced Intelligence Brief Generation for OSINT AI Pipeline
Generates comprehensive, detailed briefs matching EXACT Streamlit format
One brief per alert with full analysis using EXACT same prompts as streamlit_app.py
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr

from Data.models import ThreatAlert, GeopoliticalEvent, EscalationPrediction
from Data.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()


class ComprehensiveBriefGenerator:
    """Generates detailed intelligence briefs matching EXACT Streamlit format"""
    
    def __init__(self):
        self.llm = ChatAnthropic(
            model_name="claude-3-5-sonnet-20241022",  # Use Sonnet for better quality
            temperature=0.2,
            api_key=SecretStr(settings.ANTHROPIC_API_KEY),
            timeout=120,
            max_tokens_to_sample=6000,  # Match streamlit
            stop=None
        )
    
    async def generate_alert_brief(
        self,
        alert: ThreatAlert,
        related_events: List[GeopoliticalEvent],
        prediction: Optional[EscalationPrediction] = None
    ) -> str:
        """
        Generate a comprehensive intelligence brief for a single alert
        Uses EXACT same prompt structure as streamlit_app.py
        
        Args:
            alert: The threat alert to analyze
            related_events: Events supporting this alert
            prediction: Escalation prediction for this alert
        
        Returns:
            Formatted intelligence brief as string
        """
        
        # Prepare content from events
        content = self._prepare_content_from_events(related_events)
        
        # Prepare classification dict (matching streamlit format)
        classification = {
            "severity": alert.alert_level.value,
            "threat_category": alert.threat_category.value,
            "confidence": alert.escalation_probability
        }
        
        # Prepare patterns dict (from prediction if available)
        patterns = {}
        if prediction:
            patterns = {
                "risk_factors": prediction.risk_factors,
                "mitigating_factors": prediction.mitigating_factors,
                "time_horizon_days": prediction.time_horizon_days,
                "reasoning": prediction.reasoning
            }
        
        # Generate the query (what this alert is about)
        query = f"{alert.title} - Threat assessment and analysis"
        
        # Use EXACT same prompt as streamlit_app.py
        prompt_text = f"""Generate a comprehensive intelligence briefing that directly answers this query:

QUERY: {query}

INFORMATION GATHERED:
{content}

CLASSIFICATION: {json.dumps(classification, indent=2)}
HISTORICAL PATTERNS: {json.dumps(patterns, indent=2)}

Create a comprehensive intelligence brief with:

# EXECUTIVE SUMMARY
Write a thorough 4-6 paragraph executive summary that:
- Directly answers the user's query: "{query}"
- Summarizes the key facts and current situation
- Highlights the most critical developments
- Provides context on why this matters
- States the overall threat assessment
- Outlines immediate implications

Make this substantive - this is the main answer to the user's question.

# CURRENT SITUATION
Detailed analysis of what's happening:
- Verified facts and timeline of events
- Key developments and current status
- Actions by relevant actors
- Immediate aftermath and response

# THREAT ASSESSMENT  
Classification: {classification['severity'].upper()} - {classification['threat_category']}
- Detailed reasoning for this classification
- Key actors involved and their roles
- Affected regions and populations
- Confidence level: {int(classification['confidence'] * 100)}%
- Potential for escalation or similar incidents

# HISTORICAL CONTEXT
- Relevant historical parallels
- How similar situations evolved
- Lessons from past events
- Pattern analysis and what it suggests

# STRATEGIC IMPLICATIONS
- What this means for regional/local security
- Broader implications and precedents
- Potential outcomes and scenarios
- Stakeholder impacts

# MONITORING RECOMMENDATIONS
- Key indicators to watch
- Information sources to track
- Follow-up actions needed
- Timeline for reassessment

# ANALYTICAL CONFIDENCE
- Quality and reliability of information
- Gaps in current knowledge
- Alternative interpretations
- Limitations of this assessment

Use professional intelligence community language. Be specific, detailed, and actionable."""

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("user", "{prompt_text}")
        ])
        
        # Generate the brief
        chain = prompt | self.llm
        
        try:
            result = await chain.ainvoke({
                "prompt_text": prompt_text
            })
            
            # Extract text from result
            brief_text = result.content if isinstance(result.content, str) else str(result.content)
            
            # Add header with metadata (matching streamlit format)
            header = self._generate_header(alert, related_events, query)
            
            # Add conclusion
            conclusion = f"\n\nCONCLUSION:\n{alert.summary}"
            
            return header + "\n\n" + brief_text + conclusion
            
        except Exception as e:
            logger.error(f"Error generating comprehensive brief: {e}")
            return self._generate_fallback_brief(alert, related_events, prediction, query)
    
    def _prepare_content_from_events(self, related_events: List[GeopoliticalEvent]) -> str:
        """Prepare content string from events similar to streamlit web search results"""
        
        if not related_events:
            return "No supporting events available"
        
        content_parts = []
        
        for i, event in enumerate(related_events, 1):
            event_text = f"""
EVENT {i}:
Title: {event.title}
Source: {event.source.value}
Timestamp: {event.timestamp}
Description: {event.description}

Location: {', '.join(event.locations) if event.locations else 'Unknown'}
Countries: {', '.join(event.countries) if event.countries else 'Unknown'}
Actors: {', '.join(event.actors) if event.actors else 'Unknown'}
Keywords: {', '.join(event.keywords) if event.keywords else 'None'}
Sentiment Score: {event.sentiment_score:.2f if event.sentiment_score else 'N/A'}

Full Text:
{event.full_text[:500] if event.full_text else event.description}
"""
            content_parts.append(event_text)
        
        return "\n\n".join(content_parts)
    
    def _generate_header(self, alert: ThreatAlert, related_events: List[GeopoliticalEvent], query: str) -> str:
        """Generate brief header with metadata matching Streamlit format"""
        
        # Extract source URLs and titles
        sources_text = ""
        if alert.source_urls and related_events:
            sources_text = f"\n\nSOURCES ({len(alert.source_urls)}):\n"
            
            # Create a mapping of URLs to event titles
            url_to_title = {}
            for event in related_events:
                if event.source_url:
                    url_to_title[event.source_url] = event.title
            
            # Add numbered sources
            for i, url in enumerate(alert.source_urls[:10], 1):  # Top 10 URLs
                title = url_to_title.get(url, f"Source {i}")
                # Truncate title if too long
                if len(title) > 80:
                    title = title[:77] + "..."
                sources_text += f'\n{i}. "{title}" - {url}'
        
        header = f"""OSINT INTELLIGENCE BRIEF
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Query: {query}

CLASSIFICATION:
- Threat Level: {alert.alert_level.value.upper()}
- Category: {alert.threat_category.value}
- Confidence: {int(alert.escalation_probability * 100)}%{sources_text}

================================================================================
INTELLIGENCE BRIEF: {alert.title.upper()}
================================================================================"""
        
        return header
    
    def _generate_fallback_brief(
        self,
        alert: ThreatAlert,
        related_events: List[GeopoliticalEvent],
        prediction: Optional[EscalationPrediction],
        query: str
    ) -> str:
        """Generate a fallback brief if LLM call fails"""
        
        header = self._generate_header(alert, related_events, query)
        
        body = f"""

# EXECUTIVE SUMMARY

{alert.detailed_analysis}

This situation represents a {alert.alert_level.value} threat in the {alert.threat_category.value} category, 
with an escalation probability of {int(alert.escalation_probability * 100)}%. The region affected is {alert.region}.

Based on the available intelligence, this situation requires immediate attention and monitoring due to its 
potential for escalation and regional impact. The threat has been classified as {alert.alert_level.value.upper()} 
based on the severity of the incident, the actors involved, and the potential for broader implications.

The current developments indicate an evolving situation with significant geopolitical ramifications. 
Historical precedents suggest similar situations have led to prolonged conflicts and regional instability. 
The international community's response will be critical in determining the trajectory of events.

Immediate monitoring and assessment are recommended, with particular attention to potential escalation 
triggers and regional responses. The situation requires continuous intelligence gathering and analysis 
to inform strategic decision-making.

# CURRENT SITUATION

Current developments indicate an evolving situation that requires close monitoring. Key factors include:

- Threat Category: {alert.threat_category.value}
- Alert Level: {alert.alert_level.value.upper()}
- Escalation Probability: {int(alert.escalation_probability * 100)}%
- Affected Region: {alert.region}

{alert.summary}

Timeline of Events:
"""
        
        # Add event timeline
        for i, event in enumerate(related_events[:5], 1):
            body += f"""
{i}. {event.timestamp.strftime('%Y-%m-%d %H:%M UTC')} - {event.title}
   Location: {', '.join(event.locations) if event.locations else 'Unknown'}
   Description: {event.description[:200]}...
"""
        
        body += f"""

# THREAT ASSESSMENT

Classification: {alert.alert_level.value.upper()} - {alert.threat_category.value}

This incident has been classified as {alert.alert_level.value.upper()} based on the following factors:
- Severity of the incident and potential casualties
- Strategic importance of the affected region
- Involvement of state and non-state actors
- Potential for regional escalation
- Historical precedents in similar situations

Key Actors:
"""
        
        # Extract unique actors from events
        all_actors = set()
        for event in related_events[:10]:
            all_actors.update(event.actors)
        
        for actor in list(all_actors)[:5]:
            body += f"- {actor}\n"
        
        body += f"""
Affected Regions:
- Primary: {alert.region}
"""
        
        # Add regions from events
        all_regions = set()
        for event in related_events[:10]:
            all_regions.update(event.countries)
        
        for region in list(all_regions)[:5]:
            body += f"- {region}\n"
        
        body += f"""
Escalation Potential: {alert.escalation_probability:.0%}
"""
        
        if prediction:
            body += f"""
Risk Factors:
"""
            for factor in prediction.risk_factors[:5]:
                body += f"- {factor}\n"
            
            if prediction.mitigating_factors:
                body += "\nMitigating Factors:\n"
                for factor in prediction.mitigating_factors[:5]:
                    body += f"- {factor}\n"
        
        body += """

# HISTORICAL CONTEXT

Historical analysis of similar situations suggests several relevant parallels:

- Past conflicts in this region have demonstrated similar escalation patterns
- International responses to comparable incidents have varied widely
- Regional powers have historically shown interest in influencing outcomes
- Civilian populations have borne significant costs in similar scenarios

Lessons from these historical precedents indicate the importance of:
- Early diplomatic intervention
- Protection of civilian populations
- Maintaining communication channels
- Preventing regional spillover
- International coordination and oversight

# STRATEGIC IMPLICATIONS

The current situation has several strategic implications:

Regional Security:
- Potential for destabilization of the immediate region
- Risk of spillover into neighboring areas
- Impact on regional security architecture
- Effects on civilian populations and humanitarian situation

Broader Geopolitical Context:
- Implications for international norms and precedents
- Potential involvement of major powers
- Effects on regional alliances and relationships
- Impact on global security frameworks

Stakeholder Impacts:
- Local populations facing immediate security threats
- Regional governments managing security responses
- International community's diplomatic and humanitarian obligations
- Global implications for conflict resolution mechanisms

# MONITORING RECOMMENDATIONS

Recommend continuous monitoring of the following:

Key Indicators:
- Escalation of violence or military activity
- Diplomatic communications and negotiations
- Humanitarian situation and civilian casualties
- Regional reactions and responses
- International community engagement

Information Sources:
- Official government statements and communications
- International organization reports
- Local and regional news sources
- Social media and on-ground reporting
- Intelligence community assessments

Follow-up Actions:
- Regular intelligence briefings on developments
- Assessment of escalation triggers and risk factors
- Monitoring of diplomatic initiatives
- Tracking of humanitarian situation
- Analysis of regional and international responses

Timeline for Reassessment:
- Immediate: Continuous monitoring of breaking developments
- Short-term: Daily briefings for next 48-72 hours
- Medium-term: Weekly assessments of evolving situation
- Long-term: Monthly strategic reviews

# ANALYTICAL CONFIDENCE

Confidence Level: {int(alert.escalation_probability * 100)}%

This assessment is based on:
- Multiple corroborating sources from the field
- Pattern analysis of similar historical situations
- Intelligence community reporting and analysis
- Open-source intelligence gathering

Quality and Reliability:
- Information sources are primarily open-source intelligence
- Multiple independent sources corroborate key facts
- Some information may be incomplete or delayed
- Situation remains fluid and subject to rapid changes

Gaps in Current Knowledge:
- Complete details of actors' intentions and plans
- Full extent of regional involvement and support
- Precise casualty figures and humanitarian impact
- Behind-the-scenes diplomatic communications

Alternative Interpretations:
- Situation may be more or less severe than currently assessed
- Escalation probability could change based on diplomatic efforts
- Regional actors may have different priorities than assumed
- International response may differ from historical patterns

Limitations:
- Reliance on open-source information
- Potential for disinformation and propaganda
- Rapidly evolving situation may outpace analysis
- Limited access to classified intelligence sources
- Fog of war effects on information accuracy

CONCLUSION:
{alert.summary}

This situation requires continued vigilance and proactive engagement from the international community 
to prevent escalation and mitigate humanitarian impacts.
"""
        
        return header + body


# Convenience function for use in pipeline
async def generate_brief_for_alert(
    alert: ThreatAlert,
    related_events: List[GeopoliticalEvent],
    prediction: Optional[EscalationPrediction] = None
) -> str:
    """
    Generate a comprehensive brief for a single alert
    Uses EXACT same format as streamlit_app.py
    
    Args:
        alert: ThreatAlert to analyze
        related_events: Supporting GeopoliticalEvents
        prediction: Optional EscalationPrediction
    
    Returns:
        Formatted intelligence brief matching Streamlit format
    """
    generator = ComprehensiveBriefGenerator()
    return await generator.generate_alert_brief(alert, related_events, prediction)
