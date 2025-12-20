"""
OSINT AI Streamlit Dashboard - FIXED VERSION
Critical fix: Uses web search for specific queries (actually searches for your query)
GDELT used only for broad "what's happening now" monitoring
"""

import streamlit as st
import anthropic 
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import time
import json

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Try to import GDELT infrastructure (optional)
try:
    from Data.Gdelt_client import GDELTIngestions
    from Data.newsapi_client import NewsAPIClient
    from Data.eventregistry_client import EventRegistryClient
    GDELT_AVAILABLE = True
except ImportError:
    GDELTIngestions = None
    NewsAPIClient = None
    EventRegistryClient = None
    GDELT_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="OSINT - Geopolitical Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    padding: 1rem 0;
}
.alert-critical {
    background-color: #ffebee;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #f44336;
}
.alert-high {
    background-color: #fff3e0;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ff9800;
}
.alert-medium {
    background-color: #fff9c4;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #fdd835;
}
.alert-low {
    background-color: #e8f5e9;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #4caf50;
}
.source-card {
    background-color: #e3f2fd;
    padding: 0.75rem;
    border-radius: 0.25rem;
    margin: 0.5rem 0;
    border-left: 3px solid #2196f3;
}
.source-card strong {
    color: #1565c0;
}
.source-card small {
    color: #424242;
}
/* Make markdown links in source cards visible */
a {
    color: #1976d2 !important;
    text-decoration: underline !important;
    font-weight: 500;
}
a:hover {
    color: #0d47a1 !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'briefing_history' not in st.session_state:
    st.session_state.briefing_history = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""


class OSINTAnalyzer:
    """Main analyzer - uses web search for specific queries"""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-5-20250929"
    
    def _get_first_text_block(self, content: List[Any]) -> str:
        """Extract text from response"""
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    return block["text"]
                continue
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", "")
                if text:
                    return text
        return ""
    
    def search_and_extract_sources(self, query: str) -> tuple[List[Dict], str]:
        """
        Search for query and extract both sources and content
        Returns: (list of source dicts, combined content text)
        """
        try:
            st.info(f"🔍 Searching for: **{query}**")
            
            # Use web search to find query-specific information
            message = self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search"
                }],
                messages=[{
                    "role": "user",
                    "content": f"""Search for the most recent and relevant information about: {query}

Focus on finding:
1. Latest breaking news and developments (prioritize most recent - last 7 days)
2. Official statements and verified reports
3. Current status and updates
4. Multiple credible news sources

After searching, provide:
1. A list of your top 5-10 most relevant sources with titles and URLs (be specific with article titles)
2. A comprehensive summary of what you found

If you cannot find verified information about this specific query, clearly state that.

Format your response as:

SOURCES FOUND:
1. [Specific Article Title] - [Full URL]
2. [Specific Article Title] - [Full URL]
...

DETAILED SUMMARY:
[Comprehensive summary of all findings from the sources]

If no relevant information was found, explain what you searched for and what you found instead."""
                }]
            )
            
            response_text = self._get_first_text_block(message.content)
            
            # Parse sources and content
            sources = []
            content = ""
            
            if "SOURCES FOUND:" in response_text and "DETAILED SUMMARY:" in response_text:
                parts = response_text.split("DETAILED SUMMARY:")
                source_section = parts[0].replace("SOURCES FOUND:", "").strip()
                content = parts[1].strip()
                
                # Parse individual sources
                for line in source_section.split('\n'):
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith('-')):
                        # Extract URL if present
                        url = None
                        title = line
                        
                        # Try to extract URL from markdown links or plain text
                        if 'http' in line:
                            # Find URL
                            url_start = line.find('http')
                            url_end = line.find(' ', url_start) if ' ' in line[url_start:] else len(line)
                            if url_end == -1:
                                url_end = len(line)
                            url = line[url_start:url_end].rstrip(')').rstrip(']').rstrip(',')
                            
                            # Clean title
                            title = line[:url_start].strip()
                            if title.endswith('-'):
                                title = title[:-1].strip()
                        
                        # Remove numbering
                        title = title.lstrip('0123456789.-) ').strip()
                        
                        if title and len(title) > 5:  # Avoid parsing errors
                            sources.append({
                                "title": title,
                                "url": url,
                                "source": "Web Search",
                                "timestamp": datetime.now().isoformat(),
                                "content": ""
                            })
            else:
                # Fallback: treat entire response as content
                content = response_text
            
            # If no sources extracted, create a generic one
            if not sources:
                sources = [{
                    "title": f"Search Results for: {query}",
                    "url": None,
                    "source": "Web Search",
                    "timestamp": datetime.now().isoformat(),
                    "content": response_text
                }]
            
            st.success(f"✅ Found {len(sources)} source(s)")
            return sources, content
            
        except Exception as e:
            st.error(f"Search error: {e}")
            return [], ""
    
    def classify_threat(self, content: str, query: str) -> Dict[str, Any]:
        """Classify threat level based on content"""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=3000,
                messages=[{
                    "role": "user",
                    "content": f"""Analyze this information about "{query}" and classify the threat level.

INFORMATION: 
{content[:4000]}

You must provide ONLY a valid JSON response with no extra text before or after. The JSON must have:
{{
    "threat_category": "one of: military_buildup, terrorism, cyber_operations, wmd_activity, diplomatic_crisis, humanitarian_crisis, economic_warfare, civil_unrest, border_conflict, political_instability, disinformation, mass_violence, other",
    "severity": "critical/high/medium/low",
    "confidence": 0.0 - 1.0,
    "key_actors": ["list of countries/organizations/individuals involved"],
    "regions_affected": ["list of regions/locations"],
    "reasoning": "detailed explanation - use only single spaces, no newlines or tabs in this field"
}}

IMPORTANT: In the "reasoning" field, write everything on one line with single spaces only. Do not use newlines, tabs, or other control characters.

If this is about a specific incident, classify based on scale, casualties, and ongoing threat level."""
                }]
            )
            
            response_text = self._get_first_text_block(message.content)
            
            # Find JSON in response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                
                # Remove control characters that break JSON parsing
                import re
                # Replace control characters with spaces
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                # Replace multiple spaces with single space
                json_str = re.sub(r'\s+', ' ', json_str)
                
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as je:
                    st.warning(f"JSON parsing error: {je}")
                    # Try to extract key information manually
                    return {
                        "threat_category": "other",
                        "severity": "medium",
                        "confidence": 0.5,
                        "key_actors": [],
                        "regions_affected": [],
                        "reasoning": response_text[:500]
                    }
            else:
                return {
                    "threat_category": "other",
                    "severity": "medium",
                    "confidence": 0.5,
                    "key_actors": [],
                    "regions_affected": [],
                    "reasoning": response_text[:500]
                }
        except Exception as e:
            st.error(f"Classification error: {e}")
            return {
                "threat_category": "other",
                "severity": "medium",
                "confidence": 0.0,
                "key_actors": [],
                "regions_affected": [],
                "reasoning": f"Error during classification: {str(e)}"
            }
    
    def find_historical_patterns(self, content: str, query: str, classification: Dict) -> Dict[str, Any]:
        """Find historical patterns"""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2500,
                messages=[{
                    "role": "user",
                    "content": f"""Find historical parallels to this situation:

CURRENT SITUATION: {query}
DETAILS: {content[:2000]}
CLASSIFICATION: {classification['threat_category']} - {classification['severity']}

Identify 2-3 historical events that are similar. Provide ONLY valid JSON with no extra text:
{{
    "parallels": [
        {{
            "event": "Historical event name",
            "year": "YYYY",
            "similarity_score": 0.0 - 1.0,
            "outcome": "How it was resolved or what happened - single line, no newlines",
            "lessons": "Key lessons learned - single line, no newlines"
        }}
    ],
    "pattern_type": "escalation/de-escalation/cyclical/unprecedented",
    "analysis": "Brief analysis - single line, no newlines"
}}

IMPORTANT: All text fields must be single lines with no newlines or control characters."""
                }]
            )
            
            response_text = self._get_first_text_block(message.content)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                
                # Remove control characters
                import re
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    return {"parallels": [], "pattern_type": "unprecedented", "analysis": response_text[:300]}
            else:
                return {"parallels": [], "pattern_type": "unprecedented", "analysis": response_text[:300]}
        except Exception as e:
            return {"parallels": [], "pattern_type": "unprecedented", "analysis": f"Error: {str(e)}"}
    
    def generate_intelligence_brief(self, query: str, content: str, 
                                   classification: Dict, patterns: Dict) -> str:
        """Generate comprehensive intelligence brief"""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=6000,
                messages=[{
                    "role": "user",
                    "content": f"""Generate a comprehensive intelligence briefing that directly answers this query:

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
                }]
            )
            
            return self._get_first_text_block(message.content)
        except Exception as e:
            st.error(f"Brief generation error: {e}")
            return "Error generating brief"


def display_source_articles(sources: List[Dict]):
    """Display source articles with links"""
    st.markdown("### 📰 Source Articles")
    st.markdown("*Information gathered from the following sources:*")
    
    for i, source in enumerate(sources, 1):
        title = source.get('title', 'Untitled')
        url = source.get('url')
        timestamp = source.get('timestamp', '')
        
        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_str = dt.strftime('%Y-%m-%d %H:%M UTC')
        except:
            time_str = timestamp
        
        # Display source card with blue background
        st.markdown(f"""
        <div class="source-card">
            <strong>{i}. {title}</strong><br>
            <small>🕒 {time_str}</small>
        </div>
        """, unsafe_allow_html=True)
        
        if url:
            st.markdown(f"**🔗 Source:** [{url}]({url})")
        else:
            st.caption("🔗 No direct URL available (web search summary)")
        
        st.markdown("")


def main():
    # Header
    st.markdown('<div class="main-header">🌍 OSINT Geopolitical Intelligence Analyzer 🌍</div>', unsafe_allow_html=True)
    st.markdown("**AI-powered intelligence analysis using real-time web search**")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Get your key at console.anthropic.com"
        )
        
        st.markdown("---")
        
        st.header("Example Queries")
        example_queries = [
            "What's the current situation with China and Taiwan?",
            "Analyze recent developments in the Ukraine conflict",
            "What are the latest tensions in the Middle East?",
            "What's happening with North Korea's recent missile tests?",
            "What's the status of Iran's nuclear program?",
            "Analyze Russia's military activities in Eastern Europe",
            "What are the current US-China trade tensions?",
            "What's happening in the Red Sea shipping crisis?"
        ]
        
        for query_text in example_queries:
            if st.button(query_text, key=f"example_{query_text}", use_container_width=True):
                st.session_state.user_query = query_text
                st.rerun()
    
    # Main interface
    if not api_key:
        st.warning("⚠️ Please enter your Anthropic API key in the sidebar to proceed")
        st.info("Don't have an API key? Get one at https://console.anthropic.com/")
        
        st.header("How It Works")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔍 Smart Search")
            st.write("• Uses Claude's web search to find **query-specific** information")
            st.write("• Prioritizes most recent and relevant sources")
            st.write("• Extracts and cites original articles")
        
        with col2:
            st.markdown("### 🎯 AI Analysis")
            st.write("• Professional threat classification")
            st.write("• Historical pattern matching")
            st.write("• Comprehensive intelligence briefs")
        
        return
    
    # Initialize analyzer
    analyzer = OSINTAnalyzer(api_key)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Intelligence Query", "📊 Analysis Dashboard", "📜 Briefing History"])
    
    with tab1:
        st.header("Ask a Geopolitical or Security Question")
        
        query = st.text_area(
            "Enter your question:",
            value=st.session_state.get('user_query', ''),
            height=100,
            placeholder="Example: What are the latest developments in [specific event/region/conflict]?",
            help="Ask about any current event, conflict, or security situation.",
            key="query_input"
        )
        
        analyze_button = st.button("🚀 Generate Intelligence Brief", type="primary", use_container_width=True)
        
        if analyze_button and query:
            # Check if this is a new query
            if query != st.session_state.last_query:
                st.session_state.last_query = query
                st.session_state.processing = True
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Search and extract sources
                    status_text.text("📡 Step 1/3: Searching for relevant information...")
                    progress_bar.progress(10)
                    sources, content = analyzer.search_and_extract_sources(query)
                    
                    if not content:
                        st.error("No information found. Try rephrasing your query.")
                        st.session_state.processing = False
                        progress_bar.empty()
                        status_text.empty()
                    else:
                        progress_bar.progress(30)
                        time.sleep(0.3)
                        
                        # Step 2: Classify threat
                        status_text.text("🎯 Step 2/3: Analyzing and classifying information...")
                        progress_bar.progress(50)
                        time.sleep(0.3)
                        classification = analyzer.classify_threat(content, query)
                        
                        if classification:
                            st.success(f"✅ Classification: {classification['threat_category'].upper().replace('_', ' ')} - {classification['severity'].upper()}")
                        
                        # Step 3: Find patterns and generate brief
                        status_text.text("📚 Step 3/3: Finding patterns and generating comprehensive brief...")
                        progress_bar.progress(70)
                        time.sleep(0.3)
                        patterns = analyzer.find_historical_patterns(content, query, classification)
                        
                        progress_bar.progress(85)
                        briefing = analyzer.generate_intelligence_brief(query, content, classification, patterns)
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Analysis complete!")
                        time.sleep(0.5)
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Check if briefing indicates insufficient information
                        insufficient_keywords = [
                            "cannot responsibly generate",
                            "fabricated",
                            "contradictory or erroneous",
                            "fictional event",
                            "insufficient",
                            "no verified details"
                        ]
                        
                        is_insufficient = any(keyword.lower() in briefing.lower() for keyword in insufficient_keywords)
                        
                        if is_insufficient:
                            st.warning("⚠️ **Limited Information Available**")
                            st.info("""
The search found limited or unclear information about this query. This could mean:
- The event is very recent and not yet widely reported
- The query contains incorrect details (dates, locations, names)
- The event may not exist or information is still developing

**Suggestions:**
1. Try rephrasing your query with different keywords
2. Check if dates/locations/names are correct
3. Search for broader terms (e.g., "Australia shooting December 2024")
4. Try a different topic with verified recent events
                            """)
                        
                        # Display results
                        st.markdown("---")
                        st.header("Intelligence Briefing")
                        
                        # Key metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        severity_colors = {
                            'critical': '🔴',
                            'high': '🟠',
                            'medium': '🟡',
                            'low': '🟢'
                        }
                        severity_icon = severity_colors.get(classification['severity'].lower(), '⚪')
                        
                        with col1:
                            st.metric("Threat Level", f"{severity_icon} {classification['severity'].upper()}")
                        with col2:
                            st.metric("Category", classification['threat_category'].replace('_', ' ').title())
                        with col3:
                            st.metric("Confidence", f"{int(classification['confidence'] * 100)}%")
                        with col4:
                            st.metric("Sources", len(sources))
                        
                        # Show source articles
                        st.markdown("---")
                        display_source_articles(sources)
                        
                        # Detailed classification
                        st.markdown("---")
                        with st.expander("🎯 Detailed Threat Classification", expanded=False):
                            st.write(f"**Category:** {classification['threat_category'].replace('_', ' ').title()}")
                            st.write(f"**Severity:** {severity_icon} {classification['severity'].upper()}")
                            st.write(f"**Confidence:** {int(classification['confidence'] * 100)}%")
                            
                            if classification.get('key_actors'):
                                st.markdown("**Key Actors:**")
                                for actor in classification['key_actors']:
                                    st.write(f"- {actor}")
                            
                            if classification.get('regions_affected'):
                                st.markdown("**Regions Affected:**")
                                for region in classification['regions_affected']:
                                    st.write(f"- {region}")
                            
                            st.markdown("**Classification Reasoning:**")
                            st.write(classification.get('reasoning', 'N/A'))
                        
                        # Historical patterns
                        with st.expander("📜 Historical Patterns", expanded=False):
                            if patterns and patterns.get('parallels'):
                                for parallel in patterns['parallels']:
                                    st.markdown(f"### {parallel['event']} ({parallel['year']})")
                                    st.write(f"**Similarity Score:** {int(parallel['similarity_score'] * 100)}%")
                                    st.write(f"**Outcome:** {parallel['outcome']}")
                                    st.write(f"**Lessons:** {parallel['lessons']}")
                                    st.markdown("---")
                                
                                if patterns.get('analysis'):
                                    st.markdown("**Pattern Analysis:**")
                                    st.write(patterns['analysis'])
                            else:
                                st.write("No strong historical parallels identified.")
                        
                        # Main briefing
                        st.markdown("---")
                        st.header("📄 Full Intelligence Briefing")
                        st.markdown(briefing)
                        
                        # Save to history
                        st.session_state.briefing_history.append({
                            'timestamp': datetime.now(),
                            'query': query,
                            'classification': classification,
                            'patterns': patterns,
                            'briefing': briefing,
                            'num_sources': len(sources),
                            'sources': sources
                        })
                        
                        # Download button
                        download_content = f"""OSINT INTELLIGENCE BRIEF
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Query: {query}

CLASSIFICATION:
- Threat Level: {classification['severity'].upper()}
- Category: {classification['threat_category']}
- Confidence: {int(classification['confidence'] * 100)}%

SOURCES ({len(sources)}):
"""
                        for i, source in enumerate(sources, 1):
                            download_content += f"\n{i}. {source.get('title', 'Untitled')}"
                            if source.get('url'):
                                download_content += f"\n   URL: {source['url']}"
                        
                        download_content += f"\n\n{briefing}"
                        
                        st.download_button(
                            label="📥 Download Complete Briefing",
                            data=download_content,
                            file_name=f"intelligence_brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                        
                        # Clear query for next input
                        if 'user_query' in st.session_state:
                            del st.session_state.user_query
                        
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
                finally:
                    st.session_state.processing = False
    
    with tab2:
        st.header("📊 Analysis Dashboard")
        
        if not st.session_state.briefing_history:
            st.info("No analyses yet. Generate your first intelligence brief in the Query tab!")
        else:
            latest = st.session_state.briefing_history[-1]
            
            st.subheader("Latest Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Query:** {latest['query']}")
                st.write(f"**Time:** {latest['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            with col2:
                st.write(f"**Sources Analyzed:** {latest.get('num_sources', 'N/A')}")
            
            st.markdown("---")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Threat Assessment")
                severity_map = {'low': 25, 'medium': 50, 'high': 75, 'critical': 100}
                severity_value = severity_map.get(latest['classification']['severity'].lower(), 50)
                st.progress(severity_value / 100)
                st.write(f"**Category:** {latest['classification']['threat_category'].replace('_', ' ').title()}")
                st.write(f"**Confidence:** {int(latest['classification']['confidence'] * 100)}%")
            
            with col2:
                st.markdown("### Analysis History")
                st.metric("Total Briefings", len(st.session_state.briefing_history))
                
                # Count by severity
                severity_counts = {}
                for item in st.session_state.briefing_history:
                    sev = item['classification']['severity']
                    severity_counts[sev] = severity_counts.get(sev, 0) + 1
                
                for sev, count in severity_counts.items():
                    st.write(f"{sev.title()}: {count}")
    
    with tab3:
        st.header("📜 Briefing History")
        
        if not st.session_state.briefing_history:
            st.info("No briefing history yet. Generate analyses to see them here!")
        else:
            for idx, item in enumerate(reversed(st.session_state.briefing_history)):
                severity_icon = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}.get(
                    item['classification']['severity'].lower(), '⚪'
                )
                
                with st.expander(
                    f"{severity_icon} {item['timestamp'].strftime('%Y-%m-%d %H:%M')} - {item['query'][:60]}..."
                ):
                    st.markdown(f"**Full Query:** {item['query']}")
                    st.markdown(f"**Sources Analyzed:** {item.get('num_sources', 'N/A')}")
                    st.markdown(f"**Threat:** {item['classification']['threat_category']} - {severity_icon} {item['classification']['severity'].upper()}")
                    st.markdown(f"**Confidence:** {int(item['classification']['confidence'] * 100)}%")
                    
                    # Show sources
                    if item.get('sources'):
                        st.markdown("---")
                        st.markdown("**Sources:**")
                        for i, source in enumerate(item['sources'][:5], 1):
                            st.write(f"{i}. {source.get('title', 'Untitled')}")
                            if source.get('url'):
                                st.markdown(f"   🔗 {source['url']}")
                    
                    st.markdown("---")
                    st.markdown(item['briefing'])
                    
                    st.download_button(
                        label="📥 Download",
                        data=f"Query: {item['query']}\n\n{item['briefing']}",
                        file_name=f"brief_{item['timestamp'].strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        key=f"download_{idx}"
                    )
            
            if st.button("🗑️ Clear History"):
                st.session_state.briefing_history = []
                st.rerun()


if __name__ == "__main__":
    main()