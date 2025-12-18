import streamlit as st
import anthropic 
import requests
from datetime import datetime, timedelta
import json 
from typing import Dict, List, Any 
import time 

#Page configurations
st.set_page_config(
    page_title = "OSINT - Geopolitical Intelligence",
    page_icon = "🌍",
    layout = "wide",
    initial_sidebar_state = "expanded")

#Custom CSS
st.markdown("""
            <style>
            .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            padding: 1rem 0;
            }
            .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
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
            .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            }
            </style>
            """, unsafe_allow_html = True)

#Initializing the session state 
if 'briefing_history' not in st.session_state:
    st.session_state.briefing_history = []
if 'processing' not in st.session_state:
    st.session_state.processing = False 

class OSINTAnalyzer:
    """ Main Analyzer class that will coordinate the multi-agent workflow"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key = api_key)
        self.model = "claude-3-5-sonnet-20241022"

    def _get_first_text_block(self, content: List[Any]) -> str:
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

    def search_recent_events(self, query: str, max_results: int = 10) -> List[Dict]:
        """ Search for recent events using Anthropic's search API """
        try:
            #Using claude with web search to find the recent events
            message = self.client.messages.create(
                model = self.model, 
                max_tokens = 4000,
                tools = [{
                    "type": "web_search_20250305",
                    "name": "web_search"
                }],
                messages = [{
                    "role": "user",
                    "content": f"""Search for the most recent news and events related to: {query}

    Focus on:
    -Breaking news and develoopments in the last 7 days
    -Official statements and government actions
    -Military movements or security incidents
    -Diplomatic developments
    -Economic sanctions or trade actions

    Return a comprehensive summary of what you find. """
                }]
            )

            #Extracting the search results from the response
            events = []
            for block in message.content:
                text = None
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text = block.get("text")
                elif getattr(block, "type", None) == "text":
                    text = getattr(block, "text", "")

                if text:
                    events.append({
                        "content": text,
                        "timestamp": datetime.now().isoformat(),
                        "query": query
                    })

            return events
        except Exception as e:
            st.error(f"Error during search: {e}")
            return []
        
    def classify_threat(self, event_content: str) -> Dict[str, Any]:
        """Agent 1: Classify the threat level of the event"""
        try:
            message = self.client.messages.create(
                model = self.model,
                max_tokens = 2000,
                messages = [{
                    "role": "user",
                    "content": f""" Analyze this geopolitical event and classify its threat level as Critical, High, Medium, or Low.

    Event: 
    {event_content}

    Provide a JSON response with:
    {{
        "threat_category": "one of: military_buildup, terrorism, cyber_operations, wmd_activity, diplomatic_crisis, humanitarian_crisis, economic_warfare, civil_unrest, border_conflict, political_instability, disinformation, other",
        "severity": "low/medium/high/critical",
        "confidence": 0.0 - 1.0,
        "key_actors": ["list of countries/organizations"],
        "reasoning": "brief explanation
    }}.""" 
                }]
            )

            #Parsing response
            response_text = self._get_first_text_block(message.content)
            #Extracting JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(response_text[json_start:json_end])
            else:
                return {
                    "threat_category": "other",
                    "severity": "medium",
                    "confidence": 0.5,
                    "key_actors": [],
                    "reasoning": response_text
                }
        except Exception as e:
            st.error(f"Classification error: {str(e)}")
            return {
                "threat_category": "other",
                "severity": "medium",
                "confidence": 0.0,
                "key_actors": [],
                "reasoning": "Classification failed; see error logs."
            }
        
    def find_historical_patterns(self, event_content: str, classification: Dict) -> Dict[str, Any]:
        """ Agent 2: Matching to historical patterns"""
        try:
            message = self.client.messages.create(
                model = self.model,
                max_tokens = 2000,
                messages = [{
                    "role": "user",
                    "content": f""" Find historical parallels to this geopolitical event based on its classification:
    CURRENT EVENT:
    {event_content}

    CLASSIFICATION: {classification['threat_category']} - {classification['severity']}
    Identify 2-3 historical events that are similar in nature. Provide a JSON response with:
    {{
        "parallels": [
        {{
            "event": "Historical event name",
            "year": "YYYY",
            "similarity_score": 0.0 - 1.0,
            "outcome": "How it was resolved",
            "lessons": "Key lessons learned"
        }}
    ],
        "pattern_type": "escalation/de-escalation/cyclical/unprecedented"
    }}"""
                }]
            )

            response_text = self._get_first_text_block(message.content)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(response_text[json_start:json_end])
            else:
                return {"parallels": [], "pattern_type": "unprecedented"}
        except Exception as e:
            st.error(f"Pattern matching error: {str(e)}")
            return {"parallels": [], "pattern_type": "unprecedented"}
        
    def predict_escalation(self, event_content: str, classification: Dict, patterns: Dict) -> Dict[str, Any]:
        """ Agent 3: Predicting escalation probability"""
        try:
            message = self.client.messages.create(
                model = self.model,
                max_tokens = 2000,
                messages = [{
                    "role": "user",
                    "content": f""" Predict escalation probability for this geopolitical situation:
    EVENT: {event_content}
    THREAT: {classification['threat_category']} - {classification['severity']}
    HISTORICAL PATTERNS: {json.dumps(patterns)}

    Provide JSON with:
    {{
        "escalation_probability": 0.0 - 1.0,
        "timeframe": "24h/7d/30d",
        "escalation_triggers": ["trigger 1", "trigger 2"],
        "de-escalation_factors": ["factor 1", "factor 2"],
        "risk_assessment": "detailed explanation",
        "recommended_monitoring": ["what to monitor"]
    }}"""
                }]
            )

            response_text = self._get_first_text_block(message.content)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(response_text[json_start:json_end])
            else:
                return {
                    "escalation_probability": 0.5,
                    "timeframe": "30d",
                    "escalation_triggers": [],
                    "de-escalation_factors": [],
                    "risk_assessment": response_text,
                    "recommended_monitoring": []
                }
        except Exception as e:
            st.error(f"Escalation prediction error: {str(e)}")
            return {
                "escalation_probability": 0.0,
                "timeframe": "30d",
                "escalation_triggers": [],
                "de-escalation_factors": [],
                "risk_assessment": "Escalation prediction failed; see error logs.",
                "recommended_monitoring": []
            }
        
    def generate_intelligence_brief(self, query: str, events: List, classification: Dict,
                                    patterns: Dict, escalation: Dict) -> str:
        """ Agent 4: Generating the final intelligence brief"""

        try: 
            message = self.client.messages.create(
                model = self.model,
                max_tokens = 4000,
                messages = [{
                    "role": "user",
                    "content": f""" Generate a comprehensive and professional intelligence briefing: 
    QUERY: {query}
    EVENTS: {json.dumps([e['content'][:500] for e in events])}
    CLASSIFICATION: {json.dumps(classification)}
    HISTORICAL PATTERNS: {json.dumps(patterns)}
    ESCALATION ANALYSIS: {json.dumps(escalation)}

    Create a comprehensive intelligence brief with:

    #EXECUTIVE SUMMARY
    (2-3 sentences summarizing key findings)
    
    #SITUATION ANALYSIS
    (Current state and key developments)
    
    #THREAT ASSESSMENT
    (Severity, actors, capabilities)

    #HISTORICAL CONTEXT
    (Relevant parallels and patterns)

    #ESCALATION FORECAST
    (Probability, triggers, timeline)

    #STRATEGIC IMPLICATIONS
    (What this means for stakeholders)

    #RECOMMENDATIONS
    (Monitoring priorities and response options)

    #CONFIDENCE ASSESSMENT
    (Data quality and analytical limitations)

    Use professional intelligence community language. Be specfiic and actionable. """
                }]
            )

            return self._get_first_text_block(message.content)
        except Exception as e: 
            st.error(f"Brief generation error: {str(e)}")
            return "Error generating brief"
        
def main():
    #Header
    st.markdown('<div class="main-header">🌍 OSINT Geopolitical Intelligence Analyzer 🌍</div>', unsafe_allow_html=True)
    st.markdown("**Ask any geopolitical question and receive a comprehensive intelligence briefing powered by multi-agent AI analysis.**")

    #Sidebar for API key and settings
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input(
            "Anthropic API Key",
            type = "password",
            help = "Get your key at console.anthropic.com"
        )

        st.markdown("---")

        st.header("Example Queries")
        example_queries = [
            "What's the current situation with China and Taiwan?",
            "Analyze recent developments in the Ukraine conflict",
            "What are the latest tensions in the Middle East?",
            "Tell me about North Korea's missile program",
            "What's hahppening with Iran's nuclear program?",
            "Analyze Russia's military posture in Eastern Europe",
            "What are the current US-China trade tensions?",
            "Assess the situation in the South China Sea"
        ]

        for query in example_queries:
            if st.button(query, key = f"example_{query}", use_container_width = True):
                st.session_state.user_query = query

    #Main interface
    if not api_key:
        st.warning("Please enter your Anthropic API key in the sidebar to proceed")
        st.info("Don't have an API key? Get one at https://console.anthropic.com/")

        #Show features
        st.header("Platform Features")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Multi-Sourced Intelligence")
            st.write("Aggregates data from GDELT, NewsAPI, and EventRegistry")

        with col2:
            st.markdown("### 6-Agent Pipeline")
            st.write("Classification -> Pattern Matching -> Escalation Prediction -> Briefing")

        with col3:
            st.markdown("### Real time Analysis")
            st.write("Get up-to-date intelligence briefings on emerging geopolitical events")
        return 
    
    #Initialize analyzer
    analyzer = OSINTAnalyzer(api_key)

    #Tabs for the different features 
    tab1, tab2, tab3 = st.tabs(["Intelligence Query", "Analysis Dashboard", "Briefing History"])
    with tab1:
        st.header("Ask a Geopolitical Question")

        #Query input
        query = st.text_area(
            "Enter your question: ",
            value = st.session_state.get('query', ''),
            height = 100,
            placeholder = "Example: What are the latest developments in the Ukraine-Russia conflict?",
            help="Ask about any geopolitical topic, conflic, or region. The system will search recent news and generate a comprehensive intelligence briefing."
        )

        col1, col2, col3 = st.columns([2,1,1])
        with col1: 
            analyze_button = st.button("Generate Intelligence Brief", type = "primary", use_container_width = True)
        with col2: 
            max_sources = st.number_input("Max Sources", min_value = 3, max_value = 20, value = 10)
        with col3:
            st.write("")

        if analyze_button and query:
            st.session_state.processing = True

            #Progres tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                #Step 1: Searching for recent events
                status_text.text("Step 1: Searching for recent events...")
                progress_bar.progress(10)
                events = analyzer.search_recent_events(query, max_sources)

                if not events:
                    st.error("No recent events found. Try a different query.")
                    st.session_state.processing = False 
                    return 
                
                st.success(f"Found information from {len(events)} source(s)")

                #Step 2: Classifying threat
                status_text.text("Step 2: Classifying threat level...")
                progress_bar.progress(30)
                time.sleep(0.5) #Slight time delay for UX
                classification = analyzer.classify_threat(events[0]['content'])

                if classification:
                    st.success(f"Classification: {classification['threat_category'].upper()} - {classification['severity'].upper()}")

                #Step 3: Finding historical patterns
                status_text.text("Step 3: Analyzing historical patterns...")
                progress_bar.progress(50)
                time.sleep(0.5)
                patterns = analyzer.find_historical_patterns(events[0]['content'], classification)

                if patterns:
                    st.success(f"Found {len(patterns.get('parallels', []))} historical parallels")

                #Step 4: Predicting escalation 
                status_text.text("Step 4: Predicting escalation probability...")
                progress_bar.progress(70)
                time.sleep(0.5)
                escalation = analyzer.predict_escalation(events[0]['content'], classification, patterns)

                if escalation: 
                    prob = int(escalation['escalation_probability'] * 100)
                    st.success(f"Escalation probability: {prob}% over {escalation['timeframe']}")

                #Step 5: Generating intelligence brief
                status_text.text("Step 5: Generating intelligence briefing...")
                progress_bar.progress(90)
                briefing = analyzer.generate_intelligence_brief(query, events, classification, patterns, escalation)

                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                time.sleep(0.5)

                #Clearing progress indicators
                progress_bar.empty()
                status_text.empty()

                #Displaying results in expandable sections
                st.markdown("---")
                st.header("Intelligence Briefing")

                #Key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Threat Level", classification['severity'].upper())
                with col2: 
                    st.metric("Category", classification['threat_category'].replace('_', ' ').title())
                with col3:
                    prob_pct = int(escalation['escalation_probability'] * 100)
                    st.metric("Escalation Risk", f"{prob_pct}%")
                with col4: 
                    st.metric("Confidence", f"{int(classification['confidence'] * 100)}%")

                # Detailed analysis in expanders
                with st.expander("🎯 Threat Classification", expanded=False):
                    st.write(f"**Category:** {classification['threat_category'].replace('_', ' ').title()}")
                    st.write(f"**Severity:** {classification['severity'].upper()}")
                    st.write(f"**Key Actors:** {', '.join(classification.get('key_actors', []))}")
                    st.write(f"**Reasoning:** {classification.get('reasoning', 'N/A')}")
                
                with st.expander("📜 Historical Patterns", expanded=False):
                    if patterns and patterns.get('parallels'):
                        for parallel in patterns['parallels']:
                            st.markdown(f"### {parallel['event']} ({parallel['year']})")
                            st.write(f"**Similarity Score:** {int(parallel['similarity_score'] * 100)}%")
                            st.write(f"**Outcome:** {parallel['outcome']}")
                            st.write(f"**Lessons:** {parallel['lessons']}")
                            st.markdown("---")
                    else:
                        st.write("No strong historical parallels identified.")
                
                with st.expander("Escalation Assessment", expanded=False):
                    st.write(f"**Probability:** {int(escalation['escalation_probability'] * 100)}%")
                    st.write(f"**Timeframe:** {escalation['timeframe']}")
                    st.write(f"**Risk Assessment:** {escalation['risk_assessment']}")
                    
                    st.markdown("**Escalation Triggers:**")
                    for trigger in escalation.get('escalation_triggers', []):
                        st.write(f"- {trigger}")
                    
                    st.markdown("**De-escalation Factors:**")
                    for factor in escalation.get('de-escalation_factors', []):
                        st.write(f"- {factor}")
                    
                    st.markdown("**Monitoring Priorities:**")
                    for item in escalation.get('recommended_monitoring', []):
                        st.write(f"- {item}")
                
                # Main briefing
                st.markdown("---")
                st.header("Intelligence Briefing")
                st.markdown(briefing)
                
                # Save to history
                st.session_state.briefing_history.append({
                    'timestamp': datetime.now(),
                    'query': query,
                    'classification': classification,
                    'escalation': escalation,
                    'briefing': briefing
                })
                
                # Download button
                st.download_button(
                    label="Download Briefing",
                    data=briefing,
                    file_name=f"intelligence_brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
            finally:
                st.session_state.processing = False
    
    with tab2:
        st.header("📈 Analysis Dashboard")
        
        if not st.session_state.briefing_history:
            st.info("No analyses yet. Generate your first intelligence brief in the Query tab!")
        else:
            latest = st.session_state.briefing_history[-1]
            
            st.subheader("Latest Analysis")
            st.write(f"**Query:** {latest['query']}")
            st.write(f"**Time:** {latest['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Threat Assessment")
                severity_map = {'low': 25, 'medium': 50, 'high': 75, 'critical': 100}
                severity_value = severity_map.get(latest['classification']['severity'], 50)
                st.progress(severity_value / 100)
                st.write(f"**Category:** {latest['classification']['threat_category'].replace('_', ' ').title()}")
                st.write(f"**Confidence:** {int(latest['classification']['confidence'] * 100)}%")
            
            with col2:
                st.markdown("### Escalation Risk")
                esc_prob = latest['escalation']['escalation_probability']
                st.progress(esc_prob)
                st.write(f"**Probability:** {int(esc_prob * 100)}%")
                st.write(f"**Timeframe:** {latest['escalation']['timeframe']}")
            
            # Key actors
            if latest['classification'].get('key_actors'):
                st.markdown("### Key Actors")
                cols = st.columns(len(latest['classification']['key_actors']))
                for idx, actor in enumerate(latest['classification']['key_actors']):
                    with cols[idx]:
                        st.info(actor)
    
    with tab3:
        st.header("📜 Briefing History")
        
        if not st.session_state.briefing_history:
            st.info("No briefing history yet. Generate analyses to see them here!")
        else:
            # Show most recent first
            for idx, item in enumerate(reversed(st.session_state.briefing_history)):
                with st.expander(f"{item['timestamp'].strftime('%Y-%m-%d %H:%M')} - {item['query'][:50]}..."):
                    st.markdown(f"**Full Query:** {item['query']}")
                    st.markdown(f"**Threat:** {item['classification']['threat_category']} - {item['classification']['severity']}")
                    st.markdown(f"**Escalation Risk:** {int(item['escalation']['escalation_probability'] * 100)}%")
                    st.markdown("---")
                    st.markdown(item['briefing'])
                    
                    st.download_button(
                        label="📥 Download",
                        data=item['briefing'],
                        file_name=f"brief_{item['timestamp'].strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        key=f"download_{idx}"
                    )
            
            if st.button("Clear History"):
                st.session_state.briefing_history = []
                st.rerun()

if __name__ == "__main__":
    main()
