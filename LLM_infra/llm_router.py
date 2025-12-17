"""
LLM Router - Intelligent routing between vLLM, SGLang, and Claude API
Automatically selects the best backend based on task characteristics
"""

import os
import time
import logging
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict
from openai import OpenAI
import anthropic

from Data.settings import Settings

logger = logging.getLogger(__name__)

BACKENDS: Tuple[Literal["vllm", "sglang", "claude"], Literal["vllm", "sglang", "claude"], Literal["vllm", "sglang", "claude"]] = (
    "vllm",
    "sglang",
    "claude",
)


class BackendMetrics(TypedDict):
    requests: int
    total_time: float
    errors: int
    tokens: int


class LLMRouter:
    """Routes LLM requests to optimal backend (vLLM/SGLang/Claude)"""
    
    def __init__(self):
        self.settings = Settings()
        self.vllm_client: Optional[OpenAI] = None
        self.sglang_client: Optional[OpenAI] = None
        self.claude_client: Optional[anthropic.Anthropic] = None
        self.vllm_available = False
        self.sglang_available = False
        self.claude_available = False
        
        # Initialize clients
        self._init_vllm_client()
        self._init_sglang_client()
        self._init_claude_client()
        
        # Performance tracking
        self.metrics: Dict[Literal["vllm", "sglang", "claude"], BackendMetrics] = {
            "vllm": {"requests": 0, "total_time": 0.0, "errors": 0, "tokens": 0},
            "sglang": {"requests": 0, "total_time": 0.0, "errors": 0, "tokens": 0},
            "claude": {"requests": 0, "total_time": 0.0, "errors": 0, "tokens": 0},
        }
        
        logger.info("LLM Router initialized")
    
    def _extract_text_from_anthropic(self, blocks: Sequence[Any]) -> str:
        """Extract concatenated text from Anthropic content blocks."""
        if not blocks:
            raise ValueError("No content blocks provided from Claude response")
        texts = []
        for block in blocks:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                texts.append(text)
        if not texts:
            raise ValueError("No text content found in Claude response")
        return "".join(texts)
    
    def _clean_json_content(self, raw_content: str) -> str:
        """Remove optional markdown fences and return raw JSON text."""
        content = raw_content.strip()
        if not content:
            raise ValueError("Empty response content")
        
        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in content:
            content = content.split("```", 1)[1].split("```", 1)[0].strip()
        
        if not content:
            raise ValueError("No JSON content found in model response")
        return content
    
    def _init_vllm_client(self):
        """Initialize vLLM client (OpenAI-compatible)"""
        try:
            self.vllm_client = OpenAI(
                api_key="EMPTY",  # vLLM doesn't need API key
                base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
            )
            self.vllm_available = True
            logger.info("vLLM client initialized")
        except Exception as e:
            logger.warning(f"vLLM not available: {e}")
            self.vllm_available = False
            self.vllm_client = None
    
    def _init_sglang_client(self):
        """Initialize SGLang client (OpenAI-compatible)"""
        try:
            self.sglang_client = OpenAI(
                api_key="EMPTY",
                base_url=os.getenv("SGLANG_BASE_URL", "http://localhost:8001/v1")
            )
            self.sglang_available = True
            logger.info("SGLang client initialized")
        except Exception as e:
            logger.warning(f"SGLang not available: {e}")
            self.sglang_available = False
            self.sglang_client = None
    
    def _init_claude_client(self):
        """Initialize Anthropic Claude client"""
        try:
            self.claude_client = anthropic.Anthropic(
                api_key=self.settings.ANTHROPIC_API_KEY
            )
            self.claude_available = True
            logger.info("Claude API client initialized")
        except Exception as e:
            logger.warning(f"Claude not available: {e}")
            self.claude_available = False
            self.claude_client = None
    
    def select_backend(
        self,
        task_type: Literal["classification", "pattern_matching", "reasoning", "brief_generation"],
        auto: bool = True
    ) -> Literal["vllm", "sglang", "claude"]:
        """
        Select optimal backend based on task type
        
        Args:
            task_type: Type of task to perform
            auto: Whether to auto-select (False uses fallback logic)
        
        Returns:
            Backend name: "vllm", "sglang", or "claude"
        """
        if not auto:
            # Fallback to available backends
            if self.claude_available:
                return "claude"
            elif self.vllm_available:
                return "vllm"
            elif self.sglang_available:
                return "sglang"
            else:
                raise RuntimeError("No LLM backends available")
        
        # Task-specific routing
        if task_type == "classification":
            # vLLM is best for simple classification (high throughput)
            if self.vllm_available:
                return "vllm"
            elif self.sglang_available:
                return "sglang"
            else:
                return "claude"
        
        elif task_type == "pattern_matching":
            # SGLang excels at pattern matching with prefix caching
            if self.sglang_available:
                return "sglang"
            elif self.vllm_available:
                return "vllm"
            else:
                return "claude"
        
        elif task_type in ["reasoning", "brief_generation"]:
            # Claude is best for complex reasoning
            if self.claude_available:
                return "claude"
            elif self.sglang_available:
                return "sglang"
            else:
                return "vllm"
        
        # Default fallback
        return "claude" if self.claude_available else "vllm"
    
    def classify_event(
        self,
        title: str,
        description: str,
        locations: List[str],
        backend: Literal["vllm", "sglang", "claude", "auto"] = "auto"
    ) -> Dict[str, Any]:
        """
        Classify a geopolitical event
        
        Args:
            title: Event title
            description: Event description
            locations: List of locations
            backend: Which backend to use
        
        Returns:
            Classification result with threat_category, confidence, etc.
        """
        # Select backend
        selected_backend: Literal["vllm", "sglang", "claude"]
        if backend == "auto":
            selected_backend = self.select_backend("classification")
        else:
            selected_backend = backend
        
        # Build prompt
        prompt = f"""You are a geopolitical threat analyst. Classify this event.

Event: {title}
Description: {description}
Locations: {', '.join(locations)}

Categories: military_buildup, airspace_violation, naval_incident, border_conflict, 
drone_attack, sanctions, terrorism, civil_unrest, cyber_operation, wmd_activity, 
humanitarian_crisis, energy_security

Respond in JSON format ONLY (no markdown, no preamble):
{{
    "threat_category": "category_name",
    "confidence": 0.95,
    "severity": 0.8,
    "reasoning": "brief explanation",
    "actors": ["actor1", "actor2"]
}}"""

        start_time = time.time()
        
        try:
            if selected_backend == "claude":
                result = self._classify_with_claude(prompt)
            else:
                result = self._classify_with_local_llm(prompt, selected_backend)
            
            elapsed = time.time() - start_time
            
            # Update metrics
            self.metrics[selected_backend]["requests"] += 1
            self.metrics[selected_backend]["total_time"] += elapsed
            
            # Add metadata
            result["backend"] = selected_backend
            result["latency_ms"] = elapsed * 1000
            
            return result
        
        except Exception as e:
            self.metrics[selected_backend]["errors"] += 1
            logger.error(f"Error in classification ({selected_backend}): {e}")
            raise
    
    def _classify_with_claude(self, prompt: str) -> Dict[str, Any]:
        """Classify using Claude API"""
        if self.claude_client is None:
            raise RuntimeError("Claude client not initialized")
        
        response = self.claude_client.messages.create(
            model=self.settings.PRIMARY_LLM_MODEL,
            max_tokens=500,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse JSON from response
        import json
        raw_content = self._extract_text_from_anthropic(response.content)
        cleaned_content = self._clean_json_content(raw_content)
        result = json.loads(cleaned_content)
        
        usage = getattr(response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)
            self.metrics["claude"]["tokens"] += int(input_tokens) + int(output_tokens)
        
        return result
    
    def _classify_with_local_llm(self, prompt: str, backend: Literal["vllm", "sglang"]) -> Dict[str, Any]:
        """Classify using vLLM or SGLang"""
        client = self.vllm_client if backend == "vllm" else self.sglang_client
        
        if not client:
            raise RuntimeError(f"{backend} client not available")
        
        response = client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        
        # Parse JSON response
        import json
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("No content returned from local LLM")
        
        cleaned_content = self._clean_json_content(content)
        result = json.loads(cleaned_content)
        
        # Track tokens if available
        usage = getattr(response, "usage", None)
        if usage and getattr(usage, "total_tokens", None) is not None:
            self.metrics[backend]["tokens"] += int(usage.total_tokens)
        
        return result
    
    def analyze_pattern(
        self,
        events: List[Dict[str, Any]],
        backend: Literal["vllm", "sglang", "claude", "auto"] = "auto"
    ) -> Dict[str, Any]:
        """
        Analyze historical patterns in events
        
        Args:
            events: List of event dictionaries
            backend: Which backend to use
        
        Returns:
            Pattern analysis with historical parallels
        """
        # Select backend (SGLang preferred for pattern matching)
        selected_backend: Literal["vllm", "sglang", "claude"]
        if backend == "auto":
            selected_backend = self.select_backend("pattern_matching")
        else:
            selected_backend = backend
        
        # Build prompt with shared prefix (benefits SGLang's RadixAttention)
        base_prompt = """You are a geopolitical historian. Analyze these events and identify historical parallels.

Consider:
1. Similar past conflicts or incidents
2. Escalation patterns (what typically happens next)
3. Diplomatic precedents
4. Geographic/regional history
5. Actor behavior patterns

Events to analyze:
"""
        
        events_text = "\n\n".join([
            f"Event {i+1}: {e.get('title', 'Unknown')}\n{e.get('description', '')}"
            for i, e in enumerate(events[:5])  # Limit to 5 events
        ])
        
        prompt = base_prompt + events_text + """\n\nProvide analysis in JSON format:
{
    "historical_parallel": "Most relevant historical event",
    "similarity_score": 0.85,
    "key_similarities": ["similarity1", "similarity2"],
    "key_differences": ["difference1", "difference2"],
    "likely_outcome": "What typically happens next",
    "confidence": 0.75
}"""

        start_time = time.time()
        
        try:
            if selected_backend == "claude":
                result = self._analyze_with_claude(prompt)
            else:
                result = self._analyze_with_local_llm(prompt, selected_backend)
            
            elapsed = time.time() - start_time
            
            # Update metrics
            self.metrics[selected_backend]["requests"] += 1
            self.metrics[selected_backend]["total_time"] += elapsed
            
            result["backend"] = selected_backend
            result["latency_ms"] = elapsed * 1000
            
            return result
        
        except Exception as e:
            self.metrics[selected_backend]["errors"] += 1
            logger.error(f"Error in pattern analysis ({selected_backend}): {e}")
            raise
    
    def _analyze_with_claude(self, prompt: str) -> Dict[str, Any]:
        """Analyze using Claude API"""
        if self.claude_client is None:
            raise RuntimeError("Claude client not initialized")
        
        response = self.claude_client.messages.create(
            model=self.settings.PRIMARY_LLM_MODEL,
            max_tokens=1000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        
        import json
        raw_content = self._extract_text_from_anthropic(response.content)
        cleaned_content = self._clean_json_content(raw_content)
        
        usage = getattr(response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)
            self.metrics["claude"]["tokens"] += int(input_tokens) + int(output_tokens)
        
        return json.loads(cleaned_content)
    
    def _analyze_with_local_llm(self, prompt: str, backend: Literal["vllm", "sglang"]) -> Dict[str, Any]:
        """Analyze using vLLM or SGLang"""
        client = self.vllm_client if backend == "vllm" else self.sglang_client
        
        if not client:
            raise RuntimeError(f"{backend} client not available")
        
        response = client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        
        import json
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("No content returned from local LLM")
        
        cleaned_content = self._clean_json_content(content)
        
        return json.loads(cleaned_content)
    
    def generate_brief(
        self,
        events_summary: str,
        alerts_summary: str,
        backend: Literal["vllm", "sglang", "claude", "auto"] = "claude"
    ) -> str:
        """
        Generate intelligence brief (Claude recommended)
        
        Args:
            events_summary: Summary of processed events
            alerts_summary: Summary of alerts
            backend: Backend to use (defaults to Claude for quality)
        
        Returns:
            Generated intelligence brief text
        """
        selected_backend: Literal["vllm", "sglang", "claude"]
        if backend == "auto":
            selected_backend = self.select_backend("brief_generation")
        else:
            selected_backend = backend
        
        prompt = f"""You are a senior intelligence officer. Create a concise intelligence brief.

EVENTS SUMMARY:
{events_summary}

ALERTS SUMMARY:
{alerts_summary}

Generate a professional brief with:
1. Executive Summary (2-3 sentences)
2. Critical Developments
3. Emerging Patterns
4. Recommendations

Keep it concise and actionable."""

        start_time = time.time()
        
        try:
            if selected_backend == "claude":
                if self.claude_client is None:
                    raise RuntimeError("Claude client not initialized")
                response = self.claude_client.messages.create(
                    model=self.settings.PRIMARY_LLM_MODEL,
                    max_tokens=2000,
                    temperature=0.2,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                brief = self._extract_text_from_anthropic(response.content)
                usage = getattr(response, "usage", None)
                if usage:
                    input_tokens = getattr(usage, "input_tokens", 0)
                    output_tokens = getattr(usage, "output_tokens", 0)
                    self.metrics["claude"]["tokens"] += int(input_tokens) + int(output_tokens)
            
            else:
                client = self.vllm_client if selected_backend == "vllm" else self.sglang_client
                if client is None:
                    raise RuntimeError(f"{selected_backend} client not available")
                response = client.chat.completions.create(
                    model="default",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=2000
                )
                brief = response.choices[0].message.content
            
            elapsed = time.time() - start_time
            
            if brief is None:
                raise ValueError("No brief text returned from model")
            
            self.metrics[selected_backend]["requests"] += 1
            self.metrics[selected_backend]["total_time"] += elapsed
            
            return brief
        
        except Exception as e:
            self.metrics[selected_backend]["errors"] += 1
            logger.error(f"Error generating brief ({selected_backend}): {e}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all backends"""
        metrics_summary = {}
        
        for backend in BACKENDS:
            m = self.metrics[backend]
            if m["requests"] > 0:
                avg_latency = (m["total_time"] / m["requests"]) * 1000
                error_rate = m["errors"] / m["requests"]
            else:
                avg_latency = 0
                error_rate = 0
            
            metrics_summary[backend] = {
                "total_requests": m["requests"],
                "avg_latency_ms": round(avg_latency, 2),
                "error_rate": round(error_rate, 3),
                "total_time_s": round(m["total_time"], 2),
                "total_tokens": m["tokens"],
                "available": getattr(self, f"{backend}_available", False)
            }
        
        return metrics_summary
    
    def reset_metrics(self):
        """Reset all metrics"""
        for backend in self.metrics:
            self.metrics[backend] = {
                "requests": 0,
                "total_time": 0.0,
                "errors": 0,
                "tokens": 0
            }
        logger.info("Metrics reset")
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all backends"""
        health = {}
        
        # Check vLLM
        if self.vllm_available and self.vllm_client:
            try:
                self.vllm_client.chat.completions.create(
                    model="default",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                health["vllm"] = True
            except:
                health["vllm"] = False
        else:
            health["vllm"] = False
        
        # Check SGLang
        if self.sglang_available and self.sglang_client:
            try:
                self.sglang_client.chat.completions.create(
                    model="default",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                health["sglang"] = True
            except:
                health["sglang"] = False
        else:
            health["sglang"] = False
        
        # Check Claude
        health["claude"] = self.claude_available
        
        return health


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize router
    router = LLMRouter()
    
    # Check health
    health = router.health_check()
    print("\n🏥 Backend Health:")
    for backend, status in health.items():
        print(f"  {backend}: {'✅' if status else '❌'}")
    
    # Test classification
    print("\n📊 Testing Classification...")
    try:
        result = router.classify_event(
            title="Chinese warships conduct exercises near Taiwan",
            description="PLA Navy vessels spotted in Taiwan Strait",
            locations=["Taiwan Strait", "Taiwan"],
            backend="auto"
        )
        
        print(f"  Backend used: {result['backend']}")
        print(f"  Category: {result['threat_category']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Latency: {result['latency_ms']:.2f}ms")
    
    except Exception as e:
        print(f"  ❌ Classification failed: {e}")
    
    # Get metrics
    print("\n📈 Performance Metrics:")
    metrics = router.get_metrics()
    for backend, m in metrics.items():
        if m["total_requests"] > 0:
            print(f"\n  {backend.upper()}:")
            print(f"    Requests: {m['total_requests']}")
            print(f"    Avg Latency: {m['avg_latency_ms']:.2f}ms")
            print(f"    Error Rate: {m['error_rate']:.1%}")
