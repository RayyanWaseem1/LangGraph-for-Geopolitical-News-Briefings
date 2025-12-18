"""
Comprehensive vLLM vs SGLang Benchmark for OSINT Pipeline
Measures: TTFT, throughput, latency, tokens/sec, cost efficiency
"""

import asyncio
import time
import json
import statistics
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import httpx
from pathlib import Path

@dataclass
class BenchmarkMetrics:
    """Metrics for a single request"""
    request_id: int
    model_name: str
    backend: str  # "vllm" or "sglang"
    
    # Timing metrics
    ttft: float  # Time to First Token (ms)
    total_time: float  # Total request time (ms)
    
    # Token metrics
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    # Throughput
    tokens_per_second: float
    
    # Success/failure
    success: bool
    error_message: Optional[str] = None
    
    # Additional context
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class BenchmarkSummary:
    """Aggregate metrics for a benchmark run"""
    backend: str
    model_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    
    # TTFT statistics
    avg_ttft: float
    median_ttft: float
    p95_ttft: float
    p99_ttft: float
    
    # Total time statistics
    avg_total_time: float
    median_total_time: float
    p95_total_time: float
    
    # Throughput statistics
    avg_tokens_per_second: float
    median_tokens_per_second: float
    total_tokens_processed: int
    
    # Overall throughput
    total_duration: float  # seconds
    requests_per_second: float
    tokens_per_second_aggregate: float
    
    def to_dict(self):
        return asdict(self)


class LLMBenchmark:
    """Benchmark runner for vLLM and SGLang"""
    
    def __init__(self, vllm_url: str = "http://localhost:8000", sglang_url: str = "http://localhost:8001"):
        self.vllm_url = vllm_url
        self.sglang_url = sglang_url
        self.client = httpx.AsyncClient(timeout=300.0)
    
    async def single_request(
        self,
        backend: str,
        prompt: str,
        model: str,
        request_id: int,
        max_tokens: int = 500,
        temperature: float = 0.1
    ) -> BenchmarkMetrics:
        """Execute a single request and collect metrics"""
        
        url = f"{self.vllm_url if backend == 'vllm' else self.sglang_url}/v1/chat/completions"
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        start_time = time.time()
        ttft = None
        
        try:
            response = await self.client.post(url, json=payload)
            
            if response.status_code != 200:
                return BenchmarkMetrics(
                    request_id=request_id,
                    model_name=model,
                    backend=backend,
                    ttft=0,
                    total_time=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    tokens_per_second=0,
                    success=False,
                    error_message=f"HTTP {response.status_code}: {response.text[:200]}"
                )
            
            # For non-streaming, TTFT ≈ total time
            ttft = (time.time() - start_time) * 1000  # Convert to ms
            total_time = ttft
            
            data = response.json()
            usage = data.get("usage", {})
            
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
            
            # Tokens per second
            tokens_per_second = (completion_tokens / (total_time / 1000)) if total_time > 0 else 0
            
            return BenchmarkMetrics(
                request_id=request_id,
                model_name=model,
                backend=backend,
                ttft=ttft,
                total_time=total_time,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                tokens_per_second=tokens_per_second,
                success=True
            )
            
        except Exception as e:
            return BenchmarkMetrics(
                request_id=request_id,
                model_name=model,
                backend=backend,
                ttft=0,
                total_time=(time.time() - start_time) * 1000,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                tokens_per_second=0,
                success=False,
                error_message=str(e)
            )
    
    async def run_benchmark(
        self,
        backend: str,
        model: str,
        prompts: List[str],
        concurrent_requests: int = 1,
        max_tokens: int = 500
    ) -> tuple[List[BenchmarkMetrics], BenchmarkSummary]:
        """Run benchmark on a specific backend"""
        
        print(f"\n{'='*80}")
        print(f"Running {backend.upper()} Benchmark")
        print(f"{'='*80}")
        print(f"Model: {model}")
        print(f"Total requests: {len(prompts)}")
        print(f"Concurrent requests: {concurrent_requests}")
        print(f"Max tokens per request: {max_tokens}")
        print()
        
        start_time = time.time()
        metrics_list = []
        
        # Process in batches based on concurrency
        for i in range(0, len(prompts), concurrent_requests):
            batch = prompts[i:i+concurrent_requests]
            
            tasks = [
                self.single_request(backend, prompt, model, i+j, max_tokens)
                for j, prompt in enumerate(batch)
            ]
            
            batch_results = await asyncio.gather(*tasks)
            metrics_list.extend(batch_results)
            
            # Progress indicator
            completed = min(i + concurrent_requests, len(prompts))
            print(f"Progress: {completed}/{len(prompts)} requests completed")
        
        total_duration = time.time() - start_time
        
        # Calculate summary statistics
        successful_metrics = [m for m in metrics_list if m.success]
        
        if not successful_metrics:
            print(f"\n❌ All requests failed for {backend}")
            return metrics_list, None
        
        ttfts = [m.ttft for m in successful_metrics]
        total_times = [m.total_time for m in successful_metrics]
        tokens_per_sec = [m.tokens_per_second for m in successful_metrics]
        
        total_tokens = sum(m.total_tokens for m in successful_metrics)
        
        summary = BenchmarkSummary(
            backend=backend,
            model_name=model,
            total_requests=len(prompts),
            successful_requests=len(successful_metrics),
            failed_requests=len(prompts) - len(successful_metrics),
            
            avg_ttft=statistics.mean(ttfts),
            median_ttft=statistics.median(ttfts),
            p95_ttft=statistics.quantiles(ttfts, n=20)[18] if len(ttfts) > 1 else ttfts[0],
            p99_ttft=statistics.quantiles(ttfts, n=100)[98] if len(ttfts) > 1 else ttfts[0],
            
            avg_total_time=statistics.mean(total_times),
            median_total_time=statistics.median(total_times),
            p95_total_time=statistics.quantiles(total_times, n=20)[18] if len(total_times) > 1 else total_times[0],
            
            avg_tokens_per_second=statistics.mean(tokens_per_sec),
            median_tokens_per_second=statistics.median(tokens_per_sec),
            total_tokens_processed=total_tokens,
            
            total_duration=total_duration,
            requests_per_second=len(successful_metrics) / total_duration,
            tokens_per_second_aggregate=total_tokens / total_duration
        )
        
        return metrics_list, summary
    
    async def compare_backends(
        self,
        vllm_model: str,
        sglang_model: str,
        prompts: List[str],
        concurrent_requests: int = 1,
        max_tokens: int = 500,
        output_file: str = "benchmark_results.json"
    ):
        """Compare vLLM and SGLang performance"""
        
        print("\n" + "="*80)
        print("VLLM vs SGLANG BENCHMARK")
        print("="*80)
        print(f"Test prompts: {len(prompts)}")
        print(f"Concurrency: {concurrent_requests}")
        print()
        
        # Benchmark vLLM
        vllm_metrics, vllm_summary = await self.run_benchmark(
            "vllm", vllm_model, prompts, concurrent_requests, max_tokens
        )
        
        # Benchmark SGLang
        sglang_metrics, sglang_summary = await self.run_benchmark(
            "sglang", sglang_model, prompts, concurrent_requests, max_tokens
        )
        
        # Display comparison
        self.display_comparison(vllm_summary, sglang_summary)
        
        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "prompts_count": len(prompts),
                "concurrent_requests": concurrent_requests,
                "max_tokens": max_tokens,
                "vllm_model": vllm_model,
                "sglang_model": sglang_model
            },
            "vllm": {
                "summary": vllm_summary.to_dict() if vllm_summary else None,
                "metrics": [asdict(m) for m in vllm_metrics]
            },
            "sglang": {
                "summary": sglang_summary.to_dict() if sglang_summary else None,
                "metrics": [asdict(m) for m in sglang_metrics]
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")
        
        return results
    
    def display_comparison(self, vllm: BenchmarkSummary, sglang: BenchmarkSummary):
        """Display side-by-side comparison"""
        
        print("\n" + "="*80)
        print("BENCHMARK RESULTS COMPARISON")
        print("="*80)
        
        if not vllm or not sglang:
            print("❌ Unable to compare - one or both benchmarks failed")
            return
        
        print(f"\n{'Metric':<40} {'vLLM':<20} {'SGLang':<20} {'Winner':<10}")
        print("-" * 90)
        
        metrics = [
            ("Success Rate", 
             f"{vllm.successful_requests}/{vllm.total_requests}",
             f"{sglang.successful_requests}/{sglang.total_requests}",
             "vLLM" if vllm.successful_requests > sglang.successful_requests else "SGLang"),
            
            ("Avg TTFT (ms)", 
             f"{vllm.avg_ttft:.2f}",
             f"{sglang.avg_ttft:.2f}",
             "vLLM" if vllm.avg_ttft < sglang.avg_ttft else "SGLang"),
            
            ("P95 TTFT (ms)", 
             f"{vllm.p95_ttft:.2f}",
             f"{sglang.p95_ttft:.2f}",
             "vLLM" if vllm.p95_ttft < sglang.p95_ttft else "SGLang"),
            
            ("Avg Total Time (ms)", 
             f"{vllm.avg_total_time:.2f}",
             f"{sglang.avg_total_time:.2f}",
             "vLLM" if vllm.avg_total_time < sglang.avg_total_time else "SGLang"),
            
            ("Avg Tokens/sec", 
             f"{vllm.avg_tokens_per_second:.2f}",
             f"{sglang.avg_tokens_per_second:.2f}",
             "vLLM" if vllm.avg_tokens_per_second > sglang.avg_tokens_per_second else "SGLang"),
            
            ("Requests/sec", 
             f"{vllm.requests_per_second:.2f}",
             f"{sglang.requests_per_second:.2f}",
             "vLLM" if vllm.requests_per_second > sglang.requests_per_second else "SGLang"),
            
            ("Total Throughput (tokens/sec)", 
             f"{vllm.tokens_per_second_aggregate:.2f}",
             f"{sglang.tokens_per_second_aggregate:.2f}",
             "vLLM" if vllm.tokens_per_second_aggregate > sglang.tokens_per_second_aggregate else "SGLang"),
            
            ("Total Duration (sec)", 
             f"{vllm.total_duration:.2f}",
             f"{sglang.total_duration:.2f}",
             "vLLM" if vllm.total_duration < sglang.total_duration else "SGLang"),
        ]
        
        for metric, v_val, s_val, winner in metrics:
            winner_mark = "🏆" if winner else ""
            print(f"{metric:<40} {v_val:<20} {s_val:<20} {winner} {winner_mark}")
        
        print("\n" + "="*80)
        
        # Overall winner
        vllm_wins = sum(1 for _, _, _, w in metrics if w == "vLLM")
        sglang_wins = sum(1 for _, _, _, w in metrics if w == "SGLang")
        
        if vllm_wins > sglang_wins:
            print(f"🏆 Overall Winner: vLLM ({vllm_wins}/{len(metrics)} metrics)")
        elif sglang_wins > vllm_wins:
            print(f"🏆 Overall Winner: SGLang ({sglang_wins}/{len(metrics)} metrics)")
        else:
            print(f"🤝 Tie: Both performed equally ({vllm_wins}/{len(metrics)} metrics each)")
        
        print("="*80)


async def main():
    """Main benchmark execution"""
    
    # Test prompts (from OSINT pipeline)
    prompts = [
        "Classify this geopolitical event: 'Chinese warships conduct exercises near Taiwan'. Category, confidence, severity, and key actors in JSON format.",
        "Analyze escalation probability for this situation: Military tensions in South China Sea. Provide probability, confidence, alert level, and reasoning in JSON.",
        "Classify: 'Terrorist attack in Paris, multiple casualties'. Return JSON with category, confidence, severity, actors.",
        "Assess escalation risk: Border conflict between India and Pakistan escalates. JSON format with probability, alert level, risk factors.",
        "Classify event: 'Cyberattack targets government infrastructure in Estonia'. JSON output required.",
        "Analyze: Russian military buildup on Ukraine border. Escalation probability and alert level in JSON.",
        "Classify: 'Humanitarian crisis in Gaza as aid access blocked'. Category, severity, key actors as JSON.",
        "Escalation analysis: Nuclear facility incident in Iran raises tensions. JSON with probability and risk factors.",
        "Classify: 'Mass protests in Hong Kong over new security law'. JSON format.",
        "Assess: US-China trade tensions reach critical point. Escalation probability in JSON.",
    ]
    
    # Configuration
    benchmark = LLMBenchmark(
        vllm_url="http://localhost:8000",
        sglang_url="http://localhost:8001"
    )
    
    # Run comparison
    results = await benchmark.compare_backends(
        vllm_model="meta-llama/Llama-3.1-8B-Instruct",  # Adjust to your model
        sglang_model="meta-llama/Llama-3.1-8B-Instruct",  # Same model for fair comparison
        prompts=prompts,
        concurrent_requests=5,  # Test with concurrency
        max_tokens=500,
        output_file="vllm_vs_sglang_results.json"
    )
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    asyncio.run(main())