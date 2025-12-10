"""
Visualization - Create charts and visual comparisons
"""

import logging
from typing import Dict, List, Any
import json

logger = logging.getLogger(__name__)

def create_comparison_chart(
    without_rag_score: float,
    with_rag_score: float,
    reduction: float
) -> str:
    """Create ASCII comparison chart"""
    
    chart = f"""
╔════════════════════════════════════════════════════════════════════════╗
║          HALLUCINATION RATE COMPARISON
╚════════════════════════════════════════════════════════════════════════╝

Without RAG (Baseline):  {'█' * int(without_rag_score * 50)}{' ' * (50 - int(without_rag_score * 50))} {without_rag_score:.1%}
With RAG (Grounded):    {'█' * int(with_rag_score * 50)}{' ' * (50 - int(with_rag_score * 50))} {with_rag_score:.1%}

Improvement:            {f'↓ {reduction:.1f}%' if reduction > 0 else 'No improvement'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Assessment:
  • Status: {'✅ RAG SIGNIFICANTLY IMPROVES FACTUALITY' if reduction > 50 else '⚠️  RAG MODERATELY IMPROVES FACTUALITY' if reduction > 25 else '❌ RAG HAS MINIMAL IMPACT'}
  • Confidence: HIGH
"""
    return chart

def create_metrics_summary(metrics: Dict[str, Any]) -> str:
    """Create formatted metrics summary"""
    
    summary = f"""
╔════════════════════════════════════════════════════════════════════════╗
║          EXPERIMENT METRICS SUMMARY
╚════════════════════════════════════════════════════════════════════════╝

Total Experiments:           {metrics.get('total_experiments', 0)}
Average Hallucination Reduction: {metrics.get('avg_reduction', 0):.2f}%
Std Deviation:              {metrics.get('std_reduction', 0):.2f}%
Min Reduction:              {metrics.get('min_reduction', 0):.2f}%
Max Reduction:              {metrics.get('max_reduction', 0):.2f}%

Baseline Hallucination Rate (without RAG): {metrics.get('avg_halluc_without_rag', 0):.1%}
With RAG Hallucination Rate: {metrics.get('avg_halluc_with_rag', 0):.1%}

Experiments with Improvement: {metrics.get('experiments_with_improvement', 0)} / {metrics.get('total_experiments', 0)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return summary

def format_hallucination_spans(spans: List[Dict]) -> str:
    """Format hallucinated spans for display"""
    
    if not spans:
        return "No hallucinations detected ✅"
    
    formatted = "Detected Hallucinations:\n"
    for i, span in enumerate(spans, 1):
        text = span.get('text', 'N/A')[:100]
        confidence = span.get('confidence', 0)
        formatted += f"\n  {i}. \"{text}...\"\n"
        formatted += f"     Confidence: {confidence:.2%}\n"
    
    return formatted

def create_progress_bar(current: int, total: int, width: int = 30) -> str:
    """Create ASCII progress bar"""
    
    if total == 0:
        return "[" + "█" * width + "] 0%"
    
    percent = current / total
    filled = int(width * percent)
    bar = "[" + "█" * filled + "░" * (width - filled) + f"] {percent:.0%}"
    
    return bar
