"""
Comparison Engine - Analyze differences between RAG and non-RAG outputs
"""

import logging
from typing import Dict, List, Any
import numpy as np
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class ComparisonEngine:
    """Compares RAG vs non-RAG generation outputs"""
    
    def __init__(self):
        self.metrics = {}
    
    def compare(
        self,
        without_rag_result: Dict[str, Any],
        with_rag_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare hallucination detection results"""
        
        try:
            # Extract scores
            without_rag_score = without_rag_result.get('hallucination_score', 0.5)
            with_rag_score = with_rag_result.get('hallucination_score', 0.5)
            
            # Calculate reduction
            if without_rag_score == 0:
                reduction = 0.0
            else:
                reduction = ((without_rag_score - with_rag_score) / without_rag_score) * 100
                reduction = max(0, min(reduction, 100))  # Clamp to [0, 100]
            
            comparison = {
                'without_rag_score': float(without_rag_score),
                'with_rag_score': float(with_rag_score),
                'hallucination_reduction': float(reduction),
                'improvement': 'Significant' if reduction > 50 else 'Moderate' if reduction > 25 else 'Minimal',
                'without_rag_result': without_rag_result,
                'with_rag_result': with_rag_result,
                'summary': self._generate_summary(without_rag_score, with_rag_score, reduction)
            }
            
            return comparison
        
        except Exception as e:
            logger.error(f"Comparison error: {e}")
            return {
                'error': str(e),
                'hallucination_reduction': 0.0
            }
    
    def _generate_summary(
        self,
        without_rag_score: float,
        with_rag_score: float,
        reduction: float
    ) -> str:
        """Generate human-readable summary"""
        
        summary = f"""
Hallucination Analysis Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Without RAG (Baseline):
  • Hallucination Score: {without_rag_score:.2%}
  • Status: {'HIGH RISK' if without_rag_score > 0.7 else 'MODERATE' if without_rag_score > 0.4 else 'LOW'}

With RAG (Grounded):
  • Hallucination Score: {with_rag_score:.2%}
  • Status: {'HIGH RISK' if with_rag_score > 0.7 else 'MODERATE' if with_rag_score > 0.4 else 'LOW'}

Improvement:
  • Reduction: {reduction:.2f}%
  • Assessment: {'SIGNIFICANT ✓' if reduction > 50 else 'MODERATE ✓' if reduction > 25 else 'MINIMAL'}

Key Insight:
  {"RAG successfully grounded responses in retrieved documents, significantly reducing hallucinations." if reduction > 50 else "RAG provided some improvement in response grounding." if reduction > 25 else "RAG had minimal impact on hallucination rate."}
"""
        return summary
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (simple method)"""
        similarity = SequenceMatcher(None, text1, text2).ratio()
        return similarity
    
    def analyze_context_usage(
        self,
        context: List[str],
        answer: str
    ) -> Dict[str, Any]:
        """Analyze how well answer uses retrieved context"""
        
        if not context or not answer:
            return {
                'context_coverage': 0.0,
                'relevance': 'No context provided'
            }
        
        try:
            context_str = ' '.join(context).lower()
            answer_lower = answer.lower()
            
            # Calculate word overlap
            context_words = set(context_str.split())
            answer_words = set(answer_lower.split())
            overlap = context_words.intersection(answer_words)
            
            coverage = len(overlap) / max(len(answer_words), 1)
            
            return {
                'context_coverage': float(coverage),
                'overlapping_terms': len(overlap),
                'answer_terms': len(answer_words),
                'relevance': 'High' if coverage > 0.5 else 'Medium' if coverage > 0.25 else 'Low'
            }
        
        except Exception as e:
            logger.error(f"Context analysis error: {e}")
            return {
                'context_coverage': 0.0,
                'error': str(e)
            }
    
    def calculate_metrics_batch(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate aggregate metrics from multiple comparisons"""
        
        if not results:
            return {}
        
        reductions = [r.get('hallucination_reduction', 0) for r in results]
        scores_without = [r.get('without_rag_score', 0) for r in results]
        scores_with = [r.get('with_rag_score', 0) for r in results]
        
        return {
            'total_experiments': len(results),
            'avg_reduction': float(np.mean(reductions)),
            'std_reduction': float(np.std(reductions)),
            'min_reduction': float(np.min(reductions)),
            'max_reduction': float(np.max(reductions)),
            'avg_halluc_without_rag': float(np.mean(scores_without)),
            'avg_halluc_with_rag': float(np.mean(scores_with)),
            'experiments_with_improvement': sum(1 for r in reductions if r > 0)
        }
