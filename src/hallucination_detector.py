"""
Hallucination Detector - Using LettuceDetect
Detects and evaluates hallucinations in LLM outputs
"""

import logging
from typing import Dict, List, Optional, Any
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """Detects hallucinations using LettuceDetect or Luna models"""

    def __init__(self, model_name: str = "lettucedetect"):
        self.model_name = model_name
        self.device = "cpu"  # Using CPU for your Windows i5

        # Initialize models based on selection
        if model_name.lower() == "lettucedetect":
            self._init_lettucedetect()
        else:
            self._init_luna()

    def _init_lettucedetect(self):
        """Initialize LettuceDetect model"""
        try:
            logger.info("Loading LettuceDetect model...")

            # Official LettuceDetect usage (English ModernBERT base model)
            # Ref: docs / README
            from lettucedetect.models.inference import HallucinationDetector as LD

            self.detector = LD(
                method="transformer",
                model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1",
            )
            logger.info("✅ LettuceDetect loaded successfully")

        except ImportError:
            logger.warning("LettuceDetect not installed, using Luna fallback")
            self._init_luna()
        except Exception as e:
            logger.error(f"LettuceDetect initialization error: {e}")
            logger.info("Falling back to Luna detector")
            self._init_luna()

    def _init_luna(self):
        """Initialize Luna model as fallback"""
        try:
            logger.info("Loading Luna (DeBERTA) model...")
            model_name = "MoritzLaurer/luna"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("✅ Luna model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Luna model: {e}")
            logger.info("Using simple keyword-based detector as fallback")

    def detect(
        self,
        context: str,
        question: str,
        answer: str,
    ) -> Dict[str, Any]:
        """Detect hallucinations in answer given context and question"""
        try:
            # Use LettuceDetect if available
            if hasattr(self, "detector"):
                return self._detect_with_lettucedetect(context, question, answer)
            else:
                return self._detect_with_luna(context, question, answer)
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return self._fallback_detection(context, answer)

    def _detect_with_lettucedetect(
        self,
        context: str,
        question: str,
        answer: str,
    ) -> Dict[str, Any]:
        """Detect using LettuceDetect"""
        try:
            predictions = self.detector.predict(
                context=[context],
                question=question,
                answer=answer,
                output_format="spans",
            )

            hallucination_count = len(predictions) if predictions else 0
            answer_length = len(answer.split())
            hallucination_ratio = hallucination_count / max(answer_length, 1)

            return {
                "method": "lettucedetect",
                "hallucination_score": hallucination_ratio,
                "confidence": 0.79,  # LettuceDetect F1 score (approx, doc value)
                "hallucinated_spans": predictions or [],
                "hallucination_count": hallucination_count,
                "total_tokens": answer_length,
                "is_hallucinated": hallucination_ratio > 0.3,
            }
        except Exception as e:
            logger.error(f"LettuceDetect error: {e}")
            return self._fallback_detection(context, answer)

    def _detect_with_luna(
        self,
        context: str,
        question: str,
        answer: str,
    ) -> Dict[str, Any]:
        """Detect using Luna (DeBERTA) model"""
        try:
            # Prepare input
            input_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer: {answer}"

            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Calculate confidence
            predictions = logits.argmax(dim=-1)
            confidence = (
                torch.softmax(logits, dim=-1).max(dim=-1).values.mean().item()
            )

            # Count hallucinated tokens
            hallucinated_tokens = (predictions == 1).sum().item()
            total_tokens = predictions.shape[1]
            hallucination_score = hallucinated_tokens / max(total_tokens, 1)

            return {
                "method": "luna",
                "hallucination_score": hallucination_score,
                "confidence": confidence,
                "hallucinated_tokens": hallucinated_tokens,
                "total_tokens": total_tokens,
                "is_hallucinated": hallucination_score > 0.3,
            }
        except Exception as e:
            logger.error(f"Luna error: {e}")
            return self._fallback_detection(context, answer)

    def _fallback_detection(self, context: str, answer: str) -> Dict[str, Any]:
        """Simple keyword-based fallback detection"""
        try:
            context_words = set(context.lower().split())
            answer_words = answer.lower().split()

            unsupported_count = sum(
                1
                for word in answer_words
                if word.isalpha() and word not in context_words
            )

            hallucination_score = unsupported_count / max(len(answer_words), 1)

            return {
                "method": "keyword_fallback",
                "hallucination_score": min(hallucination_score, 1.0),
                "confidence": 0.5,
                "unsupported_words": unsupported_count,
                "total_words": len(answer_words),
                "is_hallucinated": hallucination_score > 0.3,
                "note": "Simple keyword-based fallback detection",
            }
        except Exception as e:
            logger.error(f"Fallback detection error: {e}")
            return {
                "method": "error",
                "hallucination_score": 0.5,
                "confidence": 0.0,
                "error": str(e),
            }

    def batch_detect(
        self,
        contexts: List[str],
        questions: List[str],
        answers: List[str],
    ) -> List[Dict[str, Any]]:
        """Detect hallucinations for multiple examples"""
        results = []
        for context, question, answer in zip(contexts, questions, answers):
            result = self.detect(context, question, answer)
            results.append(result)
        return results

    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "available_methods": ["lettucedetect", "luna", "fallback"],
            "description": "Hallucination detection for RAG systems",
        }
