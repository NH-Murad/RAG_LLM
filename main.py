"""
Distributed RAG System with Hallucination Detection
Main Gradio Application
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict

import gradio as gr
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import custom modules
try:
    from src.rag_system import RAGSystem
    from src.hallucination_detector import HallucinationDetector
    from src.comparison_engine import ComparisonEngine
    from src.data_handler import DataHandler
except ImportError as e:
    logger.error(f"Custom modules not found: {e}. Make sure src/ and utils/ folders exist.")
    raise


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class RAGResult:
    """Data class for RAG results"""

    query: str
    without_rag_answer: str
    with_rag_answer: str
    retrieved_context: List[str]
    without_rag_hallucinations: Dict[str, Any]
    with_rag_hallucinations: Dict[str, Any]
    reduction_percentage: float
    execution_time: float
    timestamp: str


# =============================================================================
# Core Application Class
# =============================================================================


class RAGHallucinationApp:
    """Main application class handling the RAG system and Gradio interface"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.rag_system: RAGSystem = None
        self.hallucination_detector: HallucinationDetector = None
        self.comparison_engine: ComparisonEngine = None
        self.data_handler: DataHandler = None
        self.results_history: List[RAGResult] = []

        # Initialize sub-systems
        self.initialize_systems()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file or use defaults."""
        import yaml

        default_config = {
            "ollama_model": "phi",
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": 256,
            "chunk_overlap": 50,
            "top_k": 3,
            "temperature": 0.7,
            "max_length": 512,
        }

        config_file = Path(self.config_path)
        if not config_file.exists():
            logger.warning("config.yaml not found. Using default configuration.")
            return default_config

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if cfg:
                default_config.update(cfg)
            return default_config
        except Exception as e:
            logger.error(f"Error loading config.yaml: {e}")
            return default_config

    def initialize_systems(self) -> None:
        """Initialize RAG pipeline, hallucination detector, comparison engine, etc."""
        try:
            logger.info("Starting RAG Hallucination Detection Application...")

            logger.info("Initializing RAG System...")
            self.rag_system = RAGSystem(config=self.config)

            logger.info("Initializing Hallucination Detector...")
            self.hallucination_detector = HallucinationDetector(
                model_name="lettucedetect"
            )

            logger.info("Initializing Comparison Engine...")
            self.comparison_engine = ComparisonEngine()

            logger.info("Initializing Data Handler...")
            self.data_handler = DataHandler(data_dir="data")

            logger.info("All systems initialized successfully!")

        except Exception as e:
            logger.error(f"Initialization error: {e}", exc_info=True)
            raise

    # ------------------------------------------------------------------
    # RAG Generation
    # ------------------------------------------------------------------
    def generate_without_rag(self, query: str) -> str:
        """Generate answer without RAG (baseline)."""
        try:
            answer = self.rag_system.generate_without_rag(query)
            return answer
        except Exception as e:
            logger.error(f"Error generating without RAG: {e}")
            return f"Error: {str(e)}"

    def generate_with_rag(self, query: str) -> Tuple[str, List[str]]:
        """Generate answer with RAG (grounded)."""
        try:
            answer, context = self.rag_system.generate_with_rag(query)
            return answer, context
        except Exception as e:
            logger.error(f"Error generating with RAG: {e}")
            return f"Error: {str(e)}", []

    # ------------------------------------------------------------------
    # Hallucination Detection and Comparison
    # ------------------------------------------------------------------
    def detect_hallucinations(
        self,
        query: str,
        context_docs: List[str],
        without_rag_answer: str,
        with_rag_answer: str,
    ) -> Dict[str, Any]:
        """Run hallucination detection on both answers."""
        try:
            context_str = "\n\n".join(context_docs)

            without_rag_result = self.hallucination_detector.detect(
                context=context_str,
                question=query,
                answer=without_rag_answer,
            )

            with_rag_result = self.hallucination_detector.detect(
                context=context_str,
                question=query,
                answer=with_rag_answer,
            )

            comparison = self.comparison_engine.compare(
                without_rag_result=without_rag_result,
                with_rag_result=with_rag_result,
            )

            return {
                "without_rag": without_rag_result,
                "with_rag": with_rag_result,
                "comparison": comparison,
            }
        except Exception as e:
            logger.error(f"Error in hallucination detection: {e}")
            return {
                "error": str(e),
                "without_rag": {},
                "with_rag": {},
                "comparison": {},
            }

    # ------------------------------------------------------------------
    # Main Pipeline
    # ------------------------------------------------------------------
    def run_full_pipeline(
        self,
        query: str,
        documents_text: str = "",
    ) -> Tuple[str, str, str, str, str]:
        """
        Full pipeline:
        1. Optionally add temporary document
        2. Generate answers with and without RAG
        3. Detect hallucinations
        4. Compare and log results
        """
        import time

        start_time = time.time()

        try:
            logger.info(f"Running full pipeline for query: {query!r}")

            # Optionally add a temporary document from the text area
            temp_docs: List[str] = []
            if documents_text and documents_text.strip():
                self.data_handler.add_text_document(
                    "temp_input_doc",
                    documents_text,
                    metadata={"source": "user_input"},
                )
                temp_docs.append(documents_text)
                # Basic indexing for new docs
                self.rag_system.index_documents(temp_docs)

            # 1) Generate answers
            without_rag_answer = self.generate_without_rag(query)
            with_rag_answer, context_docs = self.generate_with_rag(query)

            # 2) Detect hallucinations
            detection_results = self.detect_hallucinations(
                query=query,
                context_docs=context_docs,
                without_rag_answer=without_rag_answer,
                with_rag_answer=with_rag_answer,
            )

            comparison_result = detection_results.get("comparison", {})
            context_str = "\n\n".join(context_docs) if context_docs else ""

            # 3) Generate report
            report = self._generate_report(
                query=query,
                without_rag=without_rag_answer,
                with_rag=with_rag_answer,
                context=context_docs,
                comparison=comparison_result,
            )

            execution_time = time.time() - start_time

            # 4) Store results
            result = RAGResult(
                query=query,
                without_rag_answer=without_rag_answer,
                with_rag_answer=with_rag_answer,
                retrieved_context=context_docs,
                without_rag_hallucinations=detection_results.get("without_rag", {}),
                with_rag_hallucinations=detection_results.get("with_rag", {}),
                reduction_percentage=comparison_result.get(
                    "hallucination_reduction", 0.0
                ),
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
            )
            self.results_history.append(result)

            return (
                without_rag_answer,
                with_rag_answer,
                context_str,
                json.dumps(comparison_result, indent=2),
                report,
            )

        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ("Error", "Error", error_msg, error_msg, error_msg)

    # ------------------------------------------------------------------
    # Reporting and Statistics
    # ------------------------------------------------------------------
    def _generate_report(
        self,
        query: str,
        without_rag: str,
        with_rag: str,
        context: List[str],
        comparison: Dict[str, Any],
    ) -> str:
        """Generate detailed text report."""
        comp = comparison or {}
        without_score = comp.get("without_rag_score", 0.0)
        with_score = comp.get("with_rag_score", 0.0)
        reduction = comp.get("hallucination_reduction", 0.0)
        improvement = comp.get("improvement", "N/A")
        summary = comp.get("summary", "")

        context_preview = (
            "\n\n".join(context[:3]) if context else "No context retrieved"
        )

        report = f"""
==================== RAG HALLUCINATION ANALYSIS REPORT ====================

Query:
{query}

---------------------- BASELINE (WITHOUT RAG) ----------------------
Answer:
{without_rag}

Hallucination Score: {without_score:.3f}

------------------------ WITH RAG (GROUNDED) ------------------------
Answer:
{with_rag}

Hallucination Score: {with_score:.3f}

--------------------------- COMPARISON -----------------------------
Reduction in Hallucination Score: {reduction:.2f}%
Improvement Category: {improvement}

---------------------- RETRIEVED CONTEXT ---------------------------
{context_preview}

---------------------- SUMMARY ----------------------------
{summary}

===================================================================
"""
        return report

    def get_statistics(self) -> str:
        """Get statistics of all experiments."""
        if not self.results_history:
            return "No results available yet. Run some queries first."

        reductions = [r.reduction_percentage for r in self.results_history]

        stats = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║           STATISTICS
╚════════════════════════════════════════════════════════════════════════════╝

Total Experiments: {len(self.results_history)}
Average Hallucination Reduction: {np.mean(reductions):.2f}%
Min Reduction: {np.min(reductions):.2f}%
Max Reduction: {np.max(reductions):.2f}%
Std Deviation: {np.std(reductions):.2f}%

"""
        return stats

    def export_results(self) -> str:
        """Export results to CSV."""
        if not self.results_history:
            return "No results to export."

        try:
            data = [asdict(r) for r in self.results_history]
            df = pd.DataFrame(data)

            results_dir = Path("data/results")
            results_dir.mkdir(parents=True, exist_ok=True)
            file_path = results_dir / f"rag_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(file_path, index=False, encoding="utf-8")

            return f"Results exported successfully to {file_path}"
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return f"Error exporting results: {e}"


# =============================================================================
# Gradio Interface
# =============================================================================


def create_gradio_interface(app: RAGHallucinationApp) -> gr.Blocks:
    """Create Gradio interface."""

    with gr.Blocks(title="RAG Hallucination Detection System") as interface:
        gr.Markdown("# 🚀 Distributed RAG with Hallucination Detection")
        gr.Markdown(
            "**Compare answer quality WITH and WITHOUT RAG | Detect & Measure Hallucinations**"
        )

        with gr.Tabs():
            # Tab 1: Main Pipeline
            with gr.Tab("📊 Analysis Pipeline"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📝 Input")
                        query = gr.Textbox(
                            label="Query",
                            placeholder="Ask a question...",
                            lines=3,
                        )
                        documents = gr.Textbox(
                            label="Documents (Paste or leave empty)",
                            placeholder="Paste your documents here...",
                            lines=5,
                        )
                        run_btn = gr.Button("🔍 Run Full Pipeline", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("### 📈 Results")
                        without_rag_output = gr.Textbox(
                            label="Without RAG (Baseline)",
                            interactive=False,
                            lines=4,
                        )
                        with_rag_output = gr.Textbox(
                            label="With RAG (Grounded)",
                            interactive=False,
                            lines=4,
                        )

                context_output = gr.Textbox(
                    label="Retrieved Context",
                    interactive=False,
                    lines=3,
                )

                with gr.Row():
                    comparison_output = gr.Textbox(
                        label="Comparison Results (JSON)",
                        interactive=False,
                        lines=5,
                    )
                    report_output = gr.Textbox(
                        label="Detailed Report",
                        interactive=False,
                        lines=5,
                    )

                run_btn.click(
                    fn=app.run_full_pipeline,
                    inputs=[query, documents],
                    outputs=[
                        without_rag_output,
                        with_rag_output,
                        context_output,
                        comparison_output,
                        report_output,
                    ],
                )

            # Tab 2: Document Management
            with gr.Tab("📁 Document Management"):
                gr.Markdown("### Load Documents")
                doc_text = gr.Textbox(
                    label="Document Content",
                    placeholder="Paste document text here...",
                    lines=10,
                )
                doc_name = gr.Textbox(
                    label="Document Name",
                    placeholder="Give your document a name...",
                )
                upload_btn = gr.Button("📤 Load Document", variant="primary")
                upload_status = gr.Textbox(label="Status", interactive=False)

                def load_doc(name, content):
                    if not name or not content:
                        return "Please provide both name and content"
                    success = app.data_handler.add_text_document(name, content)
                    app.rag_system.index_documents([content])
                    return (
                        "✅ Document loaded successfully!"
                        if success
                        else "❌ Error loading document"
                    )

                upload_btn.click(
                    fn=load_doc,
                    inputs=[doc_name, doc_text],
                    outputs=upload_status,
                )

            # Tab 3: Statistics
            with gr.Tab("📊 Statistics"):
                gr.Markdown("### Experiment Results")
                stats_output = gr.Textbox(
                    label="Statistics",
                    interactive=False,
                    lines=10,
                )
                refresh_btn = gr.Button("🔄 Refresh Stats", variant="primary")

                refresh_btn.click(
                    fn=app.get_statistics,
                    outputs=stats_output,
                )

                export_btn = gr.Button("💾 Export Results", variant="secondary")
                export_status = gr.Textbox(label="Export Status", interactive=False)

                export_btn.click(
                    fn=app.export_results,
                    outputs=export_status,
                )

            # Tab 4: Help
            with gr.Tab("❓ Help & Guide"):
                gr.Markdown(
                    """
## How to Use

### 1. **Load Documents**
   - Go to "Document Management" tab  
   - Paste your document content  
   - Give it a name and click "Load Document"

### 2. **Ask Questions**
   - Return to "Analysis Pipeline"  
   - Enter your question in the Query field  
   - Optionally paste more documents (will be added temporarily)

### 3. **Run Analysis**
   - Click "Run Full Pipeline"  
   - System will:
     - Generate answer WITHOUT context (baseline)
     - Generate answer WITH retrieved context (RAG)
     - Detect hallucinations in both answers
     - Compare hallucination rates

### 4. **Interpret Results**
   - **Without RAG**: Answer based only on model knowledge  
   - **With RAG**: Answer grounded in retrieved documents  
   - **Comparison**: Shows how much RAG reduces hallucinations

### 5. **Export & Analyze**
   - Go to "Statistics" tab  
   - View overall statistics  
   - Export all results as CSV
                    """
                )

                gr.Markdown(
                    """
---
**Distributed RAG Hallucination Detection System** | Built for Research
                    """
                )

    return interface


# =============================================================================
# Main entry point
# =============================================================================


def main() -> None:
    """Main entry point."""
    try:
        app = RAGHallucinationApp()
        interface = create_gradio_interface(app)
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True,
        )
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print("Please check the logs for more details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
