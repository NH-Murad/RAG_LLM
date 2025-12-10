# 🚀 Distributed RAG with Hallucination Detection

A comprehensive Python system for investigating how **Retrieval-Augmented Generation (RAG)** mitigates hallucinations in Large Language Models using **LettuceDetect** and other state-of-the-art hallucination detection frameworks.

## 🎯 Project Goals

This system allows you to:

✅ **Compare Approaches**: Generate answers WITH and WITHOUT RAG simultaneously
✅ **Detect Hallucinations**: Use LettuceDetect (79.22% F1 score) to identify false claims
✅ **Measure Improvement**: Calculate exact percentage reduction in hallucinations
✅ **Analyze Patterns**: Understand how RAG improves factual grounding
✅ **Export Results**: Save experiments as CSV for further analysis
✅ **Lightweight**: Runs on Windows i5 CPU without heavy GPU requirements

## 📋 System Requirements

- **OS**: Windows 10/11
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 15GB for models
- **CPU**: i5+ (multi-core preferred)
- **GPU**: Optional (CPU mode fully supported)
- **Internet**: Required for first-time model downloads

## ⚡ Quick Start (5 minutes)

### 1️⃣ Install Prerequisites

```bash
# Clone or download this repository
cd rag_hallucination_detection

# Create virtual environment
python -m venv rag_env
rag_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Install Ollama

- Download from [https://ollama.ai](https://ollama.ai)
- Install and start the service
- Pull models:
  ```bash
  ollama pull phi      # Fastest (2.7B parameters)
  ollama pull mistral  # Better quality (7B parameters)
  ```
- Verify: `ollama serve` should show it running on http://localhost:11434

### 3️⃣ Run the Application

```bash
python main.py
```

The application will launch at **http://localhost:7860** in your browser.

## 📊 How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│          User Input: Question + Documents                  │
└────────┬────────────────────────────────────────────────────┘
         │
    ┌────┴─────────────────────────────────────┐
    │                                          │
    ▼                                          ▼
┌──────────────────┐                   ┌──────────────────┐
│  WITHOUT RAG     │                   │   WITH RAG       │
│ (Baseline)       │                   │  (Grounded)      │
│                  │                   │                  │
│ LLM generates    │                   │ 1. Retrieve docs │
│ answer from      │                   │ 2. Inject context│
│ knowledge        │                   │ 3. Generate      │
└────────┬─────────┘                   └────────┬─────────┘
         │                                      │
         └──────────────────┬───────────────────┘
                            │
                 ┌──────────▼──────────┐
                 │ LettuceDetect       │
                 │ Hallucination       │
                 │ Detector            │
                 └──────────┬──────────┘
                            │
            ┌───────────────┴────────────────┐
            │                                │
      ▼─────────────▼                  ▼─────────────▼
   Without RAG                        With RAG
   Hallucination: 45%                Hallucination: 15%
   
            ▼──────────────────────────────────▼
            │  Comparison & Analysis          │
            │  Reduction: 66.7% ✓             │
            └────────────────────────────────┘
```

### Workflow

1. **Load Documents**: Upload your knowledge base documents
2. **Ask Question**: Enter a query about the documents
3. **Generate Without RAG**: LLM answers from general knowledge
4. **Generate With RAG**: LLM answers grounded in retrieved context
5. **Detect Hallucinations**: LettuceDetect identifies false claims in both
6. **Compare**: View metrics and reduction percentages
7. **Export**: Save results for analysis

## 🎮 Using the Interface

### Tab 1: 📊 Analysis Pipeline
- **Input Section**:
  - Enter your question
  - Paste documents (optional - can be pre-loaded)
  - Click "Run Full Pipeline"

- **Results Section**:
  - Without RAG answer
  - With RAG answer
  - Retrieved context
  - Comparison metrics
  - Detailed report

### Tab 2: 📁 Document Management
- Load documents by name and content
- Documents persist for multiple queries
- View document statistics

### Tab 3: 📊 Statistics
- View aggregate statistics from all experiments
- Average hallucination reduction across all queries
- Export all results as CSV

### Tab 4: ❓ Help
- Detailed usage guide
- Key metrics explanation
- Troubleshooting tips

## 📈 Understanding Results

### Hallucination Score
- **0.0 - 0.3**: Low hallucination (SAFE)
- **0.3 - 0.7**: Moderate hallucination (CAUTION)
- **0.7 - 1.0**: High hallucination (HIGH RISK)

### Reduction Percentage
- **>50%**: SIGNIFICANT improvement with RAG ✅
- **25-50%**: MODERATE improvement with RAG ⚠️
- **<25%**: MINIMAL improvement with RAG ❌

### Confidence Score
- Shows how confident the detector is (higher = more reliable)
- LettuceDetect: 0.79 F1 score benchmark

## 💾 Exporting Results

Export automatically includes:
- Query and answers
- Hallucination scores (both approaches)
- Retrieved context documents
- Reduction percentage
- Timestamp

**CSV Format:**
```csv
query,without_rag_answer,with_rag_answer,without_rag_hallucin_score,with_rag_hallucin_score,reduction_pct,timestamp
"What is quantum computing?","...",..","0.45","0.15","66.7","2024-01-15T10:30:00"
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# Model Selection
ollama_model: phi          # phi (fast) or mistral (better)
embedding_model: all-MiniLM-L6-v2  # Lightweight embeddings

# RAG Settings
chunk_size: 256            # Document chunk size
top_k_retrieval: 3        # Documents to retrieve
similarity_threshold: 0.3

# Performance
use_gpu: false            # CPU mode for Windows i5
num_threads: 4
```

## 🔧 Troubleshooting

### "Cannot connect to Ollama"
```bash
# Make sure Ollama is running
ollama serve
# Should show: "Listening on http://127.0.0.1:11434"
```

### "Out of memory" errors
```yaml
# In config.yaml, use faster model
ollama_model: phi          # Instead of mistral
# OR reduce document size and chunk count
```

### "Slow inference speed"
- Use `phi` instead of `mistral` model
- Reduce `top_k_retrieval` from 3 to 2
- Ensure Ollama is optimized in settings

### LettuceDetect import error
```bash
pip install lettucedetect --upgrade
# If still failing, fallback Luna model auto-activates
```

## 📚 Sample Test Cases

Try these questions with the included sample documents:

### Science Domain
**Document**: About Quantum Computing
- Q: "What is quantum computing?"
- Q: "Explain superposition"
- Q: "What are applications of quantum computers?"

### Geography Domain
**Document**: About Denmark
- Q: "What is the capital of Denmark?"
- Q: "What is Denmark's population?"
- Q: "What is Denmark's climate like?"

## 🔬 Research Applications

Perfect for:

✅ Evaluating RAG effectiveness for your domain
✅ Benchmarking hallucination detection models
✅ Building hallucination-aware LLM systems
✅ Developing grounded QA systems
✅ Cybersecurity threat intelligence automation
✅ Verifying AI-generated content

## 📊 Expected Performance

On Windows i5 with 8GB RAM:

| Task | Speed | Accuracy |
|------|-------|----------|
| Embedding generation | 30-50 docs/sec | N/A |
| LLM inference | 5-15 tokens/sec | ~90% (phi) |
| Hallucination detection | 30-60 examples/sec | 79.22% (LettuceDetect F1) |
| Full pipeline | 5-10 seconds/query | Varies by content |

## 🎓 Key Learnings

After running experiments, you'll understand:

1. **How much RAG helps**: Real percentage improvements for your data
2. **Hallucination patterns**: Where and how models make false claims
3. **Detection reliability**: How confident hallucination detectors are
4. **Grounding effectiveness**: How well documents prevent false claims
5. **Model trade-offs**: Speed vs. quality considerations

## 📖 Connecting to Your GitHub Project

This system builds on your existing research:
- Integrates LettuceDetect (the framework from your paper references)
- Implements distributed detection (lightweight models)
- Tests hallucination reduction empirically
- Provides reproducible benchmarks

To integrate with your existing code:
```python
from main import RAGHallucinationApp

app = RAGHallucinationApp()
# Use your documents
app.load_documents(['doc1.txt', 'doc2.pdf'])
# Run analysis
results = app.run_full_pipeline(query="Your question")
```

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Areas to enhance:
- Additional hallucination detectors
- Multi-language support
- Advanced visualization
- API endpoints
- Advanced RAG techniques (multi-hop, semantic routing)

## 📧 Support

- Check the Help tab in the application
- Review troubleshooting section
- Check logs in `logs/rag_system.log`

## 🚀 Next Steps

1. ✅ Run quick start guide
2. ✅ Load sample documents
3. ✅ Test with sample questions
4. ✅ Upload your own documents
5. ✅ Run full experiments
6. ✅ Export and analyze results
7. ✅ Fine-tune based on findings

---

**Happy researching! 🧪**

For cybersecurity research and hallucination detection excellence.
