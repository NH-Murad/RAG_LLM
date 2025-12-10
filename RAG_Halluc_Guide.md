# Distributed RAG with Hallucination Detection - Setup Guide

## Project Overview
This application investigates how Distributed RAG mitigates hallucinations in LLMs by:
1. Generating answers WITHOUT RAG (baseline)
2. Generating answers WITH RAG (grounded)
3. Detecting hallucinations using LettuceDetect
4. Comparing hallucination rates between both approaches

## System Requirements
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 15GB for models
- **CPU**: i5 (multi-core benefits from parallelization)
- **OS**: Windows 10/11

## Installation Steps

### Step 1: Install Dependencies
```bash
# Create virtual environment
python -m venv rag_env
rag_env\Scripts\activate

# Install core packages
pip install ollama transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers faiss-cpu
pip install gradio pandas numpy matplotlib seaborn
pip install lettucedetect
pip install tqdm colorama
```

### Step 2: Install Ollama
- Download from https://ollama.ai
- Models needed:
  - `ollama pull phi:latest` (2.7B - fastest)
  - `ollama pull mistral:latest` (7B - better quality)
  - Start Ollama service (runs on localhost:11434)

### Step 3: Prepare Models
- LettuceDetect models auto-download on first use
- Sentence-Transformers auto-download on first use
- FAISS installs with CPU support

## Project Structure
```
rag_project/
├── main.py                          # Main Gradio application
├── config.yaml                      # Configuration file
├── src/
│   ├── rag_system.py               # RAG pipeline
│   ├── hallucination_detector.py   # Detection logic
│   ├── comparison_engine.py        # Comparison & metrics
│   └── data_handler.py             # Data management
├── data/
│   ├── documents/                  # Store your documents
│   └── results/                    # Experiment results
├── logs/                           # Application logs
└── utils/
    ├── visualization.py            # Plots & charts
    └── helpers.py                  # Utility functions
```

## How to Run

### Option 1: Gradio Web Interface (Recommended)
```bash
python main.py
# Opens at http://localhost:7860
```

### Option 2: Command Line
```bash
python src/rag_system.py --query "Your question here" --doc "document.txt"
```

## Usage Workflow

1. **Load Documents**: Upload PDFs, TXT files, or paste text
2. **Enter Question**: Ask a question about the documents
3. **Run Without RAG**: Get LLM response without context
4. **Run With RAG**: Get LLM response with retrieved context
5. **Detect Hallucinations**: LettuceDetect identifies false claims
6. **Compare Results**: View metrics and visualizations
7. **Export Report**: Save findings as PDF/CSV

## Key Metrics

- **Hallucination Rate**: % of hallucinated tokens in response
- **Confidence Score**: How confident is the detector
- **Context Relevance**: BLEU/ROUGE score vs retrieved docs
- **Reduction %**: How much RAG reduced hallucinations

## Expected Performance
- **Speed**: 2-5 seconds per query (CPU)
- **Accuracy**: 79.22% F1 score (LettuceDetect)
- **Memory**: ~4GB active during inference

## Troubleshooting

### Ollama Not Connecting
```bash
# Check if service is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

### Out of Memory
- Use smaller model (phi instead of mistral)
- Reduce batch size
- Use 4-bit quantization

### Slow Performance
- Run on CPU threading
- Reduce document chunk size
- Use smaller embedding model

## Next Steps
1. Test with provided examples
2. Upload your documents
3. Run A/B tests (with/without RAG)
4. Analyze hallucination patterns
5. Fine-tune model if needed
