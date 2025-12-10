# 🚀 Distributed RAG Hallucination Detection System - START HERE

Welcome! This document guides you through getting your complete, production-ready RAG system running.

## 📋 What You Have

A complete Python application for investigating **how Retrieval-Augmented Generation (RAG) reduces hallucinations** in Large Language Models, with integrated **LettuceDetect** hallucination detection.

**Key capabilities:**
- Generate answers WITHOUT RAG (baseline)
- Generate answers WITH RAG (grounded)
- Detect hallucinations in both using LettuceDetect
- Compare hallucination rates
- Export results for analysis

## ⚡ Getting Started (5 Minutes)

### 1. Download & Extract
Extract all files to a folder like `C:\Users\YourName\rag_project`

### 2. Open Command Prompt
```bash
cd C:\Users\YourName\rag_project
```

### 3. Run Setup
```bash
python setup.py
```

**This will automatically:**
- Create folder structure
- Download and install all dependencies (5-10 minutes)
- Check your system
- Create sample documents

### 4. Ensure Ollama is Running

Option A: If Ollama not yet installed
```bash
# Download from https://ollama.ai
# Install and run: ollama serve
```

Option B: If already installed
```bash
# Just ensure it's running
ollama serve
# Should show: "Listening on 127.0.0.1:11434"
```

### 5. Launch Application
```bash
python main.py
```

Opens at: **http://localhost:7860**

## 📖 What to Read

Read these in order:

1. **THIS FILE** (you're reading it!) - Overview
2. **README.md** - Complete documentation & usage guide
3. **IMPLEMENTATION_SUMMARY.md** - Technical details & architecture
4. **RAG_Halluc_Guide.md** - Setup troubleshooting

## 🎯 Your First Experiment (10 Minutes)

Once app is running:

1. **Go to "Document Management" tab**
   - Paste or upload a document
   - Click "Load Document"

2. **Go to "Analysis Pipeline" tab**
   - Type a question about your document
   - Click "Run Full Pipeline"

3. **See Results**
   - Compare WITHOUT RAG vs WITH RAG answers
   - View hallucination detection results
   - See reduction percentage

4. **View Statistics**
   - Go to "Statistics" tab
   - See aggregate results
   - Export as CSV

## 🔧 System Architecture

```
You ask a question
    ↓
WITHOUT RAG Path: LLM answers from knowledge
WITH RAG Path: LLM answers grounded in documents
    ↓
LettuceDetect: Detects hallucinations in both
    ↓
Comparison Engine: Calculates reduction
    ↓
Results: Shows improvement percentage
```

## 📂 Important Files

| File | Purpose | When to Edit |
|------|---------|-------------|
| **main.py** | Main application | Advanced customization |
| **config.yaml** | Settings | Change model, chunk size, etc. |
| **src/rag_system.py** | RAG pipeline | To customize retrieval |
| **src/hallucination_detector.py** | Detection | To use different detector |
| **README.md** | Documentation | Reference while using |

## 💾 Where Are My Results?

- **Results CSV**: `data/results/`
- **Logs**: `logs/rag_system.log`
- **Documents**: `data/documents/`
- **Cache**: `cache/`

## 🆘 Troubleshooting

### "Cannot connect to Ollama"
```bash
# Start Ollama service
ollama serve
```

### "Out of memory"
Edit `config.yaml`:
```yaml
ollama_model: phi  # Smaller model
```

### "Very slow"
Edit `config.yaml`:
```yaml
chunk_size: 128      # Smaller chunks
top_k_retrieval: 2   # Fewer documents
```

### Python/Package Errors
```bash
# Reinstall everything
python setup.py
```

## 🎮 Using the Interface

### Tab 1: Analysis Pipeline ✅ Main feature
1. Enter question
2. Paste documents (optional)
3. Click "Run Full Pipeline"
4. View results

### Tab 2: Document Management
Load documents that persist across multiple queries

### Tab 3: Statistics
View all results and export to CSV

### Tab 4: Help
Built-in documentation and tips

## 📊 Understanding Results

**Hallucination Score:**
- 0.0-0.3 = Safe ✅
- 0.3-0.7 = Caution ⚠️
- 0.7-1.0 = High Risk ❌

**Reduction:**
- >50% = RAG significantly helps ✅
- 25-50% = RAG somewhat helps ⚠️
- <25% = RAG minimally helps ❌

## 🚀 Recommended Workflow

**Day 1: Setup & Testing**
```
1. Run setup.py
2. Run main.py
3. Test with sample documents
4. Ask 5 test questions
```

**Day 2: Gather Data**
```
1. Upload your documents
2. Create 20 diverse questions
3. Run through system
4. Export results
```

**Day 3: Analysis**
```
1. Load CSV in Excel/Python
2. Analyze hallucination reduction patterns
3. Create visualizations
4. Document findings
```

## 💻 System Requirements Reminder

- Windows 10/11
- 8GB RAM minimum
- 15GB disk space
- Internet (for first download)
- i5 CPU (what you have!)

## 🎓 What You'll Learn

Using this system, you'll understand:

✅ How RAG actually reduces hallucinations (with real numbers)
✅ Where hallucinations occur (token-level)
✅ How to measure detector confidence
✅ Speed/quality trade-offs
✅ How to benchmark hallucination detection
✅ How to build grounded QA systems

## 🔗 Connecting to Your Research

Your GitHub project: `Hallucinations-in-RAG-Systems`

This system implements:
- ✅ LettuceDetect detector (79.22% F1)
- ✅ RAG-Sequence architecture
- ✅ Distributed (lightweight) detection
- ✅ Empirical measurement methodology
- ✅ Reproducible benchmarks

You can use this as:
1. **Standalone tool** for your research
2. **Foundation** for more advanced work
3. **Benchmarking** platform for detectors
4. **Data generation** for your papers

## 📞 Need Help?

1. **Check README.md** - Comprehensive documentation
2. **Check Help tab** - In the application
3. **Check logs** - `logs/rag_system.log`
4. **Check IMPLEMENTATION_SUMMARY.md** - Technical details

## 🎉 Next Step

Ready to start? Type in command prompt:

```bash
python setup.py
```

Then:

```bash
python main.py
```

Then open browser to: **http://localhost:7860**

---

**Happy research! 🧪**

Built for your cybersecurity research needs.
Questions? See README.md or Help tab in application.
