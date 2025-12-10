# 🎯 QUICK REFERENCE CARD

## Command Reference

```bash
# 1. FIRST TIME ONLY: Setup everything
python setup.py

# 2. Start Ollama (keep running)
ollama serve

# 3. Run the application
python main.py

# 4. Stop application
Ctrl + C
```

## Files At a Glance

| File | What It Does |
|------|-------------|
| `main.py` | The app - run this! |
| `setup.py` | Setup wizard - run once |
| `config.yaml` | Settings (model, etc) |
| `requirements.txt` | Dependencies |
| `README.md` | Full documentation |
| `START_HERE.md` | Getting started |

## Key Keyboard Shortcuts

| Action | How |
|--------|-----|
| Run pipeline | Click button or Ctrl+Enter |
| Export results | Go to Statistics tab, click Export |
| View logs | Check `logs/rag_system.log` |

## Configuration Tweaks

### If system is slow:
```yaml
# In config.yaml
ollama_model: phi           # Use small model
chunk_size: 128             # Smaller chunks
top_k_retrieval: 2          # Fewer docs
```

### If out of memory:
```yaml
ollama_model: phi
num_threads: 2              # Use fewer threads
batch_size: 16              # Smaller batches
```

### For better quality (slower):
```yaml
ollama_model: mistral       # Better model
chunk_size: 512             # Larger chunks
top_k_retrieval: 5          # More docs
```

## Understanding Output

### Hallucination Score
```
0.0-0.3 = Good ✅
0.3-0.7 = OK ⚠️
0.7-1.0 = Bad ❌
```

### Reduction %
```
>50%   = RAG great ✅
25-50% = RAG good ⚠️
<25%   = RAG minimal ❌
```

## File Locations

```
Results:    data/results/*.csv
Logs:       logs/rag_system.log
Documents:  data/documents/
Config:     config.yaml
```

## Troubleshooting Flowchart

```
Problem?
├─ Can't connect to Ollama
│  └─ Run: ollama serve
├─ Out of memory
│  └─ Use phi model in config.yaml
├─ Very slow
│  └─ Reduce chunk_size in config.yaml
├─ Import error
│  └─ Run: python setup.py
└─ Other
   └─ Check: logs/rag_system.log
```

## First Run Checklist

- [ ] Extracted all files
- [ ] Opened command prompt in folder
- [ ] Ran `python setup.py`
- [ ] Started `ollama serve`
- [ ] Ran `python main.py`
- [ ] Opened http://localhost:7860
- [ ] Loaded a document
- [ ] Asked a question
- [ ] Got results!

## Key Concepts (60 seconds)

**RAG**: Using documents to ground LLM answers
**Hallucination**: LLM making up false facts
**LettuceDetect**: Detector that finds false facts
**Without RAG**: LLM answers from memory
**With RAG**: LLM answers from documents
**Reduction**: How much RAG helps (in %)

## Quick Workflow

```
1. Load document
2. Ask question
3. See 2 answers (with & without RAG)
4. See hallucination scores
5. See improvement %
6. Repeat or export
```

## Model Selection Guide

| Model | Speed | Quality | Use When |
|-------|-------|---------|----------|
| phi | ⚡⚡⚡ | ⭐⭐ | Testing, fast results |
| mistral | ⚡⚡ | ⭐⭐⭐⭐ | Final results, quality matters |

## Embedding Model Info

Using `all-MiniLM-L6-v2`:
- 33M parameters (very small)
- Fast (~50 docs/sec)
- High quality
- Works great with RAG

## Results CSV Columns

```
query             - Your question
without_rag       - Answer without documents
with_rag          - Answer with documents  
hallucin_without  - Score (0=good, 1=bad)
hallucin_with     - Score (0=good, 1=bad)
reduction_pct     - Improvement percentage
timestamp         - When experiment ran
```

## System Status Checks

```bash
# Check Python
python --version       # Should be 3.8+

# Check Ollama
curl http://localhost:11434/api/tags

# Check logs
cat logs/rag_system.log

# Check dependencies
pip list | grep torch
```

## Common Error Solutions

| Error | Solution |
|-------|----------|
| ModuleNotFoundError | `python setup.py` |
| Connection refused | `ollama serve` |
| Out of memory | Use `phi` model |
| No models in Ollama | `ollama pull phi` |
| Port 7860 in use | Change in main.py line |

## Environment Setup (One-time)

```bash
# Create virtual environment
python -m venv rag_env

# Activate it
rag_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# You're ready!
```

## Next 24 Hours

**Hour 1:** Setup & test with samples
**Hour 2:** Upload your documents
**Hour 3:** Run 20 queries
**Hour 4:** Export results
**Hour 5+:** Analyze & publish!

---

💡 **Pro Tip**: Keep a terminal with `ollama serve` running while using the app.

🔑 **Remember**: First run is setup.py, then main.py, then browser!

✅ **You're all set!** Happy analyzing! 🚀
