"""
Quick Start Helpers
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def verify_installation() -> bool:
    """Verify all dependencies are installed"""
    
    packages = [
        'torch',
        'transformers',
        'sentence-transformers',
        'faiss-cpu',
        'gradio',
        'pandas',
        'numpy',
        'requests',
        'pyyaml'
    ]
    
    missing = []
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.error(f"Install with: pip install {' '.join(missing)}")
        return False
    
    logger.info("✅ All dependencies installed successfully!")
    return True

def check_ollama_service() -> bool:
    """Check if Ollama service is running"""
    import requests
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        logger.error("❌ Ollama service not running")
        logger.error("Please start Ollama: ollama serve")
        return False

def create_sample_documents():
    """Create sample documents for testing"""
    
    sample_docs = {
        "sample_science.txt": """
Quantum Computing:
Quantum computing is a revolutionary technology that uses the principles of quantum mechanics to process information. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or 'qubits' that can exist in a superposition of states.

Key concepts:
1. Superposition: A qubit can be 0, 1, or both simultaneously until measured.
2. Entanglement: Qubits can be entangled, meaning the state of one qubit depends on another.
3. Interference: Quantum algorithms use interference to amplify correct answers and cancel out wrong ones.

Applications:
- Drug discovery and molecular simulation
- Optimization problems
- Cryptography and security
- Machine learning and AI

Current quantum computers are still in the NISQ (Noisy Intermediate-Scale Quantum) era, with 50-1000 qubits. IBM, Google, and other companies are working towards practical quantum advantage.
""",
        "sample_geography.txt": """
Denmark:
Denmark is a Nordic country in Northern Europe, part of the Kingdom of Denmark. It consists of the Jutland Peninsula and several islands.

Geography:
- Capital: Copenhagen
- Population: ~5.9 million
- Area: 43,094 km²
- Bordering countries: Germany to the south
- Major islands: Zealand (where Copenhagen is), Funen, and Bornholm

Climate:
Denmark has a temperate oceanic climate with mild winters and cool summers. The average temperature in January is around -1°C, and in July about 17°C.

Economy:
Denmark has a highly developed economy with:
- Strong manufacturing sector
- Renewable energy (wind power)
- Fishing and agriculture
- Technology and pharmaceuticals

Culture:
- The concept of "hygge" (coziness and warmth)
- Famous for LEGO, design, and cycling culture
- Copenhagen is known for its restaurants and innovative cuisine
"""
    }
    
    data_dir = Path("data/documents")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, content in sample_docs.items():
        file_path = data_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Created sample document: {filename}")
    
    return list(sample_docs.keys())

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "rag_system.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Logging initialized. Log file: {log_file}")
