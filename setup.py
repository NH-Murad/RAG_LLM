"""
Initialization Script - Setup and verify the RAG Hallucination Detection System
Run this first to set up everything
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemSetup:
    """Setup and initialization handler"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
    
    def verify_python_version(self):
        """Check Python version (3.8+ required)"""
        logger.info("🔍 Checking Python version...")
        
        if sys.version_info < (3, 8):
            logger.error(f"❌ Python 3.8+ required. Current: {sys.version}")
            return False
        
        logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} OK")
        return True
    
    def create_directory_structure(self):
        """Create all necessary directories"""
        logger.info("📁 Creating directory structure...")
        
        directories = [
            'data/documents',
            'data/results',
            'logs',
            'cache',
            'src',
            'utils'
        ]
        
        for d in directories:
            Path(d).mkdir(parents=True, exist_ok=True)
            logger.info(f"   ✓ {d}")
        
        logger.info("✅ Directory structure created")
        return True
    
    def verify_requirements(self):
        """Verify that requirements.txt exists"""
        logger.info("📋 Checking requirements.txt...")
        
        if not Path("requirements.txt").exists():
            logger.error("❌ requirements.txt not found")
            return False
        
        logger.info("✅ requirements.txt found")
        return True
    
    def install_dependencies(self):
        """Install dependencies from requirements.txt"""
        logger.info("📦 Installing dependencies (this may take 5-10 minutes)...")
        
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
                stdout=subprocess.DEVNULL
            )
            logger.info("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    
    def verify_imports(self):
        """Verify critical imports work"""
        logger.info("🔗 Verifying imports...")
        
        critical_packages = {
            'torch': 'PyTorch',
            'transformers': 'HuggingFace Transformers',
            'sentence_transformers': 'Sentence Transformers',
            'faiss': 'FAISS',
            'gradio': 'Gradio'
        }
        
        failed = []
        for package, name in critical_packages.items():
            try:
                __import__(package)
                logger.info(f"   ✓ {name}")
            except ImportError:
                logger.warning(f"   ✗ {name} - Installation may not have completed")
                failed.append(name)
        
        if failed:
            logger.warning(f"⚠️  Some packages failed to import: {', '.join(failed)}")
            return len(failed) < 3  # Allow some failures
        
        logger.info("✅ All critical imports verified")
        return True
    
    def check_ollama_connection(self):
        """Check if Ollama service is accessible"""
        logger.info("🔌 Checking Ollama connection...")
        
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                logger.info(f"   ✓ Ollama running with {len(models)} model(s)")
                
                if models:
                    logger.info("   Available models:")
                    for model in models:
                        name = model.get('name', 'unknown')
                        logger.info(f"      - {name}")
                    return True
                else:
                    logger.warning("   ⚠️  No models installed. Please run:")
                    logger.warning("      ollama pull phi")
                    logger.warning("      ollama pull mistral")
                    return False
            else:
                logger.error(f"❌ Ollama returned status {response.status_code}")
                return False
        
        except Exception as e:
            logger.error(f"❌ Cannot connect to Ollama at localhost:11434")
            logger.error("   Please ensure Ollama is installed and running:")
            logger.error("   1. Download from https://ollama.ai")
            logger.error("   2. Run: ollama serve")
            logger.error(f"   Details: {e}")
            return False
    
    def create_sample_documents(self):
        """Create sample documents for testing"""
        logger.info("📄 Creating sample documents...")
        
        samples = {
            "sample_science.txt": """Quantum Computing Basics:
Quantum computing represents a revolutionary paradigm in computation that harnesses the principles of quantum mechanics. Unlike classical computers that process bits (binary 0 or 1), quantum computers operate with quantum bits or 'qubits' that can exist in superposition.

Key Quantum Concepts:
1. Superposition: Qubits can exist as 0, 1, or both states simultaneously until measurement
2. Entanglement: Qubits can be interconnected such that the state of one depends on another
3. Interference: Quantum algorithms leverage interference to amplify correct solutions

Current State:
Leading companies like IBM, Google, and others are developing NISQ (Noisy Intermediate-Scale Quantum) systems with 50-1000+ qubits. Google claimed "quantum advantage" in 2019, demonstrating quantum computation superiority for specific problems.""",
            
            "sample_geography.txt": """Denmark - Nordic Nation:
Denmark is a Scandinavian country located in Northern Europe, forming the Jutland Peninsula and numerous islands.

Key Facts:
- Capital: Copenhagen
- Population: Approximately 5.9 million
- Area: 43,094 square kilometers
- Neighbors: Germany to the south, Sweden and Norway across the sea

Economic Profile:
Denmark has a highly developed mixed economy with strengths in:
- Manufacturing and engineering
- Renewable energy (particularly wind power)
- Pharmaceuticals and biotechnology
- Information technology
- Design and consumer goods

Culture:
The Danish concept of "hygge" emphasizes coziness, comfort, and well-being. Copenhagen is renowned for its innovative gastronomy, harbor culture, and cycling infrastructure."""
        }
        
        docs_dir = Path("data/documents")
        for filename, content in samples.items():
            file_path = docs_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"   ✓ Created {filename}")
        
        logger.info("✅ Sample documents created")
        return True
    
    def create_init_files(self):
        """Create __init__.py files for packages"""
        logger.info("📝 Creating package files...")
        
        packages = ['src', 'utils']
        
        for pkg in packages:
            init_file = Path(pkg) / "__init__.py"
            init_file.touch()
            logger.info(f"   ✓ {init_file}")
        
        logger.info("✅ Package files created")
        return True
    
    def run_full_setup(self) -> bool:
        """Run complete setup sequence"""
        logger.info("=" * 70)
        logger.info("🚀 RAG HALLUCINATION DETECTION SYSTEM - SETUP")
        logger.info("=" * 70)
        
        steps = [
            ("Python Version Check", self.verify_python_version),
            ("Directory Structure", self.create_directory_structure),
            ("Package Files", self.create_init_files),
            ("Requirements Check", self.verify_requirements),
            ("Install Dependencies", self.install_dependencies),
            ("Import Verification", self.verify_imports),
            ("Ollama Connection", self.check_ollama_connection),
            ("Sample Documents", self.create_sample_documents),
        ]
        
        failed_steps = []
        
        for step_name, step_func in steps:
            logger.info(f"\n📋 {step_name}...")
            try:
                result = step_func()
                if not result:
                    failed_steps.append(step_name)
            except Exception as e:
                logger.error(f"❌ Error in {step_name}: {e}")
                failed_steps.append(step_name)
        
        logger.info("\n" + "=" * 70)
        
        if not failed_steps:
            logger.info("✅ SETUP COMPLETE - ALL SYSTEMS GO!")
            logger.info("\nNext steps:")
            logger.info("1. If Ollama is not running, start it: ollama serve")
            logger.info("2. Run the application: python main.py")
            logger.info("3. Open browser to: http://localhost:7860")
            return True
        else:
            logger.warning(f"⚠️  SETUP PARTIALLY COMPLETE")
            logger.warning(f"Failed steps: {', '.join(failed_steps)}")
            logger.warning("\nPlease fix these issues and run setup again.")
            
            if "Ollama Connection" in failed_steps:
                logger.warning("\n🔴 CRITICAL: Ollama not available")
                logger.warning("Please install Ollama from https://ollama.ai and run: ollama serve")
            
            return False

def main():
    """Main entry point"""
    try:
        setup = SystemSetup()
        success = setup.run_full_setup()
        
        sys.exit(0 if success else 1)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
