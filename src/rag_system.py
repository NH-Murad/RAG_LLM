"""
RAG System - Core Pipeline for Retrieval-Augmented Generation
Handles both with-RAG and without-RAG generation
"""

import logging
import torch
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import json

logger = logging.getLogger(__name__)

class RAGSystem:
    """Retrieval-Augmented Generation System"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.ollama_base_url = "http://localhost:11434"
        self.ollama_model = config.get('ollama_model', 'phi')
        self.embedding_model_name = config.get('embedding_model', 'all-MiniLM-L6-v2')
        self.chunk_size = config.get('chunk_size', 256)
        self.chunk_overlap = config.get('chunk_overlap', 50)
        self.top_k = config.get('top_k', 3)
        self.temperature = config.get('temperature', 0.7)
        self.max_length = config.get('max_length', 512)
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize FAISS index
        self.faiss_index = None
        self.documents = []
        self.document_embeddings = []
        
        self._check_ollama_connection()
    
    def _check_ollama_connection(self):
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Ollama connection successful")
                available_models = response.json().get('models', [])
                logger.info(f"Available models: {[m['name'] for m in available_models]}")
            else:
                logger.warning(f"Ollama returned status: {response.status_code}")
        except requests.exceptions.ConnectionError:
            logger.error("❌ Cannot connect to Ollama. Make sure it's running on localhost:11434")
            raise ConnectionError("Ollama service not found at http://localhost:11434")
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            i += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def index_documents(self, documents: List[str]):
        """Index documents using FAISS"""
        try:
            logger.info(f"Indexing {len(documents)} document(s)...")
            
            # Chunk and flatten documents
            all_chunks = []
            for doc in documents:
                chunks = self._chunk_text(doc)
                all_chunks.extend(chunks)
            
            logger.info(f"Created {len(all_chunks)} chunks from documents")
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embedding_model.encode(
                all_chunks,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Create FAISS index
            embedding_dim = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(embedding_dim)
            self.faiss_index.add(embeddings.astype(np.float32))
            
            self.documents = all_chunks
            self.document_embeddings = embeddings
            
            logger.info(f"✅ Successfully indexed {len(all_chunks)} chunks")
        
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            raise
    
    def retrieve_context(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """Retrieve most relevant documents for query"""
        if not self.faiss_index or len(self.documents) == 0:
            logger.warning("No documents indexed")
            return []
        
        top_k = top_k or self.top_k
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode(
                [query],
                convert_to_numpy=True
            )
            
            # Search in FAISS
            distances, indices = self.faiss_index.search(
                query_embedding.astype(np.float32),
                min(top_k, len(self.documents))
            )
            
            # Get relevant documents
            relevant_docs = [self.documents[idx] for idx in indices[0]]
            
            logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
            return relevant_docs
        
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def generate_without_rag(self, query: str) -> str:
        """Generate answer without RAG (baseline)"""
        try:
            logger.info("Generating answer WITHOUT RAG...")
            
            prompt = f"""Answer the following question based on your knowledge:

Question: {query}

Answer:"""
            
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.temperature,
                },
                timeout=120
            )
            
            if response.status_code == 200:
                answer = response.json().get('response', 'No response').strip()
                logger.info("✅ Generated answer without RAG")
                return answer
            else:
                error = f"Ollama error: {response.status_code}"
                logger.error(error)
                return error
        
        except Exception as e:
            logger.error(f"Error generating without RAG: {e}")
            return f"Error: {str(e)}"
    
    def generate_with_rag(self, query: str) -> Tuple[str, List[str]]:
        """Generate answer with RAG (grounded in context)"""
        try:
            logger.info("Generating answer WITH RAG...")
            
            # Retrieve context
            context_docs = self.retrieve_context(query)
            
            if not context_docs:
                logger.warning("No context retrieved, falling back to non-RAG")
                return self.generate_without_rag(query), []
            
            context_str = "\n\n".join(context_docs)
            
            prompt = f"""Based on the following context, answer the question accurately and factually.

Context:
{context_str}

Question: {query}

Answer (based only on the provided context):"""
            prompt += (
    "\n\nAnswer concisely. Use only the information in the context. "
    "Keep the answer under 10 short sentences."
            )

            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.temperature,
                },
                timeout=120
            )
            
            if response.status_code == 200:
                answer = response.json().get('response', 'No response').strip()
                logger.info("✅ Generated answer with RAG")
                return answer, context_docs
            else:
                error = f"Ollama error: {response.status_code}"
                logger.error(error)
                return error, context_docs
        
        except Exception as e:
            logger.error(f"Error generating with RAG: {e}")
            return f"Error: {str(e)}", []
    
    def get_status(self) -> Dict:
        """Get RAG system status"""
        return {
            'model': self.ollama_model,
            'embedding_model': self.embedding_model_name,
            'documents_indexed': len(self.documents),
            'ollama_available': self._check_ollama_connection(),
            'faiss_index_active': self.faiss_index is not None
        }
