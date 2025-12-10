"""
Data Handler - Manage documents and datasets
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class DataHandler:
    """Handle document loading, storage, and management"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.documents = {}
        self.metadata = {}
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories"""
        dirs = [
            self.data_dir / "documents",
            self.data_dir / "results",
            self.data_dir / "cache"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def load_document(self, file_path: str) -> bool:
        """Load document from file"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                logger.error(f"File not found: {file_path}")
                return False
            
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            doc_id = path.stem
            self.add_text_document(
                doc_id,
                content,
                metadata={'source': file_path, 'timestamp': datetime.now().isoformat()}
            )
            
            logger.info(f"Loaded document: {doc_id} ({len(content)} chars)")
            return True
        
        except Exception as e:
            logger.error(f"Error loading document: {e}")
            return False
    
    def add_text_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Add text document to collection"""
        try:
            if not content.strip():
                logger.warning(f"Empty content for document: {doc_id}")
                return False
            
            self.documents[doc_id] = content
            self.metadata[doc_id] = metadata or {'added_at': datetime.now().isoformat()}
            
            logger.info(f"Added document: {doc_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[str]:
        """Retrieve document by ID"""
        return self.documents.get(doc_id)
    
    def get_all_documents(self) -> List[str]:
        """Get all documents as a list"""
        return list(self.documents.values())
    
    def get_document_ids(self) -> List[str]:
        """Get all document IDs"""
        return list(self.documents.keys())
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document"""
        try:
            if doc_id in self.documents:
                del self.documents[doc_id]
                if doc_id in self.metadata:
                    del self.metadata[doc_id]
                logger.info(f"Deleted document: {doc_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def clear_all(self) -> bool:
        """Clear all documents"""
        try:
            self.documents.clear()
            self.metadata.clear()
            logger.info("Cleared all documents")
            return True
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get document statistics"""
        total_chars = sum(len(doc) for doc in self.documents.values())
        total_words = sum(len(doc.split()) for doc in self.documents.values())
        
        return {
            'total_documents': len(self.documents),
            'total_characters': total_chars,
            'total_words': total_words,
            'avg_doc_length': total_chars / len(self.documents) if self.documents else 0,
            'document_ids': self.get_document_ids()
        }
    
    def save_results(self, results: List[Dict], filename: str) -> bool:
        """Save results to file"""
        try:
            output_path = self.data_dir / "results" / filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved results to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False
    
    def load_results(self, filename: str) -> Optional[List[Dict]]:
        """Load results from file"""
        try:
            input_path = self.data_dir / "results" / filename
            with open(input_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"Loaded results from {input_path}")
            return results
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return None
    
    def export_csv(self, results: List[Dict], filename: str) -> bool:
        """Export results as CSV"""
        try:
            import csv
            output_path = self.data_dir / "results" / filename
            
            if not results:
                logger.warning("No results to export")
                return False
            
            keys = results[0].keys()
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
            
            logger.info(f"Exported results to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")
            return False
