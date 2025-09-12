"""
Student Data Processing and FAISS Index Management
Handles data loading, preprocessing, and vector index operations
"""
import os
import pandas as pd
import faiss
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sentence_transformers import SentenceTransformer
from langchain.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from .student_simulator_config import AppConfig
from .logger import get_logger

logger = get_logger(__name__)

class StudentDataProcessor:
    """Handles student data processing and FAISS index management"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.id_to_row: Optional[Dict[int, pd.Series]] = None
        self.docstore: Optional[InMemoryDocstore] = None
        self.index_to_docstore_id: Optional[Dict[int, str]] = None
        self.faiss_retriever: Optional[FAISS] = None
        
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """Load and preprocess student data from CSV"""
        try:
            data_path = data_path or self.config.data.data_path
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            logger.info(f"Loading data from: {data_path}")
            self.data = pd.read_csv(data_path)
            
            # Validate required columns
            required_columns = ['Question', 'Understanding', 'Emotion', 'Reasoning', 'Response']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Create combined text for embedding
            self.data['Combined'] = (
                "Question: " + self.data['Question'].astype(str) +
                " Understanding: " + self.data['Understanding'].astype(str) +
                " Emotion: " + self.data['Emotion'].astype(str) +
                " Reasoning: " + self.data['Reasoning'].astype(str) +
                " Response: " + self.data['Response'].astype(str)
            )
            
            # Clean and validate data
            self.data = self._clean_data(self.data)
            
            logger.info(f"Successfully loaded {len(self.data)} records")
            return self.data
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the loaded data"""
        try:
            # Remove rows with missing critical data
            initial_count = len(data)
            data = data.dropna(subset=['Question', 'Understanding', 'Emotion', 'Reasoning', 'Response'])
            
            # Remove duplicate rows
            data = data.drop_duplicates()
            
            # Clean text data
            text_columns = ['Question', 'Understanding', 'Emotion', 'Reasoning', 'Response']
            for col in text_columns:
                data[col] = data[col].astype(str).str.strip()
            
            # Remove empty responses
            data = data[data['Response'].str.len() > 0]
            
            final_count = len(data)
            removed_count = initial_count - final_count
            
            if removed_count > 0:
                logger.warning(f"Removed {removed_count} invalid records during cleaning")
            
            logger.info(f"Data cleaning completed. Final record count: {final_count}")
            return data
            
        except Exception as e:
            logger.error(f"Error during data cleaning: {e}")
            raise
    
    def build_faiss_index(self) -> Tuple[faiss.Index, SentenceTransformer, Dict[int, pd.Series], InMemoryDocstore, Dict[int, str]]:
        """Build FAISS index from the loaded data"""
        try:
            if self.data is None:
                raise ValueError("Data not loaded. Call load_data() first.")
            
            logger.info("Building FAISS index...")
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(self.config.model.embedding_model)
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embedding_model.encode(
                self.data['Combined'].tolist(),
                show_progress_bar=True,
                batch_size=32
            )
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(embeddings.astype('float32'))
            
            # Create mapping structures
            self.id_to_row = {i: row for i, row in self.data.iterrows()}
            self.docstore = InMemoryDocstore({
                str(i): row.to_dict() for i, row in self.data.iterrows()
            })
            self.index_to_docstore_id = {i: str(i) for i in range(len(self.data))}
            
            # Create FAISS retriever
            self.faiss_retriever = FAISS(
                index=self.faiss_index,
                embedding_function=HuggingFaceEmbeddings(model_name=self.config.model.embedding_model),
                docstore=self.docstore,
                index_to_docstore_id=self.index_to_docstore_id
            )
            
            logger.info(f"FAISS index built successfully with {self.faiss_index.ntotal} vectors")
            return (
                self.faiss_index,
                self.embedding_model,
                self.id_to_row,
                self.docstore,
                self.index_to_docstore_id
            )
            
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            raise
    
    def search_similar(self, query: str, k: int = None) -> List[Tuple[int, float]]:
        """Search for similar records in the FAISS index"""
        try:
            if self.faiss_index is None or self.embedding_model is None:
                raise ValueError("FAISS index not built. Call build_faiss_index() first.")
            
            k = k or min(self.config.data.max_retrieval_results, self.faiss_index.ntotal)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search in FAISS index
            distances, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
            
            # Return results as list of (index, distance) tuples
            results = [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0]) if idx != -1]
            
            logger.debug(f"Found {len(results)} similar records for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching FAISS index: {e}")
            raise
    
    def get_record_by_index(self, index: int) -> Optional[pd.Series]:
        """Get a record by its index"""
        if self.id_to_row is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return self.id_to_row.get(index)
    
    def get_records_by_indices(self, indices: List[int]) -> List[pd.Series]:
        """Get multiple records by their indices"""
        if self.id_to_row is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return [self.id_to_row.get(idx) for idx in indices if idx in self.id_to_row]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded data"""
        if self.data is None:
            return {}
        
        stats = {
            "total_records": len(self.data),
            "unique_understandings": self.data['Understanding'].nunique(),
            "unique_emotions": self.data['Emotion'].nunique(),
            "unique_reasonings": self.data['Reasoning'].nunique(),
            "avg_response_length": self.data['Response'].str.len().mean(),
            "index_size": self.faiss_index.ntotal if self.faiss_index else 0
        }
        
        return stats
    
    def validate_context_match(self, record: pd.Series, context: Dict[str, str]) -> bool:
        """Validate if a record matches the given context"""
        try:
            return (
                record['Understanding'].strip() == context['Understanding'] and
                record['Emotion'].strip() == context['Emotion'] and
                record['Reasoning'].strip() == context['Reasoning']
            )
        except Exception as e:
            logger.error(f"Error validating context match: {e}")
            return False
