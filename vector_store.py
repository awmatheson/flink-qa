import faiss
import numpy as np
from typing import List, Dict, Tuple
import json
import os
from collections import Counter

class VectorStore:
    def __init__(self, dimension: int = 384):  # 384 is the dimension for all-MiniLM-L6-v2
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.metadata = []
        self.document_chunks = {}  # Maps document name to list of chunk indices

    def add_documents(self, documents: List[Dict[str, any]]):
        """Add documents and their embeddings to the vector store."""
        for doc in documents:
            embedding = np.array(doc['embedding'], dtype=np.float32).reshape(1, -1)
            chunk_idx = len(self.documents)  # Current chunk index
            self.index.add(embedding)
            self.documents.append(doc['content'])
            self.metadata.append({
                'filename': doc['filename']
            })
            
            # Track which chunks belong to which document
            if doc['filename'] not in self.document_chunks:
                self.document_chunks[doc['filename']] = []
            self.document_chunks[doc['filename']].append(chunk_idx)

    def search(self, query_embedding: List[float], k: int = 10) -> List[Dict]:
        """Search for similar documents using a query embedding and rank by document frequency."""
        query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_vector, k)
        
        # Count occurrences of each document in the results
        doc_counts = Counter()
        doc_chunks = {}  # Store chunks for each document
        
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # FAISS returns -1 for empty slots
                doc_name = self.metadata[idx]['filename']
                doc_counts[doc_name] += 1
                
                if doc_name not in doc_chunks:
                    doc_chunks[doc_name] = []
                doc_chunks[doc_name].append({
                    'content': self.documents[idx],
                    'score': float(distances[0][i])
                })
        
        # Sort documents by frequency and average score
        results = []
        for doc_name, count in doc_counts.most_common():
            chunks = doc_chunks[doc_name]
            avg_score = sum(chunk['score'] for chunk in chunks) / len(chunks)
            results.append({
                'filename': doc_name,
                'content': '\n'.join(chunk['content'] for chunk in chunks),
                'chunk_count': count,
                'avg_score': avg_score
            })
        
        return results

    def save(self, directory: str):
        """Save the vector store to disk."""
        os.makedirs(directory, exist_ok=True)
        
        # Save the FAISS index
        faiss.write_index(self.index, os.path.join(directory, 'index.faiss'))
        
        # Save the documents, metadata, and document chunks
        with open(os.path.join(directory, 'documents.json'), 'w') as f:
            json.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'document_chunks': self.document_chunks
            }, f)

    def load(self, directory: str):
        """Load the vector store from disk."""
        # Load the FAISS index
        self.index = faiss.read_index(os.path.join(directory, 'index.faiss'))
        
        # Load the documents, metadata, and document chunks
        with open(os.path.join(directory, 'documents.json'), 'r') as f:
            data = json.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.document_chunks = data['document_chunks'] 