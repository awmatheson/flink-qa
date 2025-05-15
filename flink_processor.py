from pyflink.table import (
    EnvironmentSettings, TableEnvironment, DataTypes, Schema
)
from pyflink.table.udf import udf, udtf
from pyflink.table.types import RowType, RowField, Row
from pyflink.table.expressions import col, call
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import os
from typing import List, Dict, Any, Tuple, Iterator
import json
import numpy as np
import re
import uuid

# Global model for embedding UDF
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

@udf(result_type=DataTypes.STRING(), deterministic=True)
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        return ""

@udf(result_type=DataTypes.ARRAY(DataTypes.ROW([
    DataTypes.FIELD("text", DataTypes.STRING()),
    DataTypes.FIELD("index", DataTypes.STRING()),
    DataTypes.FIELD("total", DataTypes.STRING())
])), deterministic=True)
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Row]:
    """Split text into overlapping chunks and return as list of Row objects."""
    # Clean and normalize text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # If text is shorter than chunk_size, return it as a single chunk
    if len(text) <= chunk_size:
        return [Row(text=text, index="0", total="1")]
    
    chunks = []
    start = 0
    chunk_index = 0
    while start < len(text):
        # Find the end of the chunk
        end = start + chunk_size
        
        # If we're not at the end of the text, try to break at a sentence
        if end < len(text):
            # Look for sentence endings
            sentence_end = max(
                text.rfind('.', start, end),
                text.rfind('!', start, end),
                text.rfind('?', start, end)
            )
            if sentence_end > start + chunk_size // 2:  # Only use if it's not too far back
                end = sentence_end + 1
        
        # Add the chunk with its metadata
        chunks.append(Row(
            text=text[start:end].strip(),
            index=str(chunk_index),
            total=str(len(chunks) + 1)
        ))
        
        # Move start position, accounting for overlap
        start = end - overlap
        chunk_index += 1
    
    return chunks

@udtf(result_types=[
    DataTypes.STRING(),
    DataTypes.STRING(),
    DataTypes.STRING(),
    DataTypes.STRING(),
    DataTypes.STRING()
])
def expand_chunks(doc_id: str, filename: str, chunks: List[Row]) -> Iterator[Tuple[str, str, str, str, str]]:
    """Expand chunks into separate rows."""
    for chunk in chunks:
        yield doc_id, filename, chunk.text, chunk.index, chunk.total

@udf(result_type=DataTypes.ARRAY(DataTypes.FLOAT()), deterministic=True)
def generate_embedding(text: str) -> list:
    """Generate embedding for text using sentence-transformers with optimizations."""
    try:
        if not text or not isinstance(text, str):
            print("Warning: Empty or invalid text input")
            return [0.0] * 384  # Return zero vector of correct dimension
        
        # Clean and normalize text
        text = text.strip()
        if not text:
            print("Warning: Empty text after cleaning")
            return [0.0] * 384
            
        embedding = embedding_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Verify embedding dimension
        if len(embedding) != 384:
            print(f"Warning: Invalid embedding dimension: {len(embedding)}")
            return [0.0] * 384
            
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return [0.0] * 384  # Return zero vector of correct dimension

class FlinkProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        # Initialize Flink Table Environment with parallelism
        self.env_settings = EnvironmentSettings.in_streaming_mode()
        self.table_env = TableEnvironment.create(self.env_settings)
        # Configure environment settings
        self.table_env.get_config().set("parallelism.default", "4")
        self.table_env.get_config().set("python.fn-execution.bundle.size", "1000")
        self.table_env.get_config().set("python.fn-execution.bundle.time", "1000")
        self.table_env.get_config().set("python.fn-execution.arrow.batch.size", "1000")
        # Store chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Register UDFs
        self.table_env.create_temporary_function("extract_text", extract_text_from_pdf)
        self.table_env.create_temporary_function("chunk_text", chunk_text)
        self.table_env.create_temporary_function("expand_chunks", expand_chunks)
        self.table_env.create_temporary_function("generate_embedding", generate_embedding)

    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process documents using Flink Table API with optimizations."""
        # Create a table from the documents with optimized schema
        row_type = RowType([
            RowField("doc_id", DataTypes.STRING()),
            RowField("filename", DataTypes.STRING()),
            RowField("content", DataTypes.STRING()),
            RowField("embedding", DataTypes.ARRAY(DataTypes.FLOAT()))
        ])
        
        # Prepare data with document IDs
        table_data = [(str(uuid.uuid4()), doc['filename'], doc['content'], None) for doc in documents]
        
        # Create source table
        source_table = self.table_env.from_elements(
            table_data,
            row_type
        )
        
        # Process the documents with optimized operations
        # 1. Extract text from PDFs
        extracted_table = source_table.select(
            col("doc_id"),
            col("filename"),
            call("extract_text", col("content")).alias("extracted_text")
        )
        
        # 2. Chunk the text and expand chunks
        chunked_table = extracted_table.select(
            col("doc_id"),
            col("filename"),
            call("chunk_text", col("extracted_text"), self.chunk_size, self.chunk_overlap).alias("chunks")
        ).join_lateral(
            call("expand_chunks", col("doc_id"), col("filename"), col("chunks"))
        ).select(
            col("f0").alias("doc_id"),
            col("f1").alias("filename"),
            col("f2").alias("chunk_text"),
            col("f3").alias("chunk_index"),
            col("f4").alias("total_chunks")
        )
        
        # 3. Generate embeddings for chunks
        result_table = chunked_table.select(
            col("doc_id"),
            col("filename"),
            col("chunk_text"),
            col("chunk_index"),
            col("total_chunks"),
            call("generate_embedding", col("chunk_text")).alias("embedding")
        )
        
        # Collect results efficiently
        results = []
        with result_table.execute().collect() as results_iterator:
            for row in results_iterator:
                print('DEBUG ROW:', row, [type(x) for x in row])  # Debug print to check row structure and types
                results.append({
                    'doc_id': row[0],
                    'filename': row[1],
                    'content': row[2],
                    'chunk_index': int(row[3]),
                    'total_chunks': int(row[4]),
                    'embedding': row[5]
                })
        
        return results

    def process_directory(self, pdf_directory: str) -> List[Dict[str, Any]]:
        """Process all PDFs in a directory using Flink Table API."""
        # Get list of PDF files
        pdf_files = []
        for filename in os.listdir(pdf_directory):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(pdf_directory, filename)
                pdf_files.append({
                    'filename': filename,
                    'content': pdf_path,
                    'embedding': None
                })
        # Process the PDFs using Flink
        return self.process_documents(pdf_files) 