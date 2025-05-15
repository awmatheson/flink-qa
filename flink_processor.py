from pyflink.table import (
    EnvironmentSettings, TableEnvironment, DataTypes, Schema
)
from pyflink.table.udf import udf
from pyflink.table.types import RowType, RowField
from pyflink.table.expressions import col, call
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import os
from typing import List, Dict, Any
import json
import numpy as np

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

@udf(result_type=DataTypes.ARRAY(DataTypes.FLOAT()), deterministic=True)
def generate_embedding(text: str) -> list:
    """Generate embedding for text using sentence-transformers with optimizations."""
    try:
        if not text or not isinstance(text, str):
            return []
        embedding = embedding_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return []

class FlinkProcessor:
    def __init__(self):
        # Initialize Flink Table Environment with parallelism
        self.env_settings = EnvironmentSettings.in_streaming_mode()
        self.table_env = TableEnvironment.create(self.env_settings)
        # Configure environment settings
        self.table_env.get_config().set("parallelism.default", "4")
        self.table_env.get_config().set("python.fn-execution.bundle.size", "1000")
        self.table_env.get_config().set("python.fn-execution.bundle.time", "1000")
        self.table_env.get_config().set("python.fn-execution.arrow.batch.size", "1000")
        # Register UDFs
        self.table_env.create_temporary_function("extract_text", extract_text_from_pdf)
        self.table_env.create_temporary_function("generate_embedding", generate_embedding)

    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process documents using Flink Table API with optimizations."""
        # Create a table from the documents with optimized schema
        row_type = RowType([
            RowField("filename", DataTypes.STRING()),
            RowField("content", DataTypes.STRING()),
            RowField("embedding", DataTypes.ARRAY(DataTypes.FLOAT()))
        ])
        # Prepare data with proper types
        table_data = [(doc['filename'], doc['content'], None) for doc in documents]
        # Create source table
        source_table = self.table_env.from_elements(
            table_data,
            row_type
        )
        # Process the documents with optimized operations
        result_table = source_table.select(
            col("filename"),
            call("extract_text", col("content")).alias("extracted_text"),
            call("generate_embedding", call("extract_text", col("content"))).alias("embedding")
        )
        # Collect results efficiently
        results = []
        with result_table.execute().collect() as results_iterator:
            for row in results_iterator:
                results.append({
                    'filename': row[0],
                    'content': row[1],
                    'embedding': row[2]
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