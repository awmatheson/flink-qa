# PDF Document Processing and Question Answering System

Loosely inspired by the Langchain QA bot and PDF ingestion system - https://python.langchain.com/v0.2/docs/tutorials/pdf_qa/

This system uses Apache Flink's Python Table API to process PDF documents, extract text, generate embeddings, and provide a question-answering interface using a large language model.

## Features

- PDF text extraction and processing using Flink Table API
- Vector embeddings generation using sentence-transformers
- Document-level ranking and retrieval
- Question answering using Llama 3 model
- Optimized Flink processing pipeline with:
  - Parallel processing
  - Efficient batch operations
  - Optimized UDF implementations
  - State management for distributed processing

## Requirements

- Python 3.11+
- Apache Flink 2.x
- PyPDF2
- sentence-transformers
- llama-cpp-python
- torch
- numpy
- uv (Python package installer)

## Installation

1. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate a virtual environment:
```bash
uv venv .venv --python=3.11
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies using uv:
```bash
uv pip install -r requirements.txt
```

4. Download the Llama model:
   - Create a `models` directory in the project root:
     ```bash
     mkdir models
     ```
   - Download the Llama 3 8B Instruct model (Q4_K_M quantized version) from Hugging Face:
     ```bash
     # You'll need to have access to the model on Hugging Face
     # Place the downloaded model file in the models directory
     # The file should be named: Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
     ```

## Usage

### Processing Documents

To process PDF documents in a directory:

```bash
python main.py --pdf_dir <directory_path> --process
```

This will:
1. Extract text from PDFs
2. Generate embeddings
3. Store them in the vector database

### Question Answering

To start the question-answering interface with default settings:

```bash
python main.py --pdf_dir <directory_path> --model_path models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

### Advanced Configuration

The system supports various command-line options for customization:

```bash
python main.py \
    --pdf_dir <directory_path> \
    --model_path models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
    --input_tokens 4096 \
    --output_tokens 1024 \
    --n_ctx 4096 \
    --n_gpu_layers -1 \
    --embedding_model all-MiniLM-L6-v2
```

Command-line options:
- `--pdf_dir`: Directory containing PDF documents (required)
- `--vector_store_dir`: Directory for vector store (default: 'vector_store')
- `--process`: Process documents and create embeddings
- `--model_path`: Path to the Llama GGUF model file (default: 'models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf')
- `--input_tokens`: Maximum input tokens for context (default: 4096)
- `--output_tokens`: Maximum output tokens for answers (default: 1024)
- `--n_ctx`: Context window size for Llama model (default: 4096)
- `--n_gpu_layers`: Number of GPU layers to use (-1 for all, default: -1)
- `--embedding_model`: Model to use for embeddings (default: 'all-MiniLM-L6-v2')

## Architecture

### Flink Processor (`flink_processor.py`)
- Uses Flink Table API for efficient document processing
- Implements optimized UDFs for text extraction and embedding generation
- Configures parallel processing and batch operations
- Handles document-level ranking and retrieval

### Vector Store (`vector_store.py`)
- Manages document embeddings and metadata
- Implements similarity search
- Handles document-level ranking and deduplication

### Main Application (`main.py`)
- Orchestrates the processing pipeline
- Manages the question-answering interface
- Handles user interaction and input/output

## Optimizations

1. **Flink Processing**
   - Parallel processing with configurable parallelism
   - Optimized batch sizes for better throughput
   - Efficient UDF implementations
   - State management for distributed processing

2. **Embedding Generation**
   - Batch processing for better performance
   - Normalized embeddings for improved similarity comparison
   - Stateful processing for distributed environments

3. **Document Processing**
   - Efficient PDF text extraction
   - Document-level ranking
   - Deduplication by document name

## Cluster Deployment

TODO



## Contributing

Feel free to submit issues and enhancement requests! 
