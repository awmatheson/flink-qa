# PDF Document Processing and Question Answering System

This system uses Apache Flink's Python Table API to process PDF documents, extract text, generate embeddings, and provide a question-answering interface using a large language model.

## Features

- PDF text extraction and processing using Flink Table API
- Vector embeddings generation using sentence-transformers
- Document-level ranking and retrieval
- Question answering using FLAN-T5 model
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
- transformers
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
uv venv .venv --python=3.11 # On Windows
```

3. Install dependencies using uv:
```bash
uv pip install -r requirements.txt
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

To start the question-answering interface:

```bash
python main.py
```

The system will:
1. Load the question-answering model
2. Connect to the vector database
3. Present an interactive prompt for questions

Type 'exit' to quit the question-answering interface.

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

When deploying to a Flink cluster, consider the following:

### Model Distribution
- The sentence-transformers model and FLAN-T5 model are loaded as global variables
- Each TaskManager will load its own copy of the models
- Consider using Flink's distributed cache to share model files:
  ```python
  env.add_python_file("path/to/model/files")
  ```

### Resource Requirements
- Each TaskManager needs sufficient memory for model loading
- Recommended minimum:
  - 4GB RAM per TaskManager for sentence-transformers
  - 8GB RAM per TaskManager for FLAN-T5
  - Adjust based on batch size and parallelism

### Configuration
Add these settings to your `flink-conf.yaml`:
```yaml
python.fn-execution.memory.managed: true
python.fn-execution.memory.size: 1024mb
taskmanager.memory.process.size: 4096mb
```

### Best Practices
1. **Model Loading**
   - Use lazy loading in UDFs to avoid loading models on unused TaskManagers
   - Consider using model quantization for smaller memory footprint
   - Implement proper state management for distributed processing

2. **Resource Management**
   - Monitor memory usage across TaskManagers
   - Adjust parallelism based on available resources
   - Use appropriate batch sizes to balance throughput and memory usage

3. **Error Handling**
   - Implement retry logic for model loading failures
   - Add health checks for model availability
   - Monitor model inference performance

4. **State Management**
   - Use Flink's state management for distributed caching needs
   - Consider using RocksDB state backend for large state
   - Implement proper state cleanup and TTL policies

## Contributing

Feel free to submit issues and enhancement requests! 
