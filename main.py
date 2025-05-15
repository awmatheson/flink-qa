import os
from flink_processor import FlinkProcessor
from vector_store import VectorStore
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import torch
import argparse
from typing import List, Dict, Optional
import json

class DocumentQA:
    def __init__(
        self, 
        pdf_directory: str, 
        vector_store_dir: str,
        model_path: str = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        max_input_tokens: int = 4096,  # Llama models support larger context
        max_output_tokens: int = 1024,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        n_ctx: int = 4096,  # Context window for Llama
        n_gpu_layers: int = -1  # Use all GPU layers if available
    ):
        self.pdf_directory = pdf_directory
        self.vector_store_dir = vector_store_dir
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.vector_store = VectorStore()
        
        # Initialize the Llama model
        print(f"Loading Llama model from: {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )
        print("Model loaded successfully!")
        
        # Configurable token limits
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        print(f"Using context window of {max_input_tokens} tokens and output limit of {max_output_tokens} tokens")

    def process_documents(self):
        """Process PDF documents and create embeddings using Flink pipeline."""
        print("Processing PDF documents using Flink...")
        flink_processor = FlinkProcessor(chunk_size=1000, chunk_overlap=200)
        processed_docs = flink_processor.process_directory(self.pdf_directory)
        
        print("Storing documents in vector database...")
        self.vector_store.add_documents(processed_docs)
        self.vector_store.save(self.vector_store_dir)
        print("Processing complete!")

    def load_existing_store(self):
        """Load an existing vector store."""
        if os.path.exists(self.vector_store_dir):
            self.vector_store.load(self.vector_store_dir)
            print("Loaded existing vector store.")
        else:
            print("No existing vector store found.")

    def _prepare_context(self, results: List[Dict]) -> str:
        """Prepare context from search results, respecting token limits."""
        # Sort results by score (lower distance = better score)
        sorted_results = sorted(results, key=lambda x: x['avg_score'])
        
        # Group chunks by document
        doc_chunks = {}
        for result in sorted_results:
            doc_name = result['filename']
            if doc_name not in doc_chunks:
                doc_chunks[doc_name] = []
            doc_chunks[doc_name].append(result)
        
        # Prepare context, starting with the most relevant documents
        context_parts = []
        total_tokens = 0
        
        for doc_name, chunks in doc_chunks.items():
            # Sort chunks by their index to maintain document order
            chunks.sort(key=lambda x: x.get('chunk_index', 0))
            
            # Add document metadata and structure
            doc_content = f"Document: {doc_name}\n"
            doc_content += "Content:\n"
            
            # Add chunks with clear separation and context
            for i, chunk in enumerate(chunks):
                if i > 0:  # Add separator between chunks
                    doc_content += "\n---\n"
                doc_content += chunk['content']
            
            # Count tokens for this document (approximate)
            doc_tokens = len(doc_content.split())  # Simple approximation
            
            # If adding this document would exceed the limit, stop
            if total_tokens + doc_tokens > self.max_input_tokens - 300:  # Increased buffer for prompt
                break
                
            context_parts.append(doc_content)
            total_tokens += doc_tokens
        
        # Join all document contexts with clear separation
        return "\n\n" + "="*50 + "\n\n".join(context_parts)

    def generate_answer(self, question: str, context: str) -> str:
        """Generate an answer using the Llama model based on the question and context."""
        # Prepare the prompt with clear instructions
        prompt = f"""<|system|>
You are a helpful assistant that answers questions based on the provided context. Your task is to:
1. Analyze the context carefully
2. Extract relevant information that directly answers the question
3. Provide a clear, concise answer
4. If the answer cannot be found in the context, say "I cannot find a specific answer in the provided documents."

Context:
{context}

Question: {question}

Instructions: Focus on providing a direct answer based only on the information in the context. Do not make assumptions or include information not present in the context.
</|system|>
<|user|>
Please answer the question based on the provided context.
</|user|>
<|assistant|>"""
        
        try:
            # Generate answer using Llama
            response = self.llm(
                prompt,
                max_tokens=self.max_output_tokens,
                temperature=0.2,
                top_p=0.95,
                stop=["</|assistant|>", "<|user|>", "<|system|>"],
                echo=False
            )
            
            # Extract and return the answer
            answer = response['choices'][0]['text'].strip()
            return answer
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return "I encountered an error while generating the answer. Please try again."

    def answer_question(self, question: str, k: int = 5):
        """Answer a question using the vector store with document-based ranking and LLM."""
        try:
            # Generate embedding for the question
            question_embedding = self.embedding_model.encode(question)
            
            # Search for relevant documents
            results = self.vector_store.search(question_embedding, k=k)
            
            if not results:
                return "No relevant documents found. Please try rephrasing your question or check if the documents contain the information you're looking for."
            
            # Prepare context respecting token limits
            context = self._prepare_context(results)
            
            # Generate answer using LLM
            answer = self.generate_answer(question, context)
            
            # Format the response
            response = f"Answer: {answer}\n\n"
            response += "Based on the following documents:\n\n"
            
            # Group results by document
            doc_results = {}
            for result in results:
                doc_name = result['filename']
                if doc_name not in doc_results:
                    doc_results[doc_name] = []
                doc_results[doc_name].append(result)
            
            # Add document summaries with improved formatting
            for doc_name, chunks in doc_results.items():
                chunks.sort(key=lambda x: x.get('chunk_index', 0))
                response += f"Document: {doc_name}\n"
                response += f"Relevance: {len(chunks)} matching chunks, average score: {sum(c['avg_score'] for c in chunks)/len(chunks):.4f}\n"
                response += f"Content preview: {chunks[0]['content'][:200]}...\n\n"
            
            return response
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            return "I encountered an error while processing your question. Please try again."

def main():
    parser = argparse.ArgumentParser(description='PDF Document QA System')
    parser.add_argument('--pdf_dir', required=True, help='Directory containing PDF documents')
    parser.add_argument('--vector_store_dir', default='vector_store', help='Directory for vector store')
    parser.add_argument('--process', action='store_true', help='Process documents and create embeddings')
    parser.add_argument('--model_path', default='Meta-Llama-3-8B-Instruct.Q4_K_M.gguf',
                      help='Path to the Llama GGUF model file')
    parser.add_argument('--input_tokens', type=int, default=4096,
                      help='Maximum input tokens for context (default: 4096)')
    parser.add_argument('--output_tokens', type=int, default=1024,
                      help='Maximum output tokens for answers (default: 1024)')
    parser.add_argument('--embedding_model', default='all-MiniLM-L6-v2',
                      help='Model to use for embeddings (default: all-MiniLM-L6-v2)')
    parser.add_argument('--n_ctx', type=int, default=4096,
                      help='Context window size for Llama model (default: 4096)')
    parser.add_argument('--n_gpu_layers', type=int, default=-1,
                      help='Number of GPU layers to use (-1 for all, default: -1)')
    args = parser.parse_args()

    qa_system = DocumentQA(
        args.pdf_dir, 
        args.vector_store_dir,
        model_path=args.model_path,
        max_input_tokens=args.input_tokens,
        max_output_tokens=args.output_tokens,
        embedding_model_name=args.embedding_model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers
    )
    
    if args.process:
        qa_system.process_documents()
    else:
        qa_system.load_existing_store()
    
    print("\nQuestion Answering System Ready!")
    print("Type 'exit' to quit.")
    
    while True:
        question = input("\nEnter your question: ")
        if question.lower() == 'exit':
            break
        
        answer = qa_system.answer_question(question)
        print("\n" + answer)

if __name__ == "__main__":
    main()
