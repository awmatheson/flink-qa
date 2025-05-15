import os
from flink_processor import FlinkProcessor
from vector_store import VectorStore
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import argparse

class DocumentQA:
    def __init__(self, pdf_directory: str, vector_store_dir: str):
        self.pdf_directory = pdf_directory
        self.vector_store_dir = vector_store_dir
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_store = VectorStore()
        
        # Initialize the LLM
        print("Loading question answering model...")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        if torch.cuda.is_available():
            self.llm = self.llm.to("cuda")
        print("Model loaded successfully!")

    def process_documents(self):
        """Process PDF documents and create embeddings using Flink pipeline."""
        print("Processing PDF documents using Flink...")
        flink_processor = FlinkProcessor()
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

    def generate_answer(self, question: str, context: str) -> str:
        """Generate an answer using the LLM based on the question and context."""
        # Prepare the prompt
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        outputs = self.llm.generate(
            **inputs,
            max_length=200,
            num_beams=4,
            temperature=0.7,
            no_repeat_ngram_size=2
        )
        
        # Decode and return the answer
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    def answer_question(self, question: str, k: int = 10):
        """Answer a question using the vector store with document-based ranking and LLM."""
        # Generate embedding for the question
        question_embedding = self.embedding_model.encode(question)
        
        # Search for relevant documents
        results = self.vector_store.search(question_embedding, k=k)
        
        if not results:
            return "No relevant documents found."
        
        # Combine all relevant document content for context
        context = "\n\n".join([
            f"Document from {result['filename']}:\n{result['content']}"
            for result in results
        ])
        
        # Generate answer using LLM
        answer = self.generate_answer(question, context)
        
        # Format the response
        response = f"Answer: {answer}\n\n"
        response += "Based on the following documents:\n\n"
        for i, result in enumerate(results, 1):
            response += f"Document {i} (from {result['filename']}):\n"
            response += f"Relevance: {result['chunk_count']} matching chunks, average score: {result['avg_score']:.4f}\n"
            response += f"{result['content'][:500]}...\n\n"
        
        return response

def main():
    parser = argparse.ArgumentParser(description='PDF Document QA System')
    parser.add_argument('--pdf_dir', required=True, help='Directory containing PDF documents')
    parser.add_argument('--vector_store_dir', default='vector_store', help='Directory for vector store')
    parser.add_argument('--process', action='store_true', help='Process documents and create embeddings')
    args = parser.parse_args()

    qa_system = DocumentQA(args.pdf_dir, args.vector_store_dir)
    
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
