from pdf_processor import PDFProcessor
from embedding_service import EmbeddingService
from retrieval_service import RetrievalService
from question_answering_service import QuestionAnsweringService

def main():
    # Step 1: Extract texts from PDFs
    pdf_processor = PDFProcessor(pdf_folder='./pdfs/')
    texts = pdf_processor.extract_text_from_pdfs()

    # Initialize services only if texts are valid
    embedding_service = None
    retrieval_service = None

    # Check if any text was extracted
    if not texts or all(t is None or t.strip() == "" for t in texts):
        print("No valid text found in the PDFs. Exiting the process.")
    else:
        # Step 2: Create embeddings
        embedding_service = EmbeddingService()
        embeddings = embedding_service.embed_texts(texts)

        # Ensure embeddings were generated before proceeding
        if embeddings is not None and len(embeddings) > 0:
            # Step 3: Build FAISS index
            retrieval_service = RetrievalService(dimension=embeddings.shape[1])
            retrieval_service.build_index(embeddings)
        else:
            print("Failed to generate embeddings. Exiting the process.")

    # Only proceed if embeddings and retrieval services are initialized
    if embedding_service and retrieval_service:
        # Step 4: Get user query and retrieve relevant documents
        query = input("Enter your question: ")
        query_embedding = embedding_service.embed_texts([query])
        indices = retrieval_service.retrieve_documents(query_embedding)
        relevant_docs = [texts[i] for i in indices[0]]

        # Step 5: Generate answer using the Llama model
        question_answering_service = QuestionAnsweringService()
        context = " ".join(relevant_docs)
        answer = question_answering_service.generate_answer(query, context)
        print(f"Answer: {answer}")
    else:
        print("Cannot process the query without valid embeddings and index.")

if __name__ == "__main__":
    main()
