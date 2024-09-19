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
        return  # Exit if no valid text is found
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
            return  # Exit if embeddings failed

    # Step 4: Continuously ask user for questions
    question_answering_service = QuestionAnsweringService()

    while True:
        query = input("Enter your question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            print("Exiting the question-answering session.")
            break

        query_embedding = embedding_service.embed_texts([query])
        indices = retrieval_service.retrieve_documents(query_embedding)
        relevant_docs = [texts[i] for i in indices[0]]

        # Step 5: Generate answer using the Llama model
        context = " ".join(relevant_docs)
        answer = question_answering_service.generate_answer(query, context)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
