
import faiss
import numpy as np

class RetrievalService:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)

    def build_index(self, embeddings):
        self.index.add(np.array(embeddings))  # Add embeddings to the index

    def retrieve_documents(self, query_embedding, top_k=5):
        distances, indices = self.index.search(np.array(query_embedding), top_k)
        return indices
    