
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingService:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings

    def save_embeddings(self, embeddings, filename='embeddings.npy'):
        np.save(filename, embeddings)

    def load_embeddings(self, filename='embeddings.npy'):
        return np.load(filename)
    