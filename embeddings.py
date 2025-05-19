import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Inicializa o modelo Sentence-BERT
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def generate_embeddings(texts):
    """
    Gera embeddings a partir de uma lista de textos.
    """
    embeddings = model.encode(texts)
    return embeddings

def create_faiss_index(embeddings):
    """
    Cria o índice FAISS com os embeddings fornecidos.
    """
    dim = embeddings.shape[1]  # Dimensão do embedding
    index = faiss.IndexFlatL2(dim)  # Índice para busca L2 (distância euclidiana)
    index.add(embeddings)  # Adiciona os embeddings ao índice
    return index

def search_faiss_index(index, query, k=5):
    """
    Realiza a busca no índice FAISS.
    """
    query_embedding = model.encode([query])  # Gera o embedding para a consulta
    distances, indices = index.search(query_embedding, k)  # Realiza a busca
    return distances, indices
