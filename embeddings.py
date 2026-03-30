from sentence_transformers import SentenceTransformer
def get_embeddings(text):
    model=SentenceTransformer('all-mpnet-base-v2')
    embeddings=model.encode(text)
    return embeddings