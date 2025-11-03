import numpy as np


def is_face_duplicate(new_embedding, existing_embeddings, threshold=0.45):
    """
    Cek apakah embedding wajah sudah ada di DB
    Menggunakan cosine similarity
    threshold lebih kecil = lebih ketat
    """

    if len(existing_embeddings) == 0:
        return False

    new_emb = np.array(new_embedding)

    for emb in existing_embeddings:
        emb = np.array(emb)
        similarity = np.dot(new_emb, emb) / (np.linalg.norm(new_emb) * np.linalg.norm(emb))

        if similarity > (1 - threshold):  # misal sim > 0.55 dianggap sama
            return True

    return False
