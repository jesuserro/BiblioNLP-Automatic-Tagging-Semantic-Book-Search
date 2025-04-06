# %%
# Recomendación de libros basada en tags

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

# 1. Cargar el dataset
df = pd.read_csv('data/processed/books.csv')

# 2. Combinar título y blurb como entrada textual
df['text'] = df['book_title'].fillna('') + '. ' + df['blurb'].fillna('')

# 3. Generar embeddings para los libros
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
book_embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)

# 4. Función para recomendar libros basados en tags
def recommend_books_by_tags(input_tags, top_n=5):
    # Combinar los tags en un único string
    tags_text = ', '.join(input_tags)
    
    # Generar embeddings para los tags proporcionados
    tags_embedding = model.encode([tags_text])
    
    # Calcular la similitud coseno entre los embeddings de los libros y los tags
    similarities = cosine_similarity(tags_embedding, book_embeddings).flatten()
    
    # Obtener los índices de los libros más similares
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    # Retornar los títulos de los libros recomendados
    recommended_books = df.iloc[top_indices][['book_title', 'tags']]
    return recommended_books

# 5. Guardar modelo y objetos necesarios
joblib.dump(model, 'model/book_recommendation_by_tags.joblib')


# %%
# 5. Ejemplo de uso 1
input_tags = ["faith", "spirituality", "selfhelp"]
recommended_books = recommend_books_by_tags(input_tags, top_n=5)
print("Libros recomendados:")
print(recommended_books)

# %%
# 5. Ejemplo de uso 2
input_tags = ["adventures", "fiction", "fantasy"]
recommended_books = recommend_books_by_tags(input_tags, top_n=5)
print("Libros recomendados:")
print(recommended_books)


