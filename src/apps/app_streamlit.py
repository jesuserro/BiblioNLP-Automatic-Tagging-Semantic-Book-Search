import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="BiblioNLP - Predicción de Tags", page_icon="📚")

st.title("BiblioNLP - Predicción automática de etiquetas")
st.markdown("Introduce el título y la sinopsis de uno o más libros para predecir sus etiquetas automáticamente.")

# Cargar modelos y objetos necesarios
@st.cache_resource
def load_models():
    clf = joblib.load("model/book_tagging_pipeline.joblib")
    mlb = joblib.load("model/book_tagging_pipeline_mlb.joblib")
    
    # Use a valid Hugging Face model identifier or local folder
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  # Replace with your model
    return clf, mlb, model

clf, mlb, model = load_models()

# Función para predecir etiquetas
def predict_tags(titles, blurbs, threshold=0.3):
    texts = [t + ". " + b for t, b in zip(titles, blurbs)]
    X_test = model.encode(texts)
    preds_proba = np.array([proba[:, 1] for proba in clf.predict_proba(X_test)]).T
    preds = (preds_proba >= threshold).astype(int)
    tag_lists = mlb.inverse_transform(preds)
    return tag_lists

# Formulario de entrada
with st.form(key="tag_form"):
    num_books = st.number_input("¿Cuántos libros deseas evaluar?", min_value=1, max_value=5, value=1)

    titles = []
    blurbs = []

    for i in range(num_books):
        st.subheader(f"Libro {i + 1}")
        title = st.text_input(f"Título del libro {i + 1}", key=f"title_{i}")
        blurb = st.text_area(f"Blurb / Sinopsis del libro {i + 1}", key=f"blurb_{i}")
        titles.append(title)
        blurbs.append(blurb)

    submit_button = st.form_submit_button(label="Predecir etiquetas")

# Al hacer submit
if submit_button:
    # Validación rápida
    if any(t.strip() == "" or b.strip() == "" for t, b in zip(titles, blurbs)):
        st.warning("Por favor, completa todos los títulos y blurbs.")
    else:
        # Realizar predicción
        predicted_tags = predict_tags(titles, blurbs)
        st.success("Etiquetas predichas:")
        for i, tags in enumerate(predicted_tags):
            st.markdown(f"**Libro {i + 1}:** {titles[i]}")
            st.write(f"Etiquetas: {', '.join(tags) if tags else 'Ninguna etiqueta detectada'}")

# Ui parte 2

from sklearn.metrics.pairwise import cosine_similarity
import time  # Para simular progreso

# Cargar el modelo de recomendación
@st.cache_resource
def load_recommendation_model():
    model = joblib.load("model/book_recommendation_by_tags.joblib")
    df = pd.read_csv("data/processed/books.csv")
    
    # Generar la columna 'text' combinando 'book_title' y 'blurb'
    df['text'] = df['book_title'].fillna('') + '. ' + df['blurb'].fillna('')
    
    return model, df

recommendation_model, books_df = load_recommendation_model()

# Función para recomendar libros
def recommend_books_by_tags(input_tags, top_n=5):
    if not input_tags:
        return pd.DataFrame()  # Retornar un DataFrame vacío si no hay tags

    tags_text = ", ".join(input_tags)
    
    # Mostrar spinner mientras se generan los embeddings
    with st.spinner("Generando recomendaciones, por favor espera..."):
        tags_embedding = recommendation_model.encode([tags_text])
        book_embeddings = recommendation_model.encode(books_df['text'].tolist())
        
        # Simular progreso
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)  # Simular tiempo de procesamiento
            progress_bar.progress(i + 1)
        
        # Calcular similitudes
        similarities = cosine_similarity(tags_embedding, book_embeddings).flatten()
        top_indices = similarities.argsort()[-top_n:][::-1]
        recommended_books = books_df.iloc[top_indices][['book_title', 'tags']]
    
    return recommended_books

# Manejar estado para los resultados
if "recommended_books" not in st.session_state:
    st.session_state["recommended_books"] = None

# Formulario de entrada para recomendación
with st.form(key="recommendation_form"):
    tags_input = st.text_input("Introduce etiquetas separadas por comas", key="tags_input")
    num_recommendations = st.number_input("Número de libros a recomendar", min_value=1, max_value=10, value=5)
    recommend_button = st.form_submit_button(label="Recomendar")

# Al hacer submit en el formulario de recomendación
if recommend_button:
    if tags_input.strip() == "":
        st.warning("Por favor, introduce al menos una etiqueta.")
    else:
        input_tags = [tag.strip() for tag in tags_input.split(",")]
        st.session_state["recommended_books"] = recommend_books_by_tags(input_tags, top_n=num_recommendations)

# Mostrar resultados si existen
if st.session_state["recommended_books"] is not None:
    recommended_books = st.session_state["recommended_books"]
    if recommended_books.empty:
        st.warning("No se encontraron libros recomendados para las etiquetas proporcionadas.")
    else:
        st.success("Libros recomendados:")
        for _, row in recommended_books.iterrows():
            st.markdown(f"**Título:** {row['book_title']}")
            st.write(f"Etiquetas: {row['tags']}")