import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import time

st.set_page_config(page_title="BiblioNLP - Predicción de Tags", page_icon="📚")

st.title("BiblioNLP - Predicción automática de etiquetas")
st.markdown(
    "Introduce el título y la sinopsis de uno o más libros para predecir sus etiquetas automáticamente o para obtener recomendaciones."
)

@st.cache_resource
def load_models():
    clf = joblib.load("model/book_tagging_pipeline.joblib")
    mlb = joblib.load("model/book_tagging_pipeline_mlb.joblib")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    clustering_model = joblib.load("model/book_clustering_kmeans.joblib")
    return clf, mlb, model

clf, mlb, embedding_model = load_models()

# Crear pestañas
tab1, tab2 = st.tabs(["Predicción de etiquetas", "Recomendaciones"])

# === TAB 1 ===
with tab1:
    with st.form(key="tag_form"):
        num_books = st.number_input("¿Cuántos libros deseas evaluar?", min_value=1, max_value=5, value=1)
        titles, blurbs = [], []

        for i in range(num_books):
            st.subheader(f"Libro {i + 1}")
            title = st.text_input(f"Título del libro {i + 1}", key=f"title_{i}")
            blurb = st.text_area(f"Blurb / Sinopsis del libro {i + 1}", key=f"blurb_{i}")
            titles.append(title)
            blurbs.append(blurb)

        submit_button = st.form_submit_button(label="Predecir etiquetas")

    if submit_button:
        if any(t.strip() == "" or b.strip() == "" for t, b in zip(titles, blurbs)):
            st.warning("Por favor, completa todos los títulos y blurbs.")
        else:
            with st.spinner("Generando etiquetas y clustering..."):
                progress_bar = st.progress(0)
                texts = [t + ". " + b for t, b in zip(titles, blurbs)]
                X_test = embedding_model.encode(texts)

                for i in range(30):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)

                # Predicción de etiquetas
                preds_proba = np.array([proba[:, 1] for proba in clf.predict_proba(X_test)]).T
                preds = (preds_proba >= 0.3).astype(int)
                predicted_tags = mlb.inverse_transform(preds)

                # Predicción de clustering
                clustering_model = joblib.load("model/book_clustering_kmeans.joblib")
                clusters = clustering_model.predict(X_test)

            st.success("Resultados:")
            clustering_books_df = pd.read_csv("data/processed/clustering_books.csv")
            for i, (tags, cluster) in enumerate(zip(predicted_tags, clusters)):
                st.markdown(f"**Libro {i + 1}:** {titles[i]}")
                st.write(f"Etiquetas: {', '.join(tags) if tags else 'Ninguna etiqueta detectada'}")

                # Mostrar libros similares del mismo cluster
                cluster_books = clustering_books_df[clustering_books_df["cluster"] == cluster].head(5)
                st.markdown("**Libros similares en el mismo cluster:**")
                for idx, row in cluster_books.iterrows():
                    st.markdown(f"- {row['book_title']}")

# === TAB 2 ===
with tab2:
    with st.form(key="recommendation_form"):
        tags_input = st.text_input("Introduce etiquetas separadas por comas")
        num_recommendations = st.number_input("Número de libros a recomendar", min_value=1, max_value=10, value=5)
        recommend_button = st.form_submit_button(label="Recomendar")

    if recommend_button:
        if tags_input.strip() == "":
            st.warning("Por favor, introduce al menos una etiqueta.")
        else:
            input_tags = [tag.strip() for tag in tags_input.split(",")]

            with st.spinner("Buscando libros similares..."):
                progress_bar = st.progress(0)

                tags_text = ", ".join(input_tags)
                tags_embedding = recommendation_model.encode([tags_text])
                book_embeddings = recommendation_model.encode(books_df["text"].tolist())

                for i in range(50):
                    time.sleep(0.01)
                    progress_bar.progress(int((i + 1) * 100 / 50))

                similarities = cosine_similarity(tags_embedding, book_embeddings).flatten()
                top_indices = similarities.argsort()[-num_recommendations:][::-1]
                recommended_books = books_df.iloc[top_indices][["book_title", "tags"]]

            if recommended_books.empty:
                st.warning("No se encontraron libros recomendados para las etiquetas proporcionadas.")
            else:
                st.success("Libros recomendados:")
                for _, row in recommended_books.iterrows():
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #444; border-radius: 8px; padding: 10px; margin-bottom: 15px; background-color: #1e1e1e;">
                            <h4 style="color: #f1f1f1; margin-bottom: 5px;">📖 {row['book_title']}</h4>
                            <p style="margin: 0; color: #cccccc;"><strong>Etiquetas:</strong> 
                            <span style="color: #00aced;">{', '.join(row['tags'].split(', '))}</span></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
