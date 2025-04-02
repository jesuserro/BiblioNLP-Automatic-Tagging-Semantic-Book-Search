import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import time

# Constantes para las URLs de los modelos
TAGGING_MODEL_URL        = "model/book_tagging_pipeline.joblib"
MLB_MODEL_URL            = "model/book_tagging_pipeline_mlb.joblib"
CLUSTERING_MODEL_URL     = "model/book_clustering_kmeans.joblib"
RECOMMENDATION_MODEL_URL = "model/book_recommendation_by_tags.joblib"

# Constantes para valores por defecto
DEFAULT_BOOK_TITLE = "The Dark Interval: Letters on Loss, Grief, and Transformation"
DEFAULT_BOOK_BLURB = (
    "From the writer of the classic Letters to a Young Poet, reflections on grief and loss, collected and published here in one volume for the first time.\n\n"
    "“A great poet’s reflections on our greatest mystery.”—Billy Collins\n\n"
    "“A treasure . . . The solace Rilke offers is uncommon, uplifting and necessary.”—The Guardian\n\n"
    "Gleaned from Rainer Maria Rilke’s voluminous, never-before-translated letters to bereaved friends and acquaintances, The Dark Interval is a profound vision of the mourning process and a meditation on death’s place in our lives. Following the format of Letters to a Young Poet, this book arranges Rilke’s letters into an uninterrupted sequence, showcasing the full range of the great author’s thoughts on death and dying, as well as his sensitive and moving expressions of consolation and condolence.\n\n"
    "Presented with care and authority by master translator Ulrich Baer, The Dark Interval is a literary treasure, an indispensable resource for anyone searching for solace, comfort, and meaning in a time of grief.\n\n"
    "Praise for The Dark Interval\n\n"
    "“Even though each of these letters of condolence is personalized with intimate detail, together they hammer home Rilke’s remarkable truth about the death of another: that the pain of it can force us into a ‘deeper . . . level of life’ and render us more ‘vibrant.’ Here we have a great poet’s reflections on our greatest mystery.”—Billy Collins\n\n"
    "“As we live our lives, it is possible to feel not sadness or melancholy but a rush of power as the life of others passes into us. This rhapsodic volume teaches us that death is not a negation but a deepening experience in the onslaught of existence. What a wise and victorious book!”—Henri Cole"
)
DEFAULT_BOOK_TITLE_2 = "Messi: Edición revisada y actualizada (Biografías y memorias)"
DEFAULT_BOOK_BLURB_2 = (
    "Leo Messi es el jugador de fútbol más conocido del planeta, pero también un enigma como persona, por su hermetismo. Esta biografía, que fue publicada por primera vez en 2014, y posteriormente actualizada en 2018, se presenta de nuevo en una edición que recoge lo más relevante de los últimos años del jugador en el Fútbol Club Barcelona. \n\n"
    "En esta nueva edición, el autor repasa lo más destacado desde aquel fatídico Mundial de Brasil hasta el final de la temporada 2017/18, así como su paso por el Mundial de Rusia y por la Copa América 2021, que coincidía con el momento en que expiraba su contrato con el Fútbol Club Barcelona, y que convirtió al astro argentino en foco de todas las miradas, generando una enorme expectación.\n\n"
    "En agosto de 2021, se anunció el desenlace que parecía imposible: Messi no pudo renovar en el Barça y se anunció su fichaje por el PSG. ¿Qué pasó? ¿Cómo es posible que, queriendo quedarse, tuviera que salir?"
)
DEFAULT_TAGS_INPUT = "galaxies, spacetime, astrophysics"

st.set_page_config(page_title="BiblioNLP - Predicción de Tags", page_icon="📚")

st.title("BiblioNLP - Predicción automática de etiquetas")
st.markdown(
    "Introduce el título y la sinopsis de uno o más libros para predecir sus etiquetas automáticamente o para obtener recomendaciones."
)

@st.cache_resource
def load_models():
    clf = joblib.load(TAGGING_MODEL_URL)
    mlb = joblib.load(MLB_MODEL_URL)
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    clustering_model = joblib.load(CLUSTERING_MODEL_URL)
    return clf, mlb, model

clf, mlb, embedding_model = load_models()

# Crear pestañas
tab1, tab2 = st.tabs(["Predicción de etiquetas", "Recomendaciones"])

# === TAB 1 ===
with tab1:
    with st.form(key="tag_form"):
        titles, blurbs = [], []

        for i in range(2):  # Mostrar siempre 2 libros por defecto
            st.subheader(f"Libro {i + 1}")
            title = st.text_input(
                f"Título del libro {i + 1}",
                key=f"title_{i}",
                value=DEFAULT_BOOK_TITLE if i == 0 else (DEFAULT_BOOK_TITLE_2 if i == 1 else "")
            )
            blurb = st.text_area(
                f"Blurb / Sinopsis del libro {i + 1}",
                key=f"blurb_{i}",
                value=DEFAULT_BOOK_BLURB if i == 0 else (DEFAULT_BOOK_BLURB_2 if i == 1 else "")
            )
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
                clustering_model = joblib.load(CLUSTERING_MODEL_URL)
                clusters = clustering_model.predict(X_test)

            st.success("Resultados:")
            clustering_books_df = pd.read_csv("data/processed/clustering_books.csv")
            for i, (tags, cluster) in enumerate(zip(predicted_tags, clusters)):
                st.markdown(f"## Libro {i + 1}: {titles[i]}")
                st.write(f"Etiquetas: `{', '.join(tags)}`" if tags else "Ninguna etiqueta detectada")
                
                # Mostrar libros similares del mismo cluster
                cluster_books = clustering_books_df[clustering_books_df["cluster"] == cluster].head(5)
                st.markdown("## Libros similares en el mismo cluster:")
                for idx, row in cluster_books.iterrows():
                    st.markdown(f"- {row['book_title']}")

# === TAB 2 ===
with tab2:
    with st.form(key="recommendation_form"):
        tags_inputs = []
        for i in range(2):  # Mostrar siempre 2 sets de etiquetas por defecto
            st.subheader(f"Set de etiquetas {i + 1}")
            tags_input = st.text_input(
                f"Introduce etiquetas separadas por comas (Set {i + 1})",
                key=f"tags_input_{i}",
                value=DEFAULT_TAGS_INPUT if i == 0 else "deportes, fútbol, messi"
            )
            tags_inputs.append(tags_input)

        num_recommendations = st.number_input("Número de libros a recomendar", min_value=1, max_value=10, value=5)
        recommend_button = st.form_submit_button(label="Recomendar")

    if recommend_button:
        if any(tags.strip() == "" for tags in tags_inputs):
            st.warning("Por favor, introduce al menos una etiqueta en cada set.")
        else:
            with st.spinner("Procesando recomendaciones..."):
                recommendation_model = joblib.load(RECOMMENDATION_MODEL_URL)
                books_df = pd.read_csv("data/processed/books.csv")
                book_embeddings = recommendation_model.encode(books_df["blurb"].tolist())

                st.success("Resultados:")
                for i, tags_input in enumerate(tags_inputs):
                    input_tags = [tag.strip() for tag in tags_input.split(",")]

                    # Crear el progress bar dentro del procesamiento
                    progress_bar = st.progress(0)

                    tags_text = ", ".join(input_tags)
                    tags_embedding = recommendation_model.encode([tags_text])

                    # Actualizar el progreso
                    for j in range(50):
                        time.sleep(0.01)
                        progress_bar.progress(int((j + 1) * 100 / 50))

                    similarities = cosine_similarity(tags_embedding, book_embeddings).flatten()
                    top_indices = similarities.argsort()[-num_recommendations:][::-1]
                    recommended_books = books_df.iloc[top_indices][["book_title", "tags"]]

                    if recommended_books.empty:
                        st.warning(f"No se encontraron libros recomendados para el Set {i + 1}.")
                    else:
                        st.markdown(f"## Recomendaciones para el Set {i + 1}:")
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