import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import time
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import spacy
from collections import Counter
import sys
import subprocess
from pinecone import Pinecone, ServerlessSpec, exceptions
from spacy.lang.en.stop_words import STOP_WORDS as STOP_WORDS_EN
from spacy.lang.es.stop_words import STOP_WORDS as STOP_WORDS_ES
import re
import configparser
import os

st.set_page_config(page_title="BiblioNLP - Predicción de Tags", page_icon="📚", layout="wide")

# Leer configuración desde config.cfg
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), "../config.cfg")  # Ruta relativa al archivo config.cfg
config.read(config_path)

# Verificar si el archivo de configuración se cargó correctamente
if "pinecone" not in config:
    raise ValueError("No se encontró la sección [pinecone] en el archivo config.cfg")

# st.write(f"Python executable: {sys.executable}")
# st.write(f"Python version: {sys.version}")

# Constantes para las URLs de los modelos
TAGGING_MODEL_URL        = "model/book_tagging_pipeline.joblib"
MLB_MODEL_URL            = "model/book_tagging_pipeline_mlb.joblib"
CLUSTERING_MODEL_URL     = "model/book_clustering_kmeans.joblib"
RECOMMENDATION_MODEL_URL = "model/book_recommendation_by_tags.joblib"
# Cargar el modelo de sentimientos
SENTIMENT_MODEL_PATH     = "model/sentiment_roberta_model.joblib"
SENTIMENT_TOKENIZER_PATH = "model/sentiment_roberta_tokenizer.joblib"
sentiment_model          = joblib.load(SENTIMENT_MODEL_PATH)
sentiment_tokenizer      = joblib.load(SENTIMENT_TOKENIZER_PATH)

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
DEFAULT_BOOK_TITLE_2 = "The Flea: The Amazing Story of Leo Messi"
DEFAULT_BOOK_BLURB_2 = (
    "The captivating story of soccer legend Lionel Messi, from his first touch at age five in the streets of Rosario, Argentina, to his first goal on the Camp Nou pitch in Barcelona, Spain. The Flea tells the amazing story of a boy who was born to play the beautiful game and destined to become the world's greatest soccer player."
)
DEFAULT_TAGS_INPUT = "galaxies, spacetime, astrophysics"
TAGS_INPUT_2       = "messi, biography"

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

# Función para calcular sentimientos
def analyze_sentiments(text):
    max_len = 512
    encoded = sentiment_tokenizer(text, return_tensors="pt", max_length=max_len, truncation=True, padding="max_length")
    with torch.no_grad():
        output = sentiment_model(**encoded)
    scores = torch.softmax(output.logits, dim=1).squeeze().numpy()
    labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    return dict(zip(labels, scores))

# Función para generar gráfica de sentimientos
def plot_sentiments(sentiments):
    # Definir colores para los sentimientos
    sentiment_colors = { 
        "anger": "#b71c1c", # Rojo muy oscuro (máx. negatividad) 
        "disgust": "#d32f2f", # Rojo oscuro 
        "fear": "#ef5350", # Rojo algo más claro 
        "sadness": "#ffcdd2", # Tono salmón muy claro 
        "neutral": "#f0f0f0", # Gris muy claro (punto de transición) 
        "surprise": "#ffe082", # Amarillo pastel 
        "joy": "#ffd600" # Amarillo vivo (máx. positividad) 
    }

    # Ordenar los sentimientos de negativos a positivos
    ordered_labels = ["anger", "disgust", "fear", "sadness", "neutral", "surprise", "joy"]
    ordered_sentiments = {label: sentiments[label] for label in ordered_labels}

    # Crear la figura con mayor altura
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Asignar colores a las barras según el sentimiento
    colors = [sentiment_colors[label] for label in ordered_sentiments.keys()]
    
    # Crear el gráfico de barras
    ax.bar(ordered_sentiments.keys(), ordered_sentiments.values(), color=colors)
    
    # Configurar el título y etiquetas
    ax.set_title("Análisis de Sentimientos", fontsize=16)
    ax.set_ylabel("Puntuación", fontsize=16)
    ax.set_xlabel("Sentimientos", fontsize=16)
    
    # Aumentar el tamaño de los labels del eje X
    ax.set_xticklabels(ordered_sentiments.keys(), rotation=45, fontsize=12)
    
    # Ajustar el diseño
    plt.tight_layout()
    return fig

# Función para colorear etiquetas reales
def format_real_tags(real_tags, predicted_tags):
    formatted_tags = []
    for tag in real_tags:
        if tag in predicted_tags:
            # Verde para indicar coincidencia con las etiquetas predichas
            formatted_tags.append(f'<span style="color:green">{tag}</span>')
        else:
            # Color normal para las etiquetas que no coinciden
            formatted_tags.append(f'<span>{tag}</span>')
    return ", ".join(formatted_tags)

# Función para colorear tags predichos
def format_predicted_tags(predicted_tags, real_tags, scores):
    formatted_tags = []
    for tag, score in zip(predicted_tags, scores):
        if tag in real_tags:
            # Verde para coincidencias
            formatted_tags.append(f'<span style="color:green">{tag}</span>')
        else:
            # Amarillo (gold) para etiquetas que no coinciden
            formatted_tags.append(f'<span style="color:gold">{tag}</span>')
    return ", ".join(formatted_tags)

# Función para colorear etiquetas Pinecone
def format_pinecone_tags(pinecone_tags, real_tags):
    formatted_tags = []
    for tag in pinecone_tags:
        if tag in real_tags:
            # Verde para coincidencias con las etiquetas reales
            formatted_tags.append(f'<span style="color:green">{tag}</span>')
        else:
            # Sin color para etiquetas que no coinciden
            formatted_tags.append(f'<span style="color:gold">{tag}</span>')
    return ", ".join(formatted_tags)

# Función para colorear etiquetas sustantivas
def format_noun_tags(noun_tags, real_tags):
    formatted_tags = []
    for tag in noun_tags:
        if tag in real_tags:
            # Verde para coincidencias con las etiquetas reales
            formatted_tags.append(f'<span style="color:green">{tag}</span>')
        else:
            # Sin color para etiquetas que no coinciden
            formatted_tags.append(f'<span>{tag}</span>')
    return ", ".join(formatted_tags)

# Pinecone: Cargar modelo de spaCy para sustantivos
# Inicializar Pinecone
PINECONE_API_KEY = config["pinecone"]["api_key"]
PINECONE_ENV = config["pinecone"]["environment"]
INDEX_NAME = "book-embeddings"  # Nombre del índice usado en el notebook

# Inicializar Pinecone usando la nueva API
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in pc.list_indexes().names():
        # Crear el índice si no existe
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # Cambia esto según las dimensiones de tus embeddings
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
        )
        st.info("Índice Pinecone creado correctamente.")
    else:
        st.info("Índice Pinecone existente. Conectando...")

    # Obtener el índice
    index = pc.Index(INDEX_NAME)
except exceptions.PineconeApiException as e:
    st.error(f"Error al inicializar Pinecone: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error inesperado al inicializar Pinecone: {e}")
    st.stop()

# Función para predecir etiquetas fusionadas (Logistic Regression + Pinecone + Nouns)
# Cargar modelo de spaCy
try:
    # nlp = spacy.load("es_core_news_sm")
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "es_core_news_sm"])
    nlp = spacy.load("es_core_news_sm")

# Combinar stopwords en español e inglés
STOPWORDS_COMBINADAS = STOP_WORDS_EN.union(STOP_WORDS_ES)

# Expresión regular para sufijos verbales en español
VERBAL_SUFFIXES_ES = re.compile(r"(ar|er|ir|ado|ido|ando|iendo|ándose|iéndose|éndose)$")

# Función para predecir etiquetas fusionadas (Logistic Regression + Pinecone + Nouns)
def predict_with_ensemble(title, blurb, top_k=5, threshold=0.3, enrich_with_nouns=True, pinecone_top_tags=6):
    text = title + ". " + blurb
    embedding = embedding_model.encode([text])[0]

    # A. Logistic Regression
    probs = np.array([estimator.predict_proba(embedding.reshape(1, -1))[0][1] for estimator in clf.estimators_])
    pred_lr = (probs >= threshold).astype(int)
    pred_lr = np.array([pred_lr])
    tags_lr = list(mlb.inverse_transform(pred_lr)[0])

    # B. Pinecone (top-K vecinos más cercanos)
    pinecone_result = index.query(
        vector=embedding.tolist(),
        top_k=top_k,
        include_metadata=True,
        namespace="books"
    )

    pinecone_all_tags = []
    for match in pinecone_result.matches:
        if 'tags' in match.metadata and match.metadata['tags']:
            pinecone_all_tags += [tag.strip().lower() for tag in match.metadata['tags'].split(',')]

    pinecone_tag_counts = Counter(pinecone_all_tags)
    tags_pinecone = [tag for tag, _ in pinecone_tag_counts.most_common(pinecone_top_tags)]

    # C. Sustantivos relevantes del título y del blurb
    tags_nouns = []
    if enrich_with_nouns:
        doc = nlp(title + ". " + blurb)
        tags_nouns = sorted(set(
            token.lemma_.lower()
            for token in doc
            if token.pos_ in ["NOUN", "PROPN"]
            and token.pos_ not in ["VERB", "AUX"]
            and not VERBAL_SUFFIXES_ES.search(token.lemma_.lower())
            and token.lemma_.lower() not in STOPWORDS_COMBINADAS
            and token.is_alpha
            and len(token) > 3
        ))

    # D. Fusión final de tags
    fusion_final = sorted(set(
        [t.lower() for t in tags_lr] +
        tags_pinecone +
        tags_nouns
    ))

    return {
        "tags_logistic": sorted(tags_lr),
        "tags_pinecone": tags_pinecone,
        "tags_nouns": tags_nouns,
        "tags_fusion": fusion_final
    }

# Crear pestañas
tab_story, tab0, tab1, tab2, tab3 = st.tabs(["StoryTelling", "Overview", "Tag Generator", "Book Recommender", "To-Do List"])

# === TAB StoryTelling ===
with tab_story:
    st.title("StoryTelling")
    st.markdown("### The Motivation Behind the Project")
    st.markdown("""
    Books have always been a gateway to knowledge, imagination, and personal growth. However, in a world where millions of books are published every year, finding the right book can feel overwhelming.  
    This project was born out of a love for literature and a desire to make the process of discovering books more intuitive and meaningful.  
    """)

    st.markdown("### The Problem We Aim to Solve")
    st.markdown("""
    - **Overwhelming Choices**: With so many books available, readers often struggle to find books that truly resonate with their interests.  
    - **Lack of Personalization**: Traditional recommendation systems often fail to capture the nuanced preferences of readers.  
    - **Disconnected Metadata**: Many books lack proper tagging, making it difficult to categorize and recommend them effectively.  
    """)

    st.markdown("### Our Vision")
    st.markdown("""
    Imagine a world where every reader can effortlessly discover books that inspire, educate, and entertain them.  
    Our goal is to bridge the gap between readers and books by leveraging the power of machine learning and natural language processing to create a seamless and personalized book discovery experience.  
    """)

    st.markdown("### Why This Matters")
    st.markdown("""
    Books have the power to change lives. By helping readers find the right books, we can:  
    - Foster a deeper love for reading.  
    - Encourage lifelong learning.  
    - Connect people with stories and ideas that resonate with their unique journeys.  
    """)

    # Add an emotional image to enhance storytelling
    # st.image("img/storytelling_books.jpg", caption="Books: A Gateway to Knowledge and Imagination", use_column_width=True)


# === TAB 0 ===
with tab0:
    st.title("Overview")
    st.markdown("### How the Application Works")
    st.markdown("""
    This application is designed to predict tags for books and recommend similar books based on their descriptions. 
    It uses machine learning models and embeddings to process the input data and generate meaningful results.
    """)

    st.markdown("### Data Sources")
    st.markdown("""
    - **Processed Books Dataset**: `data/processed/books.csv`  
      Contains book titles, blurbs, and tags.  
      **Shape**: `pd.read_csv('data/processed/books.csv').shape`
    - **Clustering Dataset**: `data/processed/clustering_books.csv`  
      Contains book titles and their assigned clusters.  
      **Shape**: `pd.read_csv('data/processed/clustering_books.csv').shape`
    - **Raw Goodreads Dataset**: `data/raw/goodreads_data.csv`  
      Contains raw book data including genres and descriptions.  
      **Shape**: `pd.read_csv('data/raw/goodreads_data.csv').shape`
    """)

    st.markdown("### Models Used")
    st.markdown("""
    - **Logistic Regression**: Used for tag prediction.  
    - **SentenceTransformer**: Embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) for semantic similarity.  
    - **KMeans Clustering**: Groups books into clusters based on their embeddings.  
    - **Pinecone**: Retrieves similar books using vector search.  
    - **Sentiment Analysis**: RoBERTa-based model for analyzing the sentiment of book blurbs.
    """)

    st.markdown("### Data Transformation Processes")
    st.markdown("""
    - **Text Embedding**: Converts book titles and blurbs into numerical vectors using SentenceTransformer.  
    - **Tag Prediction**: Logistic regression predicts tags based on embeddings.  
    - **Clustering**: Groups books into clusters for similarity-based recommendations.  
    - **Pinecone Integration**: Retrieves top-K similar books using cosine similarity.  
    - **Noun Extraction**: Extracts relevant nouns from text using spaCy for additional tag enrichment.
    """)

    st.markdown("### Visualizations")
    st.markdown("Below are some visualizations of the data and model performance:")

    # Example visualization: Distribution of clusters
    clustering_books_df = pd.read_csv("data/processed/clustering_books.csv")
    cluster_counts = clustering_books_df["cluster"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(cluster_counts.index, cluster_counts.values, color="skyblue")
    ax.set_title("Distribution of Clusters")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Books")
    st.pyplot(fig)

    # Example image from the root directory
    # st.image("img/overview_diagram.png", caption="System Overview", use_column_width=True)

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
                st.markdown("### Libros similares en el mismo cluster:")
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
                value=DEFAULT_TAGS_INPUT if i == 0 else TAGS_INPUT_2
            )
            tags_inputs.append(tags_input)

        num_recommendations = st.number_input("Número de libros a recomendar", min_value=1, max_value=10, value=10)
        recommend_button = st.form_submit_button(label="Recomendar")

    # Actualizar el código de recomendaciones en tab2
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

                    if i + 1 == 2:  # Si es el Set 2
                        books_df = pd.read_csv("data/raw/goodreads_data.csv")
                        books_df.rename(columns={"Book": "book_title", "Description": "blurb", "Genres": "tags"}, inplace=True)

                        # Transformar las etiquetas en minúsculas y reemplazar espacios con guiones altos
                        books_df["tags"] = books_df["tags"].apply(
                            lambda x: ", ".join(tag.strip().lower().replace(" ", "-") for tag in eval(x))
                        )
                    else:
                        books_df = pd.read_csv("data/processed/books.csv")

                    # Calcular similitudes
                    similarities = cosine_similarity(tags_embedding, book_embeddings).flatten()
                    top_indices = similarities.argsort()[-num_recommendations:][::-1]
                    recommended_books = books_df.iloc[top_indices][["book_title", "blurb", "tags"]]
                    
                    if recommended_books.empty:
                        st.warning(f"No se encontraron libros recomendados para el Set {i + 1}.")
                    else:
                        st.markdown(f"## Recomendaciones para el Set {i + 1}:")
                        for _, row in recommended_books.iterrows():
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.markdown(f"### 📖 {row['book_title']}")

                                # Predicción de etiquetas
                                text = row["blurb"]
                                X_test = embedding_model.encode([text])
                                preds_proba = np.array([proba[:, 1] for proba in clf.predict_proba(X_test)]).T
                                preds = (preds_proba >= 0.3).astype(int)
                                predicted_tags = mlb.inverse_transform(preds)[0]
                                scores = preds_proba[0]

                                # Etiquetas reales con coincidencias resaltadas
                                real_tags = row["tags"].split(", ")
                                formatted_real_tags = format_real_tags(real_tags, predicted_tags)
                                st.markdown(f"- **Tags reales:** {formatted_real_tags}", unsafe_allow_html=True)

                                # Formatear etiquetas predichas
                                formatted_tags = format_predicted_tags(predicted_tags, real_tags, scores)
                                st.markdown(f"- **Predicted by my model:** {formatted_tags}", unsafe_allow_html=True)

                                # Predicción de etiquetas Pinecone
                                ensemble_result = predict_with_ensemble(row["book_title"], row["blurb"])
                                pinecone_tags = ensemble_result["tags_pinecone"]
                                pinecone_scores = [0.5] * len(pinecone_tags)  # Placeholder scores
                                # formatted_pinecone_tags = format_predicted_tags(pinecone_tags, real_tags, pinecone_scores)
                                formatted_pinecone_tags = format_pinecone_tags(pinecone_tags, real_tags)
                                st.markdown(f"- **Predicted by Others (Pinecone):** {formatted_pinecone_tags}", unsafe_allow_html=True)

                                # Etiquetas sustantivas
                                noun_tags = ensemble_result["tags_nouns"]
                                formatted_noun_tags = format_noun_tags(noun_tags, real_tags)
                                st.markdown(f"- **From Blurb (Lemmatized):** {formatted_noun_tags}", unsafe_allow_html=True)
                            with col2:
                                sentiments = analyze_sentiments(row["blurb"])
                                fig = plot_sentiments(sentiments)
                                st.pyplot(fig)

# === TAB 3 ===
with tab3:
    st.title("To-Do List")
    st.markdown("""
    ### Future Improvements
    - **Model Enhancements**:  
      - Fine-tune the logistic regression model for better tag prediction accuracy.  
      - Experiment with other embedding models like `all-MiniLM-L6-v2` for improved semantic understanding.
    - **Data Augmentation**:  
      - Add more diverse datasets to improve model generalization.  
      - Include multilingual datasets for better language support.
    - **UI/UX Improvements**:  
      - Add more interactive visualizations.  
      - Allow users to upload custom datasets for predictions.
    - **Performance Optimization**:  
      - Optimize Pinecone queries for faster recommendations.  
      - Reduce memory usage by batching large datasets.
    - **Additional Features**:  
      - Add a "Download Results" button for exporting predictions and recommendations.  
      - Implement a feedback loop to improve model predictions based on user input.
    """)

    # Example image for the To-Do List
    # st.image("img/todo_list.png", caption="Future Improvements", use_column_width=True)