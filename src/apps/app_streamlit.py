import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

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