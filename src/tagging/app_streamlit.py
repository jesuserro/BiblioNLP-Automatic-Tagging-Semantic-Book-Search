import streamlit as st
import requests

st.set_page_config(page_title="BiblioNLP - Predicción de Tags", page_icon="📚")

st.title("BiblioNLP - Predicción automática de etiquetas")
st.markdown("Introduce el título y la sinopsis de uno o más libros para predecir sus etiquetas automáticamente.")

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
        # Preparar datos y enviar al endpoint
        payload = {"titles": titles, "blurbs": blurbs}
        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            if response.status_code == 200:
                results = response.json()["predicted_tags"]
                st.success("Etiquetas predichas:")
                for i, tags in enumerate(results):
                    st.markdown(f"**Libro {i + 1}:** {titles[i]}")
                    st.write(f"Etiquetas: {', '.join(tags) if tags else 'Ninguna etiqueta detectada'}")
            else:
                st.error(f"Error en la API: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("No se pudo conectar al servidor FastAPI. Asegúrate de que esté corriendo en http://127.0.0.1:8000.")
