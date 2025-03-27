# 📚 BiblioNLP: Automatic Tagging & Sentiment Analysis for Books

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-green)](https://www.nltk.org/)
[![Transformers](https://img.shields.io/badge/Transformers-Hugging%20Face-orange)](https://huggingface.co/)

¡Bienvenido/a a **BiblioNLP**! Este proyecto combina **Procesamiento de Lenguaje Natural (NLP)** con análisis de reseñas y descripciones de libros para **generar etiquetas automáticamente** y proporcionar **búsquedas semánticas** en el contenido. Además, incluye un pipeline para **análisis de sentimientos**.

---

## ✨ Características Principales

- 🔎 **Búsqueda Semántica**: Encuentra libros y pasajes basados en similitud semántica, no solo por palabras clave.
- 🏷 **Generación Automática de Etiquetas**: Extrae etiquetas relevantes (topics/keywords) a partir de descripciones de libros o reseñas.
- 😊 **Análisis de Sentimientos**: Analiza emociones en descripciones de libros usando **VADER** y **RoBERTa**.
- 📊 **Embeddings**: Utiliza modelos preentrenados (ej. [Sentence-BERT](https://www.sbert.net/)) y bases vectoriales (ej. [Pinecone](https://www.pinecone.io/)) para la indexación y recuperación.
- 🗃 **Base de Datos MySQL**: Integra la información de libros, autores y tags almacenados en una base de datos relacional.

---

## 📂 Estructura del Proyecto

```plaintext
BiblioNLP-Automatic-Tagging-Semantic-Book-Search/
├── data/
│   ├── raw/             # Datos crudos (blurbs, reseñas, etc.)
│   └── processed/       # Datos procesados para uso del modelo
├── notebooks/
│   ├── eda.ipynb        # Exploratory Data Analysis
│   ├── nlp_sentiment_analysis_books_pipeline_es.ipynb  # Sentiment analysis pipeline
├── src/
│   ├── embeddings/      # Scripts para generar y almacenar embeddings
│   ├── search/          # Lógica de búsqueda semántica
│   ├── tagging/         # Módulo de generación de etiquetas
│   └── utils/           # Funciones auxiliares (preprocesamiento, etc.)
├── requirements.txt     # Python dependencies
├── LICENSE              # License file
└── README.md            # Project documentation
```

---

## 🚀 Instalación y Uso

### 1. Clona el Repositorio

```bash
git clone https://github.com/tu-usuario/BiblioNLP-Automatic-Tagging-Semantic-Book-Search.git
cd BiblioNLP
```

### 2. Instala Dependencias

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configura Variables de Entorno

```bash
export MYSQL_USER="tu_usuario"
export MYSQL_PASS="tu_contraseña"
export PINECONE_API_KEY="tu_api_key"
```

### 4. Ejecuta un Script de Prueba

#### Generación de Etiquetas

```bash
python src/tagging/main.py
```

#### Análisis de Sentimientos

Ejecuta el notebook `notebooks/nlp_sentiment_analysis_books_pipeline_es.ipynb` para realizar análisis exploratorio de sentimientos.

#### Búsqueda Semántica

```bash
python src/search/semantic_search.py --query "magia y aventuras"
```

---

## 🛠 Técnicas y Modelos

### 1. **Pipeline de Generación de Etiquetas**

- **Input**: Combina títulos y descripciones de libros en un único campo de texto.
- **Embeddings**: Generados con `SentenceTransformer` usando el modelo multilingüe `paraphrase-multilingual-MiniLM-L12-v2`.
- **Clasificador**: Modelo de **Logistic Regression** envuelto en un `MultiOutputClassifier` para clasificación multilabel.
- **Output**: Predice etiquetas relevantes (ej. `philosophy`, `science`, `to-read`).

### 2. **Pipeline de Análisis de Sentimientos**

- **VADER**:
  - Herramienta basada en reglas para análisis de sentimientos.
  - Calcula puntajes de polaridad (`positive`, `negative`, `neutral`, `compound`).
- **RoBERTa**:
  - Modelo preentrenado (`j-hartmann/emotion-english-distilroberta-base`).
  - Detecta emociones como `joy`, `sadness`, `anger`, `fear`, etc.
  - Aplicado a descripciones de libros para obtener insights emocionales.

### 3. **Funciones Reutilizables**

- **`predict_tags`**:
  - Predice etiquetas para nuevos libros basándose en sus títulos y descripciones.
  - Clasificación multilabel con umbral ajustable.
- **`mooder`**:
  - Aplica RoBERTa para extraer puntajes de emociones desde texto.

---

## 📊 Ejemplo de Resultados

### Generación de Etiquetas

```python
new_titles = ["La conspiración del universo"]
new_blurbs = ["Una historia que entrelaza ciencia, fe y filosofía para revelar los secretos de la creación."]
predicted_tags = predict_tags(new_titles, new_blurbs)
print(predicted_tags)
# Output: [('philosophy', 'science', 'to-read')]
```

### Análisis de Sentimientos

- **VADER**: Visualiza la distribución de sentimientos (`compound` scores).
- **RoBERTa**: Analiza emociones como `joy` o `sadness` en descripciones de libros.

---

## 🛠 Streamlit

```python
streamlit run src/apps/app_streamlit.py --server.runOnSave true
```

## 🛠 Tecnologías y Herramientas

- 🐍 **Python 3.9+**
- 🧠 **Modelos NLP**: [Sentence-BERT](https://www.sbert.net/), [Hugging Face Transformers](https://huggingface.co/)
- 📊 **Visualización**: Matplotlib, Seaborn
- 🗂 **Procesamiento de Datos**: Pandas, NumPy
- 🏷 **Machine Learning**: Scikit-learn
- 🗄️ **Base de Datos**: MySQL (opcional)
- 🗂 **Base Vectorial**: Pinecone (opcional)

---

## 🙌 Contribuciones

¡Las contribuciones son siempre bienvenidas!

1. Haz un fork del proyecto.
2. Crea una rama para tu feature o bugfix (`git checkout -b nombre-rama`).
3. Haz commit de tus cambios (`git commit -m 'Agrego nueva funcionalidad'`).
4. Sube la rama (`git push origin nombre-rama`).
5. Abre un Pull Request detallando tus cambios.

---

## ⚖️ Licencia

Este proyecto se distribuye bajo la licencia [MIT](LICENSE). ¡Siéntete libre de usarlo y mejorarlo!

---

> **Nota:** Este repositorio es parte de un proyecto de aprendizaje e investigación de NLP, por lo cual no garantiza escalabilidad de producción sin ajustes adicionales.

¡Gracias por visitar **BiblioNLP** y que disfrutes explorando el mundo de los libros con NLP! 🎉
