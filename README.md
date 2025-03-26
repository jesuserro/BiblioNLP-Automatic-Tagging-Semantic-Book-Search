# 📚 BiblioNLP: Automatic Tagging & Semantic Book Search

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

¡Bienvenido/a a **BiblioNLP**! Este proyecto combina **Procesamiento de Lenguaje Natural (NLP)** con análisis de reseñas y descripciones de libros para **generar etiquetas automáticamente** y proporcionar **búsquedas semánticas** en el contenido.

## ✨ Características Principales

- 🔎 **Búsqueda Semántica**: Encuentra libros y pasajes basados en similitud semántica, no solo por palabras clave.
- 🏷 **Generación Automática de Etiquetas**: Extrae etiquetas relevantes (topics/keywords) a partir de descripciones de libros o reseñas.
- 📊 **Embeddings**: Utiliza modelos pre-entrenados (ej. [Sentence-BERT](https://www.sbert.net/)) y bases vectoriales (ej. [Pinecone](https://www.pinecone.io/)) para la indexación y recuperación.
- 🗃 **Base de Datos MySQL**: Integra la información de libros, autores y tags almacenados en una base de datos relacional.

## 📂 Estructura del Proyecto

``` txt
BiblioNLP-Automatic-Tagging-Semantic-Book-Search/
├── data/
│   ├── raw/             # Datos crudos (blurbs, reseñas, etc.)
│   └── processed/       # Datos procesados para uso del modelo
├── notebooks/
│   ├── EDA.ipynb        # Exploratory Data Analysis
│   └── TagGenerator.ipynb
├── src/
│   ├── embeddings/      # Scripts para generar y almacenar embeddings
│   ├── search/          # Lógica de búsqueda semántica
│   ├── tagging/         # Módulo de generación de etiquetas
│   └── utils/           # Funciones auxiliares (preprocesamiento, etc.)
├── requirements.txt
├── LICENSE
└── README.md
```

## 🚀 Instalación y Uso

1. **Clona el repositorio**:

   ```bash
   git clone https://github.com/tu-usuario/BiblioNLP-Automatic-Tagging-Semantic-Book-Search.git
   cd BiblioNLP
   ```

2. **Instala dependencias**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   which python

   pip install -r requirements.txt
   ```

3. **Configura variables de entorno** (credenciales de MySQL, Pinecone, etc.):

   ```bash
   export MYSQL_USER="tu_usuario"
   export MYSQL_PASS="tu_contraseña"
   export PINECONE_API_KEY="tu_api_key"
   ```

4. **Ejecuta un script de prueba**:

   ```bash
   python src/tagging/main.py
   ```

   Esto generará etiquetas automáticamente para los blurbs de ejemplo.

5. **Busca pasajes** (búsqueda semántica):

   ```bash
   python src/search/semantic_search.py --query "magia y aventuras"
   ```

## 🛠 Tecnologías y Herramientas

- 🐍 **Python 3.9+**
- 🧠 **Modelos NLP**: [Sentence-BERT](https://www.sbert.net/), [Hugging Face Transformers](https://huggingface.co/)
- 🗄️ **Base de datos**: MySQL
- 🗂 **Base Vectorial**: [Pinecone](https://www.pinecone.io/)
- 🏷 **Visualización**: (opcional) [Streamlit](https://streamlit.io/) o [Flask](https://flask.palletsprojects.com/)

## 🙌 Contribuciones

¡Las contribuciones son siempre bienvenidos!

1. Haz un fork del proyecto
2. Crea una rama para tu feature o bugfix (`git checkout -b nombre-rama`)  
3. Haz commit de tus cambios (`git commit -m 'Agrego nueva funcionalidad'`)  
4. Sube la rama (`git push origin nombre-rama`)  
5. Abre un Pull Request detallando tus cambios  

## ⚖️ Licencia

Este proyecto se distribuye bajo la licencia [MIT](LICENSE). ¡Siéntete libre de usarlo y mejorarlo!

---

> **Nota:** Este repositorio es parte de un proyecto de aprendizaje e investigación de NLP, por lo cual no garantiza escalabilidad de producción sin ajustes adicionales. 

¡Gracias por visitar **BiblioNLP** y que disfrutes explorando el mundo de los libros con NLP! 🎉
