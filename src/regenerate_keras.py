# %%
# %pip install tensorflow==2.12.0

import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from termcolor import colored
import joblib

#print(f"Python version: {sys.version}")
#print(f"Tensorflow version: {tf.__version__}")

# %%
# 1. Cargar y preprocesar los datos
df = pd.read_csv("data/raw/goodreads_data_sample.csv")

# Normalizar columnas necesarias
df['Book'] = df['Book'].fillna('')
df['Description'] = df['Description'].fillna('')
df['Genres'] = df['Genres'].fillna("[]")

# Crear la columna 'text' combinando título y descripción
df['text'] = df['Book'] + ". " + df['Description']

# Convertir la columna 'Genres' de cadenas a listas reales
df['tags'] = df['Genres'].apply(eval)

# %%
# 2. Codificar las etiquetas (tags)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['tags'])

# 3. Convertir texto a embeddings
model_embed = SentenceTransformer('all-MiniLM-L6-v2')
X_embeddings = model_embed.encode(df['text'].tolist(), show_progress_bar=True)

# 4. Dividir los datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X_embeddings, y, test_size=0.2, random_state=42)

# 5. Definir el modelo Keras
input_dim = X_train.shape[1]
num_classes = y_train.shape[1]

model = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='sigmoid')  # Para clasificación multilabel
])

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# %%
# 6. Entrenar el modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# 7. Predecir etiquetas para nuevos datos
# Libros de prueba
test_books = [
    {
        "text": "Messi: Edición revisada y actualizada. Una biografía del astro argentino.",
        "expected_tags": ["Biography", "Sports", "Football", "Nonfiction", "Sports & Outdoors"]
    },
    {
        "text": "The Great Gatsby. A novel about the American dream and the roaring twenties.",
        "expected_tags": ["Classics", "Fiction", "Literature", "Romance", "Historical Fiction"]
    },
    {
        "text": "A Brief History of Time. Stephen Hawking explains the universe and black holes.",
        "expected_tags": ["Science", "Nonfiction", "Physics", "Philosophy", "History", "Astronomy"]
    },
    {
        "text": "The Catcher in the Rye. A story about teenage rebellion and identity.",
        "expected_tags": ["Classics", "Fiction", "Young Adult", "Literature", "Contemporary Fiction"]
    },
    {
        "text": "The Art of War. Ancient Chinese military strategy by Sun Tzu.",
        "expected_tags": ["Philosophy", "History", "Nonfiction", "Military", "Self-Help"]
    },
    {
        "text": "Harry Potter and the Chamber of Secrets. The second book in the Harry Potter series.",
        "expected_tags": ["Fantasy", "Fiction", "Young Adult", "Adventure", "Magic"]
    }
]

# Predecir etiquetas para los libros de prueba
for book in test_books:
    sample_embedding = model_embed.encode([book["text"]])
    predictions = model.predict(sample_embedding)
    predicted_tags = mlb.inverse_transform((predictions > 0.5).astype(int))[0]
    
    # Comparar etiquetas esperadas y predichas
    expected_tags = set(book["expected_tags"])
    predicted_tags = set(predicted_tags)
    
    # Identificar aciertos y errores
    correct_tags = expected_tags & predicted_tags
    incorrect_tags = predicted_tags - expected_tags
    missing_tags = expected_tags - predicted_tags
    
    # Pintar los resultados
    print(f"Book: {book['text']}")
    print("Expected Tags:", ", ".join(book["expected_tags"]))
    print("Predicted Tags:", ", ".join(
        [colored(tag, "green") if tag in correct_tags else tag for tag in predicted_tags]
    ))
    print("Missing Tags:", ", ".join([colored(tag, "red") for tag in missing_tags]))
    print("-" * 50)

# %%
# 8. Guardar el modelo Keras entrenado y el codificador de etiquetas como "book_tagging_keras_model..."

# Guardar el modelo Keras en el formato recomendado
model.save("model/book_tagging_keras_model.keras")  # Guardar en formato nativo de Keras

# Guardar el binarizador de etiquetas
joblib.dump(mlb, "model/book_tagging_keras_mlb_encoder.pkl")  # Guardar el binarizador


