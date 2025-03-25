## Importación de librerías
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Configuración inicial
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

# Función principal para análisis de sentimientos
def analyze_sentiments(df):
    """
    Realiza análisis de sentimientos en el campo 'blurb' del DataFrame.
    """
    # Inicializar VADER
    sia = SentimentIntensityAnalyzer()

    # Aplicar análisis de sentimientos con VADER
    print("Aplicando análisis de sentimientos con VADER...")
    df['vader_scores'] = df['blurb'].apply(lambda text: sia.polarity_scores(text))
    df[['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']] = df['vader_scores'].apply(pd.Series)

    # Configurar modelo RoBERTa
    print("Cargando modelo RoBERTa para análisis de emociones...")
    MODEL = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    # Función para obtener emociones con RoBERTa
    def mooder(text):
        max_length = 512
        encoded_text = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding="longest")
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        moods = {
            "anger": scores[0],
            "disgust": scores[1],
            "fear": scores[2],
            "joy": scores[3],
            "neutral": scores[4],
            "sadness": scores[5],
            "surprise": scores[6]
        }
        return moods

    # Aplicar análisis de emociones con RoBERTa
    print("Aplicando análisis de emociones con RoBERTa...")
    df['roberta_scores'] = df['blurb'].apply(mooder)
    roberta_scores_df = pd.json_normalize(df['roberta_scores'])
    df = pd.concat([df, roberta_scores_df], axis=1)

    return df

# Ejemplo de uso
if __name__ == "__main__":
    # Aquí se asume que el DataFrame ya ha sido cargado desde la base de datos
    # Ejemplo: df = fetch_query_as_dataframe(engine, sql_file_path)
    print("Cargando datos...")
    # Simulación de datos cargados (reemplazar con la carga real)
    data = {
        "book_id": [1, 2],
        "book_title": ["Libro A", "Libro B"],
        "authors": ["Autor A", "Autor B"],
        "tags": ["tag1, tag2", "tag3, tag4"],
        "blurb": ["Este es un gran libro sobre aventuras.", "Un libro triste sobre pérdidas."]
    }
    df = pd.DataFrame(data)

    # Realizar análisis de sentimientos
    print("Iniciando análisis de sentimientos...")
    df = analyze_sentiments(df)

    # Mostrar resultados
    print("Resultados:")
    print(df.head())