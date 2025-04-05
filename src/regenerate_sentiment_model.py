from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import os

# Define el modelo
MODEL = "j-hartmann/emotion-english-distilroberta-base"

# Crear el directorio 'model' si no existe
model_dir = "../model"
os.makedirs(model_dir, exist_ok=True)

# Cargar el modelo y el tokenizer desde Hugging Face
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Guardar el modelo y el tokenizer en formato joblib
joblib.dump(model, os.path.join(model_dir, "sentiment_roberta_model.joblib"))
joblib.dump(tokenizer, os.path.join(model_dir, "sentiment_roberta_tokenizer.joblib"))

print("Modelo y tokenizer guardados correctamente.")