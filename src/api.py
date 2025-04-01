from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
from sentence_transformers import SentenceTransformer

# Cargar modelos
clf = joblib.load("model/tag_classifier.joblib")
mlb = joblib.load("model/label_binarizer.joblib")
sbert_model = SentenceTransformer("model/sentence_transformer")

# Definición de entrada
class BookInput(BaseModel):
    titles: List[str]
    blurbs: List[str]

# Inicializar FastAPI
app = FastAPI(title="Book Tag Predictor API")

@app.post("/predict")
def predict_tags(book_input: BookInput):
    texts = [t + ". " + b for t, b in zip(book_input.titles, book_input.blurbs)]
    embeddings = sbert_model.encode(texts)
    preds = clf.predict(embeddings)
    tag_lists = mlb.inverse_transform(preds)
    return {"predicted_tags": tag_lists}
