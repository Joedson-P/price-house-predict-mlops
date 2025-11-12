from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import os
import numpy as np
import pandas as pd
import catboost

# DEfinição da estrutura de entrada
class PropertyFeatures(BaseModel):
    property_type: str = Field(..., example="apartment", description="Tipo do imóvel (apartment ou house).")
    state: str = Field(..., example="Rio de Janeiro", description="Estado do imóvel.")
    region: str = Field(..., example="Southeast", description="Região do Brasil.")
    lat: float = Field(..., example=-22.9035, description="Latitude do imóvel.")
    lon: float = Field(..., example=-43.2096, description="Longitude do imóvel.")
    area_m2: float = Field(..., example=85.5, description="Área em metros quadrados.")

# Configuração do FastAPI 
app = FastAPI(
    title = "Previsor de Preços de Imóveis",
    description = "API para prever o preço de imóveis brasileiros com base em características geográficas e físicas."
)

# Caminho absoluto para o arquivo .pkl
MODEL_PATH = os.path.join("models", "house_price_pipeline.pkl")
MODEL_PIPELINE = None

# Inicialização
@app.on_event("startup")
def load_model():
    global MODEL_PIPELINE
    try:
        MODEL_PIPELINE = joblib.load(MODEL_PATH)
        print(f"Modelo CatBoost carregado com sucesso de: {MODEL_PATH}")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        MODEL_PIPELINE = None


# DEfinição dos Endpoints

@app.get("/", summary="Verificação de Saúde")
def root():
    return {"message": "API de Previsão de Preços de Imóveis está rodando! Acesse /docs para testar."}


# PRevisão
@app.post("/predict", summary="Prever Preço do Imóvel")
def predict_price(features: PropertyFeatures):
    if MODEL_PIPELINE is None:
        return {"error": "Modelo não carregado. Verifique os logs de inicialização."}, 500

    # Converter a entrada Pydantic para DataFrame
    input_data = features.model_dump()
    df_input = pd.DataFrame([input_data])
    
    # Garantir a ordem exata das colunas que o modelo foi treinado
    expected_columns = ['property_type', 'state', 'region', 'lat', 'lon', 'area_m2']
    df_input = df_input[expected_columns]

    # Previsão em log
    price_log_pred = MODEL_PIPELINE.predict(df_input)[0]

    # Revertendo a para a escala BRL
    price_brl_pred = np.expm1(price_log_pred)

    return {
        "predicted_price_brl": round(price_brl_pred, 2),
        "model_used": "CatBoost Regressor Otimizado",
        "r2_score_training": "0.4096"
    }