# House Price Predictor MLOps

Este projeto demonstra um pipeline completo de Machine Learning (MLOps) para prever o preço de imóveis residenciais. O foco principal é a **implementação da infraestrutura de inferência** utilizando **FastAPI** e **Docker** para garantir um serviço de previsão pronto para produção.

## Destaques do Projeto

| Fase MLOps | Destaque | Resultado |
| :--- | :--- | :--- |
| **Modelo Selecionado** | **CatBoost Regressor Otimizado** | **R² de 0.4096** e RMSE de R$ 269.417,42 |
| **Serialização** | Pipeline Completa (`.pkl`) | Inclui pré-processamento (OHE e Log-Transformação) e o modelo final. |
| **Inferência** | FastAPI | API REST robusta e validada por Pydantic. |
| **Empacotamento** | Docker | Serviço empacotado e isolado, pronto para qualquer ambiente de cloud. |

## Fonte de Dados

O dataset utilizado para treinamento e teste do modelo foi obtido em: [Dados Imobiliários do Brasil](https://www.kaggle.com/datasets/ashishkumarjayswal/brasil-real-estate?select=Brasile-real-estate-dataset.csv).

## Como Rodar a API Localmente (Docker)

O método recomendado para execução é através do Docker, garantindo que todas as dependências do modelo (CatBoost, scikit-learn, etc.) sejam executadas corretamente.

1.  **Construir a Imagem:** Na pasta `api/`, execute:
    ```bash
    docker build -t price-predictor-api:v1 .
    ```
2.  **Rodar o Contêiner:** Mapeando a porta interna 8000 para a porta externa 8001:
    ```bash
    docker run -d --name price-api-container -p 8001:8000 price-predictor-api:v1
    ```
3.  **Testar a Inferência:** Acesse a documentação da API no seu navegador:
    `http://127.0.0.1:8001/docs`

## Melhorias Futuras

O R² do modelo (0.4096) é considerado baixo, mas é o limite de precisão com as *features* atuais. As principais melhorias futuras para o projeto focariam na **Engenharia de Features**:

* **Enriquecimento de Dados:** Adicionar *features* críticas de mercado imobiliário que faltam:
    * `Número de quartos/banheiros`.
    * `Idade do imóvel / Ano de construção`.
    * `Características do condomínio` (lazer, academia, portaria 24h).
* **Feature Engineering Geográfico:** Utilizar coordenadas `lat` e `lon` para calcular a distância para pontos de interesse (escolas, metrô, parques).
* **Otimização do Pipeline:** Implementar o **Optuna** ou **Bayesian Optimization** para refinar o *hyperparameter tuning* dos modelos.