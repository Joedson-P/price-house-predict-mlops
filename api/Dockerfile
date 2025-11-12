# api/Dockerfil# Base Image
FROM python:3.10-buster

# Configurações do Ambiente
WORKDIR /app

# Copiar o arquivo requirements e instalar as bibliotecas necessárias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código da aplicação e o modelo treinado para o WORKDIR /app
COPY app/ app/
COPY models/ models/

# Porta
EXPOSE 8000

# Inicialização
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]