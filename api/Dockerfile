# text-to-sql-project/api/Dockerfile

# Étape 1: Image de base
FROM python:3.11-slim

# Création d'un utilisateur non-root
RUN useradd --create-home appuser
USER appuser
WORKDIR /home/appuser

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Ajout du répertoire des binaires de l'utilisateur au PATH dès le début
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Étape 2: Installation des dépendances
COPY --chown=appuser:appuser requirements.txt .

# --- MODIFICATION MAJEURE ---
# Installation en plusieurs étapes avec gestion des erreurs réseau
# Stratégie: séparer les installations critiques et ajouter des retry
RUN \
    # 1. Mettre à jour pip avec retry
    pip install --no-cache-dir --user --upgrade pip --retries 5 && \
    # 2. Installer torch en version CPU depuis sa source officielle avec retry
    pip install --no-cache-dir --user --timeout=600 --retries 5 \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Séparer l'installation des autres paquets pour isoler les problèmes
RUN pip install --no-cache-dir --user --timeout=600 --retries 5 \
    sentence-transformers

# Installer le reste des dépendances
RUN pip install --no-cache-dir --user --timeout=600 --retries 5 \
    -r requirements.txt

# Étape 3: Copie du code de l'application
COPY --chown=appuser:appuser ./app ./app

# Étape 4: Commande d'exécution
EXPOSE 8100
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]