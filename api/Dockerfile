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
# On combine toute l'installation en une seule commande RUN pour la propreté
# et pour optimiser les couches Docker.
RUN \
    # 1. Mettre à jour pip à la dernière version pour éviter les bugs et avoir toutes les options
    pip install --no-cache-dir --user --upgrade pip && \
    # 2. Installer torch en version CPU depuis sa source officielle
    pip install --no-cache-dir --user --timeout=600 \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    # 3. Installer le reste des paquets, y compris sentence-transformers
    pip install --no-cache-dir --user --timeout=600 \
        sentence-transformers \
        -r requirements.txt

# Étape 3: Copie du code de l'application
COPY --chown=appuser:appuser ./app ./app

# Étape 4: Commande d'exécution
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]