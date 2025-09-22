# Étape 1: Image de base officielle et légère pour Python
FROM python:3.11-slim

# Création d'un utilisateur non-root pour des raisons de sécurité
# Exécuter des conteneurs en tant que root est une mauvaise pratique
RUN useradd --create-home appuser
USER appuser
WORKDIR /home/appuser/app

# Variables d'environnement pour optimiser Python dans Docker
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Étape 2: Installation des dépendances
# Copier uniquement requirements.txt permet de tirer parti du cache Docker
# si les dépendances ne changent pas.
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Ajout du répertoire des binaires de l'utilisateur au PATH
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Étape 3: Copie du code de l'application
COPY --chown=appuser:appuser ./app .

# Étape 4: Commande d'exécution
# Exposition du port sur lequel l'application écoutera
EXPOSE 8000
# Utilisation de la forme 'exec' pour que le serveur Uvicorn soit le processus principal (PID 1)
# et reçoive correctement les signaux du système (ex: SIGTERM pour un arrêt propre)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]