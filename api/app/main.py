from fastapi import FastAPI
from contextlib import asynccontextmanager
from.routers import conversation
from.services.query_orchestrator import orchestrator
import logging

# Configuration de la journalisation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère les événements de démarrage et d'arrêt de l'application.
    C'est ici que nous initialisons les modèles et les connexions.
    """
    # Au démarrage :
    logging.info("Démarrage de l'application...")
    # L'initialisation de l'orchestrateur se fait à l'import, mais on peut
    # ajouter ici des tâches de démarrage comme l'indexation initiale.
    orchestrator.index_reference_queries()
    yield
    # À l'arrêt :
    logging.info("Arrêt de l'application...")
    # On pourrait ajouter ici du code de nettoyage si nécessaire.
    

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API Text-to-SQL pour Données Économiques",
    description="Une API pour interroger une base de données économique en langage naturel.",
    version="1.0.0",
    lifespan=lifespan
)

# Inclusion du routeur de conversation
app.include_router(conversation.router, prefix="/api", tags=["Conversation"])

@app.get("/health", tags=["Health Check"])
def health_check():
    """Endpoint simple pour vérifier que l'API est en ligne."""
    return {"status": "ok"}