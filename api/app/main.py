# text-to-sql-project/api/app/main.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routers import conversation
from app.routers import forecast
from app.services.query_orchestrator import QueryOrchestrator
import logging
from fastapi.middleware.cors import CORSMiddleware

# Configuration de la journalisation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère les événements de démarrage et d'arrêt de l'application.
    """
    logging.info("Démarrage de l'application...")
    # Instanciation tardive pour garantir que les services dépendants sont prêts
    app.state.orchestrator = QueryOrchestrator()
    # Indexation initiale des requêtes de référence
    app.state.orchestrator.index_reference_queries()
    yield
    logging.info("Arrêt de l'application...")

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API Text-to-SQL pour Données Économiques",
    description="Une API pour interroger une base de données économique en langage naturel.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS (ouvrir à toutes les origines par défaut; à restreindre en prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion du routeur de conversation
app.include_router(conversation.router, prefix="/api", tags=["Conversation"])
app.include_router(forecast.router, prefix="/api/forecast", tags=["Forecast Narrative"])

@app.get("/health", tags=["Health Check"])
def health_check():
    """Endpoint simple pour vérifier que l'API est en ligne."""
    return {"status": "ok"}