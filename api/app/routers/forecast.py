# text-to-sql-project/api/app/routers/forecast.py

from fastapi import APIRouter, HTTPException, status, Request
from logging import getLogger
from app.models.schemas import (
    ForecastNarrativeRequest,
    ForecastNarrativeResponse,
    SummaryStats,
    InflationPredictionResponse,
    InflationInterpretationRequest,
    InflationInterpretationResponse,
)
from app.config import settings

router = APIRouter()
logger = getLogger(__name__)


@router.post("/narrative", response_model=ForecastNarrativeResponse)
async def generate_forecast_narrative(req: Request, body: ForecastNarrativeRequest):
    try:
        orchestrator = req.app.state.orchestrator
        narrative, stats = await orchestrator.generate_forecast_narrative(body)
        return ForecastNarrativeResponse(narrative=narrative, summary_stats=stats)
    except Exception as e:
        logger.error(f"Erreur lors de la génération de la narration de prévision : {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Une erreur est survenue lors de la génération de la narration."
        )


@router.post("/inflation/prediction", response_model=InflationPredictionResponse)
async def receive_inflation_prediction(req: Request, prediction_data: dict):
    """
    Endpoint pour recevoir les prédictions du modèle d'inflation avec explicabilité SHAP.
    Formate les données selon le schéma InflationPredictionResponse pour l'interprétation économique.
    """
    try:
        orchestrator = req.app.state.orchestrator
        formatted_prediction = await orchestrator.format_inflation_prediction(prediction_data)
        return formatted_prediction
    except Exception as e:
        logger.error(f"Erreur lors du formatage de la prédiction d'inflation : {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Une erreur est survenue lors du traitement de la prédiction d'inflation."
        )


@router.post("/inflation/interpret", response_model=InflationInterpretationResponse)
async def interpret_inflation_for_economists(req: Request, body: InflationInterpretationRequest):
    """
    Endpoint principal pour interpréter les prédictions d'inflation SHAP à destination des économistes et analystes.
    Traduit les résultats techniques en analyses économiques spécifiques à l'inflation et à la politique monétaire.
    
    Supporte:
    - conversation_id pour le suivi conversationnel
    - follow_up_question pour les questions de suivi
    - Timeout configurable via settings.LLM_TIMEOUT_INFLATION
    """
    try:
        orchestrator = req.app.state.orchestrator
        interpretation = await orchestrator.generate_inflation_interpretation(
            body, 
            timeout=settings.LLM_TIMEOUT_INFLATION
        )
        return interpretation
    except Exception as e:
        logger.error(f"Erreur lors de l'interprétation de l'inflation : {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Une erreur est survenue lors de l'interprétation de l'inflation."
        )
