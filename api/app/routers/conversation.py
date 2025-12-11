# text-to-sql-project/api/app/routers/conversation.py

from fastapi import APIRouter, HTTPException, status, Request
from app.models.schemas import (
    QuestionRequest,
    AnswerResponse,
    IndexingRequest,
    IndexingResponse,
    PullModelRequest,
    PullModelResponse,
)
from logging import getLogger

router = APIRouter()
logger = getLogger(__name__)

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest, req: Request):
    """
    Endpoint principal pour poser une question en langage naturel.
    """
    logger.info(f"Nouvelle question reçue : '{request.question}' (Conversation ID: {request.conversation_id})")
    try:
        orchestrator = req.app.state.orchestrator
        result = await orchestrator.process_user_question(request.question, conversation_id=request.conversation_id)
        return AnswerResponse(**result)
    except Exception as e:
        logger.error(f"Erreur inattendue dans l'endpoint /ask : {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Une erreur interne est survenue."
        )

@router.post("/index-queries", response_model=IndexingResponse)
async def index_queries(req: Request, request: IndexingRequest | None = None):
    """
    Endpoint administratif pour (ré)indexer les requêtes SQL de référence.
    """
    try:
        orchestrator = req.app.state.orchestrator
        queries = request.queries if request else None
        count = orchestrator.index_reference_queries(queries)
        return IndexingResponse(status="success", indexed_count=count)
    except Exception as e:
        logger.error(f"Erreur lors de l'indexation : {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Une erreur est survenue lors de l'indexation : {e}"
        )

@router.post("/pull-model", response_model=PullModelResponse)
async def pull_model(req: Request, request: PullModelRequest | None = None):
    """Endpoint pour forcer le téléchargement du modèle Ollama (pratique au démarrage)."""
    try:
        orchestrator = req.app.state.orchestrator
        model = request.model if request and request.model else None
        result = await orchestrator.pull_model(model)
        return PullModelResponse(**result)
    except Exception as e:
        logger.error(f"Erreur lors du pull du modèle : {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Une erreur est survenue lors du téléchargement du modèle : {e}"
        )