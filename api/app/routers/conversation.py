# text-to-sql-project/api/app/routers/conversation.py

from fastapi import APIRouter, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from app.models.schemas import (
    QuestionRequest,
    AnswerResponse,
    IndexingRequest,
    IndexingResponse,
    PullModelRequest,
    PullModelResponse,
)
from logging import getLogger
import io

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

@router.get("/export/csv/{query_id}")
async def export_csv(query_id: str, req: Request):
    """
    Endpoint pour télécharger les résultats d'une requête au format CSV.
    Le query_id est retourné par l'endpoint /ask.
    """
    try:
        orchestrator = req.app.state.orchestrator
        csv_data = await orchestrator.export_query_results_to_csv(query_id)
        
        if csv_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Aucune donnée trouvée pour cet identifiant. Les données peuvent avoir expiré."
            )
        
        # Créer un buffer en mémoire pour le CSV
        output = io.StringIO()
        output.write(csv_data)
        output.seek(0)
        
        # Retourner le CSV en streaming
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8-sig')),  # utf-8-sig pour Excel
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=donnees_{query_id}.csv"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'export CSV : {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Une erreur est survenue lors de l'export des données."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Une erreur est survenue lors du téléchargement du modèle : {e}"
        )