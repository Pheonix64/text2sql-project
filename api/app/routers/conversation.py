from fastapi import APIRouter, HTTPException, status
from..models.schemas import QuestionRequest, AnswerResponse, IndexingRequest, IndexingResponse
from..services.query_orchestrator import orchestrator
from logging import getLogger

router = APIRouter()
logger = getLogger(__name__)

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Endpoint principal pour poser une question en langage naturel.
    """
    logger.info(f"Nouvelle question reçue : '{request.question}'")
    try:
        result = await orchestrator.process_user_question(request.question)
        return AnswerResponse(**result)
    except Exception as e:
        logger.error(f"Erreur inattendue dans l'endpoint /ask : {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Une erreur interne est survenue."
        )

@router.post("/index-queries", response_model=IndexingResponse)
async def index_queries(request: IndexingRequest | None = None):
    """
    Endpoint administratif pour (ré)indexer les requêtes SQL de référence.
    Si aucune requête n'est fournie, les requêtes par défaut sont utilisées.
    """
    try:
        queries = request.queries if request else None
        count = orchestrator.index_reference_queries(queries)
        return IndexingResponse(status="success", indexed_count=count)
    except Exception as e:
        logger.error(f"Erreur lors de l'indexation : {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Une erreur est survenue lors de l'indexation : {e}"
        )