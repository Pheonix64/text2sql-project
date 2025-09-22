# text-to-sql-project/api/app/models/schemas.py

from pydantic import BaseModel
from typing import List

class QuestionRequest(BaseModel):
    """Schéma pour la question de l'utilisateur."""
    question: str

class AnswerResponse(BaseModel):
    """Schéma pour la réponse finale."""
    answer: str
    generated_sql: str | None = None
    sql_result: str | None = None

class IndexingRequest(BaseModel):
    """Schéma pour la requête d'indexation manuelle."""
    queries: List[str]

class IndexingResponse(BaseModel):
    """Schéma pour la réponse de l'indexation."""
    status: str
    indexed_count: int