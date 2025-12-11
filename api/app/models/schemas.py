# text-to-sql-project/api/app/models/schemas.py

from pydantic import BaseModel
from typing import List, Optional, Literal

class QuestionRequest(BaseModel):
    """Schéma pour la question de l'utilisateur."""
    question: str
    conversation_id: Optional[str] = None

class AnswerResponse(BaseModel):
    """Schéma pour la réponse finale."""
    answer: str
    generated_sql: str | None = None
    sql_result: str | None = None
    conversation_id: str | None = None

class IndexingRequest(BaseModel):
    """Schéma pour la requête d'indexation manuelle."""
    queries: List[str]

class IndexingResponse(BaseModel):
    """Schéma pour la réponse de l'indexation."""
    status: str
    indexed_count: int

class PullModelRequest(BaseModel):
    """Schéma pour demander le téléchargement d'un modèle Ollama spécifique."""
    model: str | None = None

class PullModelResponse(BaseModel):
    """Schéma pour la réponse du téléchargement du modèle."""
    status: str
    model: str | None = None
    message: str | None = None


# ==== Forecast narrative schemas ====

class ForecastPoint(BaseModel):
    date: Optional[str] = None  # ISO date ou label
    value: float

class SummaryStats(BaseModel):
    count: int
    min: float
    max: float
    mean: float
    start_value: float
    end_value: float
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class ForecastNarrativeRequest(BaseModel):
    target: Literal["liquidity", "inflation", "other"]
    horizon: Optional[str] = None
    unit: Optional[str] = None
    series: List[ForecastPoint]
    lower: Optional[List[float]] = None  # bornes inférieures (optionnel, aligné sur series)
    upper: Optional[List[float]] = None  # bornes supérieures (optionnel, aligné sur series)
    language: Literal["fr", "en"] = "fr"
    tone: Literal["professionnel", "neutre", "pédagogique"] = "professionnel"
    title: Optional[str] = None

class ForecastNarrativeResponse(BaseModel):
    narrative: str
    summary_stats: SummaryStats


# ==== Prediction and SHAP schemas ====

class InflationPredictionResponse(BaseModel):
    """
    Schéma pour les réponses du modèle de prévision d'inflation avec explicabilité SHAP.
    
    Spécialement conçu pour interpréter les prédictions d'inflation pour les économistes 
    et analystes de la BCEAO/UEMOA.
    """
    predictions: dict  # Prédictions d'inflation par période {"2024-Q1": 2.5, ...}
    global_shap_importance: dict  # Importance des facteurs inflationnistes {"taux_change": 0.35, ...}
    shap_summary_details: dict  # Métadonnées du modèle (précision, période d'entraînement, etc.)
    individual_shap_explanations: dict  # Explications SHAP par observation temporelle
    confidence_intervals: Optional[dict] = None  # Intervalles de confiance des prédictions

class InflationInterpretationRequest(BaseModel):
    """
    Requête pour l'interprétation économique des prédictions d'inflation SHAP.
    Supporte les conversations multi-tours avec conversation_id.
    """
    prediction_data: InflationPredictionResponse
    analysis_language: Literal["fr", "en"] = "fr"
    target_audience: Literal["economist", "analyst", "policymaker", "general"] = "economist"
    include_policy_recommendations: bool = True
    include_monetary_policy_analysis: bool = True  # Analyse spécifique à la politique monétaire
    focus_on_bceao_mandate: bool = True  # Focus sur le mandat de stabilité des prix de la BCEAO
    conversation_id: Optional[str] = None  # ID de conversation pour questions de suivi
    follow_up_question: Optional[str] = None  # Question de suivi optionnelle

class InflationInterpretationResponse(BaseModel):
    """
    Réponse contenant l'interprétation économique des prédictions d'inflation.
    Structure mise à jour pour correspondre aux tags XML.
    """
    reasoning: Optional[str] = None  # Raisonnement interne du modèle
    executive_summary: str  # Résumé exécutif sur les perspectives d'inflation
    monthly_analysis: Optional[str] = None  # Analyse mensuelle détaillée
    inflation_drivers: Optional[str] = None  # Facteurs inflationnistes et désinflationnistes
    price_stability_assessment: Optional[str] = None  # Évaluation stabilité des prix vs cible BCEAO
    risks: Optional[str] = None  # Risques haussiers et baissiers
    limitations: Optional[str] = None  # Limites et incertitudes
    monetary_policy_recommendations: Optional[str] = None  # Recommandations pour la BCEAO
    conversation_id: Optional[str] = None  # ID de conversation pour suivi
    _meta: Optional[dict] = None  # Métadonnées (tentatives, validation format)