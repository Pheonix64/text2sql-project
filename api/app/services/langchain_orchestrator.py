import asyncio
import json
import logging
import re
from typing import List, Optional, Tuple

import chromadb
import sqlglot
from sqlalchemy import text

from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.config import settings
from app.services.query_orchestrator import QueryOrchestrator

logger = logging.getLogger(__name__)


def _only_sql_in_tags(content: str) -> str:
    """
    Extrait la requête SQL entre balises <sql>...</sql>.
    Si les balises sont absentes, renvoie le contenu brut.
    """
    m = re.search(r"<sql>(.*?)</sql>", content, flags=re.DOTALL | re.IGNORECASE)
    sql = m.group(1).strip() if m else content.strip()
    # Normaliser les sauts de ligne/espaces
    return re.sub(r"\s+\n", "\n", sql).strip()


def _is_safe_select(sql: str) -> bool:
    """
    Sécurité stricte:
    - Une seule instruction
    - Commence par SELECT
    - Pas de DDL/DML (CREATE/INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE)
    """
    if ";" in sql.strip(";"):
        return False
    s = sql.strip().lower()
    if not s.startswith("select"):
        return False
    banned = ["insert", "update", "delete", "create", "drop", "alter", "truncate"]
    return not any(b in s for b in banned)


class LangChainQueryOrchestrator(QueryOrchestrator):
    """
    Orchestrateur basé sur LangChain, respectant la logique de QueryOrchestrator.
    - Embeddings HuggingFace (même modèle que sentence-transformers indiqué en conf)
    - Vector store Chroma (même collection)
    - LLM via ChatOllama (même host/modèle Ollama)
    """

    def __init__(self):
        super().__init__()

        # Embeddings identiques à ceux configurés (modèle sentence-transformers)
        self.lc_embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)

        # Réutiliser le ChromaDB HttpClient et la collection existante
        self.lc_chroma = Chroma(
            client=self.chroma_client,  # chromadb.HttpClient déjà initialisé par le parent
            collection_name=settings.CHROMA_COLLECTION,
            embedding_function=self.lc_embeddings,
        )

        # LLM (mêmes paramètres Ollama)
        self.lc_llm = ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.LLM_MODEL,
            temperature=0.1,
        )

        # Utilitaire SQL (pour introspection, si nécessaire dans les prompts)
        self.lc_sqldb = SQLDatabase.from_uri(settings.DATABASE_URL)

        # Chaînes/prompt pour génération SQL et réponse NL
        self._init_chains()

    def _init_chains(self):
        # Prompt de génération SQL
        self.sql_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Tu es un assistant qui génère des requêtes SQL PostgreSQL SÉCURISÉES."
                        " Règles STRICTES:\n"
                        "- Une seule instruction SQL\n"
                        "- SELECT uniquement (aucun DDL/DML)\n"
                        "- N'utilise que les colonnes/tables du schéma donné\n"
                        "- Adapte-toi aux exemples si pertinents\n\n"
                        "Schéma de la base (extrait utile):\n{schema}\n\n"
                        "Exemples de requêtes similaires:\n{examples}\n"
                        "Réponds UNIQUEMENT avec la requête entre balises <sql>...</sql>."
                    ),
                ),
                (
                    "human",
                    "Question utilisateur:\n{question}\n"
                    "N'inclus rien d'autre que la requête SQL entre <sql>...</sql>."
                ),
            ]
        )

        # Prompt de reformulation en réponse naturelle
        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Tu expliques en français, de manière claire et concise, des résultats issus d'une requête SQL.",
                ),
                (
                    "human",
                    "Question: {question}\n\nRequête SQL:\n{sql}\n\nAperçu des résultats (limités):\n{rows}\n\n"
                    "Fais une réponse factuelle et utile. Si les résultats sont vides, explique-le."
                ),
            ]
        )

        self.sql_chain = (
            {
                "schema": lambda _: self.db_schema,
                "examples": self._retrieve_examples_for_question,
                "question": RunnablePassthrough(),
            }
            | self.sql_prompt
            | self.lc_llm
            | StrOutputParser()
        )

        self.answer_chain = (
            {
                "question": lambda x: x["question"],
                "sql": lambda x: x["sql"],
                "rows": lambda x: json.dumps(x["rows"], ensure_ascii=False, indent=2),
            }
            | self.answer_prompt
            | self.lc_llm
            | StrOutputParser()
        )

    def _retrieve_examples_for_question(self, question: str) -> str:
        """
        Récupère k exemples de requêtes proches depuis la collection Chroma existante.
        """
        try:
            # Utiliser le retriever interne de LangChain
            docs = self.lc_chroma.similarity_search(question, k=5)
            if not docs:
                return "\n".join(self.reference_queries[:3])
            return "\n".join(d.page_content for d in docs)
        except Exception as e:
            logger.warning(f"Recherche d'exemples Chroma échouée, fallback: {e}")
            return "\n".join(self.reference_queries[:3])

    def index_reference_queries(self, queries: Optional[List[str]] = None) -> int:
        """
        (Ré)indexe des requêtes de référence dans la collection Chroma via LangChain.
        """
        payload = queries or self.reference_queries
        if not payload:
            return 0

        # Upsert via ids déterministes pour éviter les doublons
        import hashlib

        texts, ids, metadatas = [], [], []
        for q in payload:
            q_norm = q.strip()
            if not q_norm:
                continue
            q_id = hashlib.sha1(q_norm.encode("utf-8")).hexdigest()
            texts.append(q_norm)
            ids.append(q_id)
            metadatas.append({"kind": "reference_sql"})

        # Utiliser le sémaphore Chroma du parent
        async def _do_upsert():
            async with self.chroma_sem:
                # add_texts supporte ids/metadatas
                self.lc_chroma.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        # Exécuter la tâche dans l'event loop courant si possible
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_do_upsert())
        except RuntimeError:
            # Aucun loop en cours (appel synchrone) => exécuter directement
            asyncio.run(_do_upsert())

        return len(texts)

    async def process_user_question(self, question: str) -> dict:
        """
        Reproduit le flux:
        - vérifications (sécurité/filtrage),
        - récupération d'exemples,
        - génération SQL via LangChain,
        - validation SQL (sqlglot) + garde-fous,
        - exécution via SQLAlchemy (engine LLM read-only),
        - réponse en langage naturel via LangChain.
        """
        # Filtrage question sensible (méthode héritée)
        if self._is_question_harmful(question):
            return {
                "answer": "Désolé, je ne peux pas aider avec cette demande.",
                "generated_sql": None,
                "sql_result": None,
            }

        # Génération SQL
        async with self.llm_sem:
            raw_sql = await self.sql_chain.ainvoke(question)
        sql = _only_sql_in_tags(raw_sql)

        # Validation stricte (syntaxe + politique)
        try:
            sqlglot.parse_one(sql, read="postgres")
        except Exception as e:
            logger.error(f"SQL invalide produit par le LLM: {e} | SQL: {sql}")
            return {
                "answer": "La question a été comprise, mais la génération SQL a échoué. Veuillez reformuler.",
                "generated_sql": sql,
                "sql_result": None,
            }

        if not _is_safe_select(sql):
            logger.warning(f"SQL rejetée par les règles de sécurité: {sql}")
            return {
                "answer": "La requête générée ne respecte pas les règles de sécurité (SELECT uniquement).",
                "generated_sql": sql,
                "sql_result": None,
            }

        # Exécution SQL
        rows_preview = []
        try:
            async with self.llm_sem:  # limiter la concurrence globale
                with self.db_engine.connect() as conn:
                    result = conn.execute(text(sql))
                    # Limiter l’aperçu pour le prompt de réponse
                    cols = list(result.keys())
                    for i, r in enumerate(result.fetchall()):
                        if i >= 10:
                            break
                        rows_preview.append(dict(zip(cols, r)))
        except Exception as e:
            logger.error(f"Erreur d'exécution SQL: {e}")
            return {
                "answer": "La requête SQL a échoué lors de l'exécution.",
                "generated_sql": sql,
                "sql_result": None,
            }

        # Réponse en NL
        answer = await self.answer_chain.ainvoke({"question": question, "sql": sql, "rows": rows_preview})

        return {
            "answer": answer.strip(),
            "generated_sql": sql,
            "sql_result": json.dumps(rows_preview, ensure_ascii=False),
        }

    # Forecast: on garde la logique métier et on ajoute LangChain pour la narration
    async def generate_forecast_narrative(self, body):
        """
        Produit une narration à partir d'une série fournie, en gardant le calcul
        des stats côté Python et en générant le texte via LangChain.
        """
        # Calcul des stats (aligné avec le schéma SummaryStats)
        vals = [p.value for p in body.series if p.value is not None]
        count = len(vals)
        _min = min(vals) if vals else None
        _max = max(vals) if vals else None
        mean = sum(vals) / len(vals) if vals else None

        start_value = vals[0] if vals else None
        end_value = vals[-1] if vals else None
        start_date = body.series[0].date if body.series and body.series[0].date else None
        end_date = body.series[-1].date if body.series and body.series[-1].date else None

        stats = {
            "count": count,
            "min": _min,
            "max": _max,
            "mean": mean,
            "start_value": start_value,
            "end_value": end_value,
            "start_date": start_date,
            "end_date": end_date,
        }

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Tu es un économiste qui rédige une courte narration de prévision. Langue: {lang}, ton: {tone}.",
                ),
                (
                    "human",
                    "Cible: {target}\nHorizon: {horizon}\nUnité: {unit}\n"
                    "Série (date, valeur): {series}\n"
                    "Bornes inférieures: {lower}\nBornes supérieures: {upper}\n"
                    "Rédige une narration brève et informative pour un public professionnel.",
                ),
            ]
        )

        chain = prompt | self.lc_llm | StrOutputParser()
        narrative = await chain.ainvoke(
            {
                "lang": body.language,
                "tone": body.tone,
                "target": body.target,
                "horizon": body.horizon or "",
                "unit": body.unit or "",
                "series": [(p.date, p.value) for p in body.series],
                "lower": body.lower or [],
                "upper": body.upper or [],
            }
        )

        return narrative.strip(), stats

    # SHAP: déléguer au parent pour respecter exactement la structure attendue (schémas Pydantic)
    async def format_inflation_prediction(self, prediction_data):
        return await super().format_inflation_prediction(prediction_data)

    async def generate_inflation_interpretation(self, body):
        return await super().generate_inflation_interpretation(body)