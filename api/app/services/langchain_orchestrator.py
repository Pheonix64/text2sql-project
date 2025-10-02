# text-to-sql-project/api/app/services/langchain_orchestrator.py

import asyncio
from logging import getLogger
import json
import re

from langchain_community.llms import Ollama
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
import sqlglot
from sqlalchemy import create_engine, exc, text

from app.config import settings
from app.models.schemas import InflationInterpretationRequest, NarrativeRequest

logger = getLogger(__name__)


class LangchainOrchestrator:
    """
    Orchestrateur basé sur LangChain pour gérer les fonctionnalités de l'API.
    """
    def __init__(self):
        logger.info("Initialisation de LangchainOrchestrator...")

        self.llm = Ollama(model=settings.LLM_MODEL, base_url=settings.OLLAMA_BASE_URL)
        self.embeddings = SentenceTransformerEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)

        try:
            self.db_engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
            self.db = SQLDatabase(engine=self.db_engine)
            self.admin_db_engine = create_engine(settings.ADMIN_DATABASE_URL)
            logger.info("Connexions à la base de données PostgreSQL via LangChain réussies.")
        except Exception as e:
            logger.error(f"Erreur de connexion à la base de données : {e}")
            raise

        self.chroma_client = Chroma(
            client_settings={"host": settings.CHROMA_HOST, "port": settings.CHROMA_PORT, "ssl": False},
            collection_name=settings.CHROMA_COLLECTION,
            embedding_function=self.embeddings,
        )
        self.retriever = self.chroma_client.as_retriever(search_kwargs={"k": 5})

        self.db_schema = self._get_rich_db_schema('indicateurs_economiques_uemoa')
        self.reference_queries = [
            "SELECT pib_nominal_milliards_fcfa FROM indicateurs_economiques_uemoa WHERE date = '2022-01-01'",
            "SELECT date, exportations_biens_fob, importations_biens_fob, balance_des_biens FROM indicateurs_economiques_uemoa WHERE EXTRACT(YEAR FROM date) = 2021",
            "SELECT AVG(taux_croissance_reel_pib_pct) FROM indicateurs_economiques_uemoa WHERE date >= '2015-01-01' AND date <= '2020-01-01'",
        ]

        self._build_chains()
        logger.info("LangchainOrchestrator initialisé avec succès.")

    def _build_chains(self):
        """Construit les chaînes LCEL pour les différentes tâches."""

        # --- Chaîne de génération SQL ---
        sql_generation_template = """
### Instruction (génération SQL)
Tu es un expert SQL (PostgreSQL) et analyste de l'UEMOA. Convertis la question en une seule requête SQL SELECT.
Contraintes:
- Retourne UNIQUEMENT la requête SQL.
- Ne modifie jamais les données (pas d'INSERT, UPDATE, DELETE).
- N'utilise que les colonnes du schéma.

### Schéma et descriptions
{schema}

### Exemples de requêtes similaires
{few_shot_examples}

### Question de l'utilisateur
{question}

### Requête SQL
"""
        sql_prompt = ChatPromptTemplate.from_template(sql_generation_template)
        self.sql_generation_chain = (
            RunnablePassthrough.assign(few_shot_examples=lambda x: self.retriever.invoke(x["question"]))
            | RunnablePassthrough.assign(schema=lambda _: self.db_schema)
            | sql_prompt
            | self.llm
            | StrOutputParser()
        )

        # --- Chaîne de génération de réponse finale ---
        final_response_template = """
### Instruction (rédaction d'analyse augmentée)
Tu es un analyste économique de la BCEAO. En te basant sur la QUESTION et le RÉSULTAT SQL, rédige une analyse en français.
Structure: TL;DR, Analyse, Pistes.
Règles: Si le résultat est vide, dis-le. N'invente rien.

### Question de l'utilisateur
{question}

### Résultat SQL
{sql_result}

### Réponse
"""
        final_response_prompt = ChatPromptTemplate.from_template(final_response_template)
        self.final_response_chain = final_response_prompt | self.llm | StrOutputParser()

    async def process_user_question(self, user_question: str) -> dict:
        logger.info(f"Traitement de la question via LangChain: '{user_question}'")
        generated_sql_raw = await self.sql_generation_chain.ainvoke({"question": user_question})
        generated_sql = self._extract_sql_from_text(generated_sql_raw)

        if not generated_sql:
            return {"answer": "Je n'ai pas pu générer de requête SQL.", "generated_sql": ""}

        sql_result, error = await self._execute_sql_readonly(generated_sql)
        if error:
            return {"answer": error, "generated_sql": generated_sql}

        final_answer = await self.final_response_chain.ainvoke({"question": user_question, "sql_result": sql_result})
        return {"answer": final_answer, "generated_sql": generated_sql, "sql_result": sql_result}

    async def generate_forecast_narrative(self, body: NarrativeRequest) -> tuple[str, dict]:
        """Génère une narration pour des prévisions en utilisant un LLM via LangChain."""
        values = [p.value for p in body.series]
        stats = {"count": len(values), "min": min(values) if values else 0, "max": max(values) if values else 0}

        prompt_template = """
Tu es un analyste macro-financier de l'UEMOA. Rédige une synthèse professionnelle en {language} sur la prévision suivante.
Interprète seulement les chiffres fournis.
- Commence par un TL;DR.
- Décris la tendance générale.
- Termine par 2-3 pistes d'analyse liées à la politique monétaire.

Titre: {title}
Horizon: {horizon}
Unités: {unit}
Derniers points: {series_preview}

Réponse:
"""
        narrative_prompt = ChatPromptTemplate.from_template(prompt_template)
        narrative_chain = narrative_prompt | self.llm | StrOutputParser()

        series_preview = [{"date": p.date, "value": p.value} for p in body.series[-12:]]

        narrative = await narrative_chain.ainvoke({
            "language": body.language or "fr",
            "title": body.title or "Prévision",
            "horizon": body.horizon or "non précisé",
            "unit": body.unit or "unités",
            "series_preview": json.dumps(series_preview, indent=2, default=str)
        })
        return narrative, stats

    async def generate_inflation_interpretation(self, body: InflationInterpretationRequest) -> dict:
        """Génère une interprétation d'inflation en utilisant un LLM via LangChain."""
        prompt_template = """
Tu es l'économiste en chef de la BCEAO. Fournis une analyse narrative des prédictions d'inflation pour un(e) {audience}.
Contexte: Cible d'inflation de la BCEAO est de 1-3%.

DONNÉES DU MODÈLE:
- Prédictions: {predictions}
- Importance SHAP globale: {global_shap}
- Justification SHAP par date: {individual_shap}

RÈGLES:
- Justifie chaque affirmation avec une valeur SHAP.
- Ne mentionne que les facteurs présents dans les données.
- Structure ta réponse: Résumé, Analyse, Moteurs, Risques.

Analyse:
"""
        interpretation_prompt = ChatPromptTemplate.from_template(prompt_template)
        interpretation_chain = interpretation_prompt | self.llm | StrOutputParser()

        interpretation_text = await interpretation_chain.ainvoke({
            "audience": body.target_audience,
            "predictions": json.dumps(body.prediction_data.predictions, indent=2, default=str),
            "global_shap": json.dumps(body.prediction_data.global_shap_importance, indent=2, default=str),
            "individual_shap": json.dumps(body.prediction_data.individual_shap_explanations, indent=2, default=str)
        })

        # Le parsing de la réponse reste une logique métier simple
        return self._parse_inflation_interpretation(interpretation_text)

    def _parse_inflation_interpretation(self, text: str) -> dict:
        """Parse la sortie texte du LLM en une structure JSON (logique simplifiée)."""
        # Une implémentation plus robuste utiliserait PydanticOutputParser de LangChain
        return {"executive_summary": text[:1000], "full_analysis": text}

    def index_reference_queries(self, queries: list[str] | None = None) -> int:
        queries_to_index = queries or self.reference_queries
        if not queries_to_index:
            return 0
        ids = [f"query_{i}" for i, _ in enumerate(queries_to_index)]
        self.chroma_client.add_texts(texts=queries_to_index, ids=ids)
        logger.info(f"{len(queries_to_index)} requêtes indexées dans ChromaDB.")
        return len(queries_to_index)

    def _get_rich_db_schema(self, table_name: str) -> str:
        query = text(f"""
            SELECT c.column_name, c.data_type, pgd.description
            FROM information_schema.columns AS c
            LEFT JOIN pg_catalog.pg_stat_all_tables AS st ON c.table_schema = st.schemaname AND c.table_name = st.relname
            LEFT JOIN pg_catalog.pg_description AS pgd ON pgd.objoid = st.relid AND pgd.objsubid = c.ordinal_position
            WHERE c.table_name = :table_name ORDER BY c.ordinal_position;
        """)
        try:
            with self.admin_db_engine.connect() as connection:
                columns_result = connection.execute(query, {'table_name': table_name}).fetchall()
                schema_str = f"CREATE TABLE {table_name} (\n"
                for col in columns_result:
                    col_name, data_type, description = col
                    schema_str += f"    {col_name} {data_type}," + (f" -- {description.strip()}\n" if description else "\n")
                return schema_str.rstrip(',\n') + "\n);"
        except Exception as e:
            logger.error(f"Impossible de récupérer le schéma de la base de données : {e}")
            return f"CREATE TABLE {table_name} (...); -- Erreur"

    def _extract_sql_from_text(self, text: str) -> str:
        match = re.search(r"```(?:sql)?\s*([\s\S]*?)```", text) or re.search(r"\b(SELECT|WITH)\b[\s\S]*;", text, re.IGNORECASE)
        return match.group(1).strip() if match else text.strip()

    def _validate_sql(self, sql: str) -> bool:
        try:
            return not re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER)\b", sql, re.IGNORECASE) and len(sqlglot.parse(sql, read="postgres")) == 1
        except Exception:
            return False

    async def _execute_sql_readonly(self, sql: str) -> tuple[str | None, str | None]:
        if not self._validate_sql(sql):
            return None, "Requête SQL invalide ou non autorisée."
        try:
            def run():
                with self.db_engine.connect() as conn:
                    return [dict(row._mapping) for row in conn.execute(text(sql))]
            result = await asyncio.to_thread(run)
            return json.dumps(result, indent=2, default=str), None
        except exc.SQLAlchemyError as e:
            return None, f"Erreur d'exécution SQL: {e}"