# text-to-sql-project/api/app/services/query_orchestrator.py

# =======================================================================================
# REFACTORED TO USE LANGCHAIN FRAMEWORK
# =======================================================================================
# This file has been refactored to use LangChain components while maintaining the exact
# same business logic, API endpoints, security validations, and configurations as the
# original implementation.
#
# Key LangChain components used:
# - ChatOllama: Replaces manual ollama.AsyncClient for LLM interactions
# - HuggingFaceEmbeddings: Wrapper for sentence-transformers embeddings
# - Chroma: LangChain vector store wrapper for ChromaDB
# - PromptTemplate: Structured prompt management for SQL generation and responses
# - SQLDatabase: Wrapper for SQLAlchemy database operations
# =======================================================================================

import sqlglot
from sqlalchemy import create_engine, text, exc
from logging import getLogger
import re
import json
import asyncio

# LangChain imports - Core framework
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.utilities import SQLDatabase

# ChromaDB client for direct collection management (needed for indexing operations)
import chromadb
# Direct Ollama client for model pulling (ChatOllama doesn't expose pull method)
import ollama

from app.config import settings

logger = getLogger(__name__)

class QueryOrchestrator:
    def __init__(self):
        logger.info("Initialisation de QueryOrchestrator...")
        
        # =======================================================================================
        # LANGCHAIN EMBEDDINGS - Replaces direct SentenceTransformer usage
        # =======================================================================================
        # HuggingFaceEmbeddings wraps sentence-transformers models with LangChain interface
        # This maintains compatibility with the existing embedding model while providing
        # LangChain's standardized embedding interface
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            # Encode kwargs for consistency with original SentenceTransformer behavior
            encode_kwargs={'normalize_embeddings': False}
        )
        
        # =======================================================================================
        # CHROMADB CLIENT - Direct client for collection management (indexing operations)
        # =======================================================================================
        # We still need the direct ChromaDB client for indexing operations (add, delete)
        # that are not part of the LangChain Chroma interface
        self.chroma_client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
        self.sql_collection = self.chroma_client.get_or_create_collection(name=settings.CHROMA_COLLECTION)
        
        # =======================================================================================
        # LANGCHAIN CHROMA VECTOR STORE - Replaces manual ChromaDB query operations
        # =======================================================================================
        # Chroma vector store provides LangChain interface for similarity search
        # This will be used in process_user_question for retrieving similar SQL queries
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=settings.CHROMA_COLLECTION,
            embedding_function=self.embedding_model
        )
        
        # =======================================================================================
        # LANGCHAIN CHATOLLAMA - Replaces manual ollama.AsyncClient
        # =======================================================================================
        # ChatOllama provides LangChain's standardized chat interface for Ollama models
        # Maintains async support for non-blocking event loop operations
        # The base_url corresponds to the original OLLAMA_BASE_URL setting
        self.llm = ChatOllama(
            model=settings.LLM_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            # Preserve existing timeout behavior - LangChain will handle this via request_timeout
            # We'll handle timeouts explicitly in our async calls using asyncio.wait_for
        )
        
        # =======================================================================================
        # DIRECT OLLAMA CLIENT - For model pulling only
        # =======================================================================================
        # ChatOllama doesn't expose the pull method, so we need direct ollama.AsyncClient
        # This is only used in pull_model() for downloading models
        self.ollama_client = ollama.AsyncClient(host=settings.OLLAMA_BASE_URL)
        
        # =======================================================================================
        # CONCURRENCY CONTROL - Preserved from original implementation
        # =======================================================================================
        # Sémaphores pour limiter la concurrence par ressource
        # These semaphores ensure controlled concurrency for embeddings, vector store, and LLM
        self.embed_sem = asyncio.Semaphore(2)
        self.chroma_sem = asyncio.Semaphore(4)
        self.llm_sem = asyncio.Semaphore(2)
        
        try:
            # =======================================================================================
            # SQLALCHEMY DATABASE CONNECTIONS - Preserved from original
            # =======================================================================================
            # Connexion pour le LLM (utilisateur read-only)
            self.db_engine = create_engine(
                settings.DATABASE_URL,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
            )
            # Connexion pour l'admin (pour lire le schéma) via URL encodée
            self.admin_db_engine = create_engine(settings.ADMIN_DATABASE_URL)
            
            # =======================================================================================
            # LANGCHAIN SQLDATABASE - Wrapper for database schema introspection
            # =======================================================================================
            # SQLDatabase provides LangChain interface for database operations
            # This wraps our read-only database engine for potential future LangChain SQL chains
            self.langchain_db = SQLDatabase(
                engine=self.db_engine,
                # Specify the table we're working with
                include_tables=['indicateurs_economiques_uemoa']
            )
            
            logger.info("Connexions à la base de données PostgreSQL réussies.")
        except Exception as e:
            logger.error(f"Erreur de connexion à la base de données : {e}")
            raise

        # Get database schema using our custom method (maintains original logic)
        self.db_schema = self._get_rich_db_schema(table_name='indicateurs_economiques_uemoa')

        # Requêtes SQL d'exemple pour l'indexation sémantique
        self.reference_queries = [
            "SELECT pib_nominal_milliards_fcfa FROM indicateurs_economiques_uemoa WHERE date = '2022-01-01'",
            "SELECT date, exportations_biens_fob, importations_biens_fob, balance_des_biens FROM indicateurs_economiques_uemoa WHERE EXTRACT(YEAR FROM date) = 2021",
            "SELECT AVG(taux_croissance_reel_pib_pct) FROM indicateurs_economiques_uemoa WHERE date >= '2015-01-01' AND date <= '2020-01-01'",
            "SELECT MAX(encours_de_la_dette_pct_pib) FROM indicateurs_economiques_uemoa",
            "SELECT date, taux_inflation_moyen_annuel_ipc_pct FROM indicateurs_economiques_uemoa ORDER BY date DESC LIMIT 5",
            "SELECT SUM(recettes_fiscales) FROM indicateurs_economiques_uemoa WHERE EXTRACT(YEAR FROM date) > 2018",
            "SELECT date, solde_budgetaire_global_avec_dons FROM indicateurs_economiques_uemoa WHERE taux_inflation_moyen_annuel_ipc_pct > 5.0",
        ]
        logger.info("QueryOrchestrator initialisé avec succès.")

        # Initialisation des ensembles de mots-clés pour le routage des questions
        self._init_keyword_sets()

    def _is_question_harmful(self, text_q: str) -> bool:
        """
        Détection simple (heuristique) de contenus dangereux/illégaux.
        Si détecté, on bloque en amont sans appeler d'embed/LLM.
        """
        if not text_q:
            return False
        q = text_q.lower()
        banned_terms = [
            # ==== FRANÇAIS ====
            # Violence / armes
            "bombe", "explosif", "fabrication d'arme", "arme artisanale", "arme", "pistolet", "fusil",
            "grenade", "cocktail molotov", "mitraillette", "kalachnikov", "munitions", "couteau", "assassin",
            "meurtre", "tuer", "massacre", "terrorisme", "attentat", "prise d'otage", "violence",

            # Cybercriminalité
            "piratage", "hacker", "pirater", "craquer", "intrusion", "malware", "ransomware", 
            "phishing", "cheval de troie", "virus informatique", "backdoor", "attaque ddos",
            "keylogger", "spyware",

            # Drogues / substances
            "drogue", "stupéfiant", "cannabis", "cocaïne", "héroïne", "ecstasy", "lsd", "meth", 
            "opium", "poison", "toxicomanie",

            # Escroquerie / arnaque
            "arnaque", "escroquerie", "fraude", "blanchiment", "usurpation d'identité",

            # Contenus sensibles
            "sexe", "pornographie", "pédophilie", "inceste", "viol", "prostitution",

            # ==== ENGLISH ====
            # Violence / weapons
            "bomb", "explosive", "make a bomb", "weapon", "homemade gun", "grenade", "molotov",
            "gun", "rifle", "pistol", "ammunition", "knife", "terrorism", "attack", "shooting", 
            "murder", "kill", "massacre", "hostage",

            # Cybercrime
            "hack", "hacking", "crack", "malware", "ransomware", "phishing", "scam", "trojan",
            "virus", "ddos", "keylogger", "spyware", "backdoor",

            # Drugs / substances
            "drug", "cannabis", "cocaine", "heroin", "ecstasy", "lsd", "meth", "opium", "toxic",
            "poison",

            # Fraud / scams
            "fraud", "identity theft", "money laundering", "phishing scam",

            # Sensitive content
            "sex", "porn", "child porn", "incest", "rape", "prostitution",
        ]

        return any(term in q for term in banned_terms)

    def _init_keyword_sets(self) -> None:
        """Construit dynamiquement les ensembles de mots-clés utilisés pour filtrer les questions."""
        base_economic_keywords = {
            "uemoa", "bceao", "union économique", "union monétaire",
            "pib", "produit intérieur brut", "croissance économique",
            "inflation", "déflation", "prix", "ipc", "indice prix",
            "taux", "taux d'intérêt", "taux directeur", "politique monétaire",
            "dette", "dette publique", "encours dette", "dette pib",
            "recettes fiscales", "dépenses publiques", "budget", "solde budgétaire",
            "importations", "exportations", "balance commerciale", "biens fob",
            "réserves", "réserves internationales", "change", "devise",
            "masse monétaire", "m2", "m3", "liquidité bancaire",
            "investissement", "consommation", "épargne", "transferts",
            "diaspora", "migrants", "transferts migrants",
            "agriculture", "industrie", "services", "secteurs",
            "emploi", "chômage", "population active",
            "fcfa", "franc cfa", "zone franc", "euro",
            "indicateurs économiques", "statistiques", "données économiques",
            "contribution", "valeur ajoutée"
        }

        sql_keywords = {
            "requête", "requete", "sql", "select", "where", "group by", "order by",
            "table", "colonne", "colonnes", "base de données", "base donnees",
            "extraire", "calculer", "comparer", "analyser"
        }

        date_keywords = {
            "date", "période", "periode", "année", "annee", "mois", "trimestre",
            "dernier", "dernière", "récent", "récente", "actuel", "actuelle",
            "2020", "2021", "2022", "2023", "2024", "2025"
        }

        dynamic_keywords: set[str] = set()
        try:
            with self.admin_db_engine.connect() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT column_name,
                               COALESCE(col_description((table_schema||'.'||table_name)::regclass::oid, ordinal_position), '') AS comment
                        FROM information_schema.columns
                        WHERE table_schema = :schema AND table_name = :table
                        ORDER BY ordinal_position
                        """
                    ),
                    {"schema": "public", "table": "indicateurs_economiques_uemoa"}
                ).fetchall()

            for column_name, comment in rows:
                if not column_name:
                    continue
                column_name = column_name.lower()
                dynamic_keywords.add(column_name)
                tokens = column_name.split('_')
                dynamic_keywords.update(tokens)
                if len(tokens) >= 2:
                    dynamic_keywords.add(' '.join(tokens[:2]))
                if len(tokens) >= 3:
                    dynamic_keywords.add(' '.join(tokens[:3]))
                if comment:
                    comment_tokens = re.split(r"[^a-zA-Zàâçéèêëîïôûùüÿñæœ']+", comment.lower())
                    dynamic_keywords.update({tok for tok in comment_tokens if tok and len(tok) > 3})

            logger.info("Mots-clés dynamiques chargés depuis le schéma: %d entrées", len(dynamic_keywords))
        except Exception as exc:
            logger.warning("Impossible de générer les mots-clés dynamiques depuis le schéma: %s", exc)

        self.economic_keywords = base_economic_keywords | dynamic_keywords
        self.sql_keywords = sql_keywords
        self.date_keywords = date_keywords

    def _needs_data_retrieval(self, text_q: str) -> bool:
        """
        Heuristique stricte pour décider si la question concerne UNIQUEMENT les données économiques UEMOA/BCEAO.
        Refuse les questions générales, hors-sujet, ou trop vagues.
        """
        if not text_q or len(text_q.strip()) < 10:  # Questions trop courtes
            return False

        q = text_q.lower().strip()

        economic_count = sum(1 for kw in self.economic_keywords if kw in q)
        sql_count = sum(1 for kw in self.sql_keywords if kw in q)
        date_count = sum(1 for kw in self.date_keywords if kw in q)

        # Critères stricts pour accepter la question :
        # - Au moins 2 mots-clés économiques
        #   OU (1 mot-clé économique + référence temporelle)
        #   OU (1 mot-clé économique + intention SQL explicite)
        has_economic_focus = (
            economic_count >= 2
            or (economic_count >= 1 and date_count >= 1)
            or (economic_count >= 1 and sql_count >= 1)
        )
        has_temporal_reference = date_count >= 1 or sql_count >= 1

        return has_economic_focus and has_temporal_reference

    async def _execute_sql_readonly(self, sql: str) -> list[dict]:
        """Exécute une requête SQL en lecture seule via un thread pour ne pas bloquer l'event loop."""
        def run():
            with self.db_engine.connect() as connection:
                result_proxy = connection.execute(text(sql))
                return [dict(row._mapping) for row in result_proxy]
        try:
            return await asyncio.to_thread(run)
        except exc.SQLAlchemyError as e:
            raise e

    async def generate_forecast_narrative(self, body) -> tuple[str, dict]:
        """
        Génère une narration en LN pour un résultat de prévision fourni (sans calculer la prévision).
        - Applique une vérification de contenu (harmful) et répond poliment si besoin.
        - Utilise le LLM partagé avec les sémaphores et timeouts existants.
        Retourne (narrative, summary_stats_dict).
        """
        # 1) Sécurité du prompt/user input
        user_ctrl = f"{body.title or ''} {body.target} {body.horizon or ''}".strip()
        if self._is_question_harmful(user_ctrl):
            return "Sorry, I can't assist with that.", {
                "count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "start_value": 0.0, "end_value": 0.0,
                "start_date": None, "end_date": None,
            }

        # 2) Construire des stats simples (robustesse de base)
        values = [p.value for p in body.series]
        dates = [p.date for p in body.series if p.date]
        if values:
            vmin, vmax = min(values), max(values)
            mean = sum(values) / len(values)
            start_value, end_value = values[0], values[-1]
        else:
            vmin = vmax = mean = start_value = end_value = 0.0
        start_date = dates[0] if dates else None
        end_date = dates[-1] if dates else None
        stats = {
            "count": len(values),
            "min": float(vmin),
            "max": float(vmax),
            "mean": float(mean),
            "start_value": float(start_value),
            "end_value": float(end_value),
            "start_date": start_date,
            "end_date": end_date,
        }

        # 3) Composer un prompt sobre (FR/EN)
        lang = body.language or "fr"
        tone = body.tone or "professionnel"
        unit = body.unit or "unités"
        target = body.target
        series_preview = []
        # Limiter la taille du prompt: n'envoyer que les 12 derniers points
        tail = body.series[-12:] if len(body.series) > 12 else body.series
        for p in tail:
            series_preview.append({"date": p.date, "value": p.value})

        if lang == "fr":
            prompt = (
                "Tu es un analyste macro-financier spécialisé en politique monétaire dans l'Union Économique et Monétaire Ouest-Africaine (UEMOA/BCEAO). "
                "Rédige une synthèse claire, concise et professionnelle en français, dans un style narratif et explicatif, ton {tone}. "
                "Tu reçois ci-dessous des points de prévision pour {target}. "
                "Tu n'effectues AUCUN calcul de prévision ; tu interprètes uniquement les chiffres fournis.\n\n"

                "Consignes:\n"
                "- Commence toujours par un TL;DR (1–2 phrases).\n"
                "- Décris la tendance générale, les points remarquables et le niveau d'incertitude si des bornes sont présentes.\n"
                "- Indique clairement les unités ({unit}) et la période couverte si disponible.\n"
                "- Sois factuel et rigoureux : n'invente aucune donnée.\n"
                "- Ne suggère JAMAIS que les données sont simulées, hypothétiques ou artificielles, sauf si l'entrée le précise explicitement.\n"
                "- Ne commente pas la provenance, la fiabilité ou la nature de la source des données.\n"
                "- Ne parle pas d’un pays spécifique si non indiqué : considère par défaut que les données concernent l’ensemble de l’Union (UEMOA/BCEAO).\n"
                "- Ne rajoute aucun avertissement générique ni formulation du type 'il est important de noter que...' sauf si demandé.\n"
                "- Termine toujours par 2–3 pistes d’analyse complémentaires, en lien avec la politique monétaire (par exemple : impact potentiel sur la liquidité bancaire, la stabilité des prix, la balance extérieure).\n\n"

                f"Titre: {body.title or 'Prévision'}\n"
                f"Horizon: {body.horizon or 'non précisé'}\n"
                f"Unités: {unit}\n"
                f"Stats (approx.): {stats}\n"
                f"Derniers points (max 12): {series_preview}\n\n"
                "Réponse:"
            )

        else:
            prompt = (
                "You are a macro-financial analyst specialized in monetary policy within the West African Economic and Monetary Union (WAEMU/BCEAO). "
                "Write a clear, concise, and professional narrative in English, tone {tone}. "
                "You receive forecast points for {target}. "
                "Do NOT compute new forecasts; only interpret the provided numbers.\n\n"

                "Guidelines:\n"
                "- Always start with a TL;DR (1–2 sentences).\n"
                "- Describe the overall trend, highlight notable points, and mention the level of uncertainty if bounds are provided.\n"
                "- Clearly state the units ({unit}) and the covered period if available.\n"
                "- Be strictly factual: never invent or extrapolate numbers.\n"
                "- Never suggest the data is simulated, hypothetical, or artificial unless this is explicitly indicated in the input.\n"
                "- Do not comment on the reliability, origin, or source of the data.\n"
                "- Do not mention individual countries unless explicitly present in the input; by default, interpret the data as covering the entire WAEMU (BCEAO).\n"
                "- Do not add generic disclaimers (e.g., 'it is important to note that...') unless explicitly requested.\n"
                "- Always close with 2–3 concrete ideas for further analysis, connected to monetary policy (e.g., liquidity management, inflation outlook, external balance).\n\n"

                f"Title: {body.title or 'Forecast'}\n"
                f"Horizon: {body.horizon or 'unspecified'}\n"
                f"Units: {unit}\n"
                f"Stats (approx.): {stats}\n"
                f"Last points (max 12): {series_preview}\n\n"
                "Answer:"
            )


        # =======================================================================================
        # LANGCHAIN CHATOLLAMA - Forecast narrative generation
        # =======================================================================================
        # 4) Appel LLM avec limites et timeouts
        try:
            async with self.llm_sem:
                res = await asyncio.wait_for(
                    self.llm.ainvoke(prompt),
                    timeout=90,
                )
            # Extract content from AIMessage
            narrative = res.content.strip()
        except Exception as e:
            logger.error(f"Erreur LLM lors de la narration de prévision: {e}")
            narrative = "Désolé, une erreur est survenue lors de la génération de la narration."

        return narrative, stats

    def _extract_sql_from_text(self, text: str) -> str:
        """
        Extrait la requête SQL d'un texte potentiellement verbeux renvoyé par le LLM.
        Stratégie:
        1) Si un bloc ```sql ... ``` est présent, on l'extrait.
        2) Sinon, on cherche le premier mot-clé SELECT ou WITH et on prend jusqu'au premier ';' si présent.
        3) Trim final.
        """
        if not text:
            return ""
        # 1) Bloc Markdown ```sql ... ```
        code_block = re.search(r"```(?:sql)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        if code_block:
            candidate = code_block.group(1).strip()
            return candidate
        # 2) Heuristique SELECT/WITH
        m = re.search(r"\b(SELECT|WITH)\b[\s\S]*", text, flags=re.IGNORECASE)
        if m:
            candidate = text[m.start():]
            # couper au premier ';' si présent
            semi = candidate.find(';')
            if semi != -1:
                candidate = candidate[:semi+1]
            return candidate.strip()
        return text.strip()

    def _get_rich_db_schema(self, table_name: str) -> str:
        logger.info(f"Récupération du schéma enrichi pour la table '{table_name}'...")
        query = text(f"""
            SELECT
                c.column_name,
                c.data_type,
                pgd.description
            FROM
                information_schema.columns AS c
            LEFT JOIN
                pg_catalog.pg_stat_all_tables AS st ON c.table_schema = st.schemaname AND c.table_name = st.relname
            LEFT JOIN
                pg_catalog.pg_description AS pgd ON pgd.objoid = st.relid AND pgd.objsubid = c.ordinal_position
            WHERE
                c.table_name = :table_name
            ORDER BY
                c.ordinal_position;
        """)
        
        table_comment_query = text(f"""
            SELECT obj_description('public.{table_name}'::regclass);
        """)

        try:
            with self.admin_db_engine.connect() as connection:
                table_comment_result = connection.execute(table_comment_query).scalar_one_or_none()
                columns_result = connection.execute(query, {'table_name': table_name}).fetchall()

                schema_str = f"-- Description de la table '{table_name}': {table_comment_result}\n"
                schema_str += f"CREATE TABLE {table_name} (\n"
                for col in columns_result:
                    col_name, data_type, description = col
                    schema_str += f"    {col_name} {data_type},"
                    if description:
                        schema_str += f" -- {description}\n"
                    else:
                        schema_str += "\n"
                schema_str = schema_str.rstrip(',\n') + "\n);"
                logger.info("Schéma enrichi récupéré avec succès.")
                return schema_str
        except Exception as e:
            logger.error(f"Impossible de récupérer le schéma de la base de données : {e}")
            return f"CREATE TABLE {table_name} (...); -- Erreur: impossible de récupérer le schéma détaillé"

    def index_reference_queries(self, queries: list[str] | None = None):
        """
        Index reference SQL queries into ChromaDB for semantic similarity search.
        
        LANGCHAIN INTEGRATION:
        - Uses HuggingFaceEmbeddings (LangChain wrapper) instead of direct SentenceTransformer
        - The embed_documents method is the LangChain standard interface for batch embedding
        - Maintains original indexing logic and behavior
        """
        queries_to_index = queries or self.reference_queries
        if not queries_to_index:
            logger.warning("Aucune requête de référence à indexer.")
            return 0
            
        logger.info(f"Indexation de {len(queries_to_index)} requêtes...")
        if self.sql_collection.count() > 0:
            ids_to_delete = self.sql_collection.get().get('ids') or []
            if ids_to_delete and isinstance(ids_to_delete[0], list):
                ids_to_delete = [i for sub in ids_to_delete for i in sub]
            if ids_to_delete:
                self.sql_collection.delete(ids=ids_to_delete)

        # =======================================================================================
        # LANGCHAIN EMBEDDINGS - Using embed_documents for batch embedding
        # =======================================================================================
        # embed_documents is LangChain's standard method for batch text embedding
        # Returns list of embeddings (already as lists, not numpy arrays like SentenceTransformer)
        embeddings = self.embedding_model.embed_documents(queries_to_index)
        
        self.sql_collection.add(
            embeddings=embeddings,
            documents=queries_to_index,
            ids=[f"query_{i}" for i in range(len(queries_to_index))]
        )
        logger.info("Indexation terminée.")
        return len(queries_to_index)

    def _validate_sql(self, sql_query: str) -> bool:
        try:
            # 1) Bloquer explicitement toute instruction non-lecture (sécurité défensive)
            banned = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|MERGE|CALL|COPY|VACUUM|ANALYZE|EXPLAIN)\b", re.IGNORECASE)
            if banned.search(sql_query):
                logger.warning("Validation échouée : mot-clé non autorisé détecté (DML/DDL).")
                return False

            # 2) S'assurer qu'il n'y a qu'une seule instruction
            exprs = sqlglot.parse(sql_query, read="postgres")
            if not exprs or len(exprs) != 1:
                logger.warning("Validation échouée : aucune ou plusieurs instructions détectées.")
                return False

            expr = exprs[0]

            # 3) Aplatir un éventuel WITH (CTE) pour récupérer l'expression sous-jacente
            if isinstance(expr, sqlglot.exp.With):
                base_expr = expr.this  # le SELECT/UNION encapsulé par le WITH
            else:
                base_expr = expr

            # 4) Autoriser uniquement des requêtes de lecture : SELECT, UNION/EXCEPT/INTERSECT de SELECT
            allowed_types = (
                sqlglot.exp.Select,
                sqlglot.exp.Union,
                sqlglot.exp.Except,
                sqlglot.exp.Intersect,
            )
            if not isinstance(base_expr, allowed_types):
                logger.warning(f"Validation échouée : l'expression n'est pas une requête SELECT/UNION. Type: {type(base_expr)}")
                return False

            logger.info("Validation SQL réussie (lecture uniquement).")
            return True
        except Exception as e:
            logger.error(f"Erreur de validation SQL : {e}. Requête : {sql_query}")
            return False

    async def process_user_question(self, user_question: str) -> dict:
        # 0) Sécurité: blocage des contenus dangereux/illégaux
        if self._is_question_harmful(user_question):
            return {"answer": "Désolé, je ne peux pas traiter cette demande."}

        # 0.1) Routage d'intention: refuser les questions hors-sujet économique UEMOA/BCEAO
        if not self._needs_data_retrieval(user_question):
            return {
                "answer": (
                    "Désolé, cette question ne concerne pas les données économiques de l'UEMOA/BCEAO. "
                    "Je ne peux traiter que les questions relatives aux indicateurs économiques, "
                    "statistiques financières, ou analyses de données de l'Union Économique et Monétaire Ouest-Africaine. "
                    "Veuillez reformuler votre question pour qu'elle porte sur des données économiques spécifiques "
                    "(PIB, inflation, dette, balance commerciale, etc.) avec une période temporelle définie."
                )
            }

        # =======================================================================================
        # LANGCHAIN SIMILARITY SEARCH - Replaces manual embedding + ChromaDB query
        # =======================================================================================
        # Using Chroma's similarity_search to retrieve similar SQL examples
        # This replaces: embed_query → tolist → sql_collection.query
        async with self.chroma_sem:
            similar_docs = await asyncio.to_thread(
                self.vector_store.similarity_search,
                user_question,
                k=5
            )
        
        # Extract document content (page_content is LangChain's standard attribute)
        context_queries = "\n".join([doc.page_content for doc in similar_docs]) if similar_docs else ""

        # =======================================================================================
        # LANGCHAIN PROMPTTEMPLATE - Structured SQL generation prompt
        # =======================================================================================
        # PromptTemplate provides a structured way to manage prompts with variables
        # This replaces manual f-string formatting for better maintainability
        sql_generation_template = PromptTemplate(
            input_variables=["db_schema", "context_queries", "user_question"],
            template="""### Instruction (génération SQL)
Tu es un expert SQL (PostgreSQL) et analyste économique spécialisé en politique monétaire de l'UEMOA.  
Ton objectif : convertir la question de l'utilisateur en **UNE SEULE** requête SQL **SELECT** (ou une requête WITH ... SELECT) basée strictement sur le schéma et les descriptions fournis.

Contraintes strictes :
- **Toutes les données de la base concernent par défaut l'ensemble de l'Union (BCEAO)** et non un pays spécifique. 
→ Ne filtre un pays particulier **que si la colonne `country` (ou équivalent) est explicitement utilisée dans la question**.
- **Retourne UNIQUEMENT la requête SQL**, sans texte additionnel, sans explication, sans Markdown, sans commentaires, sans backticks. La sortie doit commencer par `SELECT` ou `WITH` et se terminer par un point-virgule (`;`).
- **Ne génère jamais** de requêtes de modification (INSERT, UPDATE, DELETE, DROP, etc.).
- Préfère les fonctions temporelles PostgreSQL (ex. `DATE_TRUNC`) pour les agrégations sur date.
- Utilise des agrégations appropriées (`AVG`, `SUM`, `COUNT`, `MAX`, `MIN`) quand la question demande des synthèses.
- **N'invente pas** de colonnes, tables ou valeurs : n'utilise que les tables/colonnes du schéma donné.
- Si la question demande une liste non agrégée potentiellement volumineuse, **limite la sortie à 1000 lignes** et ajoute un `ORDER BY` pertinent.
- Si la question demande une comparaison (périodes, zones, entités), inclure clairement la clause `GROUP BY` nécessaire.
- Si l'intention de la question est ambiguë (ex. période non précisée), choisir une hypothèse raisonnable basée sur les colonnes dates du schéma (sans commentaire dans la sortie).
- Utilise des alias clairs pour les champs agrégés (ex. `AS avg_inflation`).

### Schéma et descriptions
{db_schema}

### Exemples de requêtes similaires
{context_queries}

### Question de l'utilisateur
"{user_question}"

### Requête SQL
"""
        )
        
        sql_generation_prompt = sql_generation_template.format(
            db_schema=self.db_schema,
            context_queries=context_queries,
            user_question=user_question
        )



        # =======================================================================================
        # LANGCHAIN CHATOLLAMA - Async LLM invocation with semaphore control
        # =======================================================================================
        # ainvoke is LangChain's async interface for chat models
        # Replaces: ollama_client.generate with standardized LangChain interface
        try:
            async with self.llm_sem:
                response = await asyncio.wait_for(
                    self.llm.ainvoke(sql_generation_prompt),
                    timeout=90,
                )
            # ChatOllama returns AIMessage, extract content
            generated_sql = self._extract_sql_from_text(response.content)
        except Exception as e:
            msg = str(e).lower()
            # Si le modèle n'est pas trouvé (404), tenter un pull puis retenter une fois
            if "not found" in msg or "404" in msg:
                logger.warning(f"Modèle '{settings.LLM_MODEL}' introuvable. Tentative de téléchargement puis nouvel essai...")
                pull_res = await self.pull_model(settings.LLM_MODEL)
                if pull_res.get("status") == "success":
                    try:
                        async with self.llm_sem:
                            response = await asyncio.wait_for(
                                self.llm.ainvoke(sql_generation_prompt),
                                timeout=90,
                            )
                        generated_sql = self._extract_sql_from_text(response.content)
                    except Exception as e2:
                        logger.error(f"Échec de la génération après pull du modèle: {e2}")
                        return {"answer": "Désolé, échec de la génération SQL même après téléchargement du modèle."}
                else:
                    logger.error(f"Échec du téléchargement du modèle: {pull_res}")
                    return {"answer": "Désolé, le modèle LLM n'est pas disponible et le téléchargement a échoué."}
            else:
                logger.error(f"Erreur lors de la génération SQL par Ollama : {e}")
                return {"answer": "Désolé, une erreur est survenue lors de la génération de la requête SQL."}

        # Si aucune requête n'a pu être générée, retourner une réponse claire sans passer à la validation/exécution
        if not generated_sql or not re.search(r"^\s*(SELECT|WITH)\b", generated_sql, flags=re.IGNORECASE):
            return {
                "answer": (
                    "Je n'ai pas pu générer une requête SQL pertinente pour cette question. "
                    "Pouvez-vous préciser la période, les colonnes ou la condition souhaitée ?"
                ),
                "generated_sql": generated_sql
            }

        if not self._validate_sql(generated_sql):
            return {
                "answer": "La requête SQL générée a été jugée non sécurisée et a été bloquée.",
                "generated_sql": generated_sql
            }

        try:
            sql_result = await self._execute_sql_readonly(generated_sql)
            sql_result_str = str(sql_result)
        except exc.SQLAlchemyError as e:
            logger.error(f"Erreur lors de l'exécution de la requête SQL : {e}")
            return {
                "answer": f"Une erreur est survenue lors de l'exécution de la requête sur la base de données. L'erreur était : {str(e)}",
                "generated_sql": generated_sql
            }

        natural_language_prompt = f"""
            ### Instruction (rédaction d'analyse augmentée)
            Tu es un analyste économique de la BCEAO. En te basant **seulement** sur la QUESTION de l'utilisateur et **EXCLUSIVEMENT** sur le RÉSULTAT SQL fourni, rédige une **analyse augmentée, synthétique, explicative, exacte et rationnelle** en français.  
            **Toutes les données de la base concernent l'ensemble de l'Union (BCEAO), pas un pays spécifique.** Ne parle d’un pays particulier que si la colonne `country` (ou équivalent) est explicitement présente dans `sql_result_str` ou mentionnée dans la question.

            Ton style :
            - Narratif et évolutif : commence par un bref résumé, puis déroule progressivement l'interprétation.
            - Respecte un ton pédagogique, clair et professionnel adapté à un public analyste/chargé de décision.
            - Sois partenaire de réflexion : suggère pistes et questions suivantes sans inventer.

            Structure recommandée (sans mettre les titres comme,**TL;DR**,**Contexte et portée**, etc.):
            1. **TL;DR** : synthèse immédiate (1–2 phrases).
            2. **Contexte et portée** : préciser que les résultats concernent l'ensemble de l'Union BCEAO, sauf si `country` est utilisé → dans ce cas distinguer par pays.
            3. **Métriques clés** : mettre en avant les chiffres issus de `sql_result_str`, formatés clairement.
            4. **Interprétation raisonnée** : expliquer ce que signifient ces chiffres pour la politique monétaire de l'Union.
            5. **Méthodologie / provenance** : préciser brièvement les colonnes utilisées (ex. "agrégation mensuelle sur `month` et `avg_rate`").
            6. **Limites & hypothèses** : indiquer ce que les données ne permettent pas d’affirmer (granularité, absence de dimension pays).
            7. **Recommandations** : 2–4 idées d’analyses complémentaires.
            8. **Invitation à reformuler** : si la réponse est incomplète ou ambiguë, demander précision à l’utilisateur.

            Règles :
            - Si `sql_result_str` est vide ou insuffisant → répondre honnêtement : **"Aucune donnée exploitable trouvée — merci de préciser/affiner votre question."**
            - Ne jamais inventer de chiffres ni d’informations hors `sql_result_str`.
            - Si ambiguïté → proposer 1–2 reformulations possibles.

            ### Question de l'utilisateur
            {user_question}

            ### Résultat SQL
            {sql_result_str}

            ### Réponse
            """


        try:
            async with self.llm_sem:
                final_response = await asyncio.wait_for(
                    self.llm.ainvoke(natural_language_prompt),
                    timeout=90,
                )
            # Extract content from AIMessage
            final_answer = final_response.content
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la réponse finale par Ollama : {e}")
            return {"answer": "Désolé, une erreur est survenue lors de la formulation de la réponse finale."}

        return {
            "answer": final_answer,
            "generated_sql": generated_sql,
            "sql_result": sql_result_str
        }

    async def pull_model(self, model: str | None = None) -> dict:
        """Force le téléchargement du modèle dans Ollama."""
        target_model = model or settings.LLM_MODEL
        try:
            # Appel non-stream pour simplifier, on attend la fin du pull
            async with self.llm_sem:
                await asyncio.wait_for(self.ollama_client.pull(model=target_model), timeout=600)
            return {"status": "success", "model": target_model}
        except Exception as e:
            logger.error(f"Erreur lors du pull du modèle '{target_model}' : {e}")
            return {"status": "error", "message": str(e), "model": target_model}

    async def format_inflation_prediction(self, prediction_data: dict) -> dict:
        """
        Formate les données de prédiction d'inflation reçues du modèle externe selon le schéma InflationPredictionResponse.
        
        Args:
            prediction_data: Dictionnaire contenant les prédictions d'inflation brutes du modèle
            
        Returns:
            Dictionnaire formaté selon InflationPredictionResponse
        """
        try:
            # Validation et formatage spécifique aux prédictions d'inflation
            formatted_response = {
                "predictions": prediction_data.get("predictions", {}),
                "global_shap_importance": prediction_data.get("global_shap_importance", {}),
                "shap_summary_details": prediction_data.get("shap_summary_details", {}),
                "individual_shap_explanations": prediction_data.get("individual_shap_explanations", {}),
                "confidence_intervals": prediction_data.get("confidence_intervals", None)
            }
            
            # Validation des données d'inflation
            self._validate_inflation_data(formatted_response)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Erreur lors du formatage de la prédiction d'inflation : {e}")
            raise

    async def generate_inflation_interpretation(self, body) -> dict:
        """
        Génère une interprétation économique spécialisée des prédictions d'inflation SHAP 
        pour les économistes et analystes de la BCEAO.
        
        Args:
            body: InflationInterpretationRequest contenant les données de prédiction et paramètres
            
        Returns:
            Dictionnaire contenant l'interprétation économique formatée spécifique à l'inflation
        """
        try:
            # Extraction des données de prédiction d'inflation
            prediction_data = body.prediction_data
            language = body.analysis_language
            audience = body.target_audience
            include_policy_recs = body.include_policy_recommendations
            include_monetary_analysis = body.include_monetary_policy_analysis
            focus_bceao = body.focus_on_bceao_mandate
            
            # Construction du prompt d'interprétation spécifique à l'inflation
            interpretation_prompt = self._build_inflation_interpretation_prompt(
                prediction_data, language, audience, include_monetary_analysis, focus_bceao
            )
            
            # =======================================================================================
            # LANGCHAIN CHATOLLAMA - Inflation interpretation generation
            # =======================================================================================
            # Génération de l'interprétation via LLM
            async with self.llm_sem:
                response = await asyncio.wait_for(
                    self.llm.ainvoke(interpretation_prompt),
                    timeout=120
                )
            
            # Extract content from AIMessage
            interpretation_text = response.content.strip()
            
            # Parsing et structuration de la réponse spécifique à l'inflation
            structured_interpretation = self._parse_inflation_interpretation(
                interpretation_text, include_policy_recs
            )
            
            return structured_interpretation
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de l'interprétation d'inflation : {e}")
            raise

    def _validate_inflation_data(self, prediction_data):
        """
        Valide que les données de prédiction d'inflation sont cohérentes.
        """
        predictions = prediction_data.get("predictions", {})
        
        # Vérifier que les valeurs d'inflation sont dans une plage raisonnable
        for period, value in predictions.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Valeur d'inflation invalide pour {period}: {value}")
            if value < -10 or value > 50:  # Plage raisonnable pour l'inflation (%)
                logger.warning(f"Valeur d'inflation inhabituelle pour {period}: {value}%")
        
        # Vérifier la présence des facteurs d'inflation typiques
        shap_importance = prediction_data.get("global_shap_importance", {})
        expected_factors = ["taux_change", "prix_petrole", "masse_monetaire", "alimentation"]
        
        for factor in expected_factors:
            found = any(factor in key.lower() for key in shap_importance.keys())
            if not found:
                logger.info(f"Facteur d'inflation typique non trouvé: {factor}")

    def _build_inflation_interpretation_prompt(self, prediction_data, language, audience, include_monetary_analysis, focus_bceao):
        """
        Construit le prompt spécialisé pour l'interprétation des prédictions d'inflation.
        """
        audience_descriptions = {
            "economist": {"fr": "économiste spécialisé en politique monétaire", "en": "monetary policy economist"},
            "analyst": {"fr": "analyste inflation", "en": "inflation analyst"},
            "policymaker": {"fr": "décideur de politique monétaire", "en": "monetary policymaker"},
            "general": {"fr": "public général", "en": "general public"}
        }
        
        audience_desc = audience_descriptions[audience][language]
        institutional_line_fr = ""
        institutional_line_en = ""
        if focus_bceao:
            institutional_line_fr = "- Met en avant le mandat de stabilité des prix de la BCEAO et les obligations statutaires vis-à-vis des États membres."
            institutional_line_en = "- Highlight BCEAO's price stability mandate and statutory obligations toward member states."
        
        predictions = prediction_data.predictions
        if predictions:
            avg_inflation = sum(predictions.values()) / len(predictions)
            trend = "hausse" if list(predictions.values())[-1] > list(predictions.values())[0] else "baisse"
        else:
            avg_inflation = 0
            trend = "stable"

        # ==============================================================================
        # Traitement avancé des données SHAP pour justification
        # ==============================================================================
        
        # 1. Préparer les SHAP individuels arrondis pour la lisibilité
        individual_shap = getattr(prediction_data, 'individual_shap_explanations', None) or {}
        individual_shap_rounded: dict = {}
        for d, feats in individual_shap.items():
            try:
                # Arrondir les valeurs pour alléger le prompt et faciliter la lecture par le LLM
                individual_shap_rounded[d] = {k: round(float(v), 6) for k, v in feats.items()}
            except (ValueError, TypeError):
                individual_shap_rounded[d] = feats # Garder tel quel en cas d'erreur

        # 2. Identifier les Top N contributeurs positifs et négatifs par date
        TOP_N = 5
        top_contrib_by_date: dict = {}
        for d, feats in individual_shap_rounded.items():
            items = list(feats.items())
            # Trier pour trouver les contributeurs les plus forts (positifs et négatifs)
            pos_sorted = [it for it in sorted(items, key=lambda x: x[1], reverse=True) if it[1] > 0]
            neg_sorted = [it for it in sorted(items, key=lambda x: x[1]) if it[1] < 0]
            top_contrib_by_date[d] = {
                "top_positive": pos_sorted[:TOP_N],
                "top_negative": neg_sorted[:TOP_N],
            }

        # 3. Lister toutes les features présentes pour éviter l'hallucination de facteurs externes
        features_present = set()
        try:
            features_present.update((prediction_data.global_shap_importance or {}).keys())
        except Exception:
            pass
        for feats in individual_shap_rounded.values():
            features_present.update(feats.keys())
        features_present_list = sorted(list(features_present))

        # ==============================================================================
        # Instructions de granularité selon l'audience
        # ==============================================================================
        if language == "fr":
            audience_instructions = {
                "policymaker": (
                    "- Niveau de détail: concis et décisionnel.\n"
                    "- Mettre en avant 3–5 points clés, avec chiffres essentiels uniquement.\n"
                    "- Expliciter clairement les top contributeurs SHAP par date sans jargon inutile.\n"
                ),
                "analyst": (
                    "- Niveau de détail: intermédiaire.\n"
                    "- Pour chaque date, lister le top N positif/négatif avec valeurs SHAP et courte justification.\n"
                    "- Ajouter des liens entre facteurs et mécanismes de transmission.\n"
                ),
                "economist": (
                    "- Niveau de détail: technique et complet.\n"
                    "- Pour chaque date, expliquer chiffre par chiffre la prévision et les contributions SHAP (valeur et signe).\n"
                    "- Décrire les interactions, effets de base/saisonnalité et persistance attendue.\n"
                ),
                "general": (
                    "- Niveau de détail: pédagogique et simplifié.\n"
                    "- Expliquer avec des métaphores sobres, toujours en citant les chiffres saillants.\n"
                ),
            }[audience]
        else: # 'en'
            audience_instructions = {
                "policymaker": (
                    "- Detail level: concise and decision-oriented.\n"
                    "- Highlight 3–5 key points with only essential figures.\n"
                    "- Clearly list top SHAP contributors per date without unnecessary jargon.\n"
                ),
                "analyst": (
                    "- Detail level: intermediate.\n"
                    "- For each date, list top N positive/negative with SHAP values and brief justification.\n"
                    "- Add links between factors and transmission mechanisms.\n"
                ),
                "economist": (
                    "- Detail level: technical and thorough.\n"
                    "- For each date, explain forecast numbers and SHAP contributions (value and sign).\n"
                    "- Describe interactions, base/seasonal effects and expected persistence.\n"
                ),
                "general": (
                    "- Detail level: educational and simplified.\n"
                    "- Explain with plain language while citing salient figures.\n"
                ),
            }[audience]

        # Sérialiser les structures complexes pour une injection propre dans le prompt
        try:
            shap_individuals_str = json.dumps(individual_shap_rounded, ensure_ascii=False, indent=2)
            top_contrib_str = json.dumps(top_contrib_by_date, ensure_ascii=False, indent=2)
        except Exception:
            shap_individuals_str = str(individual_shap_rounded)
            top_contrib_str = str(top_contrib_by_date)

        # ==============================================================================
        # Construction du prompt enrichi
        # ==============================================================================
        if language == "fr":
            prompt = f"""
Tu es l'économiste en chef de la BCEAO, spécialisé dans l'analyse de l'inflation et la politique monétaire de l'UEMOA.
Tu dois fournir une analyse narrative, explicative et justifiée des prédictions d'inflation pour un(e) {audience_desc}.

CONTEXTE BCEAO/UEMOA :
- Objectif de stabilité des prix : maintenir l'inflation dans la fourchette 1%–3%
- Mandat principal : assurer la stabilité monétaire dans l'Union
{institutional_line_fr}

DONNÉES DU MODÈLE D'INFLATION :
- Prédictions d'inflation : {predictions}
- Inflation moyenne prédite : {avg_inflation:.2f}%
- Tendance : {trend}
- Importance globale des facteurs (SHAP) : {prediction_data.global_shap_importance}
- Intervalles de confiance : {getattr(prediction_data, 'confidence_intervals', 'Non disponibles')}

### DONNÉES DE JUSTIFICATION (SHAP INDIVIDUELS)

#### DONNÉES SHAP INDIVIDUELLES PAR DATE (feature: shap_value) :
{shap_individuals_str}

#### TOP CONTRIBUTEURS PAR DATE (top_positive/top_negative) :
{top_contrib_str}

### RÈGLES ET INSTRUCTIONS

#### PÉRIMÈTRE ET RÈGLES (STRICT) :
 - Zone couverte : UEMOA (Union entière). Ne pas citer de pays spécifiques.
 - Cible BCEAO : 1%–3%. L'analyse doit systématiquement se référer à cette cible.
 - Features AUTORISÉES à mentionner (exclusivement) : {features_present_list}
 - **NE PAS mentionner de facteurs absents de cette liste** (ex: prix du pétrole) sauf s'ils y figurent explicitement.
 - Toute affirmation doit être justifiée par une valeur SHAP.

#### NIVEAU DE DÉTAIL (selon l'audience) :
{audience_instructions}

#### CHECKLIST LLM (OBLIGATOIRE) :
- [ ] **Fidélité aux données** : Utiliser UNIQUEMENT les données fournies ci-dessus (ne rien inventer).
- [ ] **Justification systématique** : Citer explicitement les valeurs SHAP (feature et valeur) pour justifier chaque affirmation sur les moteurs de l'inflation.
- [ ] **Distinction Fait/Hypothèse** : Distinguer clairement les observations (valeurs) des hypothèses. Préfixer toute supposition par "Hypothèse:".
- [ ] **Gestion des données manquantes** : Si une information manque, l'indiquer clairement ("Donnée manquante").
- [ ] **Analyse par date** : Expliquer date par date (horizon par horizon) les contributions positives et négatives.

### STRUCTURE D'ANALYSE REQUISE (à suivre impérativement)

#### RÉSUMÉ EXÉCUTIF
- Perspectives d'inflation en 2-3 phrases clés.
- Position par rapport à la cible BCEAO.
- Message principal pour le Comité de Politique Monétaire.

#### ANALYSE DES DYNAMIQUES INFLATIONNISTES
- Décomposition narrative des facteurs, en se basant sur leur importance SHAP (globale et par date).
- Discussion sur les possibles effets (saisonnalité, inertie) si les données SHAP le suggèrent.

#### PRINCIPAUX MOTEURS DE L'INFLATION
- Pour chaque date/horizon : Identifier les Top contributeurs positifs et négatifs avec leur impact SHAP quantifié (citer les valeurs).
- Expliquer le mécanisme de transmission probable pour les 2-3 principaux facteurs.

#### **JUSTIFICATION CHIFFRÉE PAR DATE (OBLIGATOIRE)**
- Pour chaque date présente dans les prédictions :
    1.  Rappeler la valeur de la prévision.
    2.  Lister les contributions SHAP justifiant ce chiffre, sous la forme : "Le facteur `X` a contribué à hauteur de `valeur_shap` (poussant l'inflation à la hausse/baisse car...)".
    3.  Fournir une mini-synthèse narrative pour cette date.

#### ÉVALUATION DE LA STABILITÉ DES PRIX
- Écart quantifié par rapport à la cible de 1-3%.
- Analyse des risques de sortie de la fourchette-cible.

{"#### RECOMMANDATIONS DE POLITIQUE MONÉTAIRE" if include_monetary_analysis else ""}
{("- Orientation du taux directeur" if include_monetary_analysis else "")}
{("- Mesures complémentaires" if include_monetary_analysis else "")}

#### RISQUES INFLATIONNISTES
- Risques de hausse (basés sur les facteurs SHAP positifs les plus volatils).
- Risques de baisse/déflation (basés sur les facteurs SHAP négatifs).
- Facteurs d'incertitude du modèle.

#### CONFIANCE ET FIABILITÉ DU MODÈLE
- Précision historique et performance (si disponible dans les métadonnées).
- Limites méthodologiques (ex: facteurs non inclus, hypothèses linéaires).

#### ANALYSE DES FACTEURS EXTERNES
- Impact des facteurs identifiés comme "importés" dans les données SHAP (ex: `prix_mat_imp`, `taux_change_effectif`).
- Influence des prix des matières premières (uniquement si présent dans `features_present_list`).

Utilise un ton professionnel, narratif et pédagogique. L'objectif est d'expliquer le **"pourquoi"** derrière chaque chiffre.
"""
        else:
            # Placeholder for the English prompt. A real implementation would mirror the French structure.
            prompt = f"""
You are the Chief Economist of BCEAO, specialized in inflation analysis and monetary policy for WAEMU.
You must provide a narrative, explanatory, and justified analysis of inflation forecasts for a {audience_desc}.

BCEAO/WAEMU CONTEXT:
- Price stability objective: maintain inflation within the 1%–3% band.
- Primary mandate: ensure monetary stability in the Union.
{institutional_line_en}

INFLATION MODEL DATA:
- Inflation predictions: {predictions}
- Average predicted inflation: {avg_inflation:.2f}%
- Trend: {trend}
- Global SHAP factor importance: {prediction_data.global_shap_importance}
- Confidence intervals: {getattr(prediction_data, 'confidence_intervals', 'Not Available')}

### JUSTIFICATION DATA (INDIVIDUAL SHAP)

#### INDIVIDUAL SHAP DATA BY DATE (feature: shap_value):
{shap_individuals_str}

#### TOP CONTRIBUTORS BY DATE (top_positive/top_negative):
{top_contrib_str}

### RULES AND INSTRUCTIONS

#### SCOPE AND RULES (STRICT):
 - Area covered: WAEMU (entire Union). Do not mention specific countries.
 - BCEAO Target: 1%–3%. The analysis must consistently refer to this target.
 - ALLOWED features to mention (exclusively): {features_present_list}
 - **DO NOT mention factors absent from this list** (e.g., oil prices) unless they are explicitly listed.
 - Every assertion must be justified by a SHAP value.

#### LEVEL OF DETAIL (by audience):
{audience_instructions}

#### LLM CHECKLIST (MANDATORY):
- [ ] **Data Fidelity**: Use ONLY the data provided above (do not invent anything).
- [ ] **Systematic Justification**: Explicitly cite SHAP values (feature and value) to justify every claim about inflation drivers.
- [ ] **Fact vs. Hypothesis**: Clearly distinguish observations (values) from hypotheses. Prefix any assumption with "Assumption:".
- [ ] **Handling Missing Data**: If information is missing, state it clearly ("Missing data").
- [ ] **Per-Date Analysis**: Explain, date by date (horizon by horizon), the positive and negative contributions.

### REQUIRED ANALYSIS STRUCTURE (must be followed)

#### EXECUTIVE SUMMARY
- Inflation outlook in 2-3 key sentences.
- Position relative to the BCEAO target.
- Main message for the Monetary Policy Committee.

#### ANALYSIS OF INFLATION DYNAMICS
- Narrative breakdown of factors, based on their SHAP importance (global and by date).
- Discussion of possible effects (seasonality, inertia) if suggested by SHAP data.

#### KEY INFLATION DRIVERS
- For each date/horizon: Identify the Top positive and negative contributors with their quantified SHAP impact (cite the values).
- Explain the likely transmission mechanism for the top 2-3 factors.

#### **PER-DATE NUMERICAL JUSTIFICATION (MANDATORY)**
- For each date present in the predictions:
    1.  State the forecast value.
    2.  List the SHAP contributions justifying this figure, in the format: "The `X` factor contributed `shap_value` (pushing inflation up/down because...)".
    3.  Provide a brief narrative summary for that date.

#### PRICE STABILITY ASSESSMENT
- Quantified deviation from the 1-3% target.
- Analysis of risks of exiting the target band.

{"#### MONETARY POLICY RECOMMENDATIONS" if include_monetary_analysis else ""}
{("- Policy rate guidance" if include_monetary_analysis else "")}
{("- Complementary measures" if include_monetary_analysis else "")}

#### INFLATION RISKS
- Upside risks (based on the most volatile positive SHAP factors).
- Downside/deflation risks (based on negative SHAP factors).
- Model uncertainty factors.

#### MODEL CONFIDENCE AND RELIABILITY
- Historical accuracy and performance (if available in metadata).
- Methodological limitations (e.g., excluded factors, linear assumptions).

#### ANALYSIS OF EXTERNAL FACTORS
- Impact of factors identified as "imported" in the SHAP data (e.g., `imp_mat_price`, `effective_exchange_rate`).
- Influence of commodity prices (only if present in `features_present_list`).

Use a professional, narrative, and educational tone. The goal is to explain the **"why"** behind each number.
"""

        return prompt

    def _parse_inflation_interpretation(self, interpretation_text, include_policy_recs):
        """
        Parse et structure la réponse d'interprétation d'inflation générée par le LLM.
        """
        # Initialisation avec des valeurs par défaut spécifiques à l'inflation
        parsed = {
            "executive_summary": "",
            "inflation_analysis": "",
            "key_inflation_drivers": [],
            "price_stability_assessment": "",
            "monetary_policy_recommendations": None,
            "inflation_risks": [],
            "model_confidence": "",
            "target_deviation_analysis": "",
            "external_factors_impact": ""
        }
        
        try:
            # Découpage par sections
            sections = re.split(r'####\s*', interpretation_text)
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                    
                # Identification des sections spécifiques à l'inflation
                first_line = section.split('\n')[0].strip().lower()
                
                if any(keyword in first_line for keyword in ["résumé", "summary", "exécutif", "executive"]):
                    parsed["executive_summary"] = self._extract_section_content(section)
                elif any(keyword in first_line for keyword in ["dynamiques", "dynamics"]) and "inflation" in first_line:
                    parsed["inflation_analysis"] = self._extract_section_content(section)
                elif any(keyword in first_line for keyword in ["moteurs", "drivers", "principaux", "key"]):
                    parsed["key_inflation_drivers"] = self._extract_list_items(section)
                elif any(keyword in first_line for keyword in ["stabilité", "stability", "prix", "price"]):
                    parsed["price_stability_assessment"] = self._extract_section_content(section)
                elif any(keyword in first_line for keyword in ["recommandations", "recommendations", "monétaire", "monetary"]):
                    if include_policy_recs:
                        parsed["monetary_policy_recommendations"] = self._extract_section_content(section)
                elif any(keyword in first_line for keyword in ["risques", "risks"]) and "inflation" in first_line:
                    parsed["inflation_risks"] = self._extract_list_items(section)
                elif any(keyword in first_line for keyword in ["confiance", "confidence", "fiabilité", "reliability"]):
                    parsed["model_confidence"] = self._extract_section_content(section)
                elif any(keyword in first_line for keyword in ["externes", "external", "facteurs", "factors"]):
                    parsed["external_factors_impact"] = self._extract_section_content(section)
                elif any(keyword in first_line for keyword in ["écart", "deviation", "cible", "target"]):
                    parsed["target_deviation_analysis"] = self._extract_section_content(section)
            
            # Si pas de recommandations demandées, on met None
            if not include_policy_recs:
                parsed["monetary_policy_recommendations"] = None
                
        except Exception as e:
            logger.error(f"Erreur lors du parsing de l'interprétation d'inflation : {e}")
            # En cas d'erreur, on met tout le texte dans le résumé exécutif
            parsed["executive_summary"] = interpretation_text[:500] + "..." if len(interpretation_text) > 500 else interpretation_text
        
        return parsed


    def _extract_section_content(self, section_text):
        """Extrait le contenu principal d'une section en supprimant le titre."""
        lines = section_text.split('\n')
        # Supprime la première ligne qui contient généralement le titre
        content_lines = lines[1:] if len(lines) > 1 else lines
        return '\n'.join(content_lines).strip()

    def _extract_list_items(self, section_text):
        """Extrait les éléments d'une liste à partir d'une section."""
        items = []
        content = self._extract_section_content(section_text)
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            # Détecte les listes avec -, *, •, ou numérotées
            if line and (line.startswith('-') or line.startswith('*') or line.startswith('•') or 
                        (len(line) > 2 and line[0].isdigit() and line[1] in ['.', ')'])):
                # Nettoie le marqueur de liste
                clean_item = re.sub(r'^[-*•]\s*|^\d+[.)]\s*', '', line).strip()
                if clean_item:
                    items.append(clean_item)
        
        # Si aucun item de liste n'est trouvé, on retourne le contenu brut splitté par ligne
        if not items and content:
            return [l.strip() for l in content.split('\n') if l.strip()]

        return items