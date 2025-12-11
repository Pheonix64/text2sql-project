# query_orchestrator.py
"""
QueryOrchestrator (LangChain-first, production-ready)
- Charge schéma enrichi depuis PostgreSQL (comments)
- Indexe des exemples question<->SQL depuis examples.json vers Chroma
- Pipeline: similarity -> SQL generation (LLM) -> validate -> execute -> NL analysis (LLM)
- Gestion stricte des sémaphores et appels async-safe
- Contient stubs/fonctions complètes pour forecasting/interpretation d'inflation
"""

from __future__ import annotations
import re
import json
import asyncio
import textwrap
import uuid
from typing import Any, Dict, List, Optional, Tuple
from logging import getLogger
import datetime
import sqlglot
from sqlalchemy import create_engine, text, exc
from datetime import datetime


# LangChain components
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.utilities import SQLDatabase

# Chroma direct client & Ollama direct client (for pull)
import chromadb
import ollama

# Replace with your app config module
from app.config import settings

logger = getLogger(__name__)


DEFAULT_EXAMPLES_PATH = "/home/appuser/docs/examples.json"


class QueryOrchestrator:
    def __init__(self, db_tables: Optional[List[str]] = None, examples_path: str = DEFAULT_EXAMPLES_PATH):
        logger.info("Initialisation de QueryOrchestrator...")
        # ---------------- resources & semaphores ----------------
        self.embed_sem = asyncio.Semaphore(2)
        self.chroma_sem = asyncio.Semaphore(4)
        self.llm_sem = asyncio.Semaphore(2)

        # ---------------- embeddings & vector store -------------
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": False},
        )
        self.chroma_client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
        self.sql_collection = self.chroma_client.get_or_create_collection(name=settings.CHROMA_COLLECTION)
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=settings.CHROMA_COLLECTION,
            embedding_function=self.embedding_model,
        )

        # ---------------- LLM & ollama client -------------------
        self.llm = ChatOllama(model=settings.LLM_MODEL, base_url=settings.OLLAMA_BASE_URL)
        self.ollama_client = ollama.AsyncClient(host=settings.OLLAMA_BASE_URL)

        # ---------------- DB engines ---------------------------
        try:
            self.db_engine = create_engine(
                settings.DATABASE_URL,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
            )
            self.admin_db_engine = create_engine(settings.ADMIN_DATABASE_URL)
        except Exception as e:
            logger.error("Erreur création engines DB: %s", e)
            raise

        # allow multiple tables
        self.db_tables = db_tables or ["indicateurs_economiques_uemoa"]

        try:
            self.langchain_db = SQLDatabase(engine=self.db_engine, include_tables=self.db_tables)
        except Exception as e:
            logger.warning("Impossible d'initialiser SQLDatabase: %s", e)
            self.langchain_db = None

        # examples file path
        self.examples_path = examples_path

        # schema will be chargé par initialize_context
        self.db_schema: str = ""

        # prompt templates centralisés
        self.sql_generation_template = PromptTemplate(
            input_variables=["db_schema", "context_queries", "user_question", "chat_history", "correction_instruction"],
            template=self._sql_generation_template_text(),
        )
        self.natural_language_template = PromptTemplate(
            input_variables=["user_question", "sql_result_str"],
            template=self._natural_language_template_text(),
        )

        # runnables built later
        self._build_runnables()

        # reference example queries (backup)
        self.reference_queries = [
            "SELECT pib_nominal_milliards_fcfa FROM indicateurs_economiques_uemoa WHERE date = '2022-01-01';",
            "SELECT date, exportations_biens_fob, importations_biens_fob, balance_des_biens FROM indicateurs_economiques_uemoa WHERE EXTRACT(YEAR FROM date) = 2021;",
            "SELECT AVG(taux_croissance_reel_pib_pct) FROM indicateurs_economiques_uemoa WHERE date >= '2015-01-01' AND date <= '2020-12-31';",
        ]

        # init keyword sets
        self._init_keyword_sets()

        # Conversational memory for Text-to-SQL
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
        
        # Conversational memory for Inflation Interpretation
        # Stores: {conversation_id: {"last_interpretation": {...}, "questions": [...]}}
        self.inflation_conversations: Dict[str, Dict[str, Any]] = {}

        logger.info("QueryOrchestrator initialisé.")

    # ------------------------ Initialization helpers -------------------------
    async def initialize_context(self, index_examples: bool = True, examples_list: Optional[List[Dict[str, str]]] = None):
        """
        Charge le schéma enrichi depuis la base et indexe les exemples (si demandé).
        Appeler au démarrage du service pour préparer les prompts/contextes.
        """
        logger.info("Chargement du schéma enrichi depuis la base...")
        self.db_schema = self._get_rich_db_schema_for_tables(self.db_tables)

        # si on souhaite indexer : charger depuis examples.json ou utiliser examples_list
        if index_examples:
            if examples_list:
                examples = examples_list
            else:
                examples = self._load_or_create_examples_file(self.examples_path)
            # indexe dans Chroma (embedding batch)
            try:
                self.index_reference_queries(examples)
            except Exception as e:
                logger.warning("Indexation des exemples échouée: %s", e)

    def _load_or_create_examples_file(self, path: str) -> List[Dict[str, str]]:
        """Charge examples.json si présent, sinon crée un fichier avec exemples par défaut et le retourne."""
        default_examples = [
            {
                "question": "Quel est le taux de croissance réel du PIB de l'UEMOA pour l'année 2022 ?",
                "sql": "SELECT date, taux_croissance_reel_pib_pct FROM indicateurs_economiques_uemoa WHERE date = '2022-01-01';"
            },
            {
                "question": "Quelle a été l'inflation moyenne annuelle entre 2015 et 2020 ?",
                "sql": "SELECT AVG(taux_inflation_moyen_annuel_ipc_pct) AS avg_inflation FROM indicateurs_economiques_uemoa WHERE date BETWEEN '2015-01-01' AND '2020-12-31';"
            },
            {
                "question": "Liste des 5 dernières années du PIB nominal (en milliards FCFA).",
                "sql": "SELECT date, pib_nominal_milliards_fcfa FROM indicateurs_economiques_uemoa ORDER BY date DESC LIMIT 5;"
            },
            {
                "question": "Encours de la dette en % du PIB pour 2020 et 2021.",
                "sql": "SELECT date, encours_de_la_dette_pct_pib FROM indicateurs_economiques_uemoa WHERE date IN ('2020-01-01', '2021-01-01') ORDER BY date;"
            },
            {
                "question": "Balance commerciale (export - import) pour 2019.",
                "sql": "SELECT date, balance_des_biens FROM indicateurs_economiques_uemoa WHERE date = '2019-01-01';"
            },
            {
                "question": "Comment l'inflation a-t-elle évolué au cours des 5 dernières années ?",
                "sql": "SELECT date, taux_inflation_moyen_annuel_ipc_pct FROM indicateurs_economiques_uemoa ORDER BY date DESC LIMIT 5;"
            },
            {
                "question": "Évolution du taux d'inflation entre 2018 et 2023.",
                "sql": "SELECT date, taux_inflation_moyen_annuel_ipc_pct FROM indicateurs_economiques_uemoa WHERE EXTRACT(YEAR FROM date) BETWEEN 2018 AND 2023 ORDER BY date;"
            },
            {
                "question": "Quel est le taux d'inflation en glissement annuel pour 2022 ?",
                "sql": "SELECT date, taux_inflation_glissement_annuel_pct FROM indicateurs_economiques_uemoa WHERE EXTRACT(YEAR FROM date) = 2022;"
            },
            {
                "question": "Évolution du PIB nominal de 2018 à 2024.",
                "sql": "SELECT date, pib_nominal_milliards_fcfa FROM indicateurs_economiques_uemoa WHERE EXTRACT(YEAR FROM date) BETWEEN 2018 AND 2024 ORDER BY date;"
            },
            {
                "question": "Quelle est la masse monétaire M2 en 2021 ?",
                "sql": "SELECT date, agregats_monnaie_masse_monetaire_m2 FROM indicateurs_economiques_uemoa WHERE EXTRACT(YEAR FROM date) = 2021;"
            },
            {
                "question": "Quelle est la progression du PIB entre 2020 et 2024 ?",
                "sql": "SELECT date, pib_nominal_milliards_fcfa FROM indicateurs_economiques_uemoa WHERE EXTRACT(YEAR FROM date) BETWEEN 2020 AND 2024 ORDER BY date;"
            },
            {
                "question": "Quel est le taux de croissance moyen du PIB sur les 5 dernières années ?",
                "sql": "SELECT AVG(taux_croissance_reel_pib_pct) AS croissance_moyenne FROM indicateurs_economiques_uemoa ORDER BY date DESC LIMIT 5;"
            },
            {
                "question": "Compare le PIB de 2020 et 2024.",
                "sql": "SELECT date, pib_nominal_milliards_fcfa FROM indicateurs_economiques_uemoa WHERE EXTRACT(YEAR FROM date) IN (2020, 2024) ORDER BY date;"
            },
        ]
        try:
            with open(path, "r", encoding="utf-8") as f:
                examples = json.load(f)
                logger.info("Chargé %d exemples depuis %s", len(examples), path)
                return examples
        except FileNotFoundError:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(default_examples, f, ensure_ascii=False, indent=2)
            logger.info("Fichier d'exemples non trouvé. Création de %s avec exemples par défaut.", path)
            return default_examples
        except Exception as e:
            logger.warning("Impossible de charger/écrire %s: %s. Retour d'exemples par défaut.", path, e)
            return default_examples

    # ------------------------ Prompt templates -------------------------------
    def _sql_generation_template_text(self) -> str:
        return textwrap.dedent(
            """\
            ### Instruction (génération SQL avec raisonnement)
            Tu es un expert SQL (PostgreSQL) et analyste économique spécialisé en politique monétaire de l'UEMOA.
            
            **ÉTAPE 1 - RAISONNEMENT (Chain of Thought)**
            Avant de générer le SQL, analyse la question étape par étape :
            1. Quelles informations sont demandées ? (indicateur, période, agrégation...)
            2. Quelles colonnes du schéma correspondent à ces informations ?
            3. Quelle logique SQL appliquer ? (filtrage, agrégation, tri, calcul...)
            
            Écris ton raisonnement entre les balises <raisonnement> et </raisonnement>.
            
            **ÉTAPE 2 - GÉNÉRATION SQL**
            Après le raisonnement, génère la requête SQL entre les balises <sql> et </sql>.

            ⚠️ CONTRAINTES CRITIQUES ⚠️
            1. N'INVENTE JAMAIS de noms de colonnes ! Utilise UNIQUEMENT les colonnes listées dans le schéma ci-dessous.
            2. Pour l'inflation, la colonne s'appelle : taux_inflation_moyen_annuel_ipc_pct (PAS "taux_inflation_pct" ni "inflation")
            3. Pour le PIB, la colonne s'appelle : pib_nominal_milliards_fcfa
            4. Pour la croissance, la colonne s'appelle : taux_croissance_reel_pib_pct
            5. Les données sont annuelles avec une date au format 'YYYY-01-01' (ex: '2022-01-01' pour l'année 2022)
            6. Pour filtrer par année, utilise : EXTRACT(YEAR FROM date) = 2022 ou date = '2022-01-01'
            7. Ne génère JAMAIS d'instruction de modification (INSERT/UPDATE/DELETE/DROP...)
            
            📊 RÈGLE D'OR : VALEURS EXACTES 📊
            - Retourne TOUJOURS les valeurs exactes de la base de données (date + valeur).
            - PRIVILÉGIE les requêtes simples qui retournent les données brutes plutôt que des calculs complexes.
            - Pour une évolution/progression, retourne simplement les valeurs année par année avec ORDER BY date.
            - L'analyse et les calculs seront faits par l'assistant, pas par SQL.
            
            ⚠️ SYNTAXE SQL INTERDITE ⚠️
            - JAMAIS de syntaxe comme colonne[condition] (ex: pib[annee=2024] est INVALIDE)
            - JAMAIS de références de tableau avec crochets []
            - Pour comparer des valeurs entre années, utilise :
              * Soit une simple requête avec WHERE et ORDER BY
              * Soit LAG/LEAD pour les variations
              * Soit des sous-requêtes séparées
            
            Exemple CORRECT pour progression entre 2020 et 2024 :
            SELECT date, pib_nominal_milliards_fcfa FROM indicateurs_economiques_uemoa WHERE EXTRACT(YEAR FROM date) IN (2020, 2024) ORDER BY date;

            ### Schéma de la base de données (COLONNES DISPONIBLES)
            {db_schema}

            ### Exemples de requêtes similaires (COPIE les noms de colonnes de ces exemples)
            {context_queries}

            ### Historique de la conversation
            {chat_history}

            ### Question de l'utilisateur
            "{user_question}"

            ### Instruction de correction (si applicable)
            {correction_instruction}

            ### Réponse (raisonnement puis SQL entre balises)
            """
        )

    def _natural_language_template_text(self) -> str:
        return textwrap.dedent(
            """
                Tu es un analyste économique expert à la BCEAO. En te basant SEULEMENT sur la QUESTION de l'utilisateur et EXCLUSIVEMENT sur le RÉSULTAT SQL fourni, rédige une analyse synthétique, claire et rationnelle en français.

                Toutes les données concernent l'ensemble de l'Union (BCEAO) sauf mention explicite de 'country'. Ne jamais inventer de chiffres hors du résultat SQL.

                ⚠️ CAS PARTICULIERS À GÉRER ⚠️
                
                1. Si le RÉSULTAT SQL est vide, ne contient que "[]" ou ne retourne aucune ligne :
                   → Réponds EXACTEMENT : "Les données demandées ne sont pas disponibles dans notre base pour la période ou l'indicateur spécifié. Notre base couvre les indicateurs macroéconomiques de l'UEMOA de 2005 à 2024. Pourriez-vous vérifier la période ou l'indicateur demandé ?"
                   → Ne jamais inventer de chiffres dans ce cas.
                
                2. Si la question de l'utilisateur est ambiguë, trop vague, hors sujet économique, ou incompréhensible :
                   → Réponds EXACTEMENT : "Votre question nécessite des précisions pour que je puisse vous fournir une analyse pertinente. Pourriez-vous reformuler en précisant :
                   - L'indicateur économique souhaité (PIB, inflation, dette, balance commerciale, etc.)
                   - La période concernée (année ou plage d'années entre 2005 et 2024)
                   - Le pays ou si c'est pour l'ensemble de l'UEMOA
                   - Le type d'analyse attendu (valeur, évolution, comparaison, moyenne, etc.)"
                
                3. Si les données sont partielles (certaines années demandées absentes du résultat) :
                   → Mentionne explicitement les années pour lesquelles les données sont disponibles.
                   → Indique les années manquantes si pertinent.

                RÈGLES POUR LA RÉPONSE (si des données sont disponibles) :
                - Commencer par 3 à 4 phrases résumant l'information principale.
                - Donner le contexte et la portée.
                - Présenter les chiffres clés issus du résultat SQL (langage pour communiquer avec une base de données).
                - Proposer une interprétation raisonnée.
                - Expliquer brièvement la méthodologie et/ou les colonnes utilisées.
                - Mentionner les limites et/ou hypothèses éventuelles.
                - Se terminer par 2 à 4 recommandations pratiques.
                - Ne jamais divulguer la requête SQL ni le résultat brut.
                - Ne jamais inventer de données ni extrapoler au-delà du résultat SQL.
                - Ne jamais faire des répétitions inutiles.
                - Tout ce qui est un montant doit être en chiffres exacts avec unités (FCFA) (ex: 1234.56 milliards FCFA).
                - Le PIB est toujours en milliards FCFA.

                La réponse doit être rédigée comme un rapport synthétique fluide, destiné à un décideur, et ne jamais contenir de titres ou sous-titres visibles.

                ### Question
                {user_question}

                ### Résultat SQL
                {sql_result_str}

                ### Réponse
                """
                        )

    # ------------------------ Runnables builders --------------------------------
    def _build_runnables(self):
        # Runnable to generate SQL (async)
        async def _run_sql_generation(inputs: Dict[str, Any]) -> Dict[str, Any]:
            prompt = self.sql_generation_template.format(
                db_schema=inputs.get("db_schema", self.db_schema),
                context_queries=inputs.get("context_queries", ""),
                user_question=inputs.get("user_question", ""),
                chat_history=inputs.get("chat_history", ""),
                correction_instruction=inputs.get("correction_instruction", ""),
            )
            llm_text = await self._call_llm(prompt)
            sql = self._extract_sql_from_text(llm_text)
            reasoning = self._extract_reasoning_from_text(llm_text)
            if reasoning:
                logger.info(f"Raisonnement LLM: {reasoning[:200]}...")  # Log les 200 premiers caractères
            return {"generated_sql": sql, "llm_text": llm_text, "reasoning": reasoning}

        # Runnable to validate/execute and produce final answer
        async def _run_response_generation(inputs: Dict[str, Any]) -> Dict[str, Any]:
            generated_sql = inputs.get("generated_sql", "")
            user_question = inputs.get("user_question", "")
            # Si le résultat est déjà fourni (par la boucle de retry), on l'utilise
            sql_result = inputs.get("sql_result")

            if sql_result is None:
                # validate
                if not generated_sql or not re.search(r"^\s*(SELECT|WITH)\b", generated_sql, flags=re.IGNORECASE):
                    raise ValueError("Aucune requête SELECT/WITH générée.")
                if not self._validate_sql(generated_sql):
                    raise ValueError("La requête SQL générée a été jugée non sécurisée.")

                # execute
                sql_result = await self._validate_and_execute_sql(generated_sql)
            
            sql_result_str = str(sql_result)

            # NL prompt
            nl_prompt = self.natural_language_template.format(
                user_question=user_question,
                sql_result_str=sql_result_str
            )
            final_text = await self._call_llm(nl_prompt)
            return {"final_answer": final_text, "sql_result": sql_result}

        self.sql_generation_runnable = RunnableLambda(_run_sql_generation)
        self.response_generation_runnable = RunnableLambda(_run_response_generation)

    # ------------------------ Core LLM / similarity / SQL helpers ----------------
    async def _call_llm(self, prompt: str, timeout: int = 90) -> str:
        """Appel LLM avec sémaphore, fallback pull_model si modèle absent."""
        try:
            async with self.llm_sem:
                res = await asyncio.wait_for(self.llm.ainvoke(prompt), timeout=timeout)
            return res.content if hasattr(res, "content") else str(res)
        except Exception as e:
            msg = str(e).lower()
            logger.warning("LLM call error: %s", e)
            if "not found" in msg or "404" in msg:
                logger.info("LLM model not found, attempting to pull...")
                pull_res = await self.pull_model()
                if pull_res.get("status") == "success":
                    async with self.llm_sem:
                        res = await asyncio.wait_for(self.llm.ainvoke(prompt), timeout=timeout)
                    return res.content if hasattr(res, "content") else str(res)
            raise

    async def _similarity_search(self, question: str, k: int = 5) -> List[str]:
        async with self.chroma_sem:
            docs = await asyncio.to_thread(self.vector_store.similarity_search, question, k)
        return [getattr(d, "page_content", str(d)) for d in docs]

    async def _validate_and_execute_sql(self, sql: str) -> List[Dict[str, Any]]:
        if not sql or not self._validate_sql(sql):
            raise ValueError("La requête SQL générée est invalide ou non sécurisée.")
        return await self._execute_sql_readonly(sql)

    # ------------------------ Public pipeline -----------------------------------
    async def process_user_question(self, user_question: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        # 0) Ensure conversation_id exists and check history
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            logger.info(f"Génération d'un nouveau conversation_id: {conversation_id}")
        
        has_history = False
        if conversation_id in self.conversations and self.conversations[conversation_id]:
            has_history = True

        # 1) Security and routing
        if self._is_question_harmful(user_question):
            return {"answer": "Désolé, je ne peux pas traiter cette demande.", "conversation_id": conversation_id}
        
        # On passe has_history pour assouplir le filtrage si on est dans une conversation
        if not self._needs_data_retrieval(user_question, has_history=has_history):
            return {"answer": "Désolé, cette question ne concerne pas les données économiques de l'UEMOA/BCEAO.", "conversation_id": conversation_id}

        # 2) Similarity context
        try:
            similar_docs = await self._similarity_search(user_question, k=5)
            context_queries = "\n".join(similar_docs)
        except Exception as e:
            logger.warning("Similarity search failed: %s", e)
            context_queries = ""

        # 3) Chat History Management
        chat_history_str = ""
        if has_history:
            history = self.conversations.get(conversation_id, [])
            # On garde les 5 derniers échanges pour le contexte
            recent_history = history[-5:]
            for turn in recent_history:
                chat_history_str += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

        # 4) Generate SQL with Retry Loop (Auto-Correction)
        MAX_RETRIES = 3
        correction_instruction = ""
        generated_sql = ""
        sql_result = []
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                if attempt > 0:
                    logger.info(f"Tentative de correction SQL {attempt + 1}/{MAX_RETRIES}")
                
                sql_res = await self.sql_generation_runnable.ainvoke({
                    "user_question": user_question,
                    "db_schema": self.db_schema,
                    "context_queries": context_queries,
                    "chat_history": chat_history_str,
                    "correction_instruction": correction_instruction
                })
                generated_sql = sql_res.get("generated_sql", "")

                # 3) Sanity & validate
                if not generated_sql or not re.search(r"^\s*(SELECT|WITH)\b", generated_sql, flags=re.IGNORECASE):
                    raise ValueError("Pas de requête SQL valide générée (SELECT/WITH manquant).")

                if not self._validate_sql(generated_sql):
                    raise ValueError("Requête SQL jugée non sécurisée par le validateur.")

                # 4) Execute (Try to run the query)
                sql_result = await self._validate_and_execute_sql(generated_sql)
                
                # Si on arrive ici, c'est un succès
                last_error = None
                break

            except Exception as e:
                logger.warning(f"Échec tentative {attempt + 1}: {e}")
                last_error = e
                # On prépare l'instruction de correction pour la prochaine itération
                # On inclut le schéma pour rappeler les colonnes disponibles
                correction_instruction = (
                    f"⚠️ ERREUR À CORRIGER ⚠️\n"
                    f"La requête précédente a échoué avec l'erreur suivante :\n"
                    f"```\n{e}\n```\n"
                    f"Requête générée (incorrecte) : {generated_sql}\n\n"
                    f"RAPPEL DU SCHÉMA DISPONIBLE:\n{self.db_schema}\n\n"
                    f"Analyse l'erreur et utilise UNIQUEMENT les colonnes listées ci-dessus pour corriger la requête."
                )
        
        if last_error:
            return {
                "answer": f"Désolé, je n'ai pas réussi à générer une requête valide après {MAX_RETRIES} tentatives. Erreur technique : {last_error}",
                "generated_sql": generated_sql,
                "conversation_id": conversation_id
            }

        # 5) NL generation (using the successful sql_result)
        try:
            response_res = await self.response_generation_runnable.ainvoke({
                "generated_sql": generated_sql,
                "user_question": user_question,
                "sql_result": sql_result  # Pass the result we already got
            })
            final_answer = response_res.get("final_answer", "")
            
            # 6) Update History
            if conversation_id:
                if conversation_id not in self.conversations:
                    self.conversations[conversation_id] = []
                self.conversations[conversation_id].append({
                    "user": user_question,
                    "assistant": final_answer
                })

        except Exception as e:
            logger.error("Erreur pendant génération réponse finale: %s", e)
            return {"answer": "Une erreur est survenue lors de la formulation de la réponse.", "generated_sql": generated_sql, "conversation_id": conversation_id}

        return {"answer": final_answer, "generated_sql": generated_sql, "sql_result": str(sql_result), "conversation_id": conversation_id}

    # ------------------------ Indexing / examples --------------------------------
    def index_reference_queries(self, examples: Optional[List[Dict[str, str]]] = None) -> int:
        """
        Indexe exemples (list[{"question":..,"sql":..}]) dans Chroma.
        Si None, indexe self.reference_queries (simple list of SQL).
        """
        if examples is None:
            examples = [{"question": None, "sql": q} for q in self.reference_queries]

        # prepare documents: combine question + sql to help retrieval
        docs = []
        for i, ex in enumerate(examples):
            content = f"Question: {ex.get('question') or ''}\nRequête SQL: {ex.get('sql')}"
            docs.append({"id": f"ex_{i}", "content": content})

        # embed & add via chroma client for determinism
        embeddings = self.embedding_model.embed_documents([d["content"] for d in docs])
        # ensure we clear previous
        if self.sql_collection.count() > 0:
            ids_to_delete = self.sql_collection.get().get("ids") or []
            if ids_to_delete and isinstance(ids_to_delete[0], list):
                ids_to_delete = [i for sub in ids_to_delete for i in sub]
            if ids_to_delete:
                self.sql_collection.delete(ids=ids_to_delete)
        self.sql_collection.add(
            embeddings=embeddings,
            documents=[d["content"] for d in docs],
            ids=[d["id"] for d in docs],
        )
        logger.info("Indexation terminée (%d exemples).", len(docs))
        return len(docs)

    # ------------------------ DB schema extraction -------------------------------
    def _get_rich_db_schema_for_tables(self, table_names: List[str]) -> str:
        pieces = []
        for table in table_names:
            pieces.append(self._get_rich_db_schema(table))
        return "\n\n".join(pieces)

    def _get_rich_db_schema(self, table_name: str) -> str:
        """Lit information_schema + pg_description pour retourner un CREATE TABLE commenté."""
        query = text("""
            SELECT
                c.column_name,
                c.data_type,
                pgd.description
            FROM information_schema.columns AS c
            LEFT JOIN
                pg_catalog.pg_statio_all_tables AS st ON c.table_schema = st.schemaname AND c.table_name = st.relname
            LEFT JOIN
                pg_catalog.pg_description AS pgd ON pgd.objoid = st.relid AND pgd.objsubid = c.ordinal_position
            WHERE c.table_name = :table_name
            ORDER BY c.ordinal_position;
        """)
        table_comment_query = text(f"SELECT obj_description('public.{table_name}'::regclass);")
        try:
            with self.admin_db_engine.connect() as conn:
                table_comment = conn.execute(table_comment_query).scalar_one_or_none()
                cols = conn.execute(query, {"table_name": table_name}).fetchall()
            schema_str = f"-- Description de la table '{table_name}': {table_comment}\nCREATE TABLE {table_name} (\n"
            for col in cols:
                col_name, data_type, description = col
                line = f"    {col_name} {data_type}"
                if description:
                    line += f" -- {description}"
                schema_str += line + ",\n"
            schema_str = schema_str.rstrip(",\n") + "\n);"
            return schema_str
        except Exception as e:
            logger.error("Impossible de récupérer le schéma pour %s: %s", table_name, e)
            return f"CREATE TABLE {table_name} (...); -- Erreur récupération schéma: {e}"

    # ------------------------ SQL validation & execution ------------------------
    def _extract_sql_from_text(self, text: str) -> str:
        """Extrait le SQL depuis la réponse du LLM, en priorité depuis les balises <sql>."""
        if not text:
            return ""
        
        # 1. Priorité aux balises <sql>...</sql> (Chain of Thought format)
        sql_tag = re.search(r"<sql>([\s\S]*?)</sql>", text, flags=re.IGNORECASE)
        if sql_tag:
            candidate = sql_tag.group(1).strip()
            # Nettoyer les éventuels blocs markdown à l'intérieur
            candidate = re.sub(r"```(?:sql)?\s*", "", candidate)
            candidate = re.sub(r"```", "", candidate)
            return candidate.strip()
        
        # 2. Blocs de code markdown ```sql ... ```
        code_block = re.search(r"```(?:sql)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        if code_block:
            candidate = code_block.group(1).strip()
            return candidate
        
        # 3. Fallback : chercher SELECT ou WITH directement
        m = re.search(r"\b(SELECT|WITH)\b[\s\S]*", text, flags=re.IGNORECASE)
        if m:
            candidate = text[m.start():]
            semi = candidate.find(';')
            if semi != -1:
                candidate = candidate[:semi+1]
            return candidate.strip()
        
        return text.strip()

    def _extract_reasoning_from_text(self, text: str) -> str:
        """Extrait le raisonnement depuis les balises <raisonnement>."""
        if not text:
            return ""
        reasoning_tag = re.search(r"<raisonnement>([\s\S]*?)</raisonnement>", text, flags=re.IGNORECASE)
        if reasoning_tag:
            return reasoning_tag.group(1).strip()
        return ""

    def _validate_sql(self, sql_query: str) -> bool:
        try:
            banned = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|MERGE|CALL|COPY|VACUUM|ANALYZE|EXPLAIN)\b", re.IGNORECASE)
            if banned.search(sql_query):
                logger.warning("Validation échouée : mot-clé non autorisé détecté.")
                return False
            exprs = sqlglot.parse(sql_query, read="postgres")
            if not exprs or len(exprs) != 1:
                logger.warning("Validation échouée : aucune ou plusieurs instructions détectées.")
                return False
            expr = exprs[0]
            if isinstance(expr, sqlglot.exp.With):
                base_expr = expr.this
            else:
                base_expr = expr
            allowed = (sqlglot.exp.Select, sqlglot.exp.Union, sqlglot.exp.Except, sqlglot.exp.Intersect)
            if not isinstance(base_expr, allowed):
                logger.warning("Validation échouée : expression non autorisée.")
                return False
            return True
        except Exception as e:
            logger.error("Erreur validation SQL: %s", e)
            return False

    async def _execute_sql_readonly(self, sql: str) -> List[Dict[str, Any]]:
        def run():
            with self.db_engine.connect() as connection:
                result_proxy = connection.execute(text(sql))
                return [dict(row._mapping) for row in result_proxy]
        try:
            return await asyncio.to_thread(run)
        except exc.SQLAlchemyError as e:
            logger.error("Erreur exécution SQL: %s", e)
            raise

    # ------------------------ Forecasting & inflation interpretation -----------
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

    async def generate_inflation_interpretation(self, body, timeout: int = 120) -> dict:
        """
        Génère une interprétation économique spécialisée des prédictions d'inflation SHAP 
        pour les économistes et analystes de la BCEAO.
        
        Args:
            body: InflationInterpretationRequest contenant les données de prédiction et paramètres
            timeout: Timeout en secondes pour l'appel LLM (par défaut 120)
            
        Returns:
            Dictionnaire contenant l'interprétation économique formatée spécifique à l'inflation
        """
        try:
            # Extraction des données de prédiction d'inflation
            prediction_data = body.prediction_data
            audience = body.target_audience
            include_policy_recs = body.include_policy_recommendations
            include_monetary_analysis = body.include_monetary_policy_analysis
            focus_bceao = body.focus_on_bceao_mandate
            print(prediction_data)
            # Construction du prompt d'interprétation spécifique à l'inflation
            interpretation_prompt = self._build_inflation_interpretation_prompt(
                prediction_data, audience, include_monetary_analysis, focus_bceao
            )
            
            # =======================================================================================
            # LANGCHAIN CHATOLLAMA - Inflation interpretation generation
            # =======================================================================================
            # Génération de l'interprétation via LLM
            async with self.llm_sem:
                response = await asyncio.wait_for(
                    self.llm.ainvoke(interpretation_prompt),
                    timeout=timeout
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
    
    def _validate_inflation_data(self, prediction_data) -> Dict[str, Any]:
        """
        Valide que les données de prédiction d'inflation sont cohérentes.
        Inclut des validations de série temporelle.
        
        Returns:
            Dict contenant les résultats de validation et les warnings
        """
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": []
        }
        
        predictions = prediction_data.get("predictions", {})
        
        if not predictions:
            validation_result["errors"].append("Aucune prédiction fournie")
            validation_result["is_valid"] = False
            return validation_result
        
        # 1. Vérifier que les valeurs d'inflation sont dans une plage raisonnable
        for period, value in predictions.items():
            if not isinstance(value, (int, float)):
                validation_result["errors"].append(f"Valeur d'inflation invalide pour {period}: {value}")
                validation_result["is_valid"] = False
            elif value < -10 or value > 50:  # Plage raisonnable pour l'inflation (%)
                validation_result["warnings"].append(f"Valeur d'inflation inhabituelle pour {period}: {value}%")
        
        # 2. Validation de série temporelle - Dates ordonnées et sans doublons
        dates = list(predictions.keys())
        if len(dates) != len(set(dates)):
            validation_result["warnings"].append("Dates en double détectées dans les prédictions")
        
        # Essayer de parser et trier les dates
        try:
            parsed_dates = []
            for date_str in dates:
                # Supporter plusieurs formats: YYYY-MM, YYYY-MM-DD, YYYY-Q1
                if "-Q" in date_str:
                    # Format trimestre: 2024-Q1 -> 2024-01
                    year, quarter = date_str.split("-Q")
                    month = (int(quarter) - 1) * 3 + 1
                    parsed_dates.append((date_str, datetime(int(year), month, 1)))
                elif len(date_str) == 7:  # YYYY-MM
                    parsed_dates.append((date_str, datetime.strptime(date_str, "%Y-%m")))
                elif len(date_str) == 10:  # YYYY-MM-DD
                    parsed_dates.append((date_str, datetime.strptime(date_str, "%Y-%m-%d")))
                else:
                    validation_result["warnings"].append(f"Format de date non reconnu: {date_str}")
            
            # Vérifier l'ordre chronologique
            if parsed_dates:
                sorted_dates = sorted(parsed_dates, key=lambda x: x[1])
                original_order = [d[0] for d in parsed_dates]
                sorted_order = [d[0] for d in sorted_dates]
                if original_order != sorted_order:
                    validation_result["warnings"].append("Les dates ne sont pas en ordre chronologique")
                
                # Vérifier les gaps (plus de 3 mois entre deux prédictions)
                for i in range(1, len(sorted_dates)):
                    diff_days = (sorted_dates[i][1] - sorted_dates[i-1][1]).days
                    if diff_days > 95:  # ~3 mois
                        validation_result["warnings"].append(
                            f"Gap temporel important détecté entre {sorted_dates[i-1][0]} et {sorted_dates[i][0]}"
                        )
        except Exception as e:
            validation_result["warnings"].append(f"Impossible de valider l'ordre des dates: {str(e)}")
        
        # 3. Vérifier la cohérence des variations (pas de sauts > 10 points)
        values = list(predictions.values())
        for i in range(1, len(values)):
            if abs(values[i] - values[i-1]) > 10:
                validation_result["warnings"].append(
                    f"Variation abrupte d'inflation détectée: {values[i-1]:.2f}% → {values[i]:.2f}%"
                )
        
        # 4. Vérifier la présence des facteurs d'inflation typiques
        shap_importance = prediction_data.get("global_shap_importance", {})
        expected_factors = ["taux_change", "prix_petrole", "masse_monetaire", "alimentation"]
        
        missing_factors = []
        for factor in expected_factors:
            found = any(factor in key.lower() for key in shap_importance.keys())
            if not found:
                missing_factors.append(factor)
        
        if missing_factors:
            validation_result["warnings"].append(f"Facteurs d'inflation typiques non trouvés: {', '.join(missing_factors)}")
        
        # 5. Vérifier que les SHAP individuels correspondent aux dates de prédiction
        individual_shap = prediction_data.get("individual_shap_explanations", {})
        if individual_shap:
            shap_dates = set(individual_shap.keys())
            pred_dates = set(predictions.keys())
            if shap_dates != pred_dates:
                missing_in_shap = pred_dates - shap_dates
                extra_in_shap = shap_dates - pred_dates
                if missing_in_shap:
                    validation_result["warnings"].append(f"SHAP manquants pour les dates: {missing_in_shap}")
                if extra_in_shap:
                    validation_result["warnings"].append(f"SHAP supplémentaires non associés: {extra_in_shap}")
        
        # Log des résultats
        if validation_result["errors"]:
            for error in validation_result["errors"]:
                logger.error(f"Validation inflation: {error}")
        if validation_result["warnings"]:
            for warning in validation_result["warnings"]:
                logger.warning(f"Validation inflation: {warning}")
        
        return validation_result

    def _build_inflation_interpretation_prompt(self, prediction_data, audience, include_monetary_analysis, focus_bceao):
        """
        Construit le prompt spécialisé pour l'interprétation des prédictions d'inflation.
        """
        audience_descriptions = {
            "economist": {"fr": "économiste spécialisé en politique monétaire", "en": "monetary policy economist"},
            "analyst": {"fr": "analyste inflation", "en": "inflation analyst"},
            "policymaker": {"fr": "décideur de politique monétaire", "en": "monetary policymaker"},
            "general": {"fr": "public général", "en": "general public"}
        }
        
        audience_desc = audience_descriptions[audience]
        institutional_line_fr = ""
        if focus_bceao:
            institutional_line_fr = "- Met en avant le mandat de stabilité des prix de la BCEAO et les obligations statutaires vis-à-vis des États membres."
            
        
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

        audience_instructions = {
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
        
        prompt = f"""
                    Rôle et Mission :
                    Tu es l'économiste en chef de la BCEAO. Ta mission est d’analyser les prévisions mensuelles d’inflation pour l’UEMOA.

                    Objectif :
                    Fournir une analyse narrative claire, détaillée et rigoureusement justifiée des prévisions d’inflation pour {audience_desc}, **en utilisant uniquement les données fournies**.

                    Contexte :
                    - Mandat BCEAO : stabilité des prix, croissance économique, solidité du système financier.
                    - Objectif d’inflation annuel : 1-3 %.

                    Données disponibles :
                    - Prédictions mensuelles : {predictions}
                    - Contributions SHAP par mois : {shap_individuals_str}  
                    - Inflation moyenne : {avg_inflation:.2f}%
                    - Tendance générale : {trend}  
                    - Variables disponibles : {features_present_list}  
                    - Principaux facteurs : {top_contrib_str}

                    Instructions importantes :
                    1. **Toujours utiliser les valeurs fournies** sans les modifier et sans changer leur signe.
                    2. Remplacer systématiquement les placeholders AAAA-MM par les dates exactes.
                    3. Explications mois par mois : indiquer date réelle, inflation prévue, contributions SHAP et interprétation (SHAP positif = inflationniste, SHAP négatif = désinflationniste).
                    4. Ne jamais utiliser de données externes ou inventer des chiffres.
                    5. Distinguer clairement l’inflation mensuelle prévue et l’inflation annuelle cible BCEAO.
                    6. Signaler toute donnée manquante nécessaire à une analyse complète.

                    Structure recommandée de l’analyse :
                    1. **Résumé exécutif** : message clé, tendances générales.
                    2. **Évolution mensuelle** : analyse mois par mois avec valeurs exactes et contributions SHAP.
                    3. **Facteurs de l’inflation** : moteurs inflationnistes et désinflationnistes, avec explications simples basées sur les SHAP.
                    4. **Justification chiffrée** :
                    - Date réelle
                    - Inflation prévue
                    - Liste des facteurs SHAP et impact
                    - Effet potentiel sur la trajectoire annuelle
                    5. **Évaluation de la stabilité des prix** : comparaison de l’inflation moyenne avec l’objectif BCEAO.
                    6. **Risques inflationnistes** : facteurs positifs et négatifs, valeurs exactes.
                    7. **Limites et incertitudes** : basées uniquement sur les variables fournies.
                    8. **Recommandations de politique monétaire** (optionnel) : justifiées par l’analyse.

                    Rappel final :
                    - Utiliser uniquement les données fournies.
                    - Ne jamais changer le signe des valeurs.
                    - Expliquer clairement mois par mois, avec SHAP et inflation exacte.
                    - Suivre scrupuleusement cette structure.
                    - Rédiger en français, sous forme de texte fluide, sans titres visibles et sans répétitions et tu dois utiliser un français plus humain.
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
                elif any(keyword in first_line for keyword in ["recommandations", "monétaire"]):
                    if include_policy_recs:
                        parsed["monetary_policy_recommendations"] = self._extract_section_content(section)
                elif any(keyword in first_line for keyword in ["risques", "risks"]) and "inflation" in first_line:
                    parsed["inflation_risks"] = self._extract_list_items(section)
                elif any(keyword in first_line for keyword in ["confiance", "fiabilité"]):
                    parsed["model_confidence"] = self._extract_section_content(section)
                elif any(keyword in first_line for keyword in ["externes", "facteurs"]):
                    parsed["external_factors_impact"] = self._extract_section_content(section)
                elif any(keyword in first_line for keyword in ["écart", "cible"]):
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

        print("Extracted list items:", items)
        return items
        
    # ------------------------ pull model ---------------------------------------
    async def pull_model(self, model: Optional[str] = None) -> Dict[str, Any]:
        target_model = model or settings.LLM_MODEL
        try:
            async with self.llm_sem:
                await asyncio.wait_for(self.ollama_client.pull(model=target_model), timeout=600)
            return {"status": "success", "model": target_model}
        except Exception as e:
            logger.error("Erreur pull model: %s", e)
            return {"status": "error", "message": str(e), "model": target_model}

    # ------------------------ Heuristics / routing --------------------------------
    def _init_keyword_sets(self) -> None:
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
        
        sql_keywords = {"select", "where", "group by", "order by", "sql"}
        date_keywords = {"date", "année", "mois", "trimestre", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"}
        dynamic_keywords = set()
        try:
            with self.admin_db_engine.connect() as conn:
                rows = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = :table"), {"table": self.db_tables[0]}).fetchall()
            for (col,) in rows:
                dynamic_keywords.add(col.lower())
        except Exception:
            pass
        self.economic_keywords = base_economic_keywords | dynamic_keywords
        self.sql_keywords = sql_keywords
        self.date_keywords = date_keywords

    def _is_question_harmful(self, text_q: str) -> bool:
        if not text_q:
            return False
        q = text_q.lower()
        banned_terms = [
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
            "sexe", "pornographie", "pédophilie", "inceste", "viol", "prostitution"
        ]
        return any(term in q for term in banned_terms)

    def _needs_data_retrieval(self, text_q: str, has_history: bool = False) -> bool:
        """
        Heuristique stricte pour décider si la question concerne UNIQUEMENT les données économiques UEMOA/BCEAO.
        Refuse les questions générales, hors-sujet, ou trop vagues.
        Si has_history est True, on est plus tolérant (questions de suivi).
        """
        if not text_q or len(text_q.strip()) < 2:  # Questions vides ou trop courtes
            return False

        q = text_q.lower().strip()

        # Si on a un historique, on accepte les questions courtes de suivi (ex: "Et en 2024 ?")
        if has_history:
            # On vérifie juste qu'il y a un minimum de contenu pertinent (date ou mot clé ou juste une phrase)
            # Pour l'instant, on accepte presque tout si ce n'est pas vide, car le contexte peut donner du sens à tout.
            return True

        if len(text_q.strip()) < 5:
            return False

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
