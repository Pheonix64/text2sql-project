# text-to-sql-project/api/app/services/query_orchestrator.py

import ollama
import chromadb
import sqlglot
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text, exc
from logging import getLogger
import re

from app.config import settings

logger = getLogger(__name__)

class QueryOrchestrator:
    def __init__(self):
        logger.info("Initialisation de QueryOrchestrator...")
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        self.chroma_client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
        self.sql_collection = self.chroma_client.get_or_create_collection(name=settings.CHROMA_COLLECTION)
        # Client asynchrone pour éviter de bloquer l'event loop
        self.ollama_client = ollama.AsyncClient(host=settings.OLLAMA_BASE_URL)
        
        try:
            # Connexion pour le LLM (utilisateur read-only)
            self.db_engine = create_engine(settings.DATABASE_URL)
            # Connexion pour l'admin (pour lire le schéma) via URL encodée
            self.admin_db_engine = create_engine(settings.ADMIN_DATABASE_URL)
            logger.info("Connexions à la base de données PostgreSQL réussies.")
        except Exception as e:
            logger.error(f"Erreur de connexion à la base de données : {e}")
            raise

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

        embeddings = self.embedding_model.encode(queries_to_index).tolist()
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
        question_embedding = self.embedding_model.encode(user_question).tolist()
        results = self.sql_collection.query(query_embeddings=[question_embedding], n_results=5)
        # ChromaDB renvoie une liste de listes de documents (par requête)
        documents = results.get('documents') or []
        if documents and isinstance(documents[0], list):
            flat_docs = [doc for sub in documents for doc in sub]
        else:
            flat_docs = documents
        context_queries = "\n".join(flat_docs) if flat_docs else ""

        sql_generation_prompt = f"""
            ### Instruction (génération SQL)
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
            {self.db_schema}

            ### Exemples de requêtes similaires
            {context_queries}

            ### Question de l'utilisateur
            "{user_question}"

            ### Requête SQL
            """



        try:
            response = await self.ollama_client.generate(model=settings.LLM_MODEL, prompt=sql_generation_prompt)
            generated_sql = self._extract_sql_from_text(response['response'])
        except Exception as e:
            msg = str(e).lower()
            # Si le modèle n'est pas trouvé (404), tenter un pull puis retenter une fois
            if "not found" in msg or "404" in msg:
                logger.warning(f"Modèle '{settings.LLM_MODEL}' introuvable. Tentative de téléchargement puis nouvel essai...")
                pull_res = await self.pull_model(settings.LLM_MODEL)
                if pull_res.get("status") == "success":
                    try:
                        response = await self.ollama_client.generate(model=settings.LLM_MODEL, prompt=sql_generation_prompt)
                        generated_sql = self._extract_sql_from_text(response['response'])
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
            with self.db_engine.connect() as connection:
                result_proxy = connection.execute(text(generated_sql))
                sql_result = [dict(row._mapping) for row in result_proxy]
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
            final_response = await self.ollama_client.generate(model=settings.LLM_MODEL, prompt=natural_language_prompt)
            final_answer = final_response['response']
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
            await self.ollama_client.pull(model=target_model)
            return {"status": "success", "model": target_model}
        except Exception as e:
            logger.error(f"Erreur lors du pull du modèle '{target_model}' : {e}")
            return {"status": "error", "message": str(e), "model": target_model}