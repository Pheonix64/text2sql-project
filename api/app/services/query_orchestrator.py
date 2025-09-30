# text-to-sql-project/api/app/services/query_orchestrator.py

import ollama
import chromadb
import sqlglot
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text, exc
from logging import getLogger
import re
import asyncio

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
        # Sémaphores pour limiter la concurrence par ressource
        self.embed_sem = asyncio.Semaphore(2)
        self.chroma_sem = asyncio.Semaphore(4)
        self.llm_sem = asyncio.Semaphore(2)
        
        try:
            # Connexion pour le LLM (utilisateur read-only)
            self.db_engine = create_engine(
                settings.DATABASE_URL,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
            )
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

    def _needs_data_retrieval(self, text_q: str) -> bool:
        """
        Heuristique rapide pour décider si la question vise les données économiques (→ embeddings + SQL)
        ou si c'est hors-scope (→ réponse générale par LLM sans SQL).
        """
        if not text_q:
            return False
        q = text_q.lower()
        keywords = [
            # Domaines économiques / termes du schéma probable
            "uemoa", "bceao", "pib", "inflation", "taux", "dette", "recettes", "importations",
            "exportations", "balance", "solde", "date", "indicateurs", "economiques", "économiques",
            # SQL-intent
            "select", "requete", "requête", "sql", "table", "colonne", "colonnes", "where", "group by",
            # Noms de colonnes courants
            "taux_croissance_reel_pib_pct", "taux_inflation_moyen_annuel_ipc_pct",
            "encours_de_la_dette_pct_pib", "solde_budgetaire_global_hors_dons",
            "exportations_biens_fob", "importations_biens_fob",
        ]
        return any(kw in q for kw in keywords)

    async def _answer_general_question(self, user_question: str) -> str:
        """Répond aux questions générales (hors extraction SQL) via LLM, en français."""
        prompt = f"""
            Tu es un assistant utile et pédagogique qui répond en français de manière concise et exacte.
            Réponds à la question suivante de façon simple et factuelle.

            Question: {user_question}
            Réponse:
        """
        try:
            async with self.llm_sem:
                res = await asyncio.wait_for(
                    self.ollama_client.generate(model=settings.LLM_MODEL, prompt=prompt),
                    timeout=60,
                )
            return res.get('response', '').strip() or "Je n'ai pas de réponse claire à fournir pour cette question."
        except Exception as e:
            logger.error(f"Erreur LLM en mode général: {e}")
            return "Désolé, une erreur est survenue lors de la génération de la réponse."

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


        # 4) Appel LLM avec limites et timeouts
        try:
            async with self.llm_sem:
                res = await asyncio.wait_for(
                    self.ollama_client.generate(model=settings.LLM_MODEL, prompt=prompt),
                    timeout=90,
                )
            narrative = res.get("response", "").strip()
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
        # 0) Sécurité: blocage des contenus dangereux/illégaux
        if self._is_question_harmful(user_question):
            return {"answer": "Désolé, je ne peux pas traiter cette demande."}

        # 0.1) Routage d'intention: si la question n'est pas orientée données/SQL, répondre directement via LLM
        if not self._needs_data_retrieval(user_question):
            answer = await self._answer_general_question(user_question)
            return {"answer": answer}

        async with self.embed_sem:
            question_embedding_arr = await asyncio.to_thread(self.embedding_model.encode, user_question)
        question_embedding = question_embedding_arr.tolist()
        async with self.chroma_sem:
            results = await asyncio.to_thread(
                self.sql_collection.query,
                query_embeddings=[question_embedding],
                n_results=5,
            )
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
            async with self.llm_sem:
                response = await asyncio.wait_for(
                    self.ollama_client.generate(model=settings.LLM_MODEL, prompt=sql_generation_prompt),
                    timeout=90,
                )
            generated_sql = self._extract_sql_from_text(response['response'])
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
                                self.ollama_client.generate(model=settings.LLM_MODEL, prompt=sql_generation_prompt),
                                timeout=90,
                            )
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
                    self.ollama_client.generate(model=settings.LLM_MODEL, prompt=natural_language_prompt),
                    timeout=90,
                )
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
            
            # Génération de l'interprétation via LLM
            async with self.llm_sem:
                response = await asyncio.wait_for(
                    self.ollama_client.generate(
                        model=settings.LLM_MODEL,
                        prompt=interpretation_prompt
                    ),
                    timeout=120
                )
            
            interpretation_text = response.get("response", "").strip()
            
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
        
        
        # Calcul de la moyenne des prédictions pour contexte
        predictions = prediction_data.predictions
        if predictions:
            avg_inflation = sum(predictions.values()) / len(predictions)
            trend = "hausse" if list(predictions.values())[-1] > list(predictions.values())[0] else "baisse"
        else:
            avg_inflation = 0
            trend = "stable"
        
        if language == "fr":
            prompt = f"""
Tu es l'économiste en chef de la BCEAO, spécialisé dans l'analyse de l'inflation et la politique monétaire de l'UEMOA.
Tu dois fournir une analyse approfondie des prédictions d'inflation pour un(e) {audience_desc}.

CONTEXTE BCEAO/UEMOA :
- Objectif de stabilité des prix : maintenir l'inflation autour de 3% (fourchette 1-3%)
- Mandat principal : assurer la stabilité monétaire dans l'Union
- Instruments : taux directeur, réserves obligatoires, opérations d'open market
- Défis : économies dépendantes des matières premières, taux de change fixe avec l'Euro
{institutional_line_fr}

DONNÉES DU MODÈLE D'INFLATION :
- Prédictions d'inflation : {predictions}
- Inflation moyenne prédite : {avg_inflation:.2f}%
- Tendance : {trend}
- Facteurs explicatifs SHAP : {prediction_data.global_shap_importance}
- Intervalles de confiance : {getattr(prediction_data, 'confidence_intervals', 'Non disponibles')}
- Métadonnées du modèle : {prediction_data.shap_summary_details}

STRUCTURE D'ANALYSE REQUISE :
### RÉSUMÉ EXÉCUTIF
- Perspectives d'inflation en 2-3 phrases clés
- Position par rapport à la cible BCEAO
- Message principal pour le Comité de Politique Monétaire

### ANALYSE DES DYNAMIQUES INFLATIONNISTES  
- Décomposition des facteurs par ordre d'importance SHAP
- Distinction entre inflation importée et domestique
- Analyse des effets de base et saisonniers

### PRINCIPAUX MOTEURS DE L'INFLATION
- Top 5 des facteurs explicatifs avec impact quantifié
- Mécanismes de transmission identifiés
- Persistance attendue de chaque facteur

### ÉVALUATION DE LA STABILITÉ DES PRIX
- Écart par rapport à l'objectif de 3%
- Risques de dérapage inflationniste
- Probabilité d'atteinte de la cible

{"### RECOMMANDATIONS DE POLITIQUE MONÉTAIRE" if include_monetary_analysis else ""}
{"- Orientation du taux directeur" if include_monetary_analysis else ""}
{"- Mesures complémentaires (réserves obligatoires, communication)" if include_monetary_analysis else ""}
{"- Coordination avec les politiques budgétaires nationales" if include_monetary_analysis else ""}

### RISQUES INFLATIONNISTES
- Risques de hausse (chocs externes, tensions domestiques)
- Risques de déflation  
- Facteurs d'incertitude du modèle

### CONFIANCE ET FIABILITÉ DU MODÈLE
- Précision historique et performance
- Limites méthodologiques
- Scénarios alternatifs

### ANALYSE DES FACTEURS EXTERNES
- Impact du taux de change EUR/FCFA
- Influence des prix des matières premières
- Effets des politiques monétaires internationales

Utilise un ton professionnel adapté à l'audience BCEAO. Référence l'objectif de stabilité des prix et le mandat institutionnel.
"""
        else:
            prompt = f"""
You are the Chief Economist of BCEAO, specialized in inflation analysis and monetary policy for WAEMU.
You must provide in-depth inflation forecast analysis for a {audience_desc}.

BCEAO/WAEMU CONTEXT:
- Price stability objective: maintain inflation around 3% (1-3% range)
- Primary mandate: ensure monetary stability in the Union
- Instruments: policy rate, reserve requirements, open market operations
- Challenges: commodity-dependent economies, fixed exchange rate with Euro
{institutional_line_en}

INFLATION MODEL DATA:
- Inflation predictions: {predictions}
- Average predicted inflation: {avg_inflation:.2f}%
- Trend: {trend}
- SHAP explanatory factors: {prediction_data.global_shap_importance}
- Confidence intervals: {getattr(prediction_data, 'confidence_intervals', 'Not available')}
- Model metadata: {prediction_data.shap_summary_details}

REQUIRED ANALYSIS STRUCTURE:
### EXECUTIVE SUMMARY
- Inflation outlook in 2-3 key sentences
- Position relative to BCEAO target
- Main message for Monetary Policy Committee

### INFLATION DYNAMICS ANALYSIS
- Factor breakdown by SHAP importance order
- Distinction between imported and domestic inflation
- Base effects and seasonal analysis

### KEY INFLATION DRIVERS
- Top 5 explanatory factors with quantified impact
- Identified transmission mechanisms
- Expected persistence of each factor

### PRICE STABILITY ASSESSMENT
- Deviation from 3% objective
- Inflationary derailment risks
- Probability of target achievement

{"### MONETARY POLICY RECOMMENDATIONS" if include_monetary_analysis else ""}
{"- Policy rate guidance" if include_monetary_analysis else ""}
{"- Complementary measures (reserve requirements, communication)" if include_monetary_analysis else ""}
{"- Coordination with national fiscal policies" if include_monetary_analysis else ""}

### INFLATION RISKS
- Upside risks (external shocks, domestic tensions)
- Deflation risks
- Model uncertainty factors

### MODEL CONFIDENCE AND RELIABILITY
- Historical accuracy and performance
- Methodological limitations
- Alternative scenarios

### EXTERNAL FACTORS ANALYSIS
- EUR/FCFA exchange rate impact
- Commodity price influence
- International monetary policy effects

Use a professional tone adapted to BCEAO audience. Reference price stability objective and institutional mandate.
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
            sections = interpretation_text.split("###")
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                    
                # Identification des sections spécifiques à l'inflation
                section_lower = section.lower()
                
                if any(keyword in section_lower for keyword in ["résumé", "summary", "exécutif", "executive"]):
                    parsed["executive_summary"] = self._extract_section_content(section)
                elif any(keyword in section_lower for keyword in ["dynamiques", "dynamics", "analyse", "analysis"]) and "inflation" in section_lower:
                    parsed["inflation_analysis"] = self._extract_section_content(section)
                elif any(keyword in section_lower for keyword in ["moteurs", "drivers", "facteurs", "factors"]) and any(keyword in section_lower for keyword in ["inflation", "principaux", "key"]):
                    parsed["key_inflation_drivers"] = self._extract_list_items(section)
                elif any(keyword in section_lower for keyword in ["stabilité", "stability", "prix", "price"]):
                    parsed["price_stability_assessment"] = self._extract_section_content(section)
                elif any(keyword in section_lower for keyword in ["recommandations", "recommendations", "monétaire", "monetary", "politique", "policy"]):
                    if include_policy_recs:
                        parsed["monetary_policy_recommendations"] = self._extract_section_content(section)
                elif any(keyword in section_lower for keyword in ["risques", "risks"]) and "inflation" in section_lower:
                    parsed["inflation_risks"] = self._extract_list_items(section)
                elif any(keyword in section_lower for keyword in ["confiance", "confidence", "fiabilité", "reliability", "modèle", "model"]):
                    parsed["model_confidence"] = self._extract_section_content(section)
                elif any(keyword in section_lower for keyword in ["externes", "external", "facteurs", "factors"]):
                    parsed["external_factors_impact"] = self._extract_section_content(section)
                elif any(keyword in section_lower for keyword in ["écart", "deviation", "cible", "target"]):
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
        lines = section_text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Détecte les listes avec -, *, •, ou numérotées
            if line and (line.startswith('-') or line.startswith('*') or line.startswith('•') or 
                        (len(line) > 2 and line[0].isdigit() and line[1] in ['.', ')'])):
                # Nettoie le marqueur de liste
                clean_item = re.sub(r'^[-*•]\s*|^\d+[.)]\s*', '', line).strip()
                if clean_item:
                    items.append(clean_item)
        
        return items