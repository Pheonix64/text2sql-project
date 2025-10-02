# text-to-sql-project/api/app/services/langchain_orchestrator.py

import asyncio
from logging import getLogger
import json
import re

# Imports mis à jour pour les paquets modulaires LangChain
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
import sqlglot
from sqlalchemy import create_engine, exc, text

from app.config import settings
from app.models.schemas import InflationInterpretationRequest, ForecastNarrativeRequest

logger = getLogger(__name__)


class LangchainOrchestrator:
    """
    Orchestrateur basé sur LangChain pour gérer les fonctionnalités de l'API.
    """
    def __init__(self):
        logger.info("Initialisation de LangchainOrchestrator...")

        # 1. Initialisation des composants avec les nouvelles classes
        self.llm = ChatOllama(model=settings.LLM_MODEL, base_url=settings.OLLAMA_BASE_URL)
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
        # Sémaphore pour limiter la concurrence des appels LLM
        self.llm_sem = asyncio.Semaphore(2)

        # 2. Connexion à la base de données PostgreSQL (inchangé)
        try:
            self.db_engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
            self.db = SQLDatabase(engine=self.db_engine)
            self.admin_db_engine = create_engine(settings.ADMIN_DATABASE_URL)
            logger.info("Connexions à la base de données PostgreSQL via LangChain réussies.")
        except Exception as e:
            logger.error(f"Erreur de connexion à la base de données : {e}")
            raise

        # 3. Connexion à ChromaDB avec la nouvelle syntaxe d'initialisation
        self.chroma_client = Chroma(
            collection_name=settings.CHROMA_COLLECTION,
            embedding_function=self.embeddings,
            # Arguments `host` et `port` passés directement
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
        )
        self.retriever = self.chroma_client.as_retriever(search_kwargs={"k": 5})

        self.db_schema = self._get_rich_db_schema('indicateurs_economiques_uemoa')
        # Prépare les mots-clés pour filtrage/routage d'intention
        self._init_keyword_sets()
        self.reference_queries = [
            "SELECT pib_nominal_milliards_fcfa FROM indicateurs_economiques_uemoa WHERE date = '2022-01-01'",
            "SELECT date, exportations_biens_fob, importations_biens_fob, balance_des_biens FROM indicateurs_economiques_uemoa WHERE EXTRACT(YEAR FROM date) = 2021",
            "SELECT AVG(taux_croissance_reel_pib_pct) FROM indicateurs_economiques_uemoa WHERE date >= '2015-01-01' AND date <= '2020-01-01'",
        ]

        self._build_chains()
        logger.info("LangchainOrchestrator initialisé avec succès.")

    def _build_chains(self):
        """Construit les chaînes LCEL pour les différentes tâches."""

        sql_generation_template = """
### Instruction (génération SQL)
Tu es un expert SQL (PostgreSQL) et analyste de l'UEMOA. Convertis la question en une seule requête SQL SELECT.
Contraintes:
- Retourne UNIQUEMENT la requête SQL, sans texte additionnel ni Markdown.
- Ne modifie jamais les données (pas d'INSERT, UPDATE, DELETE, ALTER, DROP, CREATE, etc.).
- N'utilise que les colonnes du schéma fourni.
- Par défaut, les données concernent l'Union entière (BCEAO/UEMOA) — ne filtre un pays que si explicitement demandé et si la colonne existe.
- Si la question demande une liste potentiellement volumineuse, limite la sortie à 1000 lignes et ajoute un ORDER BY pertinent.
- Utilise des agrégations appropriées (AVG, SUM, COUNT, MAX, MIN) et GROUP BY si nécessaire.
- Pour les agrégations temporelles, privilégie DATE_TRUNC.
- Utilise des alias clairs pour les champs agrégés (ex. AS avg_inflation).
- En cas d'ambiguïté temporelle, choisis une hypothèse raisonnable basée sur les colonnes de date, sans commentaire.

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
        # 0) Sécurité & routage
        if self._is_question_harmful(user_question):
            return {"answer": "Désolé, je ne peux pas traiter cette demande."}
        if not self._needs_data_retrieval(user_question):
            return {
                "answer": (
                    "Désolé, cette question ne concerne pas les données économiques de l'UEMOA/BCEAO. "
                    "Veuillez reformuler avec un indicateur économique et une période (ex: PIB 2021)."
                )
            }

        # 1) Génération SQL avec retry si modèle absent
        try:
            async with self.llm_sem:
                generated_sql_raw = await self.sql_generation_chain.ainvoke({"question": user_question})
        except Exception as e:
            msg = str(e).lower()
            if "not found" in msg or "404" in msg:
                pull_res = await self.pull_model(settings.LLM_MODEL)
                if pull_res.get("status") == "success":
                    try:
                        async with self.llm_sem:
                            generated_sql_raw = await self.sql_generation_chain.ainvoke({"question": user_question})
                    except Exception as e2:
                        logger.error(f"Échec génération SQL après pull du modèle: {e2}")
                        return {"answer": "Désolé, échec de la génération SQL après téléchargement du modèle."}
                else:
                    logger.error(f"Échec du téléchargement du modèle: {pull_res}")
                    return {"answer": "Désolé, le modèle LLM n'est pas disponible et le téléchargement a échoué."}
            else:
                logger.error(f"Erreur génération SQL: {e}")
                return {"answer": "Désolé, une erreur est survenue lors de la génération de la requête SQL."}
        generated_sql = self._extract_sql_from_text(generated_sql_raw)

        if not generated_sql:
            return {"answer": "Je n'ai pas pu générer de requête SQL.", "generated_sql": ""}

        # Validation stricte
        if not self._validate_sql(generated_sql):
            return {"answer": "La requête SQL générée a été bloquée pour des raisons de sécurité.", "generated_sql": generated_sql}

        sql_result, error = await self._execute_sql_readonly(generated_sql)
        if error:
            return {"answer": error, "generated_sql": generated_sql}

        async with self.llm_sem:
            final_answer = await self.final_response_chain.ainvoke({"question": user_question, "sql_result": sql_result})
        return {"answer": final_answer, "generated_sql": generated_sql, "sql_result": sql_result}

    async def generate_forecast_narrative(self, body: ForecastNarrativeRequest) -> tuple[str, dict]:
        """Génère une narration pour des prévisions en utilisant un LLM via LangChain."""
        values = [p.value for p in body.series]
        dates = [p.date for p in body.series]
        count = len(values)
        _min = min(values) if values else 0.0
        _max = max(values) if values else 0.0
        mean = (sum(values) / count) if count else 0.0
        start_value = values[0] if count else 0.0
        end_value = values[-1] if count else 0.0
        start_date = dates[0] if dates else None
        end_date = dates[-1] if dates else None

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

        async with self.llm_sem:
            narrative = await narrative_chain.ainvoke({
            "language": body.language or "fr",
            "title": body.title or "Prévision",
            "horizon": body.horizon or "non précisé",
            "unit": body.unit or "unités",
            "series_preview": json.dumps(series_preview, indent=2, default=str)
        })
        stats = {
            "count": count,
            "min": float(_min),
            "max": float(_max),
            "mean": float(mean),
            "start_value": float(start_value),
            "end_value": float(end_value),
            "start_date": start_date,
            "end_date": end_date,
        }
        return narrative, stats

    async def generate_inflation_interpretation(self, body: InflationInterpretationRequest) -> dict:
        """Génère une interprétation d'inflation en utilisant un LLM via LangChain.

        Enrichi: nuance par audience, top-N SHAP par date et contraintes BCEAO/UEMOA.
        Sortie: JSON strict conforme à InflationInterpretationResponse.
        """
        # Pré-traitement des données
        predictions = body.prediction_data.predictions or {}
        global_shap = body.prediction_data.global_shap_importance or {}
        individual_raw = body.prediction_data.individual_shap_explanations or {}
        confidence_intervals = getattr(body.prediction_data, 'confidence_intervals', None)

        # Aplatir et arrondir les SHAP individuels (supporte structure avec feature_contributions)
        individual_shap_flat: dict[str, dict[str, float]] = {}
        for d, entry in individual_raw.items():
            feats_map = entry.get('feature_contributions') if isinstance(entry, dict) and 'feature_contributions' in entry else entry
            flat = {}
            if isinstance(feats_map, dict):
                for k, v in feats_map.items():
                    try:
                        flat[k] = round(float(v), 6)
                    except Exception:
                        continue
            individual_shap_flat[d] = flat

        # Top N contributeurs positifs/négatifs par date
        TOP_N = 5
        top_contrib_by_date: dict[str, dict[str, list[tuple[str, float]]]] = {}
        for d, feats in individual_shap_flat.items():
            items = list(feats.items())
            pos_sorted = [it for it in sorted(items, key=lambda x: x[1], reverse=True) if it[1] > 0]
            neg_sorted = [it for it in sorted(items, key=lambda x: x[1]) if it[1] < 0]
            top_contrib_by_date[d] = {
                "top_positive": pos_sorted[:TOP_N],
                "top_negative": neg_sorted[:TOP_N],
            }

        # Liste consolidée de features autorisées (anti-hallucinations)
        features_present = set(global_shap.keys())
        for feats in individual_shap_flat.values():
            features_present.update(feats.keys())
        features_present_list = sorted(features_present)

        # Règles spécifiques audience/langue
        language = getattr(body, 'analysis_language', 'fr')
        audience = getattr(body, 'target_audience', 'economist')
        include_policy = getattr(body, 'include_monetary_policy_analysis', True)

        if language == 'fr':
            audience_instructions = {
                "policymaker": (
                    "- Niveau de détail: concis et décisionnel.\n"
                    "- Mettre en avant 3–5 points clés avec chiffres essentiels.\n"
                    "- Lister clairement les top SHAP par date, sans jargon inutile.\n"
                ),
                "analyst": (
                    "- Niveau de détail: intermédiaire.\n"
                    "- Pour chaque date: top N positifs/négatifs avec valeurs SHAP + courte justification.\n"
                    "- Relier facteurs et mécanismes de transmission.\n"
                ),
                "economist": (
                    "- Niveau de détail: technique et complet.\n"
                    "- Pour chaque date: expliquer chiffres et contributions SHAP (valeur et signe).\n"
                    "- Mentionner effets de base/saisonnalité si suggérés par les données.\n"
                ),
                "general": (
                    "- Niveau de détail: pédagogique et simplifié.\n"
                    "- Expliquer en langage clair, tout en citant les chiffres saillants.\n"
                ),
            }.get(audience, "")
            context_line = (
                "- Cible de stabilité des prix BCEAO: 1–3%.\n"
                "- Zone: UEMOA (Union entière). Ne pas citer de pays spécifiques sauf si les données le justifient."
            )
        else:
            audience_instructions = {
                "policymaker": (
                    "- Detail level: concise and decision-oriented.\n"
                    "- Highlight 3–5 key points with essential figures.\n"
                    "- Clearly list top SHAP by date without unnecessary jargon.\n"
                ),
                "analyst": (
                    "- Detail level: intermediate.\n"
                    "- For each date: top N positive/negative with SHAP values + short justification.\n"
                    "- Link factors to transmission mechanisms.\n"
                ),
                "economist": (
                    "- Detail level: technical and thorough.\n"
                    "- For each date: explain forecast and SHAP contributions (value and sign).\n"
                    "- Mention base/seasonal effects if suggested by data.\n"
                ),
                "general": (
                    "- Detail level: educational and simplified.\n"
                    "- Use plain language while citing salient figures.\n"
                ),
            }.get(audience, "")
            context_line = (
                "- BCEAO price stability target: 1–3%.\n"
                "- Area: WAEMU (entire union). Do not cite countries unless evidence supports it."
            )

        # Construire le prompt JSON strict
        prompt_template = """
Tu es l'économiste en chef de la BCEAO. En t'appuyant UNIQUEMENT sur les données fournies, génère une INTERPRÉTATION JSON STRICTE.
RETOURNE UNIQUEMENT un JSON valide sans texte additionnel, avec les clés EXACTES suivantes:
{
  "executive_summary": string,
  "inflation_analysis": string,
  "key_inflation_drivers": string[],
  "price_stability_assessment": string,
  "monetary_policy_recommendations": string | null,
  "inflation_risks": string[],
  "model_confidence": string,
  "target_deviation_analysis": string,
  "external_factors_impact": string
}

Contexte et règles:
{context_line}
- Mentions AUTORISÉES de features (exclusives): {features_present_list}
- Toute affirmation doit être justifiée par des valeurs SHAP (globales ou par date).
- Adapter le niveau de détail au public: 
{audience_instructions}
{policy_line}

DONNÉES:
- Prédictions (par période): {predictions}
- Intervalles de confiance (si disponibles): {confidence_intervals}
- Importance SHAP globale: {global_shap}
- TOP contributeurs par date (positifs/négatifs): {top_contrib_by_date}
"""
        policy_line = ("- Inclure des recommandations de politique monétaire" if include_policy else "")

        interpretation_prompt = ChatPromptTemplate.from_template(prompt_template)
        interpretation_chain = interpretation_prompt | self.llm | StrOutputParser()

        async with self.llm_sem:
            raw_text = await interpretation_chain.ainvoke({
                "context_line": context_line,
                "audience_instructions": audience_instructions,
                "policy_line": policy_line,
                "features_present_list": json.dumps(features_present_list, ensure_ascii=False),
                "predictions": json.dumps(predictions, ensure_ascii=False),
                "confidence_intervals": json.dumps(confidence_intervals, ensure_ascii=False, default=str) if confidence_intervals else "null",
                "global_shap": json.dumps(global_shap, ensure_ascii=False),
                "top_contrib_by_date": json.dumps(top_contrib_by_date, ensure_ascii=False),
            })

        parsed = self._safe_json_from_text(raw_text)
        if not parsed:
            # Fallback minimal compliant structure
            summary = raw_text[:500]
            parsed = {
                "executive_summary": summary,
                "inflation_analysis": raw_text,
                "key_inflation_drivers": [],
                "price_stability_assessment": "",
                "monetary_policy_recommendations": None,
                "inflation_risks": [],
                "model_confidence": "",
                "target_deviation_analysis": "",
                "external_factors_impact": "",
            }
        return parsed

    def _safe_json_from_text(self, text: str) -> dict | None:
        """Extrait et charge un JSON depuis un texte LLM, sinon None."""
        try:
            # Essayer d'abord le texte complet
            return json.loads(text)
        except Exception:
            pass
        try:
            # Rechercher un bloc JSON dans des balises
            match = re.search(r"\{[\s\S]*\}")
            if match:
                return json.loads(match.group(0))
        except Exception:
            return None
        return None

    async def pull_model(self, model: str | None = None) -> dict:
        """Télécharge un modèle Ollama via l'API HTTP, sans dépendance externe.

        POST {base}/api/pull {"name": model, "stream": false}
        """
        model_name = model or settings.LLM_MODEL
        url = f"{settings.OLLAMA_BASE_URL}/api/pull"
        payload = {"name": model_name, "stream": False}
        try:
            import urllib.request
            req = urllib.request.Request(url, method="POST")
            req.add_header("Content-Type", "application/json")
            data = json.dumps(payload).encode("utf-8")
            with urllib.request.urlopen(req, data=data, timeout=120) as resp:
                content = resp.read().decode("utf-8")
                return {"status": "success", "model": model_name, "message": content}
        except Exception as e:
            logger.error(f"Échec du pull du modèle '{model_name}': {e}")
            return {"status": "error", "model": model_name, "message": str(e)}

    async def format_inflation_prediction(self, prediction_data: dict) -> dict:
        """Formate des données brutes de prédiction en InflationPredictionResponse."""
        predictions = prediction_data.get("predictions", {})
        global_shap = prediction_data.get("global_shap_importance", {})
        shap_summary = prediction_data.get("shap_summary_details", {})
        individual_shap = prediction_data.get("individual_shap_explanations", {})
        ci = prediction_data.get("confidence_intervals")
        return {
            "predictions": predictions,
            "global_shap_importance": global_shap,
            "shap_summary_details": shap_summary,
            "individual_shap_explanations": individual_shap,
            "confidence_intervals": ci,
        }

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
        """Validation stricte: interdit DDL/DML, une seule instruction, et SELECT/UNION/EXCEPT/INTERSECT uniquement."""
        try:
            banned = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|MERGE|CALL|COPY|VACUUM|ANALYZE|EXPLAIN)\b", re.IGNORECASE)
            if banned.search(sql):
                return False
            exprs = sqlglot.parse(sql, read="postgres")
            if not exprs or len(exprs) != 1:
                return False
            expr = exprs[0]
            base_expr = expr.this if isinstance(expr, sqlglot.exp.With) else expr
            allowed = (sqlglot.exp.Select, sqlglot.exp.Union, sqlglot.exp.Except, sqlglot.exp.Intersect)
            return isinstance(base_expr, allowed)
        except Exception:
            return False

    def _init_keyword_sets(self) -> None:
        """Construit des ensembles de mots-clés pour le routage thématique."""
        base_economic = {
            "uemoa", "bceao", "union économique", "union monétaire", "pib", "produit intérieur brut",
            "croissance", "inflation", "ipc", "taux", "taux d'intérêt", "politique monétaire", "dette",
            "budget", "balance commerciale", "exportations", "importations", "réserves", "change",
            "masse monétaire", "liquidité", "consommation", "investissement", "emploi", "chômage", "fcfa",
            "zone franc", "indicateurs économiques"
        }
        sql_keywords = {"requête", "requete", "sql", "select", "where", "group by", "order by", "table", "colonne"}
        date_keywords = {"date", "année", "annee", "mois", "trimestre", "2019", "2020", "2021", "2022", "2023", "2024", "2025"}

        dynamic_keywords: set[str] = set()
        try:
            with self.admin_db_engine.connect() as conn:
                rows = conn.execute(text(
                    """
                    SELECT column_name,
                           COALESCE(col_description((table_schema||'.'||table_name)::regclass::oid, ordinal_position), '') AS comment
                    FROM information_schema.columns
                    WHERE table_schema = :schema AND table_name = :table
                    ORDER BY ordinal_position
                    """
                ), {"schema": "public", "table": "indicateurs_economiques_uemoa"}).fetchall()
            for column_name, comment in rows:
                if not column_name:
                    continue
                name = column_name.lower()
                dynamic_keywords.add(name)
                tokens = name.split('_')
                dynamic_keywords.update(tokens)
                if len(tokens) >= 2:
                    dynamic_keywords.add(' '.join(tokens[:2]))
                if len(tokens) >= 3:
                    dynamic_keywords.add(' '.join(tokens[:3]))
                if comment:
                    for tok in re.split(r"[^a-zA-Zàâçéèêëîïôûùüÿñæœ']+", comment.lower()):
                        if tok and len(tok) > 3:
                            dynamic_keywords.add(tok)
        except Exception as exc:
            logger.warning(f"Impossible de générer les mots-clés dynamiques: {exc}")

        self.economic_keywords = base_economic | dynamic_keywords
        self.sql_keywords = sql_keywords
        self.date_keywords = date_keywords

    def _is_question_harmful(self, text_q: str) -> bool:
        if not text_q:
            return False
        q = text_q.lower()
        banned_terms = [
            # FR
            "bombe", "explosif", "arme", "pistolet", "fusil", "grenade", "cocktail molotov", "assassin", "meurtre", "tuer",
            "terrorisme", "attentat", "prise d'otage", "violence", "piratage", "hacker", "craquer", "malware", "ransomware",
            "phishing", "cheval de troie", "virus informatique", "backdoor", "ddos", "keylogger", "spyware", "drogue", "stupéfiant",
            "cannabis", "cocaïne", "héroïne", "ecstasy", "lsd", "meth", "opium", "poison", "arnaque", "escroquerie", "fraude",
            "blanchiment", "usurpation d'identité", "sexe", "pornographie", "pédophilie", "inceste", "viol", "prostitution",
            # EN
            "bomb", "explosive", "weapon", "grenade", "molotov", "gun", "rifle", "pistol", "ammunition", "knife", "terrorism",
            "attack", "shooting", "murder", "kill", "massacre", "hostage", "hack", "hacking", "crack", "trojan", "virus",
            "phishing scam", "money laundering", "identity theft", "sex", "porn", "child porn", "rape", "prostitution"
        ]
        return any(term in q for term in banned_terms)

    def _needs_data_retrieval(self, text_q: str) -> bool:
        if not text_q or len(text_q.strip()) < 10:
            return False
        q = text_q.lower().strip()
        eco = sum(1 for kw in getattr(self, 'economic_keywords', set()) if kw in q)
        sqlc = sum(1 for kw in getattr(self, 'sql_keywords', set()) if kw in q)
        datec = sum(1 for kw in getattr(self, 'date_keywords', set()) if kw in q)
        return (eco >= 2) or (eco >= 1 and (datec >= 1 or sqlc >= 1))

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