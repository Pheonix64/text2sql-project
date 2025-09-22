import ollama
import chromadb
import sqlglot
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text, exc
from logging import getLogger

from..config import settings

logger = getLogger(__name__)

class QueryOrchestrator:
    def __init__(self):
        logger.info("Initialisation de QueryOrchestrator...")
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        self.chroma_client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
        self.sql_collection = self.chroma_client.get_or_create_collection(name=settings.CHROMA_COLLECTION)
        self.ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        
        try:
            # Utilisation d'un moteur de connexion différent pour l'admin (schéma) et le llm (requêtes)
            self.db_engine = create_engine(settings.DATABASE_URL)
            # Connexion avec l'utilisateur admin pour récupérer le schéma
            admin_db_url = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.POSTGRES_DB}"
            self.admin_db_engine = create_engine(admin_db_url)
            logger.info("Connexions à la base de données PostgreSQL réussies.")
        except Exception as e:
            logger.error(f"Erreur de connexion à la base de données : {e}")
            raise

        # Récupération dynamique du schéma et des commentaires
        self.db_schema = self._get_rich_db_schema(table_name='indicateurs_economiques_uemoa')
        
        # Mise à jour des requêtes de référence pour la nouvelle table
        self.reference_queries = logger.info("QueryOrchestrator initialisé avec succès.")

    def _get_rich_db_schema(self, table_name: str) -> str:
        """
        Récupère dynamiquement le schéma de la table, y compris les commentaires
        sur la table et les colonnes pour enrichir le contexte du LLM.
        """
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
                schema_str += ");"
                logger.info("Schéma enrichi récupéré avec succès.")
                return schema_str
        except Exception as e:
            logger.error(f"Impossible de récupérer le schéma de la base de données : {e}")
            return f"CREATE TABLE {table_name} (...); -- Erreur: impossible de récupérer le schéma détaillé"

    def index_reference_queries(self, queries: list[str] | None = None):
        """Indexe les requêtes SQL de référence dans ChromaDB."""
        queries_to_index = queries or self.reference_queries
        if not queries_to_index:
            logger.warning("Aucune requête de référence à indexer.")
            return 0
            
        logger.info(f"Indexation de {len(queries_to_index)} requêtes...")
        if self.sql_collection.count() > 0:
            ids_to_delete = self.sql_collection.get()['ids']
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
        """Validation statique de la requête SQL (inchangée)."""
        try:
            parsed = sqlglot.parse_one(sql_query, read="postgres")
            if not isinstance(parsed, sqlglot.exp.Select):
                logger.warning(f"Validation échouée : la requête n'est pas un SELECT. Requête : {sql_query}")
                return False
            logger.info(f"Validation SQL réussie pour : {sql_query}")
            return True
        except Exception as e:
            logger.error(f"Erreur de validation SQL : {e}. Requête : {sql_query}")
            return False

    async def process_user_question(self, user_question: str) -> dict:
        """Exécute le pipeline complet pour répondre à une question."""
        question_embedding = self.embedding_model.encode(user_question).tolist()
        results = self.sql_collection.query(query_embeddings=[question_embedding], n_results=5)
        context_queries = "\n".join(results['documents']) if results['documents'] else ""

        # Le prompt est maintenant beaucoup plus riche grâce au schéma dynamique
        sql_generation_prompt = f"""
        ### Instruction
        Tu es un expert en SQL PostgreSQL et un analyste économique. Ton objectif est de convertir la question de l'utilisateur en une seule requête SQL SELECT en te basant sur le schéma et les descriptions ci-dessous.
        Utilise des fonctions temporelles comme DATE_TRUNC, et des agrégations comme AVG, SUM, MAX, MIN, etc., si nécessaire.
        Ne génère JAMAIS de requêtes qui modifient les données (INSERT, UPDATE, DELETE).

        ### Schéma et Descriptions de la Base de Données
        {self.db_schema}

        ### Exemples de Requêtes Similaires
        {context_queries}

        ### Question de l'Utilisateur
        "{user_question}"

        ### Requête SQL
        """

        try:
            response = await self.ollama_client.generate(model=settings.LLM_MODEL, prompt=sql_generation_prompt)
            generated_sql = response['response'].strip().replace("```sql", "").replace("```", "").strip()
        except Exception as e:
            logger.error(f"Erreur lors de la génération SQL par Ollama : {e}")
            return {"answer": "Désolé, une erreur est survenue lors de la génération de la requête SQL."}

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
        ### Instruction
        En te basant sur la question de l'utilisateur et le résultat de la requête SQL, formule une réponse claire, concise et professionnelle en français.
        Si le résultat est une liste ou contient des chiffres, présente-le de manière lisible.

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

# Instance unique de l'orchestrateur pour l'application
orchestrator = QueryOrchestrator()