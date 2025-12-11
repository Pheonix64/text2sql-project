# ⚙️ Guide de Configuration

Guide complet pour configurer le projet Text-to-SQL.

---

## 📋 Table des Matières

1. [Variables d'Environnement](#1-variables-denvironnement)
2. [Configuration Docker](#2-configuration-docker)
3. [Configuration PostgreSQL](#3-configuration-postgresql)
4. [Configuration LLM](#4-configuration-llm)
5. [Configuration ChromaDB](#5-configuration-chromadb)
6. [Personnalisation](#6-personnalisation)

---

## 1. Variables d'Environnement

Le fichier `.env` contient toutes les variables de configuration.

### Structure Complète

```env
# ==================== API ====================
API_PORT=8008                              # Port d'exposition de l'API

# ==================== PostgreSQL ====================
# Base de données
POSTGRES_DB=economic_data                  # Nom de la base de données
POSTGRES_USER=postgres                     # Utilisateur administrateur
POSTGRES_PASSWORD=postgrespwd123!          # Mot de passe admin
DB_HOST=postgres-db                        # Hôte (nom du service Docker)
DB_PORT=5432                               # Port PostgreSQL

# Utilisateur LLM (lecture seule)
LLM_USER=llm_user                          # Utilisateur pour l'API
LLM_PASSWORD=/-+3Vd9$!D@12                # Mot de passe LLM

# ==================== ChromaDB ====================
CHROMA_HOST=chroma-db                      # Hôte ChromaDB
CHROMA_PORT=8000                           # Port interne (ne pas modifier)
CHROMA_EXTERNAL_PORT=8088                  # Port externe (host)
CHROMA_COLLECTION=sql_reference_queries    # Nom de la collection

# ==================== Ollama ====================
OLLAMA_HOST=ollama                         # Hôte Ollama
OLLAMA_PORT=11434                          # Port Ollama
LLM_MODEL=mistral:7b                       # Modèle à utiliser

# ==================== Embeddings ====================
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# ==================== Timeouts LLM ====================
LLM_TIMEOUT_SQL=90                         # Timeout pour génération SQL (secondes)
LLM_TIMEOUT_INFLATION=120                  # Timeout pour interprétation inflation (secondes)
LLM_TIMEOUT_NARRATIVE=60                   # Timeout pour génération narrative (secondes)
```

### Description Détaillée

#### API_PORT
- **Par défaut :** `8008`
- **Description :** Port sur lequel l'API FastAPI sera accessible
- **Modification :** Changez si le port 8008 est déjà utilisé

```env
API_PORT=9000  # Utiliser le port 9000 à la place
```

#### POSTGRES_USER / POSTGRES_PASSWORD
- **Rôle :** Utilisateur administrateur PostgreSQL
- **Usage :** Initialisation de la base, gestion du schéma
- **Sécurité :** ⚠️ Changez le mot de passe en production !

```env
POSTGRES_PASSWORD=VotreMotDePasseSecurise123!@#
```

#### LLM_USER / LLM_PASSWORD
- **Rôle :** Utilisateur en **lecture seule** pour l'API
- **Sécurité :** Limite les risques, ne peut pas modifier les données
- **Note :** Caractères spéciaux supportés (URL-encoding automatique)

#### LLM_MODEL
- **Par défaut :** `mistral:7b`
- **Alternatives :**
  - `llama2:7b` - Plus rapide mais moins précis
  - `llama2:13b` - Meilleur mais plus lent
  - `mixtral:8x7b` - Le plus performant mais nécessite plus de RAM
  - `codellama:7b` - Spécialisé pour le code SQL

```env
LLM_MODEL=llama2:13b  # Pour de meilleurs résultats
```

#### EMBEDDING_MODEL_NAME
- **Par défaut :** `sentence-transformers/all-MiniLM-L6-v2`
- **Alternatives :**
  - `all-mpnet-base-v2` - Plus performant mais plus lourd
  - `paraphrase-multilingual-MiniLM-L12-v2` - Multilingue
  - `all-MiniLM-L12-v2` - Plus précis que L6

#### Timeouts LLM (Configurables)

Les timeouts sont configurables pour chaque type d'opération LLM :

| Variable | Par défaut | Description |
|----------|------------|-------------|
| `LLM_TIMEOUT_SQL` | 90 | Timeout pour la génération SQL |
| `LLM_TIMEOUT_INFLATION` | 120 | Timeout pour l'interprétation d'inflation SHAP |
| `LLM_TIMEOUT_NARRATIVE` | 60 | Timeout pour la génération narrative |

**Exemple d'ajustement :**
```env
# Pour des modèles plus lents ou des analyses complexes
LLM_TIMEOUT_INFLATION=180  # 3 minutes
LLM_TIMEOUT_SQL=120        # 2 minutes
```

**Note :** Le timeout d'inflation est plus long par défaut car l'interprétation économique génère plus de texte et nécessite une analyse approfondie des facteurs SHAP.

---

## 2. Configuration Docker

### docker-compose.yml

#### Modifier les Ports

```yaml
services:
  api-fastapi:
    ports:
      - "${API_PORT}:8000"      # Changez API_PORT dans .env
  
  postgres-db:
    ports:
      - "5433:5432"             # Changer le port host si 5432 est occupé
  
  chroma-db:
    ports:
      - "${CHROMA_EXTERNAL_PORT}:8000"
  
  ollama:
    ports:
      - "11435:11434"           # Changer si nécessaire
```

#### Ajuster les Ressources

**Pour Ollama (si vous avez un GPU) :**

```yaml
ollama:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1              # Nombre de GPUs
            capabilities: [gpu]
```

**Pour limiter la RAM :**

```yaml
api-fastapi:
  deploy:
    resources:
      limits:
        memory: 2G              # Maximum 2GB
      reservations:
        memory: 1G              # Minimum 1GB
```

#### Désactiver le GPU

Si vous n'avez pas de GPU NVIDIA :

```yaml
ollama:
  # Commentez ou supprimez la section deploy
  # deploy:
  #   resources:
  #     reservations:
  #       devices:
  #         - driver: nvidia
  #           count: 1
  #           capabilities: [gpu]
```

---

## 3. Configuration PostgreSQL

### Modifier le Script d'Initialisation

Fichier : `postgres/init.sql`

#### Ajouter des Données

```sql
-- Insérer des données supplémentaires
INSERT INTO indicateurs (pays, indicateur, annee, valeur) VALUES
  ('Bénin', 'PIB', 2024, 18500000000),
  ('Bénin', 'Inflation', 2024, 2.1);
```

#### Créer des Utilisateurs Supplémentaires

```sql
-- Utilisateur analytics (lecture + agrégations)
CREATE USER analytics_user WITH PASSWORD 'analytics_pwd';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO analytics_user;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO analytics_user;
```

#### Modifier les Permissions

```sql
-- Donner des droits d'écriture (⚠️ Attention)
GRANT INSERT, UPDATE ON indicateurs TO llm_user;
```

### Changer le CSV Source

Fichier : `docker-compose.yml`

```yaml
postgres-db:
  volumes:
    - ./postgres/indiceconomique_long_v4.csv:/docker-entrypoint-initdb.d/data.csv
    # Changez pour votre propre fichier:
    - ./postgres/mes_donnees.csv:/docker-entrypoint-initdb.d/data.csv
```

---

## 4. Configuration LLM

### Changer de Modèle

#### Méthode 1 : Fichier .env

```env
LLM_MODEL=llama2:13b
```

Puis redémarrer :
```bash
docker-compose restart api-fastapi
```

#### Méthode 2 : Via l'API

```bash
curl -X POST "http://localhost:8008/api/pull-model" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama2:13b"}'
```

#### Méthode 3 : Manuellement dans Ollama

```bash
docker exec -it ollama ollama pull llama2:13b
docker exec -it ollama ollama list  # Vérifier
```

### Optimiser les Performances

Dans `api/app/services/query_orchestrator.py` :

```python
# Paramètres du modèle
self.llm = ChatOllama(
    model=settings.LLM_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
    temperature=0.1,      # Plus bas = plus déterministe
    num_predict=512,      # Nombre max de tokens
    top_k=10,             # Limitation du sampling
    top_p=0.9             # Nucleus sampling
)
```

### Utiliser plusieurs modèles

```python
# Dans query_orchestrator.py
self.llm_sql = ChatOllama(model="codellama:7b")  # Pour SQL
self.llm_analysis = ChatOllama(model="mistral:7b")  # Pour analyse
```

---

## 5. Configuration ChromaDB

### Changer le Port

```env
CHROMA_EXTERNAL_PORT=9000  # Port externe
```

### Persister les Données

Les données ChromaDB sont déjà persistées via Docker volumes :

```yaml
volumes:
  chroma-data:
    driver: local
```

Pour sauvegarder manuellement :

```bash
docker cp chroma-db:/chroma/chroma ./chroma_backup
```

### Réinitialiser ChromaDB

```bash
# Supprimer le volume
docker-compose down
docker volume rm text-to-sql-project_chroma-data

# Recréer
docker-compose up -d

# Réindexer les exemples
curl -X POST "http://localhost:8008/api/index-queries"
```

---

## 6. Personnalisation

### Ajouter des Exemples SQL

Fichier : `docs/examples.json`

```json
[
  {
    "question": "Inflation moyenne UEMOA 2023",
    "sql": "SELECT AVG(valeur) FROM indicateurs WHERE zone='UEMOA' AND indicateur='Inflation' AND annee=2023"
  },
  {
    "question": "Top 5 pays par PIB",
    "sql": "SELECT pays, valeur FROM indicateurs WHERE indicateur='PIB' ORDER BY valeur DESC LIMIT 5"
  }
]
```

Puis réindexer :
```bash
curl -X POST "http://localhost:8008/api/index-queries"
```

### Personnaliser les Prompts

Fichier : `api/app/services/query_orchestrator.py`

```python
# Prompt de génération SQL
sql_generation_prompt = PromptTemplate.from_template("""
Tu es un expert SQL. Voici le schéma de la base :
{schema}

Exemples similaires :
{examples}

Question : {question}

Génère UNIQUEMENT la requête SQL entre les balises <sql></sql>.
Utilise les noms exacts des tables et colonnes.
""")
```

### Modifier le Timeout

Fichier : `docker-compose.yml`

```yaml
api-fastapi:
  environment:
    - REQUEST_TIMEOUT=60  # 60 secondes
```

Dans le code Python (FastAPI) :

```python
from fastapi import FastAPI
import uvicorn

app = FastAPI()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=60  # Timeout
    )
```

### Activer les Logs Détaillés

```env
# .env
LOG_LEVEL=DEBUG
```

Dans `api/app/main.py` :

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,  # ou INFO, WARNING, ERROR
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Mode Développement vs Production

**Développement (rechargement automatique) :**

```yaml
# docker-compose.yml
api-fastapi:
  command: ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
  volumes:
    - ./api/app:/home/appuser/app  # Montage du code
```

**Production (optimisé) :**

```yaml
api-fastapi:
  command: ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
  # Pas de volume monté
```

---

## 📚 Exemples de Configurations

### Configuration Minimale (Dev)

```env
API_PORT=8008
POSTGRES_DB=economic_data
POSTGRES_USER=postgres
POSTGRES_PASSWORD=simple123
LLM_USER=llm_user
LLM_PASSWORD=llm123
CHROMA_HOST=chroma-db
CHROMA_PORT=8000
CHROMA_EXTERNAL_PORT=8088
OLLAMA_HOST=ollama
OLLAMA_PORT=11434
LLM_MODEL=mistral:7b
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

### Configuration Production

```env
API_PORT=8008
POSTGRES_DB=bceao_economic_data
POSTGRES_USER=admin_bceao
POSTGRES_PASSWORD=SecureP@ssw0rd!2024#BCEAO
LLM_USER=readonly_llm
LLM_PASSWORD=R34d0nly!LLM#2024
CHROMA_HOST=chroma-db
CHROMA_PORT=8000
CHROMA_EXTERNAL_PORT=8088
OLLAMA_HOST=ollama
OLLAMA_PORT=11434
LLM_MODEL=mixtral:8x7b
EMBEDDING_MODEL_NAME=all-mpnet-base-v2
LOG_LEVEL=INFO
```

---

## 🔒 Sécurité

### Checklist de Sécurité

- [ ] Changer tous les mots de passe par défaut
- [ ] Utiliser des mots de passe forts (16+ caractères)
- [ ] Ne jamais commiter le fichier `.env`
- [ ] Limiter l'accès réseau (firewall)
- [ ] Utiliser HTTPS en production
- [ ] Limiter les permissions SQL de `llm_user`
- [ ] Activer l'authentification API si nécessaire
- [ ] Sauvegarder régulièrement les données

### Exemple : Ajouter Authentification API

```python
# api/app/main.py
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key != "votre_cle_secrete":
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.post("/api/ask")
async def ask_question(
    request: QuestionRequest,
    req: Request,
    api_key: str = Depends(verify_api_key)
):
    # ... votre code
```

---

**[⬆ Retour en haut](#-guide-de-configuration)**
