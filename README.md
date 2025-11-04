# 🚀 Text-to-SQL - Assistant d'Analyse de Données Économiques

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-purple)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue)

**Un système intelligent pour interroger des bases de données économiques en langage naturel**

</div>

---

## 📋 Table des Matières

- [Vue d'ensemble](#-vue-densemble)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Démarrage Rapide](#-démarrage-rapide)
- [Utilisation](#-utilisation)
- [Documentation](#-documentation)
- [Configuration](#-configuration)
- [Développement](#-développement)

---

## 🎯 Vue d'ensemble

Ce projet est une **API REST intelligente** qui permet d'interroger une base de données économiques en **langage naturel** grâce à :
- **LangChain** pour l'orchestration des LLMs
- **Ollama** avec Mistral pour la génération SQL et l'analyse
- **ChromaDB** pour la recherche sémantique d'exemples SQL
- **PostgreSQL/TimescaleDB** pour le stockage des données économiques
- **SHAP** pour l'interprétation des prédictions d'inflation

### Cas d'usage
- 📊 Requêtes en langage naturel sur données économiques
- 🔍 Génération automatique de requêtes SQL
- 📈 Analyse et interprétation de prévisions économiques
- 💡 Explicabilité des modèles de prédiction d'inflation

---

## ✨ Fonctionnalités

### 🗣️ Text-to-SQL Conversationnel
- Convertit questions en SQL valide
- Recherche sémantique d'exemples similaires
- Validation et exécution sécurisée
- Réponses en langage naturel

### 📊 Analyse de Prévisions
- Génération de narratifs économiques
- Interprétation de prédictions d'inflation
- Explicabilité SHAP pour économistes

### 🔐 Sécurité
- Utilisateur SQL en lecture seule
- Validation SQLGlot des requêtes
- Gestion des erreurs robuste

---

## 🏗️ Architecture

```
┌─────────────────┐
│   Client Web    │
└────────┬────────┘
         │ HTTP/REST
         ▼
┌─────────────────────────────────────┐
│        API FastAPI (Port 8008)      │
│  ┌─────────────────────────────┐   │
│  │  QueryOrchestrator          │   │
│  │  - LangChain Pipeline       │   │
│  │  - SQL Generation           │   │
│  │  - Result Analysis          │   │
│  └─────────────────────────────┘   │
└──┬────────┬────────┬───────────────┘
   │        │        │
   ▼        ▼        ▼
┌──────┐ ┌──────┐ ┌──────────────┐
│Ollama│ │Chroma│ │ PostgreSQL   │
│:11434│ │:8088 │ │ TimescaleDB  │
│      │ │      │ │    :5432     │
└──────┘ └──────┘ └──────────────┘
```

### Stack Technologique

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| **API** | FastAPI + Uvicorn | Serveur REST |
| **LLM** | Ollama (Mistral 7B) | Génération SQL & Analyse |
| **Orchestration** | LangChain | Pipeline Text-to-SQL |
| **Embeddings** | Sentence-Transformers | Recherche sémantique |
| **Vector DB** | ChromaDB | Stockage d'exemples SQL |
| **Database** | PostgreSQL + TimescaleDB | Données économiques |
| **Conteneurisation** | Docker Compose | Déploiement |

---

## 🚀 Démarrage Rapide

### Prérequis

- Docker Desktop installé et en cours d'exécution
- 8 GB RAM minimum (16 GB recommandé)
- 10 GB d'espace disque disponible

### Installation en 3 étapes

#### 1️⃣ Cloner le projet

```bash
git clone https://github.com/Pheonix64/text2sql-project.git
cd text-to-sql-project
```

#### 2️⃣ Configurer l'environnement

Le fichier `.env` est déjà configuré avec des valeurs par défaut :

```env
# API
API_PORT=8008

# PostgreSQL
POSTGRES_DB=economic_data
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgrespwd123!

# LLM User (read-only)
LLM_USER=llm_user
LLM_PASSWORD=/-+3Vd9$!D@12

# ChromaDB
CHROMA_HOST=chroma-db
CHROMA_PORT=8000
CHROMA_EXTERNAL_PORT=8088

# Ollama
OLLAMA_HOST=ollama
OLLAMA_PORT=11434
LLM_MODEL=mistral:7b

# Embeddings
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

#### 3️⃣ Démarrer les services

```bash
docker-compose up -d
```

**Temps de démarrage initial :** 5-10 minutes (téléchargement des modèles)

### Vérification

```bash
# Vérifier le statut des services
docker-compose ps

# Vérifier les logs
docker logs api-fastapi --tail 50

# Tester l'API
curl http://localhost:8008/health
```

**Résultat attendu :**
```json
{"status": "ok"}
```

---

## 💻 Utilisation

### Accès aux Services

| Service | URL | Description |
|---------|-----|-------------|
| **API Documentation** | http://localhost:8008/docs | Interface Swagger interactive |
| **API Alternative** | http://localhost:8008/redoc | Documentation ReDoc |
| **API Health** | http://localhost:8008/health | Statut de l'API |
| **ChromaDB** | http://localhost:8088 | Base vectorielle |
| **PostgreSQL** | localhost:5432 | Base de données |
| **Ollama** | http://localhost:11434 | Serveur LLM |

### Exemple Simple - Poser une Question

```bash
curl -X POST "http://localhost:8008/api/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quelle est l'\''évolution du PIB en 2023?"
  }'
```

**Réponse :**
```json
{
  "answer": "En 2023, le PIB a augmenté de 2.5%...",
  "sql_query": "SELECT annee, valeur FROM indicateurs WHERE indicateur='PIB' AND annee=2023",
  "result_data": [...],
  "metadata": {
    "execution_time": "1.2s",
    "rows_returned": 4
  }
}
```

### Exemples via Python

```python
import requests

# Poser une question
response = requests.post(
    "http://localhost:8008/api/ask",
    json={"question": "Quel est le taux d'inflation moyen des 5 dernières années?"}
)

result = response.json()
print(f"Réponse: {result['answer']}")
print(f"SQL généré: {result['sql_query']}")
```

---

## 📚 Documentation

### Guides Complets

- **[📖 Guide Utilisateur](docs/GUIDE_UTILISATEUR.md)** - Tutoriel complet avec exemples
- **[🔌 Référence API](docs/API_REFERENCE.md)** - Documentation détaillée des endpoints
- **[🏗️ Architecture](docs/ARCHITECTURE_DIAGRAM.md)** - Diagrammes et composants
- **[🔧 Guide de Configuration](docs/CONFIGURATION.md)** - Variables d'environnement

### Documentation Technique

- **[⚙️ LangChain Integration](docs/LANGCHAIN_INDEX.md)** - Pipeline et orchestration
- **[📊 SHAP & Prédictions](docs/SHAP_PREDICTION_GUIDE.md)** - Interprétation des modèles
- **[🧪 Guide de Tests](docs/TESTING_GUIDE.md)** - Tests et validation
- **[🔄 Refactoring](docs/REFACTORING_SUMMARY.md)** - Historique des modifications

### Exemples

- **[examples.json](docs/examples.json)** - Exemples de requêtes SQL
- **[example_shap_response.json](docs/example_shap_response.json)** - Exemple de réponse SHAP

---

## ⚙️ Configuration

### Variables d'Environnement Clés

#### Base de Données

```env
POSTGRES_DB=economic_data          # Nom de la base
POSTGRES_USER=postgres             # Utilisateur admin
POSTGRES_PASSWORD=votreMotDePasse  # Mot de passe admin
LLM_USER=llm_user                  # Utilisateur read-only pour LLM
LLM_PASSWORD=votreMotDePasse       # Mot de passe LLM
```

#### Modèle LLM

```env
LLM_MODEL=mistral:7b              # Modèle Ollama (alternatives: llama2, mixtral)
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

#### Ports

```env
API_PORT=8008                      # Port de l'API
CHROMA_EXTERNAL_PORT=8088          # Port ChromaDB externe
OLLAMA_PORT=11434                  # Port Ollama
```

### Changer le Modèle LLM

```bash
# 1. Modifier .env
LLM_MODEL=llama2:13b

# 2. Redémarrer les services
docker-compose restart api-fastapi

# 3. Télécharger le modèle (optionnel)
curl -X POST "http://localhost:8008/api/pull-model" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama2:13b"}'
```

---

## 🛠️ Développement

### Structure du Projet

```
text-to-sql-project/
├── api/                                # Application FastAPI
│   ├── Dockerfile                      # Image Docker API
│   ├── requirements.txt                # Dépendances Python
│   └── app/
│       ├── main.py                     # Point d'entrée FastAPI
│       ├── config.py                   # Configuration
│       ├── models/
│       │   └── schemas.py              # Modèles Pydantic
│       ├── routers/
│       │   ├── conversation.py         # Endpoints Text-to-SQL
│       │   └── forecast.py             # Endpoints Prévisions
│       └── services/
│           ├── query_orchestrator.py   # Orchestrateur principal
│           └── langchain_orchestrator.py
├── postgres/
│   ├── init.sql                        # Script d'initialisation
│   └── indiceconomique_long_v4.csv     # Données économiques
├── docs/                               # Documentation complète
├── docker-compose.yml                  # Configuration Docker
├── .env                                # Variables d'environnement
└── README.md                           # Ce fichier
```

### Commandes Docker Utiles

```bash
# Démarrer les services
docker-compose up -d

# Arrêter les services
docker-compose down

# Voir les logs
docker-compose logs -f api-fastapi

# Reconstruire l'API
docker-compose build --no-cache api-fastapi
docker-compose up -d api-fastapi

# Redémarrer un service
docker-compose restart api-fastapi

# Accéder à un conteneur
docker exec -it api-fastapi bash
docker exec -it postgres-db psql -U postgres -d economic_data
```

### Mode Développement

Pour activer le rechargement automatique :

```yaml
# Dans docker-compose.yml (déjà configuré)
api-fastapi:
  volumes:
    - ./api/app:/home/appuser/app
  command: ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

### Tests

```bash
# Tester la santé de l'API
curl http://localhost:8008/health

# Tester une requête
curl -X POST "http://localhost:8008/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Test"}'

# Réindexer les exemples
curl -X POST "http://localhost:8008/api/index-queries"
```

---

## 🔧 Résolution de Problèmes

### ChromaDB ne démarre pas

```bash
# Vérifier les logs
docker logs chroma-db

# Recréer le conteneur
docker-compose up -d --force-recreate chroma-db
```

### L'API ne peut pas se connecter à ChromaDB

Vérifier que `CHROMA_PORT=8000` dans `.env` (port interne, pas 8088)

### Le modèle Ollama ne se télécharge pas

```bash
# Télécharger manuellement
docker exec -it ollama ollama pull mistral:7b

# Ou via l'API
curl -X POST "http://localhost:8008/api/pull-model"
```

### Erreur de mémoire

Augmenter la RAM allouée à Docker Desktop (Settings > Resources > Memory)

---

## 📊 Endpoints API - Résumé

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/ask` | **Poser une question** |
| `POST` | `/api/index-queries` | Réindexer les exemples SQL |
| `POST` | `/api/pull-model` | Télécharger modèle LLM |
| `POST` | `/api/forecast/narrative` | Générer narration économique |
| `POST` | `/api/forecast/inflation/prediction` | Prédiction inflation |
| `POST` | `/api/forecast/inflation/interpret` | Interpréter inflation |

👉 **[Documentation API Complète](docs/API_REFERENCE.md)**

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

---

## 📝 Licence

Ce projet est développé dans le cadre d'un stage à la BCEAO.

---

## 👥 Auteurs

- **Stage BCEAO** - Développement initial

---

## 🙏 Remerciements

- **LangChain** pour l'orchestration LLM
- **Ollama** pour le serving local de LLMs
- **ChromaDB** pour la base vectorielle
- **FastAPI** pour le framework web
- **TimescaleDB** pour les données temporelles

---

<div align="center">

**[⬆ Retour en haut](#-text-to-sql---système-danalyse-de-données-économiques)**

Made with ❤️ at BCEAO

</div>
