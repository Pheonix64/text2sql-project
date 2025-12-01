# 🔌 API Reference - Text-to-SQL

Documentation complète de tous les endpoints de l'API Text-to-SQL.

**Base URL :** `http://localhost:8008`

---

## 📋 Table des Matières

1. [Health Check](#1-health-check)
2. [Text-to-SQL Endpoints](#2-text-to-sql-endpoints)
3. [Forecast Endpoints](#3-forecast-endpoints)
4. [Administration Endpoints](#4-administration-endpoints)
5. [Modèles de Données](#5-modèles-de-données)
6. [Codes d'Erreur](#6-codes-derreur)

---

## 1. Health Check

### `GET /health`

Vérifier l'état de l'API.

#### Requête

```http
GET /health HTTP/1.1
Host: localhost:8008
```

#### Réponse

**Status: 200 OK**
```json
{
  "status": "ok"
}
```

#### Exemple cURL

```bash
curl http://localhost:8008/health
```

---

## 2. Text-to-SQL Endpoints

### `POST /api/ask`

**Endpoint principal** pour poser une question en langage naturel.

#### Description

Convertit une question en SQL, exécute la requête et retourne une réponse en langage naturel.

#### Requête

**Headers:**
```http
Content-Type: application/json
```

**Body:**
```json
{
  "question": "string"  // Question en langage naturel (requis)
}
```

#### Réponse

**Status: 200 OK**
```json
{
  "answer": "string",           // Réponse en langage naturel
  "generated_sql": "string | null",  // Requête SQL générée (peut être null)
  "sql_result": "string | null"      // Résultats SQL en format string (peut être null)
}
```

**Note** : Les champs retournés correspondent exactement au schéma `AnswerResponse` défini dans `api/app/models/schemas.py`

#### Exemples

**Exemple 1 : Question Simple**

```bash
curl -X POST "http://localhost:8008/api/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quel est le taux d'\''inflation en 2021 ?"
  }'
```

**Réponse :**
```json
{
  "answer": "En 2021, l'UEMOA a enregistré un taux d'inflation moyen de 3,9%.",
  "generated_sql": "SELECT taux_inflation_moyen_annuel_ipc_pct FROM indicateurs_economiques_uemoa WHERE date = '2021-01-01';",
  "sql_result": "[{\"taux_inflation_moyen_annuel_ipc_pct\": 3.9}]"
}
```

**Réponse :**
```json
{
  "answer": "En 2021, l'UEMOA a enregistré un taux d'inflation moyen de 3,9%, dépassant légèrement l'objectif de stabilité des prix de la BCEAO fixé à 3%.",
  "generated_sql": "SELECT taux_inflation_moyen_annuel_ipc_pct FROM indicateurs_economiques_uemoa WHERE date = '2021-01-01';",
  "sql_result": "[{\"taux_inflation_moyen_annuel_ipc_pct\": 3.9}]"
}
```

**Note importante** : La table utilisée est `indicateurs_economiques_uemoa`, pas `indicateurs`. Voir le schéma complet dans `postgres/init.sql`.

**Exemple 2 : Agrégation**

```bash
curl -X POST "http://localhost:8008/api/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quel est le taux d'\''inflation moyen en 2023?"
  }'
```

**Exemple 3 : Évolution Temporelle**

```bash
curl -X POST "http://localhost:8008/api/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Évolution du PIB du Burkina Faso depuis 2020"
  }'
```

#### Codes de Statut

| Code | Description |
|------|-------------|
| `200` | Succès |
| `400` | Requête invalide (question manquante) |
| `500` | Erreur serveur (génération SQL, exécution, etc.) |

---

## 3. Forecast Endpoints

### `POST /api/forecast/narrative`

Générer une narration économique pour une prévision.

#### Requête

**Body:**
```json
{
  "indicator": "string",        // Indicateur économique (requis)
  "period": "string",           // Période (ex: "2024-Q1") (requis)
  "country": "string",          // Pays (optionnel)
  "additional_context": "string" // Contexte supplémentaire (optionnel)
}
```

#### Réponse

**Status: 200 OK**
```json
{
  "narrative": "string",        // Narration générée
  "summary_stats": {            // Statistiques résumées
    "mean": 0.0,
    "median": 0.0,
    "std": 0.0,
    "min": 0.0,
    "max": 0.0
  }
}
```

#### Exemple

```bash
curl -X POST "http://localhost:8008/api/forecast/narrative" \
  -H "Content-Type: application/json" \
  -d '{
    "indicator": "PIB",
    "period": "2024-Q1",
    "country": "Côte d'\''Ivoire"
  }'
```

**Réponse :**
```json
{
  "narrative": "Au premier trimestre 2024, le PIB de la Côte d'Ivoire a enregistré une croissance robuste de 6,2%, portée principalement par le secteur agricole et les investissements dans les infrastructures...",
  "summary_stats": {
    "mean": 6.2,
    "median": 6.1,
    "std": 0.5
  }
}
```

---

### `POST /api/forecast/inflation/prediction`

Recevoir et formater des prédictions d'inflation avec explicabilité SHAP.

#### Requête

**Body:**
```json
{
  "prediction_value": 0.0,      // Valeur prédite
  "shap_values": {              // Valeurs SHAP par feature
    "feature1": 0.0,
    "feature2": 0.0
  },
  "base_value": 0.0,            // Valeur de base
  "features": {                 // Valeurs des features
    "feature1": "value1",
    "feature2": "value2"
  }
}
```

#### Réponse

**Status: 200 OK**
```json
{
  "formatted_prediction": {
    "prediction": 0.0,
    "shap_interpretation": {...}
  }
}
```

#### Exemple

```bash
curl -X POST "http://localhost:8008/api/forecast/inflation/prediction" \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_value": 2.5,
    "shap_values": {
      "oil_price": 0.8,
      "exchange_rate": -0.3,
      "money_supply": 0.5
    },
    "base_value": 2.0
  }'
```

---

### `POST /api/forecast/inflation/interpret`

Interpréter les prédictions d'inflation pour les économistes.

#### Requête

**Body:**
```json
{
  "prediction_data": {          // Données de prédiction (requis)
    "predicted_inflation": 0.0,
    "shap_values": {...},
    "features": {...}
  },
  "context": "string",          // Contexte économique (optionnel)
  "target_audience": "string"   // Public cible (optionnel)
}
```

#### Réponse

**Status: 200 OK**
```json
{
  "economic_interpretation": "string",  // Interprétation pour économistes
  "policy_recommendations": "string",   // Recommandations politiques
  "risk_assessment": "string",         // Évaluation des risques
  "key_drivers": [                     // Facteurs clés
    {
      "factor": "string",
      "impact": 0.0,
      "interpretation": "string"
    }
  ]
}
```

#### Exemple

```bash
curl -X POST "http://localhost:8008/api/forecast/inflation/interpret" \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_data": {
      "predicted_inflation": 2.5,
      "shap_values": {
        "oil_price": 0.8,
        "exchange_rate": -0.3,
        "money_supply": 0.5
      }
    },
    "context": "Analyse pour politique monétaire BCEAO",
    "target_audience": "Comité de politique monétaire"
  }'
```

**Réponse :**
```json
{
  "economic_interpretation": "La prévision d'inflation de 2,5% pour la période indique une pression inflationniste modérée. L'analyse SHAP révèle que l'augmentation des prix du pétrole (+0,8 point) constitue le principal facteur haussier...",
  "policy_recommendations": "Face à cette dynamique, la BCEAO pourrait envisager un maintien de sa politique monétaire actuelle. Toutefois, une vigilance particulière s'impose concernant l'évolution des prix énergétiques...",
  "risk_assessment": "Risque modéré. La stabilité du taux de change (-0,3 point) joue un rôle stabilisateur, mais la volatilité des prix du pétrole représente un facteur de risque significatif.",
  "key_drivers": [
    {
      "factor": "Prix du pétrole",
      "impact": 0.8,
      "interpretation": "Impact haussier significatif dû à la hausse des cours mondiaux"
    },
    {
      "factor": "Taux de change",
      "impact": -0.3,
      "interpretation": "Impact baissier grâce à l'appréciation du FCFA"
    },
    {
      "factor": "Masse monétaire",
      "impact": 0.5,
      "interpretation": "Pression modérée liée à l'expansion du crédit"
    }
  ]
}
```

---

## 4. Administration Endpoints

### `POST /api/index-queries`

Indexer ou réindexer les exemples de requêtes SQL dans ChromaDB.

#### Requête

**Body (optionnel):**
```json
{
  "queries": [                  // Exemples personnalisés (optionnel)
    {
      "question": "string",     // Question en français
      "sql": "string"           // Requête SQL correspondante
    }
  ]
}
```

Si aucun body n'est fourni, les exemples par défaut de `examples.json` sont indexés.

#### Réponse

**Status: 200 OK**
```json
{
  "status": "success",
  "indexed_count": 0            // Nombre d'exemples indexés
}
```

#### Exemples

**Indexation par défaut :**

```bash
curl -X POST "http://localhost:8008/api/index-queries"
```

**Réponse :**
```json
{
  "status": "success",
  "indexed_count": 3
}
```

**Indexation personnalisée :**

```bash
curl -X POST "http://localhost:8008/api/index-queries" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      {
        "question": "PIB moyen UEMOA 2023",
        "sql": "SELECT AVG(valeur) FROM indicateurs WHERE zone='\''UEMOA'\'' AND indicateur='\''PIB'\'' AND annee=2023"
      },
      {
        "question": "Inflation Sénégal 2023",
        "sql": "SELECT valeur FROM indicateurs WHERE pays='\''Sénégal'\'' AND indicateur='\''Inflation'\'' AND annee=2023"
      }
    ]
  }'
```

---

### `POST /api/pull-model`

Télécharger ou mettre à jour un modèle LLM Ollama.

#### Requête

**Body (optionnel):**
```json
{
  "model": "string"             // Nom du modèle Ollama (optionnel)
}
```

Si aucun modèle n'est spécifié, le modèle configuré dans `.env` est utilisé.

#### Réponse

**Status: 200 OK**
```json
{
  "status": "success" | "error",
  "model": "string",            // Modèle téléchargé
  "message": "string"           // Message de statut
}
```

#### Exemples

**Télécharger le modèle par défaut :**

```bash
curl -X POST "http://localhost:8008/api/pull-model"
```

**Télécharger un modèle spécifique :**

```bash
curl -X POST "http://localhost:8008/api/pull-model" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2:13b"
  }'
```

**Réponse :**
```json
{
  "status": "success",
  "model": "llama2:13b",
  "message": "Modèle llama2:13b téléchargé avec succès"
}
```

---

## 5. Modèles de Données

### QuestionRequest

```typescript
{
  question: string  // Requis
}
```

### AnswerResponse

```typescript
{
  answer: string,
  sql_query: string,
  result_data: Array<Record<string, any>>,
  metadata?: {
    execution_time?: string,
    rows_returned?: number,
    similar_examples_found?: number
  }
}
```

### ForecastNarrativeRequest

```typescript
{
  indicator: string,           // Requis
  period: string,              // Requis
  country?: string,            // Optionnel
  additional_context?: string  // Optionnel
}
```

### ForecastNarrativeResponse

```typescript
{
  narrative: string,
  summary_stats: {
    mean?: number,
    median?: number,
    std?: number,
    min?: number,
    max?: number
  }
}
```

### IndexingRequest

```typescript
{
  queries?: Array<{
    question: string,
    sql: string
  }>
}
```

### IndexingResponse

```typescript
{
  status: "success" | "error",
  indexed_count: number
}
```

### PullModelRequest

```typescript
{
  model?: string  // Optionnel
}
```

### PullModelResponse

```typescript
{
  status: "success" | "error",
  model: string,
  message: string
}
```

---

## 6. Codes d'Erreur

### Codes HTTP

| Code | Signification | Description |
|------|---------------|-------------|
| `200` | OK | Requête réussie |
| `400` | Bad Request | Paramètres manquants ou invalides |
| `500` | Internal Server Error | Erreur serveur (SQL, LLM, DB) |
| `503` | Service Unavailable | Service temporairement indisponible |

### Format des Erreurs

```json
{
  "detail": "Description de l'erreur"
}
```

### Exemples d'Erreurs

**400 - Question manquante :**
```json
{
  "detail": "Le champ 'question' est requis"
}
```

**500 - Erreur SQL :**
```json
{
  "detail": "Erreur lors de l'exécution de la requête SQL"
}
```

**500 - LLM indisponible :**
```json
{
  "detail": "Le service LLM n'est pas accessible"
}
```

---

## 7. Limites et Quotas

| Limite | Valeur |
|--------|--------|
| Taille max requête | 1 MB |
| Timeout requête | 30 secondes |
| Longueur max question | 500 caractères |
| Résultats max par requête | 1000 lignes |

---

## 8. Documentation Interactive

### Swagger UI

Accédez à la documentation interactive : **http://localhost:8008/docs**

### ReDoc

Documentation alternative : **http://localhost:8008/redoc**

### OpenAPI Schema

Schéma OpenAPI : **http://localhost:8008/openapi.json**

---

## 9. Exemples d'Intégration

### Python

```python
import requests

class Text2SQLClient:
    def __init__(self, base_url="http://localhost:8008"):
        self.base_url = base_url
    
    def ask(self, question: str):
        return requests.post(
            f"{self.base_url}/api/ask",
            json={"question": question}
        ).json()
    
    def forecast_narrative(self, indicator: str, period: str, country: str = None):
        return requests.post(
            f"{self.base_url}/api/forecast/narrative",
            json={
                "indicator": indicator,
                "period": period,
                "country": country
            }
        ).json()

# Utilisation
client = Text2SQLClient()
result = client.ask("Quel est le PIB du Sénégal?")
print(result['answer'])
```

### JavaScript/TypeScript

```typescript
class Text2SQLClient {
  constructor(private baseUrl: string = 'http://localhost:8008') {}
  
  async ask(question: string): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });
    return response.json();
  }
}

// Utilisation
const client = new Text2SQLClient();
const result = await client.ask('Quel est le PIB du Sénégal?');
console.log(result.answer);
```

---

## 10. Support

Pour toute question ou problème :

- 📖 [Guide Utilisateur](GUIDE_UTILISATEUR.md)
- 🏗️ [Architecture](ARCHITECTURE_DIAGRAM.md)
- 🐛 [Issues GitHub](https://github.com/Pheonix64/text2sql-project/issues)

---

**[⬆ Retour en haut](#-api-reference---text-to-sql)**
