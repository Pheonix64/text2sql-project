# 📖 Guide Utilisateur - Text-to-SQL API

Ce guide vous accompagne pas à pas pour utiliser efficacement l'API Text-to-SQL.

---

## 📋 Table des Matières

1. [Introduction](#1-introduction)
2. [Premiers Pas](#2-premiers-pas)
3. [Utilisation Text-to-SQL](#3-utilisation-text-to-sql)
4. [Analyse de Prévisions](#4-analyse-de-prévisions)
5. [Cas d'Usage Avancés](#5-cas-dusage-avancés)
6. [Bonnes Pratiques](#6-bonnes-pratiques)
7. [Résolution de Problèmes](#7-résolution-de-problèmes)

---

## 1. Introduction

### Qu'est-ce que Text-to-SQL ?

Text-to-SQL permet d'interroger une base de données en **langage naturel** au lieu d'écrire du SQL. 

**Exemple :**
- ❌ Ancien : `SELECT AVG(valeur) FROM indicateurs WHERE indicateur='PIB' AND annee >= 2020`
- ✅ Nouveau : "Quel est le PIB moyen depuis 2020 ?"

### Comment ça marche ?

```
Question en français
        ↓
   Recherche d'exemples similaires (ChromaDB)
        ↓
   Génération SQL par LLM (Ollama/Mistral)
        ↓
   Validation de la requête (SQLGlot)
        ↓
   Exécution sécurisée (PostgreSQL)
        ↓
   Analyse des résultats par LLM
        ↓
   Réponse en langage naturel
```

---

## 2. Premiers Pas

### 2.1 Vérifier que l'API fonctionne

```bash
curl http://localhost:8008/health
```

**Réponse attendue :**
```json
{"status": "ok"}
```

### 2.2 Accéder à la documentation interactive

Ouvrez votre navigateur : **http://localhost:8008/docs**

Vous verrez l'interface **Swagger UI** avec tous les endpoints disponibles.

### 2.3 Première requête simple

**Interface Swagger :**
1. Cliquez sur `POST /api/ask`
2. Cliquez sur "Try it out"
3. Entrez :
   ```json
   {
     "question": "Bonjour"
   }
   ```
4. Cliquez sur "Execute"

**Ligne de commande :**
```bash
curl -X POST "http://localhost:8008/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Bonjour"}'
```

---

## 3. Utilisation Text-to-SQL

### 3.1 Endpoint Principal : `/api/ask`

C'est l'endpoint le plus important pour poser des questions.

#### Structure de la Requête

```json
{
  "question": "Votre question en français"
}
```

#### Structure de la Réponse

```json
{
  "answer": "Réponse en langage naturel",
  "sql_query": "SELECT ... FROM ... WHERE ...",
  "result_data": [...],
  "metadata": {
    "execution_time": "1.5s",
    "rows_returned": 10,
    "similar_examples_found": 3
  }
}
```

### 3.2 Exemples de Questions

#### Question Simple

**Requête :**
```bash
curl -X POST "http://localhost:8008/api/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quelle est la population de la France en 2023?"
  }'
```

**Réponse :**
```json
{
  "answer": "La population de la France en 2023 est de 67,8 millions d'habitants.",
  "sql_query": "SELECT pays, annee, valeur FROM indicateurs WHERE pays='France' AND indicateur='Population' AND annee=2023",
  "result_data": [
    {
      "pays": "France",
      "annee": 2023,
      "valeur": 67800000
    }
  ],
  "metadata": {
    "execution_time": "0.8s",
    "rows_returned": 1
  }
}
```

#### Question avec Agrégation

**Requête :**
```json
{
  "question": "Quel est le PIB moyen des pays de l'UEMOA en 2023?"
}
```

**SQL Généré :**
```sql
SELECT AVG(valeur) as pib_moyen 
FROM indicateurs 
WHERE indicateur='PIB' 
  AND annee=2023 
  AND pays IN ('Bénin', 'Burkina Faso', 'Côte d''Ivoire', 'Guinée-Bissau', 'Mali', 'Niger', 'Sénégal', 'Togo')
```

#### Question Temporelle

**Requête :**
```json
{
  "question": "Quelle est l'évolution de l'inflation au Sénégal depuis 2020?"
}
```

**SQL Généré :**
```sql
SELECT annee, valeur 
FROM indicateurs 
WHERE pays='Sénégal' 
  AND indicateur='Inflation' 
  AND annee >= 2020 
ORDER BY annee
```

#### Question Comparative

**Requête :**
```json
{
  "question": "Comparer le taux de croissance du PIB entre le Bénin et le Togo en 2023"
}
```

### 3.3 Exemples Python

#### Script Basique

```python
import requests
import json

API_URL = "http://localhost:8008"

def ask_question(question: str):
    """Poser une question à l'API"""
    response = requests.post(
        f"{API_URL}/api/ask",
        json={"question": question}
    )
    return response.json()

# Exemple d'utilisation
result = ask_question("Quel est le PIB du Sénégal en 2023?")

print(f"Question: {result.get('question', 'N/A')}")
print(f"Réponse: {result['answer']}")
print(f"SQL: {result['sql_query']}")
print(f"Données: {json.dumps(result['result_data'], indent=2)}")
```

#### Script avec Gestion d'Erreurs

```python
import requests
from typing import Optional, Dict, Any

class Text2SQLClient:
    def __init__(self, base_url: str = "http://localhost:8008"):
        self.base_url = base_url
        
    def ask(self, question: str) -> Optional[Dict[str, Any]]:
        """Poser une question avec gestion d'erreurs"""
        try:
            response = requests.post(
                f"{self.base_url}/api/ask",
                json={"question": question},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            print("⏱️ Timeout - La requête a pris trop de temps")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur: {e}")
            return None
    
    def health_check(self) -> bool:
        """Vérifier si l'API est accessible"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

# Utilisation
client = Text2SQLClient()

if client.health_check():
    print("✅ API accessible")
    result = client.ask("Quel est le taux d'inflation moyen en 2023?")
    if result:
        print(f"Réponse: {result['answer']}")
else:
    print("❌ API non accessible")
```

#### Requêtes Multiples

```python
questions = [
    "Quel est le PIB du Sénégal?",
    "Quelle est l'évolution de l'inflation depuis 2020?",
    "Comparer les taux de croissance des pays de l'UEMOA"
]

for question in questions:
    print(f"\n📊 Question: {question}")
    result = client.ask(question)
    if result:
        print(f"✅ {result['answer']}")
        print(f"   SQL: {result['sql_query']}")
    print("-" * 80)
```

---

## 4. Analyse de Prévisions

### 4.0 Export des Données en CSV

**Nouveau** : Vous pouvez maintenant télécharger les données brutes de vos questions au format CSV.

Endpoint : `GET /api/export/csv/{query_id}`

**Comment ça marche :**

1. Lorsque vous posez une question via `/api/ask`, la réponse inclut un `query_id`
2. Utilisez ce `query_id` pour télécharger les données en CSV
3. Les données sont disponibles pendant 30 minutes

**Exemple complet :**

```python
import requests

# 1. Poser une question
response = requests.post(
    "http://localhost:8008/api/ask",
    json={"question": "Quelle est l'évolution du PIB entre 2015 et 2020?"}
)

result = response.json()
print(f"Réponse : {result['answer']}")
print(f"Query ID : {result['query_id']}")

# 2. Télécharger le CSV
if result.get('query_id'):
    csv_url = f"http://localhost:8008/api/export/csv/{result['query_id']}"
    csv_response = requests.get(csv_url)
    
    # Sauvegarder le fichier
    with open("donnees_pib.csv", "wb") as f:
        f.write(csv_response.content)
    
    print("✅ Données exportées dans donnees_pib.csv")
```

**Via le navigateur :**

Après avoir obtenu le `query_id`, ouvrez simplement :
```
http://localhost:8008/api/export/csv/VOTRE_QUERY_ID
```

Le fichier CSV se téléchargera automatiquement.

**Format du CSV :**
- Encodage UTF-8 avec BOM (compatible Excel)
- En-têtes de colonnes inclus
- Nom du fichier : `donnees_{query_id}.csv`

---

### 4.1 Génération de Narration Économique

Endpoint : `POST /api/forecast/narrative`

**Utilisation :**
```bash
curl -X POST "http://localhost:8008/api/forecast/narrative" \
  -H "Content-Type: application/json" \
  -d '{
    "indicator": "PIB",
    "period": "2024-Q1",
    "country": "Sénégal"
  }'
```

**Réponse :**
```json
{
  "narrative": "Au premier trimestre 2024, le PIB du Sénégal a connu une croissance de 3.5%...",
  "summary_stats": {
    "mean": 3.5,
    "median": 3.4,
    "std": 0.2
  }
}
```

### 4.2 Interprétation de Prédictions d'Inflation

Endpoint : `POST /api/forecast/inflation/interpret`

**Cas d'usage :** Comprendre les facteurs qui influencent les prévisions d'inflation

**Configuration du Timeout :**
Le timeout pour cet endpoint est configurable via la variable `LLM_TIMEOUT_INFLATION` (défaut: 120 secondes).
Cette valeur peut être ajustée dans le fichier `.env` ou `config.py` pour des analyses plus complexes.

**Exemple :**
```python
import requests

# Données de prédiction avec valeurs SHAP
prediction_data = {
    "predicted_inflation": 2.5,
    "shap_values": {
        "oil_price": 0.8,
        "exchange_rate": -0.3,
        "money_supply": 0.5
    },
    "base_value": 2.0
}

response = requests.post(
    "http://localhost:8008/api/forecast/inflation/interpret",
    json={
        "prediction_data": prediction_data,
        "context": "Analyse pour politique monétaire"
    },
    timeout=150  # Timeout client (recommandé > LLM_TIMEOUT_INFLATION)
)

interpretation = response.json()
print(interpretation['economic_interpretation'])
```

---

## 5. Cas d'Usage Avancés

### 5.1 Réindexation des Exemples SQL

Si vous ajoutez de nouveaux exemples de requêtes, réindexez-les :

```bash
curl -X POST "http://localhost:8008/api/index-queries"
```

**Avec des exemples personnalisés :**
```bash
curl -X POST "http://localhost:8008/api/index-queries" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      {
        "question": "Inflation moyenne UEMOA",
        "sql": "SELECT AVG(valeur) FROM indicateurs WHERE zone='\''UEMOA'\'' AND indicateur='\''Inflation'\''"
      }
    ]
  }'
```

### 5.2 Téléchargement de Modèles LLM

Télécharger un nouveau modèle Ollama :

```bash
curl -X POST "http://localhost:8008/api/pull-model" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama2:13b"}'
```

### 5.3 Interface Web Simple (HTML/JavaScript)

```html
<!DOCTYPE html>
<html>
<head>
    <title>Text-to-SQL Interface</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; }
        #question { width: 100%; padding: 10px; font-size: 16px; }
        #result { margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 5px; }
        button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>💬 Text-to-SQL Interface</h1>
    
    <input type="text" id="question" placeholder="Posez votre question...">
    <button onclick="askQuestion()">Envoyer</button>
    
    <div id="result"></div>

    <script>
        async function askQuestion() {
            const question = document.getElementById('question').value;
            const resultDiv = document.getElementById('result');
            
            resultDiv.innerHTML = '⏳ Traitement en cours...';
            
            try {
                const response = await fetch('http://localhost:8008/api/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question})
                });
                
                const data = await response.json();
                
                resultDiv.innerHTML = `
                    <h3>✅ Réponse :</h3>
                    <p>${data.answer}</p>
                    <h4>📝 SQL Généré :</h4>
                    <code>${data.sql_query}</code>
                    <h4>📊 Données :</h4>
                    <pre>${JSON.stringify(data.result_data, null, 2)}</pre>
                `;
            } catch (error) {
                resultDiv.innerHTML = `❌ Erreur: ${error.message}`;
            }
        }
    </script>
</body>
</html>
```

---

## 6. Bonnes Pratiques

### 6.1 Formulation des Questions

✅ **BON :**
- "Quel est le PIB du Sénégal en 2023 ?"
- "Évolution de l'inflation au Burkina Faso depuis 2020"
- "Comparer le taux de croissance entre le Bénin et le Togo"

❌ **À ÉVITER :**
- "PIB" (trop vague)
- "Donne-moi toutes les données" (trop large)
- Questions avec fautes de frappe importantes

### 6.2 Performance

- Les premières requêtes peuvent être plus lentes (chargement des modèles)
- Utilisez des questions spécifiques pour de meilleures performances
- Évitez les requêtes retournant des milliers de lignes

### 6.3 Sécurité

- L'utilisateur SQL utilisé est **en lecture seule**
- Les requêtes sont **validées** avant exécution
- Pas de risque d'injection SQL grâce à la validation

---

## 7. Résolution de Problèmes

### Problème : "Connection Error"

**Cause :** L'API n'est pas accessible

**Solution :**
```bash
# Vérifier que l'API tourne
docker-compose ps

# Vérifier les logs
docker logs api-fastapi

# Redémarrer si nécessaire
docker-compose restart api-fastapi
```

### Problème : "Timeout"

**Cause :** La requête prend trop de temps

**Solutions :**
- Simplifier la question
- Vérifier que le modèle LLM est chargé
- Augmenter le timeout dans votre code client

### Problème : "Mauvaise réponse SQL"

**Cause :** Le modèle n'a pas compris la question

**Solutions :**
- Reformuler la question plus clairement
- Ajouter des exemples similaires via `/api/index-queries`
- Vérifier que la question correspond aux données disponibles

### Problème : "Empty Result"

**Cause :** La requête SQL est valide mais ne retourne aucun résultat

**Solutions :**
- Vérifier que les données existent dans la base
- Ajuster les critères de la question (années, pays, etc.)

---

## 📞 Support

Pour plus d'aide :
- 📚 [Documentation API](API_REFERENCE.md)
- 🏗️ [Architecture](ARCHITECTURE_DIAGRAM.md)
- 🔧 [Configuration](../README.md#configuration)

---

**[⬆ Retour en haut](#-guide-utilisateur---text-to-sql-api)**
