# 🔌 API Reference - Text-to-SQL (Version corrigée)

Documentation complète et rigoureuse de tous les endpoints de l'API Text-to-SQL, basée sur le code source réel.

**Base URL :** `http://localhost:8008`

**Note importante** : Cette documentation a été générée à partir du code source (`api/app/routers/`, `api/app/models/schemas.py`) pour garantir l'exactitude.

---

## 📋 Table des Matières

1. [Health Check](#1-health-check)
2. [Text-to-SQL Endpoints](#2-text-to-sql-endpoints)
3. [Forecast Endpoints](#3-forecast-endpoints)
4. [Administration Endpoints](#4-administration-endpoints)
5. [Modèles de Données (Schemas Pydantic)](#5-modèles-de-données)
6. [Codes d'Erreur](#6-codes-derreur)
7. [Table de la base de données](#7-table-de-la-base-de-données)

---

## 1. Health Check

### `GET /health`

Vérifier l'état de santé de l'API.

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

**Code source** : `api/app/main.py`, ligne 35

---

## 2. Text-to-SQL Endpoints

### `POST /api/ask`

**Endpoint principal** pour poser une question en langage naturel et obtenir une réponse basée sur les données économiques UEMOA.

#### Description

Convertit une question en français en requête SQL PostgreSQL, l'exécute sur la table `indicateurs_economiques_uemoa`, et retourne une réponse en langage naturel.

**Pipeline** :
1. Recherche sémantique d'exemples similaires (ChromaDB)
2. Génération SQL par LLM (Mistral via Ollama)
3. Validation SQL (SQLGlot + regex sécurité)
4. Exécution SQL (PostgreSQL en lecture seule)
5. Analyse des résultats par LLM
6. Réponse en langage naturel

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

**Schema Pydantic** : `QuestionRequest` (`api/app/models/schemas.py`, lignes 7-9)

#### Réponse

**Status: 200 OK**
```json
{
  "answer": "string",               // Réponse en langage naturel
  "generated_sql": "string | null", // Requête SQL générée (peut être null)
  "sql_result": "string | null"     // Résultats SQL stringifiés (peut être null)
}
```

**Schema Pydantic** : `AnswerResponse` (`api/app/models/schemas.py`, lignes 11-15)

#### Exemples

**Exemple 1 : Question simple sur l'inflation**

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
  "answer": "En 2021, l'UEMOA a enregistré un taux d'inflation moyen de 3,9%, dépassant légèrement l'objectif de stabilité des prix de la BCEAO fixé à 3%. Cette hausse s'explique par les tensions sur les prix des denrées alimentaires et de l'énergie dans un contexte de reprise post-COVID-19.",
  "generated_sql": "SELECT taux_inflation_moyen_annuel_ipc_pct FROM indicateurs_economiques_uemoa WHERE date = '2021-01-01';",
  "sql_result": "[{\"taux_inflation_moyen_annuel_ipc_pct\": 3.9}]"
}
```

**Exemple 2 : Question avec agrégation**

```bash
curl -X POST "http://localhost:8008/api/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quelle est la croissance moyenne du PIB entre 2015 et 2022 ?"
  }'
```

**Réponse :**
```json
{
  "answer": "Entre 2015 et 2022, l'UEMOA a enregistré une croissance moyenne du PIB de 5,8%. Cette performance témoigne de la résilience des économies de la zone malgré les chocs successifs (crise énergétique, pandémie COVID-19).",
  "generated_sql": "SELECT AVG(taux_croissance_reel_pib_pct) AS avg_croissance FROM indicateurs_economiques_uemoa WHERE date BETWEEN '2015-01-01' AND '2022-12-31';",
  "sql_result": "[{\"avg_croissance\": 5.8}]"
}
```

**Exemple 3 : Évolution temporelle**

```bash
curl -X POST "http://localhost:8008/api/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Comment a évolué la dette publique entre 2018 et 2022 ?"
  }'
```

#### Codes de Statut

| Code | Description |
|------|-------------|
| `200` | Succès - Réponse générée |
| `400` | Requête invalide (question manquante ou vide) |
| `500` | Erreur serveur (génération SQL échouée, exécution impossible, etc.) |

#### Notes importantes

- La table interrogée est **`indicateurs_economiques_uemoa`**, pas `indicateurs`
- Les dates sont au format `'AAAA-01-01'` (ex: `'2021-01-01'`)
- L'utilisateur SQL est en **lecture seule** (`llm_user`)
- Seules les requêtes `SELECT` sont autorisées (validation multi-niveaux)

**Code source** : `api/app/routers/conversation.py`, lignes 14-26

---

### `POST /api/index-queries`

Réindexer les exemples de requêtes SQL de référence dans ChromaDB.

#### Description

Permet de recharger les exemples SQL stockés dans `docs/examples.json` ou d'indexer des exemples personnalisés. Ces exemples sont utilisés pour la recherche sémantique (Few-Shot Learning).

#### Requête

**Body (optionnel):**
```json
{
  "queries": [                  // Exemples personnalisés (optionnel)
    "string"                    // Requête SQL brute
  ]
}
```

**Schema Pydantic** : `IndexingRequest` (`api/app/models/schemas.py`, lignes 17-19)

Si aucun body n'est fourni, les exemples par défaut de `docs/examples.json` sont indexés.

#### Réponse

**Status: 200 OK**
```json
{
  "status": "success",
  "indexed_count": 0            // Nombre d'exemples indexés
}
```

**Schema Pydantic** : `IndexingResponse` (`api/app/models/schemas.py`, lignes 21-24)

#### Exemples

**Indexation par défaut (exemples.json) :**

```bash
curl -X POST "http://localhost:8008/api/index-queries"
```

**Réponse :**
```json
{
  "status": "success",
  "indexed_count": 39
}
```

**Indexation personnalisée :**

```bash
curl -X POST "http://localhost:8008/api/index-queries" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "SELECT taux_inflation_moyen_annuel_ipc_pct FROM indicateurs_economiques_uemoa WHERE date = '\''2023-01-01'\'';",
      "SELECT AVG(pib_nominal_milliards_fcfa) FROM indicateurs_economiques_uemoa WHERE date >= '\''2020-01-01'\'';"
    ]
  }'
```

**Code source** : `api/app/routers/conversation.py`, lignes 29-40

---

### `POST /api/pull-model`

Télécharger ou mettre à jour un modèle LLM Ollama.

#### Description

Déclenche le téléchargement d'un modèle Ollama (ex: `mistral:7b`, `llama2:13b`). Utile lors de la première installation ou pour changer de modèle.

#### Requête

**Body (optionnel):**
```json
{
  "model": "string"             // Nom du modèle Ollama (optionnel)
}
```

**Schema Pydantic** : `PullModelRequest` (`api/app/models/schemas.py`, lignes 26-28)

Si aucun modèle n'est spécifié, le modèle configuré dans `.env` (`LLM_MODEL`) est utilisé.

#### Réponse

**Status: 200 OK**
```json
{
  "status": "success" | "error",
  "model": "string",            // Modèle téléchargé
  "message": "string"           // Message de statut (en cas d'erreur)
}
```

**Schema Pydantic** : `PullModelResponse` (`api/app/models/schemas.py`, lignes 30-34)

#### Exemples

**Télécharger le modèle par défaut :**

```bash
curl -X POST "http://localhost:8008/api/pull-model"
```

**Réponse :**
```json
{
  "status": "success",
  "model": "mistral:7b",
  "message": null
}
```

**Télécharger un modèle spécifique :**

```bash
curl -X POST "http://localhost:8008/api/pull-model" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2:13b"
  }'
```

**Code source** : `api/app/routers/conversation.py`, lignes 43-56

---

## 3. Forecast Endpoints

### `POST /api/forecast/narrative`

Générer une narration économique à partir de séries temporelles de prévisions.

#### Description

Transforme des données de prévision (série temporelle avec valeurs numériques) en un récit économique en français adapté à différents publics (professionnels, pédagogiques, neutres).

**Cas d'usage** :
- Narration pour prévisions de liquidité bancaire
- Analyse narrative de prévisions d'inflation
- Synthèse de projections macroéconomiques

#### Requête

**Body:**
```json
{
  "target": "liquidity" | "inflation" | "other",  // Type de prévision (requis)
  "horizon": "string",                            // Horizon temporel (optionnel, ex: "3 mois", "2024-Q2")
  "unit": "string",                               // Unité de mesure (optionnel, ex: "milliards FCFA", "%")
  "series": [                                     // Série temporelle (requis)
    {
      "date": "string",                           // Date ISO ou label (optionnel)
      "value": 0.0                                // Valeur prévue (requis)
    }
  ],
  "lower": [0.0],                                 // Bornes inférieures IC à 95% (optionnel, aligné sur series)
  "upper": [0.0],                                 // Bornes supérieures IC à 95% (optionnel, aligné sur series)
  "language": "fr" | "en",                        // Langue (défaut: "fr")
  "tone": "professionnel" | "neutre" | "pédagogique",  // Ton (défaut: "professionnel")
  "title": "string"                               // Titre optionnel
}
```

**Schema Pydantic** : `ForecastNarrativeRequest` (`api/app/models/schemas.py`, lignes 37-47)

#### Réponse

**Status: 200 OK**
```json
{
  "narrative": "string",        // Narration générée en français
  "summary_stats": {            // Statistiques résumées
    "count": 0,                 // Nombre de points de données
    "min": 0.0,                 // Valeur minimale
    "max": 0.0,                 // Valeur maximale
    "mean": 0.0,                // Moyenne arithmétique
    "start_value": 0.0,         // Valeur initiale
    "end_value": 0.0,           // Valeur finale
    "start_date": "string",     // Date de début (optionnel)
    "end_date": "string"        // Date de fin (optionnel)
  }
}
```

**Schema Pydantic** : `ForecastNarrativeResponse` (`api/app/models/schemas.py`, lignes 49-51)

#### Exemple

```bash
curl -X POST "http://localhost:8008/api/forecast/narrative" \
  -H "Content-Type: application/json" \
  -d '{
    "target": "inflation",
    "horizon": "Premier trimestre 2024",
    "unit": "pourcentage (%)",
    "series": [
      {"date": "2024-01", "value": 2.3},
      {"date": "2024-02", "value": 2.5},
      {"date": "2024-03", "value": 2.7}
    ],
    "lower": [2.0, 2.2, 2.4],
    "upper": [2.6, 2.8, 3.0],
    "language": "fr",
    "tone": "professionnel",
    "title": "Prévisions d'\''inflation UEMOA - T1 2024"
  }'
```

**Réponse :**
```json
{
  "narrative": "Les prévisions d'inflation pour le premier trimestre 2024 montrent une tendance haussière modérée, avec une progression de 2,3% en janvier à 2,7% en mars. Cette trajectoire demeure compatible avec l'objectif de stabilité des prix de la BCEAO (1-3%), bien que proche de la limite supérieure. L'intervalle de confiance suggère une incertitude limitée, les valeurs pouvant osciller entre 2,0% et 3,0%. Cette dynamique inflationniste reflète les pressions persistantes sur les prix alimentaires et énergétiques dans la zone UEMOA.",
  "summary_stats": {
    "count": 3,
    "min": 2.3,
    "max": 2.7,
    "mean": 2.5,
    "start_value": 2.3,
    "end_value": 2.7,
    "start_date": "2024-01",
    "end_date": "2024-03"
  }
}
```

**Code source** : `api/app/routers/forecast.py`, lignes 14-26

---

### `POST /api/forecast/inflation/prediction`

Recevoir et formater des prédictions d'inflation avec explicabilité SHAP.

#### Description

Endpoint destiné à recevoir les prédictions du **modèle d'inflation externe** (probablement un modèle ML Python) avec les valeurs SHAP associées. Le système valide et formate ces données selon le schéma standardisé `InflationPredictionResponse`.

**Utilisation typique** : Ce endpoint est appelé par le service de prédiction ML après génération des forecasts mensuels.

#### Requête

**Body:**
```json
{
  "predictions": {                          // Prédictions par période (requis)
    "2024-01": 2.5,
    "2024-02": 2.7,
    "2024-03": 2.9
  },
  "global_shap_importance": {               // Importance globale des features (requis)
    "taux_change": 0.35,
    "prix_petrole": 0.45,
    "masse_monetaire": 0.15,
    "prix_alimentation": 0.05
  },
  "shap_summary_details": {                 // Métadonnées du modèle (requis)
    "model_version": "1.2.0",
    "training_period": "2010-2023",
    "accuracy_metrics": {...}
  },
  "individual_shap_explanations": {         // Explications SHAP par observation temporelle (requis)
    "2024-01": {
      "taux_change": 0.4,
      "prix_petrole": 0.8,
      "masse_monetaire": -0.2,
      "prix_alimentation": 0.3
    },
    "2024-02": {
      "taux_change": 0.5,
      "prix_petrole": 0.9,
      "masse_monetaire": -0.1,
      "prix_alimentation": 0.4
    }
  },
  "confidence_intervals": {                 // Intervalles de confiance (optionnel)
    "2024-01": {"lower": 2.2, "upper": 2.8}
  }
}
```

**Schema Pydantic** : `InflationPredictionResponse` (`api/app/models/schemas.py`, lignes 57-69)

#### Réponse

**Status: 200 OK**

Retourne les mêmes données après validation et formatage.

```json
{
  "predictions": {...},
  "global_shap_importance": {...},
  "shap_summary_details": {...},
  "individual_shap_explanations": {...},
  "confidence_intervals": {...}
}
```

#### Validation

La méthode `_validate_inflation_data()` vérifie :
- Les valeurs d'inflation sont numériques et dans une plage raisonnable (-10% à +50%)
- La présence de facteurs inflationnistes typiques (taux de change, prix pétrole, masse monétaire, alimentation)

**Code source** : 
- Endpoint : `api/app/routers/forecast.py`, lignes 29-43
- Validation : `api/app/services/query_orchestrator.py`, lignes 576-594

---

### `POST /api/forecast/inflation/interpret`

**Endpoint principal** pour interpréter les prédictions d'inflation SHAP à destination des économistes BCEAO.

#### Description

Traduit les résultats techniques SHAP en **analyses économiques détaillées** spécifiques à l'inflation et à la politique monétaire. Cet endpoint utilise un **prompt LLM spécialisé** (voir `PROMPTS_DOCUMENTATION.md` section 4) pour générer des interprétations adaptées au public cible.

**Cas d'usage** :
- Briefing mensuel du Comité de Politique Monétaire
- Rapports d'analyse inflation pour économistes
- Communication vulgarisée pour décideurs politiques

#### Paramètres de Configuration

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `LLM_TIMEOUT_INFLATION` | int | 120 | Timeout en secondes pour l'appel LLM (configurable dans `.env` ou `config.py`) |

**Note** : Le timeout est plus long pour cet endpoint car l'interprétation économique génère plus de texte et nécessite une analyse approfondie des facteurs SHAP.

#### Requête

**Body:**
```json
{
  "prediction_data": {                      // Données de prédiction SHAP (requis)
    "predictions": {...},
    "global_shap_importance": {...},
    "shap_summary_details": {...},
    "individual_shap_explanations": {...},
    "confidence_intervals": {...}
  },
  "analysis_language": "fr" | "en",         // Langue (défaut: "fr")
  "target_audience": "economist" | "analyst" | "policymaker" | "general",  // Public cible (défaut: "economist")
  "include_policy_recommendations": true,   // Inclure recommandations (défaut: true)
  "include_monetary_policy_analysis": true, // Analyse politique monétaire (défaut: true)
  "focus_on_bceao_mandate": true            // Focus mandat BCEAO (défaut: true)
}
```

**Schema Pydantic** : `InflationInterpretationRequest` (`api/app/models/schemas.py`, lignes 71-78)

#### Réponse

**Status: 200 OK**
```json
{
  "executive_summary": "string",                // Résumé exécutif sur les perspectives d'inflation
  "inflation_analysis": "string",               // Analyse détaillée des dynamiques inflationnistes
  "key_inflation_drivers": ["string"],          // Principaux facteurs identifiés par SHAP
  "price_stability_assessment": "string",       // Évaluation au regard de l'objectif de stabilité des prix
  "monetary_policy_recommendations": "string | null",  // Recommandations pour la BCEAO (si include_policy_recommendations=true)
  "inflation_risks": ["string"],                // Risques inflationnistes identifiés
  "model_confidence": "string",                 // Niveau de confiance du modèle de prévision
  "target_deviation_analysis": "string",        // Analyse des écarts par rapport à la cible d'inflation
  "external_factors_impact": "string"           // Impact des facteurs externes (pétrole, taux de change, etc.)
}
```

**Schema Pydantic** : `InflationInterpretationResponse` (`api/app/models/schemas.py`, lignes 80-92)

#### Exemple Complet

```bash
curl -X POST "http://localhost:8008/api/forecast/inflation/interpret" \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_data": {
      "predictions": {
        "2024-01": 2.3,
        "2024-02": 2.5,
        "2024-03": 2.7
      },
      "global_shap_importance": {
        "prix_petrole": 0.45,
        "taux_change": 0.30,
        "masse_monetaire": 0.15,
        "prix_alimentation": 0.10
      },
      "shap_summary_details": {
        "model_version": "1.0",
        "training_period": "2015-2023",
        "r2_score": 0.89
      },
      "individual_shap_explanations": {
        "2024-01": {
          "prix_petrole": 0.8,
          "taux_change": 0.4,
          "masse_monetaire": -0.2,
          "prix_alimentation": 0.3
        },
        "2024-02": {
          "prix_petrole": 0.9,
          "taux_change": 0.5,
          "masse_monetaire": -0.1,
          "prix_alimentation": 0.2
        },
        "2024-03": {
          "prix_petrole": 1.0,
          "taux_change": 0.6,
          "masse_monetaire": 0.0,
          "prix_alimentation": 0.1
        }
      },
      "confidence_intervals": {
        "2024-01": {"lower": 2.0, "upper": 2.6},
        "2024-02": {"lower": 2.2, "upper": 2.8},
        "2024-03": {"lower": 2.4, "upper": 3.0}
      }
    },
    "analysis_language": "fr",
    "target_audience": "economist",
    "include_policy_recommendations": true,
    "include_monetary_policy_analysis": true,
    "focus_on_bceao_mandate": true
  }'
```

**Réponse** (extrait) :

```json
{
  "executive_summary": "Les prévisions d'inflation pour le premier trimestre 2024 affichent une tendance haussière modérée, passant de 2,3% en janvier à 2,7% en mars, avec une moyenne de 2,5%. Cette trajectoire reste compatible avec l'objectif BCEAO de 1-3%, mais nécessite une vigilance accrue face à la dynamique des prix pétroliers.",
  
  "inflation_analysis": "Janvier 2024 : 2,3%. L'analyse SHAP révèle que les prix du pétrole contribuent à hauteur de +0,8 point de pourcentage (pp), constituant le principal moteur inflationniste. Le taux de change FCFA/USD ajoute +0,4 pp, reflétant les tensions sur les marchés des changes. La masse monétaire exerce un effet désinflationniste modéré (-0,2 pp), tandis que les prix alimentaires contribuent positivement (+0,3 pp).\n\nFévrier 2024 : 2,5%. La pression inflationniste s'intensifie avec une contribution pétrolière accrue (+0,9 pp) et un impact plus marqué du taux de change (+0,5 pp). La masse monétaire devient neutre (-0,1 pp), suggérant une transmission plus directe des chocs externes.\n\nMars 2024 : 2,7%. La tendance haussière se confirme avec une contribution pétrolière atteignant +1,0 pp. Le taux de change poursuit son impact inflationniste (+0,6 pp), tandis que la masse monétaire devient neutre (0,0 pp) et les prix alimentaires se stabilisent (+0,1 pp).",
  
  "key_inflation_drivers": [
    "Prix du pétrole (importance globale: 0,45) - Principal facteur inflationniste sur l'ensemble de la période avec une contribution croissante de +0,8 pp à +1,0 pp",
    "Taux de change FCFA/USD (importance: 0,30) - Pression inflationniste modérée mais persistante via le canal des importations (+0,4 pp à +0,6 pp)",
    "Masse monétaire M2 (importance: 0,15) - Effet désinflationniste en janvier (-0,2 pp) devenant neutre en mars (0,0 pp)",
    "Prix alimentaires (importance: 0,10) - Contribution positive mais décroissante (+0,3 pp à +0,1 pp)"
  ],
  
  "price_stability_assessment": "L'inflation moyenne prévue de 2,5% pour le trimestre s'inscrit dans la fourchette cible de la BCEAO (1-3%), bien que proche de la limite supérieure. La trajectoire haussière observée (de 2,3% à 2,7%) suggère un risque de dépassement au-delà du trimestre si les tensions pétrolières persistent. La confiance du modèle (R² = 0,89) est élevée, confortant la fiabilité de ces prévisions.",
  
  "monetary_policy_recommendations": "Dans le contexte actuel, la BCEAO devrait maintenir une posture de vigilance active :\n\n1. Taux directeur : Maintien du statu quo à court terme, l'inflation restant dans la fourchette cible. Toutefois, préparer un scénario de resserrement si l'inflation dépasse durablement 2,8%.\n\n2. Réserves obligatoires : Envisager une augmentation marginale (0,5-1 pp) pour absorber l'excès de liquidités si la masse monétaire redevient contributrice.\n\n3. Communication : Signaler clairement la volonté de la BCEAO de maintenir l'inflation sous contrôle, notamment via le canal du taux de change.\n\n4. Surveillance renforcée : Focus sur l'évolution des prix pétroliers internationaux et du taux de change FCFA/USD.\n\n5. Coordination régionale : Renforcer les mécanismes de stabilisation des prix alimentaires via les politiques budgétaires nationales.",
  
  "inflation_risks": [
    "Risque haussier majeur : Persistance ou amplification de la hausse des prix du pétrole (contribution déjà à +1,0 pp en mars)",
    "Risque modéré : Dépréciation continue du FCFA face au dollar, augmentant le coût des importations",
    "Risque limité : Expansion excessive du crédit bancaire (actuellement neutre mais à surveiller)",
    "Risque baissier : Stabilisation ou baisse des prix alimentaires dans la région"
  ],
  
  "model_confidence": "Élevé. Le coefficient de détermination R² de 0,89 indique que le modèle explique 89% de la variance de l'inflation observée sur la période d'entraînement (2015-2023). Les intervalles de confiance à 95% sont relativement étroits (±0,3 pp en moyenne), témoignant d'une précision satisfaisante des prévisions ponctuelles.",
  
  "target_deviation_analysis": "La cible d'inflation de la BCEAO est fixée entre 1% et 3% en glissement annuel. Les prévisions pour janvier (2,3%), février (2,5%) et mars (2,7%) s'inscrivent toutes dans cette fourchette, mais avec une proximité croissante de la limite supérieure. L'écart par rapport au point médian de la cible (2%) passe de +0,3 pp à +0,7 pp sur le trimestre. Si cette tendance se poursuit au-delà de mars, un dépassement du seuil de 3% pourrait survenir au deuxième trimestre, nécessitant potentiellement une réponse de politique monétaire.",
  
  "external_factors_impact": "Les facteurs externes dominent largement la dynamique inflationniste prévue :\n\n1. Prix du pétrole (facteur externe) : Contribution cumulée de 75% de l'importance globale. Reflète la dépendance énergétique de la zone UEMOA et la transmission rapide des chocs pétroliers via les prix des carburants et de l'électricité.\n\n2. Taux de change (facteur semi-externe) : Contribution de 30%. La parité fixe FCFA/EUR protège partiellement, mais l'exposition au dollar (via les importations hors zone euro) reste significative.\n\n3. Facteurs internes (masse monétaire, prix alimentaires) : Contribution résiduelle de 25%, suggérant une capacité limitée des autorités monétaires et budgétaires nationales à contrer les chocs externes à court terme."
}
```

**Code source** : 
- Endpoint : `api/app/routers/forecast.py`, lignes 46-60
- Génération interprétation : `api/app/services/query_orchestrator.py`, méthode `generate_inflation_interpretation()` (lignes 529-574)
- Construction du prompt : `api/app/services/query_orchestrator.py`, méthode `_build_inflation_interpretation_prompt()` (lignes 596-699)

---

## 4. Administration Endpoints

Voir section 2 pour `/api/index-queries` et `/api/pull-model`.

---

## 5. Modèles de Données

### Schemas Pydantic

Tous les schemas sont définis dans `api/app/models/schemas.py`.

#### Text-to-SQL Schemas

```python
class QuestionRequest(BaseModel):
    """Schéma pour la question de l'utilisateur."""
    question: str

class AnswerResponse(BaseModel):
    """Schéma pour la réponse finale."""
    answer: str
    generated_sql: str | None = None
    sql_result: str | None = None

class IndexingRequest(BaseModel):
    """Schéma pour la requête d'indexation manuelle."""
    queries: List[str]

class IndexingResponse(BaseModel):
    """Schéma pour la réponse de l'indexation."""
    status: str
    indexed_count: int

class PullModelRequest(BaseModel):
    """Schéma pour demander le téléchargement d'un modèle Ollama spécifique."""
    model: str | None = None

class PullModelResponse(BaseModel):
    """Schéma pour la réponse du téléchargement du modèle."""
    status: str
    model: str | None = None
    message: str | None = None
```

#### Forecast Schemas

```python
class ForecastPoint(BaseModel):
    date: Optional[str] = None  # ISO date ou label
    value: float

class SummaryStats(BaseModel):
    count: int
    min: float
    max: float
    mean: float
    start_value: float
    end_value: float
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class ForecastNarrativeRequest(BaseModel):
    target: Literal["liquidity", "inflation", "other"]
    horizon: Optional[str] = None
    unit: Optional[str] = None
    series: List[ForecastPoint]
    lower: Optional[List[float]] = None
    upper: Optional[List[float]] = None
    language: Literal["fr", "en"] = "fr"
    tone: Literal["professionnel", "neutre", "pédagogique"] = "professionnel"
    title: Optional[str] = None

class ForecastNarrativeResponse(BaseModel):
    narrative: str
    summary_stats: SummaryStats
```

#### Inflation SHAP Schemas

```python
class InflationPredictionResponse(BaseModel):
    """Schéma pour les réponses du modèle de prévision d'inflation avec explicabilité SHAP."""
    predictions: dict  # {"2024-Q1": 2.5, ...}
    global_shap_importance: dict  # {"taux_change": 0.35, ...}
    shap_summary_details: dict  # Métadonnées du modèle
    individual_shap_explanations: dict  # Explications SHAP par observation temporelle
    confidence_intervals: Optional[dict] = None  # Intervalles de confiance

class InflationInterpretationRequest(BaseModel):
    """Requête pour l'interprétation économique des prédictions d'inflation SHAP."""
    prediction_data: InflationPredictionResponse
    analysis_language: Literal["fr", "en"] = "fr"
    target_audience: Literal["economist", "analyst", "policymaker", "general"] = "economist"
    include_policy_recommendations: bool = True
    include_monetary_policy_analysis: bool = True
    focus_on_bceao_mandate: bool = True

class InflationInterpretationResponse(BaseModel):
    """Réponse contenant l'interprétation économique des prédictions d'inflation."""
    executive_summary: str
    inflation_analysis: str
    key_inflation_drivers: List[str]
    price_stability_assessment: str
    monetary_policy_recommendations: Optional[str] = None
    inflation_risks: List[str]
    model_confidence: str
    target_deviation_analysis: str
    external_factors_impact: str
```

---

## 6. Codes d'Erreur

### Codes HTTP Standards

| Code | Nom | Description | Exemple |
|------|-----|-------------|---------|
| `200` | OK | Succès | Réponse générée avec succès |
| `400` | Bad Request | Requête invalide | Question vide ou manquante |
| `500` | Internal Server Error | Erreur serveur | Échec génération SQL, exécution DB impossible |

### Messages d'Erreur Typiques

**Question vide :**
```json
{
  "detail": "Question is required and cannot be empty"
}
```

**Génération SQL échouée :**
```json
{
  "answer": "Je n'ai pas pu générer une requête SQL pertinente pour cette question. Pouvez-vous préciser la période, les colonnes ou la condition souhaitée ?",
  "generated_sql": "",
  "sql_result": null
}
```

**SQL non sécurisé (bloqué par validation) :**
```json
{
  "answer": "La requête SQL générée a été jugée non sécurisée et a été bloquée.",
  "generated_sql": "SELECT * FROM ...; DROP TABLE ...",
  "sql_result": null
}
```

**Exécution SQL échouée :**
```json
{
  "answer": "Une erreur est survenue lors de l'exécution ou de la formulation de la réponse.",
  "generated_sql": "SELECT ... FROM ...",
  "sql_result": null
}
```

---

## 7. Table de la base de données

### Table : `indicateurs_economiques_uemoa`

**Type** : Hypertable TimescaleDB (optimisée pour séries temporelles)

**Description** : Contient les principaux indicateurs macroéconomiques et financiers pour la zone UEMOA sur une base annuelle.

**Schema complet** (extrait de `postgres/init.sql`) :

```sql
CREATE TABLE indicateurs_economiques_uemoa (
    date DATE NOT NULL,  -- Format 'AAAA-01-01'
    
    -- PIB et croissance
    pib_nominal_milliards_fcfa REAL,
    poids_secteur_primaire_pct REAL,
    poids_secteur_secondaire_pct REAL,
    poids_secteur_tertiaire_pct REAL,
    taux_croissance_reel_pib_pct REAL,
    contribution_croissance_primaire REAL,
    contribution_croissance_secondaire REAL,
    contribution_croissance_tertiaire REAL,
    
    -- Épargne et investissement
    epargne_interieure_milliards_fcfa REAL,
    taux_epargne_interieure_pct REAL,
    taux_epargne_interieure_publique_pct REAL,
    investissement_milliards_fcfa REAL,
    taux_investissement_pct REAL,
    taux_investissement_public_pct REAL,
    
    -- Inflation
    taux_inflation_moyen_annuel_ipc_pct REAL,
    taux_inflation_glissement_annuel_pct REAL,
    
    -- Finances publiques
    recettes_totales_et_dons REAL,
    recettes_totales_hors_dons REAL,
    recettes_fiscales REAL,
    recettes_fiscales_pct_pib REAL,
    depenses_totales_et_prets_nets REAL,
    depenses_courantes REAL,
    investissements_sur_ressources_internes REAL,
    solde_primaire_base_sur_recettes_fiscales_pct REAL,
    solde_budgetaire_de_base REAL,
    solde_budgetaire_global_avec_dons REAL,
    solde_budgetaire_global_hors_dons REAL,
    
    -- Dette publique
    encours_de_la_dette REAL,
    encours_de_la_dette_pct_pib REAL,
    service_de_la_dette_regle REAL,
    service_de_la_dette_interets REAL,
    
    -- Balance commerciale
    exportations_biens_fob REAL,
    importations_biens_fob REAL,
    balance_des_biens REAL,
    
    -- Compte courant
    compte_transactions_courantes REAL,
    balance_courante_sur_pib_pct REAL,
    balance_courante_hors_dons_publics REAL,
    balance_courante_hors_dons_sur_pib_pct REAL,
    solde_global_apres_ajustement REAL,
    financement_exceptionnel REAL,
    degre_ouverture_pct REAL,
    
    -- Agrégats monétaires
    agregats_monnaie_actifs_exterieurs_nets REAL,
    agregats_monnaie_creances_interieures REAL,
    agregats_monnaie_creances_autres_secteurs REAL,
    agregats_monnaie_masse_monetaire_m2 REAL,
    actifs_exterieurs_nets_bceao_avoirs_officiels REAL,
    taux_couverture_emission_monetaire REAL
);

-- Index TimescaleDB
SELECT create_hypertable('indicateurs_economiques_uemoa', 'date');
```

**Colonnes les plus utilisées** :

| Colonne | Type | Description |
|---------|------|-------------|
| `date` | DATE | Date au format 'AAAA-01-01' (ex: '2021-01-01') |
| `pib_nominal_milliards_fcfa` | REAL | PIB nominal en milliards de FCFA |
| `taux_croissance_reel_pib_pct` | REAL | Taux de croissance annuel du PIB réel en % |
| `taux_inflation_moyen_annuel_ipc_pct` | REAL | Taux d'inflation moyen annuel (IPC) en % |
| `recettes_fiscales` | REAL | Total des recettes fiscales en milliards FCFA |
| `encours_de_la_dette_pct_pib` | REAL | Dette publique en % du PIB |
| `exportations_biens_fob` | REAL | Exportations de biens (FOB) en milliards FCFA |
| `importations_biens_fob` | REAL | Importations de biens (FOB) en milliards FCFA |
| `balance_des_biens` | REAL | Solde commercial (Exportations - Importations) |
| `agregats_monnaie_masse_monetaire_m2` | REAL | Masse monétaire M2 en milliards FCFA |

**Données disponibles** : Années 2005-2022 (voir `postgres/indiceconomique_long_v4.csv`)

**Utilisateurs SQL** :
- `postgres` : Administrateur (full access)
- `llm_user` : Utilisateur read-only pour l'API (mot de passe : `/-+3Vd9$!D@12`)

---

## 📚 Références

### Code source

- **Main** : `api/app/main.py`
- **Routers** : `api/app/routers/conversation.py`, `api/app/routers/forecast.py`
- **Schemas** : `api/app/models/schemas.py`
- **Orchestrator** : `api/app/services/query_orchestrator.py`
- **Config** : `api/app/config.py`

### Documentation connexe

- **README** : `README.md`
- **Guide utilisateur** : `docs/GUIDE_UTILISATEUR.md`
- **Documentation des prompts** : `docs/PROMPTS_DOCUMENTATION.md`
- **Configuration** : `docs/CONFIGURATION.md`
- **Exemples SQL** : `docs/examples.json`

---

**Document créé le** : 1er décembre 2025  
**Version** : 1.0 (corrigée d'après le code source)  
**Auteur** : Stage BCEAO - Système Text-to-SQL UEMOA  
**Licence** : Confidentiel BCEAO
