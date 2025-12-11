# Guide d'utilisation des endpoints d'inflation SHAP

## Vue d'ensemble

Ce guide explique comment utiliser les endpoints dédiés à la prévision d'inflation intégrant l'explicabilité SHAP, spécifiquement adaptés pour les économistes et analystes de la BCEAO/UEMOA.

## Endpoints disponibles

### 1. `/api/forecast/narrative` (existant)
Génère une narration pour des prévisions économiques simples.

### 2. `/api/forecast/inflation/prediction`
Reçoit et formate les prédictions du modèle d'inflation avec explicabilité SHAP.

**Corps de la requête :**
```json
{
  "predictions": {
    "2024-01": 2.5,
    "2024-02": 2.8,
    "2024-03": 3.1
  },
  "global_shap_importance": {
    "taux_change": 0.35,
    "prix_petrole": 0.28,
    "masse_monetaire": 0.22,
    "balance_commerciale": 0.15
  },
  "shap_summary_details": {
    "model_type": "RandomForest",
    "feature_count": 15,
    "observation_count": 1000,
    "accuracy": 0.87
  },
  "individual_shap_explanations": {
    "observation_1": {
      "taux_change": 0.4,
      "prix_petrole": -0.2,
      "masse_monetaire": 0.1
    }
  },
  "confidence_intervals": {
    "2024-01": {"lower": 2.1, "upper": 2.9}
  }
}
```

### 3. `/api/forecast/inflation/interpret`
Génère une interprétation économique détaillée des prédictions d'inflation SHAP.

**Timeout :** Configurable via `LLM_TIMEOUT_INFLATION` dans la configuration (défaut: 120 secondes). Cette valeur peut être ajustée pour des analyses plus complexes nécessitant plus de temps de traitement.

**Corps de la requête :
```json
{
  "prediction_data": {
    // Données de prédiction (format InflationPredictionResponse)
  },
  "analysis_language": "fr",
  "target_audience": "economist",
  "include_policy_recommendations": true,
  "include_monetary_policy_analysis": true,
  "focus_on_bceao_mandate": true
}
```

**Réponse :**
```json
{
  "executive_summary": "Résumé exécutif pour les décideurs...",
  "inflation_analysis": "Analyse détaillée des dynamiques inflationnistes...",
  "key_inflation_drivers": [
    "Taux de change : impact positif de 35%",
    "Prix du pétrole : influence modérée de 28%",
    "Masse monétaire : effet inflationniste de 22%"
  ],
  "price_stability_assessment": "Position par rapport à la cible de 3%...",
  "monetary_policy_recommendations": "Actions suggérées pour le taux directeur...",
  "inflation_risks": [
    "Volatilité des taux de change",
    "Instabilité des prix des matières premières"
  ],
  "model_confidence": "Élevé (87% de précision)",
  "target_deviation_analysis": "Écart de +0.5 pt par rapport à la cible",
  "external_factors_impact": "Analyse du choc pétrolier et du change"
}
```

## Indicateur supporté

- `inflation` : Analyse de l'inflation (objectif de stabilité des prix de la BCEAO)

## Audiences cibles

- `economist` : Économiste spécialisé
- `analyst` : Analyste financier
- `policymaker` : Décideur politique
- `general` : Public général

## Exemple d'utilisation complète

### 1. Envoyer les prédictions du modèle
```bash
curl -X POST "http://localhost:8000/api/forecast/inflation/prediction" \
-H "Content-Type: application/json" \
-d '{
  "predictions": {"2024-Q1": 2.5, "2024-Q2": 2.8},
  "global_shap_importance": {
    "taux_change": 0.35,
    "prix_petrole": 0.28,
    "masse_monetaire": 0.22
  },
  "shap_summary_details": {"accuracy": 0.87},
  "individual_shap_explanations": {},
  "confidence_intervals": {}
}'
```

### 2. Demander l'interprétation économique
```bash
curl -X POST "http://localhost:8000/api/forecast/inflation/interpret" \
-H "Content-Type: application/json" \
-d '{
  "prediction_data": {
    /* Résultat de l'étape 1 */
  },
  "analysis_language": "fr",
  "target_audience": "economist",
  "include_policy_recommendations": true,
  "include_monetary_policy_analysis": true,
  "focus_on_bceao_mandate": true
}'
```

## Avantages pour les économistes

1. **Explicabilité** : Comprendre quels facteurs influencent les prédictions
2. **Interprétation contextualisée** : Analyses adaptées au contexte UEMOA/BCEAO
3. **Recommandations actionables** : Suggestions de politiques monétaires concrètes
4. **Évaluation des risques** : Identification des facteurs d'incertitude
5. **Accessibilité** : Langage adapté aux différentes audiences

## Intégration avec les modèles existants

Ces endpoints peuvent être facilement intégrés avec :
- Modèles de prévision d'inflation
- Systèmes d'early warning macroéconomique
- Outils d'aide à la décision de politique monétaire
- Tableaux de bord économiques

## Support multilingue

- Français (par défaut)
- Anglais

Les interprétations sont automatiquement adaptées selon la langue choisie et incluent la terminologie économique appropriée.