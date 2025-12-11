# Intégration du Modèle de Forecasting d'Inflation avec Explicabilité SHAP

## Vue d'ensemble

Ce système permet d'intégrer facilement le modèle de prévision d'inflation avec explicabilité SHAP pour fournir des analyses économiques adaptées aux économistes et analystes de la BCEAO/UEMOA.

## Architecture

```
Modèle Inflation SHAP → API Text-to-SQL → Interprétation Inflation → Économiste/Analyste
          ↓                ↓                   ↓                 ↓
    Prédictions      Format dédié      Analyse LLM BCEAO   Politique monétaire
```

## Format de réponse du modèle

Votre modèle d'inflation doit retourner une réponse au format suivant :

```python
class InflationPredictionResponse(BaseModel):
    predictions: dict  # Prévisions d'inflation {"2024-Q1": 2.5, ...}
    global_shap_importance: dict  # Importance des facteurs d'inflation
    shap_summary_details: dict  # Métadonnées du modèle (précision, période, etc.)
    individual_shap_explanations: dict  # Explications SHAP par période
    confidence_intervals: dict | None
```

### Exemple concret

Voir le fichier `example_shap_response.json` pour un exemple complet.

## Endpoints disponibles

### 1. `/api/forecast/inflation/prediction`
- **Objectif** : Recevoir et valider les prédictions d'inflation
- **Input** : Données brutes du modèle d'inflation
- **Output** : Données formatées selon `InflationPredictionResponse`

### 2. `/api/forecast/inflation/interpret`
- **Objectif** : Générer l'interprétation économique de l'inflation
- **Input** : `InflationInterpretationRequest`
- **Output** : `InflationInterpretationResponse`
- **Timeout** : Configurable via `LLM_TIMEOUT_INFLATION` (défaut: 120 secondes)

## Types d'analyses supportées

### Indicateur économique pris en charge
- **Inflation** : Analyse des facteurs inflationnistes et de la stabilité des prix

### Audiences cibles
- **Économiste** : Analyse technique approfondie
- **Analyste** : Focus sur les métriques financières
- **Décideur** : Synthèse exécutive et recommandations
- **Public général** : Vulgarisation accessible

## Avantages de l'explicabilité SHAP

### Pour les économistes
1. **Transparence** : Comprendre les mécanismes du modèle
2. **Validation** : Vérifier la cohérence économique
3. **Interprétation** : Traduire les résultats en termes économiques
4. **Robustesse** : Évaluer la stabilité des prédictions

### Pour les décideurs
1. **Confiance** : Justification des recommandations
2. **Actionabilité** : Leviers d'intervention identifiés
3. **Priorisation** : Focus sur les facteurs les plus importants
4. **Communication** : Explication claire aux parties prenantes

## Exemples d'interprétation

### Facteur : Taux de change EUR/FCFA
- **Importance SHAP** : 0.342 (34.2%)
- **Interprétation économique** : "La dépréciation du FCFA face à l'Euro exerce une pression inflationniste significative via l'augmentation des coûts d'importation"
- **Recommandation** : "Surveiller étroitement les tensions sur le taux de change et considérer des interventions pour stabiliser la parité"

### Facteur : Prix du pétrole Brent
- **Importance SHAP** : 0.276 (27.6%)
- **Interprétation économique** : "La hausse des prix pétroliers impacte directement l'inflation via les coûts de transport et d'énergie"
- **Recommandation** : "Anticiper les effets de second tour par une communication proactive sur la politique monétaire"

## Intégration avec votre modèle

### Étape 1 : Préparer votre modèle
Assurez-vous que votre modèle de ML peut générer :
- Des prédictions temporelles
- Des valeurs SHAP globales et individuelles
- Des métadonnées sur la performance

### Étape 2 : Adapter le format de sortie
Utilisez la classe `InflationPredictionResponse` pour formater vos résultats :

```python
prediction_response = {
    "predictions": your_predictions,
    "global_shap_importance": your_shap_values.mean(0),
    "shap_summary_details": your_model_metadata,
    "individual_shap_explanations": your_individual_shap,
    "confidence_intervals": your_confidence_intervals
}
```

### Étape 3 : Envoyer à l'API
```python
import requests

response = requests.post(
    "http://your-api/api/forecast/inflation/prediction",
    json=prediction_response
)
```

### Étape 4 : Demander l'interprétation
```python
interpretation_request = {
    "prediction_data": response.json(),
    "analysis_language": "fr",
    "target_audience": "economist",
    "include_policy_recommendations": True,
    "include_monetary_policy_analysis": True,
    "focus_on_bceao_mandate": True
}

interpretation = requests.post(
    "http://your-api/api/forecast/inflation/interpret",
    json=interpretation_request
)
```

## Bonnes pratiques

### Qualité des features
1. **Noms explicites** : Utilisez des noms de variables compréhensibles
2. **Documentation** : Ajoutez des descriptions dans `feature_descriptions`
3. **Unités** : Précisez les unités de mesure
4. **Périodicité** : Indiquez la fréquence temporelle

### Performance du modèle
1. **Métriques** : Incluez R², MAE, RMSE dans `shap_summary_details`
2. **Validation** : Documentez la méthode de validation croisée
3. **Stabilité** : Testez la robustesse sur différentes périodes
4. **Diagnostics** : Incluez les tests de résidus

### Interprétation économique
1. **Contexte** : Adaptez l'analyse au contexte UEMOA/BCEAO
2. **Audience** : Ajustez le niveau technique selon le public
3. **Actualité** : Intégrez les développements économiques récents
4. **Politiques** : Proposez des recommandations concrètes

## Support et développement

### Tests
Utilisez le script `test_shap_client.py` pour valider votre intégration.

### Documentation
Consultez `SHAP_PREDICTION_GUIDE.md` pour des détails d'utilisation.

### Exemple complet
Voir `example_shap_response.json` pour un format de réponse détaillé.

## Évolutions futures

- Support de modèles de deep learning (LIME, Integrated Gradients)
- Visualisations interactives des explications SHAP
- Analyses de sensibilité et stress tests
- Intégration avec des tableaux de bord temps réel
- APIs pour d'autres banques centrales africaines