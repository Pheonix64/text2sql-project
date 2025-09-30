#!/usr/bin/env python3
"""
Script d'exemple pour tester les endpoints d'interprétation SHAP
pour les économistes et analystes de la BCEAO/UEMOA.
"""

import requests
import json
from typing import Dict, Any

class ShapPredictionClient:
    """Client pour interagir avec les endpoints de prédiction SHAP."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def send_prediction(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Envoie les données de prédiction d'inflation au serveur."""
        url = f"{self.base_url}/api/forecast/inflation/prediction"
        response = self.session.post(url, json=prediction_data)
        response.raise_for_status()
        return response.json()
    
    def get_inflation_interpretation(self,
                                     prediction_data: Dict[str, Any],
                                     language: str = "fr",
                                     audience: str = "economist",
                                     include_policy_recs: bool = True,
                                     include_monetary_analysis: bool = True) -> Dict[str, Any]:
        """Demande l'interprétation économique des prédictions d'inflation."""
        url = f"{self.base_url}/api/forecast/inflation/interpret"

        request_data = {
            "prediction_data": prediction_data,
            "analysis_language": language,
            "target_audience": audience,
            "include_policy_recommendations": include_policy_recs,
            "include_monetary_policy_analysis": include_monetary_analysis,
            "focus_on_bceao_mandate": True
        }

        response = self.session.post(url, json=request_data)
        response.raise_for_status()
        return response.json()

def create_sample_prediction_data() -> Dict[str, Any]:
    """Crée des données d'exemple pour tester l'API."""
    return {
        "predictions": {
            "2024-Q1": 2.5,
            "2024-Q2": 2.8,
            "2024-Q3": 3.1,
            "2024-Q4": 2.9
        },
        "global_shap_importance": {
            "taux_change_eur_fcfa": 0.35,
            "prix_petrole_brent": 0.28,
            "masse_monetaire_m2": 0.22,
            "balance_commerciale": 0.15
        },
        "shap_summary_details": {
            "model_type": "RandomForestRegressor",
            "feature_count": 15,
            "observation_count": 1000,
            "r2_score": 0.87,
            "mean_absolute_error": 0.42,
            "training_period": "2010-2023"
        },
        "individual_shap_explanations": {
            "2024-Q1": {
                "taux_change_eur_fcfa": 0.4,
                "prix_petrole_brent": -0.2,
                "masse_monetaire_m2": 0.3,
                "balance_commerciale": 0.1
            },
            "2024-Q2": {
                "taux_change_eur_fcfa": 0.5,
                "prix_petrole_brent": 0.1,
                "masse_monetaire_m2": 0.2,
                "balance_commerciale": 0.0
            }
        }
    }

def main():
    """Fonction principale pour tester l'API."""
    client = ShapPredictionClient()
    
    # Création des données d'exemple
    sample_data = create_sample_prediction_data()
    
    print("=== Test des endpoints de prédiction d'inflation SHAP ===\n")
    
    try:
        # Test 1: Envoi des prédictions
        print("1. Envoi des données de prédiction...")
        formatted_prediction = client.send_prediction(sample_data)
        print("✓ Prédictions formatées avec succès\n")
        
        # Test 2: Interprétation pour économiste en français
        print("2. Génération d'interprétation pour économiste (FR)...")
        interpretation_fr = client.get_inflation_interpretation(
            prediction_data=formatted_prediction,
            language="fr",
            audience="economist",
            include_policy_recs=True,
            include_monetary_analysis=True
        )
        
        print("✓ Interprétation générée:")
        print(f"  - Résumé exécutif: {interpretation_fr['executive_summary'][:100]}...")
        print(f"  - Facteurs d'inflation clés: {len(interpretation_fr['key_inflation_drivers'])} identifiés")
        print(f"  - Recommandations monétaires: {'Incluses' if interpretation_fr['monetary_policy_recommendations'] else 'Non incluses'}\n")
        
        # Test 3: Interprétation pour décideur en anglais
        print("3. Génération d'interprétation pour décideur (EN)...")
        interpretation_en = client.get_inflation_interpretation(
            prediction_data=formatted_prediction,
            language="en",
            audience="policymaker",
            include_policy_recs=True,
            include_monetary_analysis=True
        )
        
        print("✓ Interprétation générée:")
        print(f"  - Executive summary: {interpretation_en['executive_summary'][:100]}...")
        print(f"  - Key inflation drivers: {len(interpretation_en['key_inflation_drivers'])} identified")
        print(f"  - Monetary policy recommendations: {'Included' if interpretation_en['monetary_policy_recommendations'] else 'Not included'}\n")
        
        # Sauvegarde des résultats
        results = {
            "sample_data": sample_data,
            "formatted_prediction": formatted_prediction,
            "interpretation_economist_fr": interpretation_fr,
            "interpretation_policymaker_en": interpretation_en
        }
        
        with open("shap_prediction_test_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("✓ Résultats sauvegardés dans 'shap_prediction_test_results.json'")
        print("\n=== Tests terminés avec succès ===")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur de requête: {e}")
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")

if __name__ == "__main__":
    main()