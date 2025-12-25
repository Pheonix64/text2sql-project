#!/usr/bin/env python3
"""
Script de test pour la fonctionnalité d'export CSV
Démontre l'utilisation de l'endpoint /api/export/csv

Usage:
    python test_csv_export.py
"""

import requests
import pandas as pd
from io import StringIO
import sys
from pathlib import Path

# Configuration
API_URL = "http://localhost:8008"
OUTPUT_DIR = Path("exports")

# Créer le dossier de sortie
OUTPUT_DIR.mkdir(exist_ok=True)

def print_section(title: str):
    """Affiche une section avec formatage."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_ask_and_export(question: str, filename: str):
    """
    Teste la fonctionnalité complète : question -> export CSV
    
    Args:
        question: Question en langage naturel
        filename: Nom du fichier CSV de sortie
    """
    print(f"📝 Question : {question}")
    
    # Étape 1 : Poser la question
    print("   Envoi de la question à l'API...")
    try:
        response = requests.post(
            f"{API_URL}/api/ask",
            json={"question": question},
            timeout=60
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur lors de la requête : {e}")
        return False
    
    result = response.json()
    
    # Afficher la réponse
    print(f"   Réponse : {result['answer'][:100]}...")
    print(f"   SQL généré : {result.get('generated_sql', 'N/A')[:80]}...")
    
    # Étape 2 : Vérifier la présence du query_id
    query_id = result.get('query_id')
    if not query_id:
        print("⚠️  Aucun query_id reçu - Pas de données à exporter")
        return False
    
    print(f"   Query ID : {query_id}")
    
    # Étape 3 : Télécharger le CSV
    print("   Téléchargement du CSV...")
    try:
        csv_response = requests.get(
            f"{API_URL}/api/export/csv/{query_id}",
            timeout=30
        )
        
        if csv_response.status_code == 404:
            print("❌ Données non trouvées ou expirées")
            return False
        
        csv_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur lors du téléchargement CSV : {e}")
        return False
    
    # Étape 4 : Sauvegarder et analyser
    output_path = OUTPUT_DIR / filename
    
    # Sauvegarder le CSV
    with open(output_path, "wb") as f:
        f.write(csv_response.content)
    
    print(f"✅ CSV sauvegardé : {output_path}")
    
    # Charger dans pandas pour analyse
    try:
        df = pd.read_csv(StringIO(csv_response.text))
        print(f"   Lignes : {len(df)}, Colonnes : {len(df.columns)}")
        print(f"   Colonnes : {', '.join(df.columns[:5])}")
        
        # Afficher un aperçu
        print("\n   Aperçu des données :")
        print(df.head(3).to_string(index=False))
        
    except Exception as e:
        print(f"⚠️  Impossible de charger le CSV dans pandas : {e}")
    
    print()
    return True

def test_health_check():
    """Vérifie que l'API est accessible."""
    print_section("Vérification de l'API")
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API accessible et opérationnelle")
            return True
        else:
            print(f"❌ API retourne le code {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Impossible de contacter l'API : {e}")
        print(f"   Assurez-vous que Docker est démarré et que l'API tourne sur {API_URL}")
        return False

def test_invalid_query_id():
    """Teste le comportement avec un query_id invalide."""
    print_section("Test avec Query ID Invalide")
    
    fake_query_id = "invalid123"
    print(f"📝 Test avec query_id : {fake_query_id}")
    
    try:
        response = requests.get(
            f"{API_URL}/api/export/csv/{fake_query_id}",
            timeout=10
        )
        
        if response.status_code == 404:
            print("✅ Erreur 404 correctement retournée")
            print(f"   Message : {response.json().get('detail', 'N/A')}")
            return True
        else:
            print(f"⚠️  Code de statut inattendu : {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return False

def main():
    """Point d'entrée principal."""
    print("\n" + "="*60)
    print("  TEST D'EXPORT CSV - Text-to-SQL API")
    print("="*60)
    
    # Vérification initiale
    if not test_health_check():
        print("\n❌ L'API n'est pas accessible. Arrêt des tests.")
        sys.exit(1)
    
    # Tests de questions avec données
    print_section("Test 1 : Évolution du PIB")
    success1 = test_ask_and_export(
        question="Quelle est l'évolution du PIB entre 2015 et 2020?",
        filename="pib_evolution.csv"
    )
    
    print_section("Test 2 : Taux d'inflation")
    success2 = test_ask_and_export(
        question="Donne-moi les taux d'inflation depuis 2010",
        filename="inflation_historique.csv"
    )
    
    print_section("Test 3 : Balance commerciale")
    success3 = test_ask_and_export(
        question="Quelle est la balance commerciale pour les années 2018 à 2022?",
        filename="balance_commerciale.csv"
    )
    
    # Test avec query_id invalide
    test_invalid_query_id()
    
    # Résumé
    print_section("Résumé des Tests")
    
    total_tests = 3
    successful_tests = sum([success1, success2, success3])
    
    print(f"Tests réussis : {successful_tests}/{total_tests}")
    print(f"Fichiers exportés dans : {OUTPUT_DIR.absolute()}")
    
    if successful_tests == total_tests:
        print("\n🎉 Tous les tests ont réussi !")
    else:
        print(f"\n⚠️  {total_tests - successful_tests} test(s) ont échoué")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
