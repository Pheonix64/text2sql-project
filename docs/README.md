# 📚 Documentation - Index

Documentation complète du projet Text-to-SQL.

---

## 🚀 Démarrage Rapide

👉 **[README Principal](../README.md)** - Guide de démarrage et vue d'ensemble

---

## 📖 Guides Utilisateur

| Document | Description |
|----------|-------------|
| **[Guide Utilisateur](GUIDE_UTILISATEUR.md)** | Tutoriel complet avec exemples pratiques |
| **[Référence API](API_REFERENCE.md)** | Documentation détaillée de tous les endpoints |
| **[Configuration](CONFIGURATION.md)** | Guide de configuration et personnalisation |

---

## 🏗️ Documentation Technique

### Architecture

| Document | Description |
|----------|-------------|
| **[Architecture Diagram](ARCHITECTURE_DIAGRAM.md)** | Schémas et architecture du système |
| **[LangChain Index](LANGCHAIN_INDEX.md)** | Documentation du pipeline LangChain |
| **[LangChain Refactoring](LANGCHAIN_REFACTORING.md)** | Détails de l'implémentation LangChain |
| **[LangChain Quick Start](LANGCHAIN_QUICK_START.md)** | Guide rapide LangChain |

### Fonctionnalités Spécifiques

| Document | Description |
|----------|-------------|
| **[SHAP Prediction Guide](SHAP_PREDICTION_GUIDE.md)** | Interprétation des modèles avec SHAP |
| **[Forecasting Integration](FORECASTING_INTEGRATION.md)** | Intégration des prévisions économiques |
| **[Testing Guide](TESTING_GUIDE.md)** | Guide de tests et validation |

### Historique

| Document | Description |
|----------|-------------|
| **[Refactoring Summary](REFACTORING_SUMMARY.md)** | Résumé des modifications et refactoring |

---

## 📊 Exemples et Données

| Fichier | Description |
|---------|-------------|
| **[examples.json](examples.json)** | Exemples de requêtes SQL pour indexation |
| **[example_shap_response.json](example_shap_response.json)** | Exemple de réponse SHAP pour prédictions |

---

## 🎯 Par Cas d'Usage

### Je veux utiliser l'API

1. **[README Principal](../README.md)** - Installation et démarrage
2. **[Guide Utilisateur](GUIDE_UTILISATEUR.md)** - Exemples d'utilisation
3. **[Référence API](API_REFERENCE.md)** - Documentation des endpoints

### Je veux configurer le projet

1. **[Configuration](CONFIGURATION.md)** - Variables d'environnement
2. **[Architecture Diagram](ARCHITECTURE_DIAGRAM.md)** - Comprendre l'architecture

### Je veux comprendre le code

1. **[LangChain Index](LANGCHAIN_INDEX.md)** - Pipeline et orchestration
2. **[LangChain Refactoring](LANGCHAIN_REFACTORING.md)** - Implémentation détaillée
3. **[Architecture Diagram](ARCHITECTURE_DIAGRAM.md)** - Vue d'ensemble

### Je veux intégrer les prévisions

1. **[SHAP Prediction Guide](SHAP_PREDICTION_GUIDE.md)** - Modèles SHAP
2. **[Forecasting Integration](FORECASTING_INTEGRATION.md)** - Intégration prévisions
3. **[Référence API](API_REFERENCE.md#3-forecast-endpoints)** - Endpoints forecast

### Je veux tester

1. **[Testing Guide](TESTING_GUIDE.md)** - Guide de tests
2. **[Guide Utilisateur](GUIDE_UTILISATEUR.md#6-bonnes-pratiques)** - Bonnes pratiques

---

## 📑 Structure de la Documentation

```
docs/
├── README.md                           # Cet index
├── GUIDE_UTILISATEUR.md               # Guide utilisateur complet
├── API_REFERENCE.md                   # Référence API
├── CONFIGURATION.md                   # Guide de configuration
├── ARCHITECTURE_DIAGRAM.md            # Architecture du système
├── LANGCHAIN_INDEX.md                 # Documentation LangChain
├── LANGCHAIN_REFACTORING.md           # Refactoring LangChain
├── LANGCHAIN_QUICK_START.md           # Quick Start LangChain
├── SHAP_PREDICTION_GUIDE.md           # Guide SHAP
├── FORECASTING_INTEGRATION.md         # Intégration prévisions
├── TESTING_GUIDE.md                   # Guide de tests
├── REFACTORING_SUMMARY.md             # Historique refactoring
├── examples.json                       # Exemples SQL
└── example_shap_response.json         # Exemple SHAP
```

---

## 🔍 Recherche Rapide

### Endpoints API

- Poser une question : [API Reference - /api/ask](API_REFERENCE.md#post-apiask)
- Prévisions : [API Reference - Forecast](API_REFERENCE.md#3-forecast-endpoints)
- Administration : [API Reference - Admin](API_REFERENCE.md#4-administration-endpoints)

### Configuration

- Variables d'environnement : [Configuration - Variables](CONFIGURATION.md#1-variables-denvironnement)
- Docker : [Configuration - Docker](CONFIGURATION.md#2-configuration-docker)
- LLM : [Configuration - LLM](CONFIGURATION.md#4-configuration-llm)

### Exemples

- Python : [Guide Utilisateur - Python](GUIDE_UTILISATEUR.md#33-exemples-python)
- cURL : [Guide Utilisateur - Questions](GUIDE_UTILISATEUR.md#32-exemples-de-questions)
- JavaScript : [API Reference - Intégration](API_REFERENCE.md#9-exemples-dintégration)

---

## 📞 Support

- 🐛 **Issues :** [GitHub Issues](https://github.com/Pheonix64/text2sql-project/issues)
- 📧 **Contact :** Stage BCEAO
- 📖 **Wiki :** [Documentation Complète](../README.md)

---

## 🔄 Mises à Jour

**Dernière mise à jour :** Novembre 2025

**Changelog :**
- ✅ Réorganisation de la documentation
- ✅ Création de guides structurés
- ✅ Documentation complète de l'API
- ✅ Exemples pratiques ajoutés
- ✅ Fix ChromaDB healthcheck
- ✅ Fix langchain-huggingface deprecation

---

**[⬆ Retour au README](../README.md)**
