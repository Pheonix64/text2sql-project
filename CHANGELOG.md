# 📝 CHANGELOG

Historique des modifications du projet Text-to-SQL.

---

## [1.1.0] - Novembre 2025

### ✨ Améliorations Majeures

#### Documentation
- 📚 **Réorganisation complète** de la documentation dans `/docs`
- 📖 Nouveau **[Guide Utilisateur](docs/GUIDE_UTILISATEUR.md)** complet avec exemples
- 🔌 Nouvelle **[Référence API](docs/API_REFERENCE.md)** détaillée
- ⚙️ Nouveau **[Guide de Configuration](docs/CONFIGURATION.md)**
- 📑 Création d'un **[Index de Documentation](docs/README.md)**
- 🎨 README principal mis à jour avec badges et structure claire

#### Corrections Techniques
- ✅ Fix ChromaDB healthcheck (changement vers test TCP simple)
- ✅ Fix configuration ports ChromaDB (séparation port interne/externe)
- ✅ Fix langchain-huggingface deprecation warning
- ✅ Amélioration Docker build avec retry logic (--retries 5)
- ✅ Split installation pip en plusieurs étapes pour meilleure isolation

#### Nettoyage
- 🧹 Déplacement de tous les fichiers .md vers `/docs`
- 🧹 Déplacement des exemples JSON vers `/docs`
- 🧹 Suppression des fichiers de test obsolètes
- 🧹 Structure de projet clarifiée

### 🔧 Changements Techniques

#### Dependencies
- ➕ Ajout de `langchain-huggingface>=0.1.0`
- 🔄 Migration de `langchain_community.embeddings.HuggingFaceEmbeddings` vers `langchain_huggingface.HuggingFaceEmbeddings`

#### Configuration
- ➕ Ajout de `CHROMA_EXTERNAL_PORT` pour séparer ports interne/externe
- 🔧 Mise à jour `CHROMA_PORT=8000` (port interne)
- 📝 Amélioration des commentaires dans `.env`

#### Docker
- 🐳 Healthcheck ChromaDB simplifié (TCP check au lieu de Python)
- 🐳 Ajout de `--retries 5` à toutes les commandes pip install
- 🐳 Split installation pip en 4 RUN layers distinctes
- 🐳 Ajout de `start_period` au healthcheck ChromaDB

#### Code
- 📝 Mise à jour du chemin `examples.json` → `docs/examples.json`
- 🔄 Import statements modernisés

### 📊 Fichiers Déplacés

```
Racine → docs/:
- ARCHITECTURE_DIAGRAM.md
- FORECASTING_INTEGRATION.md
- LANGCHAIN_INDEX.md
- LANGCHAIN_QUICK_START.md
- LANGCHAIN_REFACTORING.md
- REFACTORING_SUMMARY.md
- SHAP_PREDICTION_GUIDE.md
- TESTING_GUIDE.md
- examples.json
- example_shap_response.json
```

### 📚 Nouvelle Documentation

```
docs/:
+ README.md                  # Index de la documentation
+ GUIDE_UTILISATEUR.md       # Guide complet utilisateur
+ API_REFERENCE.md           # Référence API détaillée
+ CONFIGURATION.md           # Guide de configuration
```

---

## [1.0.0] - Octobre 2025

### 🎉 Version Initiale

#### Fonctionnalités
- ✨ API REST Text-to-SQL avec FastAPI
- 🤖 Intégration LangChain pour orchestration LLM
- 🧠 Support Ollama (Mistral 7B)
- 🔍 Recherche sémantique avec ChromaDB
- 📊 Base de données PostgreSQL/TimescaleDB
- 🔐 Utilisateur SQL en lecture seule
- ✅ Validation SQL avec SQLGlot

#### Endpoints
- `POST /api/ask` - Questions en langage naturel
- `POST /api/index-queries` - Indexation d'exemples
- `POST /api/pull-model` - Téléchargement de modèles
- `POST /api/forecast/narrative` - Génération de narratifs
- `POST /api/forecast/inflation/prediction` - Prédictions inflation
- `POST /api/forecast/inflation/interpret` - Interprétation SHAP
- `GET /health` - Health check

#### Architecture
- 🐳 Déploiement Docker Compose
- 🔄 Pipeline LangChain complet
- 📦 4 services: API, PostgreSQL, ChromaDB, Ollama
- 🌐 Réseau Docker bridge
- 💾 Volumes persistants

#### Documentation Initiale
- README.md basique
- Documentation technique LangChain
- Guides d'architecture
- Exemples SQL

---

## Format

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/),
et ce projet adhère au [Semantic Versioning](https://semver.org/lang/fr/).

### Types de Changements

- `✨ Ajouté` pour les nouvelles fonctionnalités
- `🔧 Modifié` pour les changements aux fonctionnalités existantes
- `❌ Déprécié` pour les fonctionnalités qui seront retirées
- `🗑️ Retiré` pour les fonctionnalités retirées
- `✅ Corrigé` pour les corrections de bugs
- `🔒 Sécurité` en cas de vulnérabilités

---

## Roadmap Future

### Version 1.2.0 (Prévu)
- [ ] Interface web interactive
- [ ] Support de modèles LLM supplémentaires
- [ ] Cache des requêtes fréquentes
- [ ] Métriques et monitoring
- [ ] Tests automatisés complets

### Version 2.0.0 (Futur)
- [ ] Support multi-langues (EN, FR)
- [ ] API d'authentification
- [ ] Gestion des utilisateurs
- [ ] Dashboard analytics
- [ ] Export des résultats (PDF, Excel)

---

**[⬆ Retour au README](README.md)**
