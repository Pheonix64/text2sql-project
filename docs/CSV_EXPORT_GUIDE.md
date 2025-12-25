# 📥 Guide d'Export CSV

Guide complet pour télécharger les données brutes de vos requêtes au format CSV.

---

## 📋 Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Comment ça fonctionne](#comment-ça-fonctionne)
3. [Exemples d'utilisation](#exemples-dutilisation)
4. [Intégration Frontend](#intégration-frontend)
5. [Limitations et Bonnes Pratiques](#limitations-et-bonnes-pratiques)

---

## Vue d'ensemble

La fonctionnalité d'export CSV vous permet de télécharger les données brutes résultant de vos questions en langage naturel.

### Cas d'usage

- 📊 Analyse approfondie dans Excel/LibreOffice
- 📈 Création de graphiques personnalisés
- 🔄 Intégration avec d'autres outils (Power BI, Tableau)
- 💾 Sauvegarde locale des résultats
- 📤 Partage des données avec des collègues

---

## Comment ça fonctionne

### Flux de travail

```
1. Poser une question
   ↓
2. Recevoir la réponse + query_id
   ↓
3. Utiliser query_id pour télécharger le CSV
   ↓
4. Données disponibles pendant 30 minutes
```

### Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/ask` | POST | Poser une question, reçoit un `query_id` |
| `/api/export/csv/{query_id}` | GET | Télécharger le CSV avec le `query_id` |

---

## Exemples d'utilisation

### Exemple 1 : Ligne de Commande (cURL)

```bash
# Étape 1 : Poser une question
curl -X POST "http://localhost:8008/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Quelle est l'\''évolution du PIB entre 2015 et 2020?"}' \
  | jq -r '.query_id' > query_id.txt

# Étape 2 : Télécharger le CSV
QUERY_ID=$(cat query_id.txt)
curl "http://localhost:8008/api/export/csv/$QUERY_ID" \
  --output donnees_pib.csv

echo "✅ Données exportées dans donnees_pib.csv"
```

### Exemple 2 : Python Simple

```python
import requests

# Configuration
API_URL = "http://localhost:8008"

# 1. Poser une question
response = requests.post(
    f"{API_URL}/api/ask",
    json={"question": "Quelle est l'évolution du PIB entre 2015 et 2020?"}
)

result = response.json()
print(f"Réponse : {result['answer']}\n")

# 2. Télécharger le CSV
query_id = result.get('query_id')
if query_id:
    csv_response = requests.get(f"{API_URL}/api/export/csv/{query_id}")
    
    with open("donnees_pib.csv", "wb") as f:
        f.write(csv_response.content)
    
    print(f"✅ Données exportées dans donnees_pib.csv")
    print(f"   Query ID : {query_id}")
else:
    print("❌ Aucun query_id reçu")
```

### Exemple 3 : Python avec Pandas

```python
import requests
import pandas as pd
from io import StringIO

API_URL = "http://localhost:8008"

def ask_and_download(question: str) -> pd.DataFrame:
    """Pose une question et retourne un DataFrame pandas."""
    
    # 1. Poser la question
    response = requests.post(
        f"{API_URL}/api/ask",
        json={"question": question}
    )
    result = response.json()
    
    print(f"Réponse : {result['answer']}\n")
    
    # 2. Télécharger et charger dans pandas
    query_id = result.get('query_id')
    if not query_id:
        print("❌ Aucune donnée disponible")
        return None
    
    csv_response = requests.get(f"{API_URL}/api/export/csv/{query_id}")
    csv_data = csv_response.text
    
    # Charger dans pandas
    df = pd.read_csv(StringIO(csv_data))
    
    print(f"✅ {len(df)} lignes chargées")
    return df

# Utilisation
df = ask_and_download("Quelle est l'évolution du PIB entre 2015 et 2020?")

if df is not None:
    # Analyse avec pandas
    print("\nAperçu des données :")
    print(df.head())
    
    print("\nStatistiques :")
    print(df.describe())
    
    # Sauvegarder
    df.to_excel("donnees_pib.xlsx", index=False)
    print("\n✅ Exporté vers Excel : donnees_pib.xlsx")
```

### Exemple 4 : JavaScript/TypeScript

```javascript
// Fonction pour poser une question et télécharger le CSV
async function askAndDownloadCSV(question) {
    const API_URL = 'http://localhost:8008';
    
    try {
        // 1. Poser la question
        const response = await fetch(`${API_URL}/api/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question })
        });
        
        const result = await response.json();
        console.log('Réponse :', result.answer);
        
        // 2. Télécharger le CSV
        if (result.query_id) {
            const csvResponse = await fetch(
                `${API_URL}/api/export/csv/${result.query_id}`
            );
            const blob = await csvResponse.blob();
            
            // Créer un lien de téléchargement
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `donnees_${result.query_id}.csv`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            
            console.log('✅ CSV téléchargé');
        }
    } catch (error) {
        console.error('❌ Erreur :', error);
    }
}

// Utilisation
askAndDownloadCSV("Quelle est l'évolution du PIB entre 2015 et 2020?");
```

---

## Intégration Frontend

### React Component

```jsx
import React, { useState } from 'react';

function CSVExporter() {
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState('');
    const [queryId, setQueryId] = useState(null);
    const [loading, setLoading] = useState(false);
    
    const API_URL = 'http://localhost:8008';
    
    const handleAsk = async () => {
        setLoading(true);
        try {
            const response = await fetch(`${API_URL}/api/ask`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question })
            });
            
            const result = await response.json();
            setAnswer(result.answer);
            setQueryId(result.query_id);
        } catch (error) {
            console.error('Erreur:', error);
        } finally {
            setLoading(false);
        }
    };
    
    const handleDownloadCSV = async () => {
        if (!queryId) return;
        
        const csvUrl = `${API_URL}/api/export/csv/${queryId}`;
        const link = document.createElement('a');
        link.href = csvUrl;
        link.download = `donnees_${queryId}.csv`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };
    
    return (
        <div className="csv-exporter">
            <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Posez votre question..."
                disabled={loading}
            />
            <button onClick={handleAsk} disabled={loading}>
                {loading ? 'Traitement...' : 'Poser la question'}
            </button>
            
            {answer && (
                <div className="result">
                    <h3>Réponse :</h3>
                    <p>{answer}</p>
                    
                    {queryId && (
                        <button onClick={handleDownloadCSV}>
                            📥 Télécharger CSV
                        </button>
                    )}
                </div>
            )}
        </div>
    );
}

export default CSVExporter;
```

---

## Limitations et Bonnes Pratiques

### Limitations

| Limitation | Valeur | Note |
|------------|--------|------|
| **Durée de stockage** | 30 minutes | Les données expirent après ce délai |
| **Taille des résultats** | 1000 lignes max | Définie par la limite SQL |
| **Encodage** | UTF-8 avec BOM | Compatible Excel et LibreOffice |

### Bonnes Pratiques

#### 1. Gestion des Erreurs

```python
import requests

response = requests.post(
    "http://localhost:8008/api/ask",
    json={"question": "Votre question"}
)

result = response.json()

# Vérifier si un query_id existe
if not result.get('query_id'):
    print("⚠️  Pas de données à exporter")
    print(f"Réponse : {result['answer']}")
else:
    # Télécharger le CSV
    csv_response = requests.get(
        f"http://localhost:8008/api/export/csv/{result['query_id']}"
    )
    
    if csv_response.status_code == 404:
        print("❌ Données expirées ou introuvables")
    elif csv_response.status_code == 200:
        # Sauvegarder
        with open("donnees.csv", "wb") as f:
            f.write(csv_response.content)
        print("✅ Export réussi")
```

#### 2. Téléchargement Immédiat

Les données sont disponibles pendant 30 minutes. Pour éviter l'expiration :

```python
# ✅ BON : Téléchargement immédiat
result = ask_question(question)
if result.get('query_id'):
    download_csv(result['query_id'])

# ❌ ÉVITER : Attendre trop longtemps
result = ask_question(question)
time.sleep(1800)  # 30 minutes
download_csv(result['query_id'])  # Risque d'expiration
```

#### 3. Réutilisation des Données

```python
import pandas as pd

# Charger une seule fois
df = download_and_load_csv(query_id)

# Réutiliser
df.to_excel("rapport.xlsx")
df.to_json("rapport.json")
df.to_html("rapport.html")
```

#### 4. Questions sans Données

Certaines questions ne retournent pas de données exploitables :

```python
# Questions qui ne retournent PAS de query_id
questions_sans_donnees = [
    "Bonjour",
    "Quelle est la capitale de la France?",
    "Explique-moi l'inflation"
]

# Questions qui retournent un query_id
questions_avec_donnees = [
    "Quelle est l'évolution du PIB entre 2015 et 2020?",
    "Donne-moi les taux d'inflation depuis 2010",
    "Liste les exportations par année"
]
```

---

## FAQ

### Q : Les données sont-elles persistées définitivement ?

**R :** Non, les données sont stockées temporairement pendant **30 minutes** pour optimiser les ressources serveur. Après expiration, le cache est automatiquement nettoyé.

### Q : Puis-je télécharger le CSV plusieurs fois ?

**R :** Oui, tant que le `query_id` n'a pas expiré (30 minutes), vous pouvez télécharger le CSV autant de fois que nécessaire.

### Q : Quel est le format du CSV ?

**R :** Le CSV est encodé en UTF-8 avec BOM, compatible avec Excel, LibreOffice, et tous les outils standards. Les colonnes correspondent exactement aux colonnes de la table PostgreSQL.

### Q : Que se passe-t-il si je n'utilise pas le query_id ?

**R :** Rien. Le cache est automatiquement nettoyé après 30 minutes. Aucune action manuelle n'est requise.

### Q : Puis-je personnaliser le format d'export ?

**R :** Actuellement, seul le format CSV est supporté. Pour d'autres formats (Excel, JSON), chargez le CSV dans pandas et exportez :

```python
import pandas as pd

df = pd.read_csv("donnees.csv")
df.to_excel("donnees.xlsx", index=False)
df.to_json("donnees.json", orient='records')
```

---

## Support

Pour plus d'aide :
- 📚 [Documentation API](API_REFERENCE.md)
- 📖 [Guide Utilisateur](GUIDE_UTILISATEUR.md)
- 🐛 [Issues GitHub](https://github.com/Pheonix64/text2sql-project/issues)

### Troubleshooting

#### Erreur "No module named 'pandas'"

Si vous voyez cette erreur dans les logs Docker, pandas n'est pas installé. Solutions :

**Solution 1 - Rebuild complet (recommandé pour production) :**
```bash
docker-compose down
docker-compose build api-fastapi
docker-compose up -d
```

**Solution 2 - Installation rapide (développement) :**
```bash
docker exec api-fastapi pip install pandas
docker-compose restart api-fastapi
```

**Vérification de l'installation :**
```bash
docker exec api-fastapi python -c "import pandas; print(pandas.__version__)"
```

#### Erreur 404 lors du téléchargement

Les données ont expiré (30 minutes). Reposez la question pour obtenir un nouveau `query_id`.

#### Le bouton CSV n'apparaît pas

Le `query_id` est `null`, ce qui signifie que la question n'a pas généré de données SQL (question conversationnelle).

---

**Dernière mise à jour** : 25 décembre 2025
