Auteurs: Gaël LOPES DA SILVA, Arthur FOUCAULT, Guillaume LECHEVALIER.

# Prédiction du décrochage et de la réussite académique

L'idée de ce projet est simple: à partir du dossier d'un étudiant (situation familiale, parcours scolaire, résultats des deux premiers semestres, contexte macro-économique), prévoir s'il risque de décrocher, de rester inscrit ou de diplomer.

Le sujet précise que les performances prédictives ne sont pas évaluées. Le modèle reste donc volontairement simple, une régression logistique. Ce qui compte ici, c'est toute la chaine autour du modèle, de l'ingestion des données jusqu'à la supervision du service, en passant par les controles qualité, l'orchestration et la tracabilité. Le tout reproductible de bout en bout.

## Les données

On s'appuie sur le dataset public "Predict Students' Dropout and Academic Success", publié par l'UCI Machine Learning Repository (dataset n° 697, licence Creative Commons BY 4.0, librement utilisable pour la recherche et l'enseignement). Il décrit 4425 étudiants avec 36 variables numériques et une cible à trois classes: 1421 Dropout, 794 Enrolled et 2209 Graduate. Aucune donnée n'est versionnée dans le dépot, le pipeline télécharge le fichier à chaque exécution.

Le dataset est un instantané historique, figé. Pour montrer une chaine qui vit avec ses données, on le rejoue par lots, ce que le sujet autorise. Il est découpé en 5 lots d'environ 800 lignes, et chaque exécution consomme un lot de plus, cumulé avec les précédents. On simule ainsi l'arrivée régulière de nouveaux étudiants, comme le ferait une vraie source d'inscriptions en production.

## Comment la chaine s'articule

La pipeline Prefect tire les ficelles. Elle télécharge et normalise les données, les découpe en lots, fait passer les controles qualité, prépare les jeux d'entrainement et de test, entraine le modèle, l'évalue puis décide s'il part en production. Chaque entrainement est tracé dans MLflow avec ses paramètres, ses métriques et ses artefacts, et le modèle rejoint le registre avec un tag de cycle de vie.

L'API FastAPI, elle, charge la version taggée production et la sert. Elle valide les entrées, enregistre chaque prédiction en SQLite avec sa version et sa latence, accepte les feedbacks des utilisateurs et expose ses métriques au format Prometheus. Si une étape échoue, la pipeline s'arrête, une alerte est enregistrée, et le modèle en production n'est pas touché. Prometheus et Grafana surveillent le service, et Locust permet de le tester en charge.

## Prérequis

Il vous faut Python 3.11 ou plus récent, uv (recommandé) ou pip, et Docker uniquement pour la supervision. Les commandes du Makefile se lancent depuis la racine du dépot (les chemins par défaut data/ et artifacts/ sont relatifs au répertoire courant). Le Makefile utilise des commandes Unix: sous Windows, passez par WSL ou adaptez les cibles en PowerShell (se renseigner en ligne ou demander à une LLM en cas de problème sous Windows).

## Installation

```bash
make install
```

Cette commande crée l'environnement virtuel, installe les dépendances aux versions figées de requirements.txt (un parcours unique et reproductible, utilisé aussi par la CI) et copie le fichier de configuration. Sans uv, pip fait l'affaire.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps
cp .env.example .env
```

La copie du .env n'est pas optionnelle: c'est là que se règlent les seuils de mise en service et la durée du cache du modèle servi.

## Démonstration complète

Tout ce qui suit a été exécuté et vérifié. Les commandes se lancent depuis la racine du dépot, dans des terminaux séparés quand c'est indiqué.

On commence par le serveur MLflow, qui gère le suivi des expérimentations et le registre de modèles (terminal 1).

```bash
make mlflow-server
```

Puis on lance la pipeline d'entrainement (terminal 2). Le premier appel entraine sur le premier lot et met en place une première version en production. Les appels suivants simulent l'arrivée des données: le modèle se réentraine sur les lots cumulés, et une nouvelle version ne passe en production que si elle bat l'actuelle.

```bash
make retrain N=1
make retrain N=3
make retrain N=5
```

On démarre ensuite l'API de prédiction (terminal 3). Le mode multiprocess de Prometheus est activé, les compteurs de tous les workers sont agrégés sur /metrics.

```bash
make serve-api WORKERS=2
```

Une prédiction se fait ainsi, avec un exemple d'étudiant fourni dans le dépot.

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @load_tests/exemple_etudiant.json
```

La réponse contient la classe prédite, les probabilités par classe, la version du modèle utilisée et la latence. On peut ensuite renvoyer la classe réelle observée, ce qui alimente le suivi de la qualité du service.

```bash
curl -X POST localhost:8000/feedback/1 -H "Content-Type: application/json" -d '{"classe_reelle": "Graduate"}'
```

L'API expose aussi /health (le processus répond), /ready (le modèle de production est effectivement chargeable, 503 sinon), /model/info (la version réellement servie par le cache), /stats (volumes, distributions, taux d'erreur, erreurs 422 incluses) et /pipeline/status (état du dernier réentrainement Prefect).

Pour montrer que la chaine tient face à un problème, la démo d'incident exécute le flow Prefect avec un lot corrompu, injecté après la construction des lots. Les controles qualité doivent faire passer la flow en état Failed et déclencher une alerte, sans créer de version ni toucher au modèle en production, qui continue de servir des prédictions avant, pendant et après l'incident.

```bash
make demo-incident
```

La preuve complète (état Prefect Failed, registre avant/après, prédictions de production avant/après) est sauvegardée dans artifacts/demo_incident/.

Le retour arrière redéploie la version précédente du modèle, en s'appuyant sur l'historique explicite des déploiements (artifacts/registry/deployments.json).

```bash
make rollback
```

La supervision se lance avec Docker. Le dashboard Grafana est provisionné automatiquement (http://localhost:3000, identifiant admin, mot de passe decrochage). La configuration utilise network_mode: host et suppose un environnement Docker Linux (sous Windows, Docker Desktop avec WSL2).

```bash
make monitoring
```

Enfin, les tests de charge s'exécutent avec Locust, soit via son interface web, soit en mode chiffré pour le protocole de dimensionnement. Le scénario garantit 90% de prédictions unitaires et 10% de lots de 20 étudiants, tirés à chaque requête. Les CSV de résultats sont conservés dans load_tests/results/ comme preuves.

```bash
make loadtest
make loadtest-headless USERS=50 SPAWN_RATE=5 DURATION=60s
```

Les résultats chiffrés, le dimensionnement recommandé et les limites connues sont détaillés dans RAPPORT.md.

## Commandes utiles

Quelques rappels, dans l'ordre d'un usage courant:

- `make install`: environnement virtuel et dépendances figées (requirements.txt)
- `make test`: les 67 tests automatisés (pytest), dont un test d'intégration réel avec le registre MLflow et un modèle entrainé
- `make lint`: la qualité de code (ruff)
- `make mlflow-server`: le serveur MLflow (suivi et registre)
- `make retrain N=x`: la pipeline complète sur x lots
- `make serve-pipeline`: le déclenchement automatique hebdomadaire (scheduler Prefect)
- `make serve-api WORKERS=x`: l'API de prédiction (métriques agrégées entre workers)
- `make demo-incident`: la démo d'incident (flow Prefect complète)
- `make rollback`: le retour arrière vers la version précédente
- `make loadtest`: le test de charge (interface Locust)
- `make monitoring`: Prometheus et Grafana

## Organisation du code

Le code métier vit dans le package src/dropout_pred. Le module config.py centralise la configuration lue dans le fichier .env. Le dossier data s'occupe de l'ingestion, du découpage en lots, des controles qualité et du prétraitement. Le dossier training gère l'entrainement et la tracabilité MLflow, evaluation les métriques et la règle de mise en service. Les modules registry.py, alerting.py et rollback.py gèrent respectivement le registre de modèles, les alertes et le retour arrière. Le dossier api contient l'API FastAPI avec ses schémas de validation, sa persistance et ses métriques.

A coté du package, vous trouverez pipelines/retrain.py pour la flow Prefect, scripts/demo_incident.py pour la démo d'incident, load_tests/ pour les scénarios de charge, monitoring/ pour la configuration Prometheus et Grafana, et tests/ pour les tests unitaires et d'API. La CI GitHub Actions se trouve dans .github/workflows/ci.yml.

## Tests et intégration continue

```bash
make test
make lint
```

Les tests couvrent les controles qualité (types, entiers, domaines catégoriels), le prétraitement, le registre de modèles (promotion, historique, retour arrière), l'état des traitements et l'API: validation des entrées, erreurs 422 persistées, pannes de chargement comptées en 500 et persistées, readiness, et un test d'intégration qui entraine un vrai modèle, l'enregistre dans un registre MLflow local et le sert sans aucun mock. La CI rejoue ruff et pytest à chaque push et sur chaque pull request, avec les dépendances figées de requirements.txt.

## Configuration

Toute la configuration se trouve dans le fichier .env, à copier depuis .env.example: adresses des services, seuils de mise en service (MIN_F1_MACRO et MIN_RECALL_DROPOUT), durée du cache du modèle servi, paramètres SMTP des alertes. Aucun secret n'est versionné dans ce dépot.
