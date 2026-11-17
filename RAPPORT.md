# Choix techniques et résultats OPS

Ce document résume les décisions prises, les preuves que la chaine fonctionne de bout en bout, les résultats des tests de charge et la taille de machine recommandée. Toutes les mesures viennent d'exécutions réelles, sur un ordinateur de développement (Intel Core i5-10210U, 4 coeurs / 8 threads, 16 Go de RAM, sous Linux). Les commandes pour refaire les mêmes exécutions sont dans le README.

## 1. Le sujet

L'objectif est de prédire si un étudiant va décrocher, continuer ses études ou réussir, à partir de son dossier (dataset UCI n° 697: 4 425 lignes, 36 variables, 3 classes). Le sujet précise qu'on n'évalue pas la performance du modèle, nous avons donc pris un modèle simple et passé l'essentiel de l'effort sur la chaîne autour du modèle. Le dataset est une photo figée du passé, pour faire comme si de nouvelles données arrivaient en continu, nous le redécoupons en 5 lots d'environ 800 lignes (sans le jeu de validation) et nous les rejouons un par un. Le sujet l'autorise.

## 2. Choix techniques, et pourquoi

Le code est un package Python découpé en modules, sans aucun notebook. Toute la configuration est centralisée dans un fichier .env, lu par pydantic-settings. C'est la structure vue en cours, et elle garantit qu'on n'a jamais besoin de régler quelque chose à la main pour que ça marche. Les dépendances sont figées dans requirements.txt, versions directes et transitives. Le Makefile et la CI installent exactement ces versions, donc tous les environnements sont identiques. Coté qualité nous avons mis en place, 67 tests pytest, le linter ruff, et une CI GitHub Actions qui rejoue les deux à chaque push. Si un test ou le lint casse, la CI casse.

Coté données, on va chercher les lots quand on en a besoin. Le fichier source contient un BOM UTF-8 et des noms de colonnes avec des tabulations parasites. L'ingestion corrige ça une bonne fois pour toutes, puis calcule une empreinte SHA-256 des données. Cette empreinte est enregistrée dans chaque run MLflow, on sait donc toujours quelles données exactes ont servi. Les controles qualité sont dix règles bloquantes écrites dans un module interne (schéma, volume, valeurs manquantes, doublons, types numériques, plages de valeurs, valeurs discrètes, domaines catégoriels, cible, représentativité des classes) plutôt qu'un framework externe. Les règles restent lisibles, testées une par une, et chaque controle produit un rapport JSON gardé comme preuve. Les types sont vérifiés explicitement. Par exemple, un texte dans une colonne numérique que la conversion transformerait silencieusement en NaN est bloqué. Sont bloqués aussi, une valeur non finie, un nombre à virgule dans une colonne d'entiers, ou une valeur interdite pour une variable binaire ou catégorielle (les domaines viennent du codebook UCI, par exemple Gender ne peut valoir que 0 ou 1, et l'API le rejette aussi via pydantic). Si une règle n'est pas respectée, tout s'arrête avant l'entrainement. Le prétraitement (imputation, standardisation, modèle) est un Pipeline scikit-learn enregistré avec le modèle, ce qui est appliqué en production est exactement ce qui a servi à l'entrainement, et une colonne inconnue en entrée est simplement ignorée, sans faire tomber le service.

L'orchestration est faite avec Prefect 3. La flow enchaîne ingestion, découpage en lots, contrôles, prétraitement, entrainement, évaluation, puis la décision de mise en production. L'ingestion est retentée 5 fois en cas de problème, le prétraitement est mis en cache une heure, et toute erreur déclenche une alerte. MLflow sert de carnet de bord. Chaque run y enregistre ses paramètres, l'empreinte des données, les métriques, le rapport de contrôles et la matrice de confusion en image. Pour la mise en production, on reprend le choix vu en cours, c'est à dire, chaque version du registre porte un tag stage (release_candidate, production ou archive), et l'API charge le modèle via ce tag. Promouvoir un modèle ne demande ni redéploiement ni changement de code, et les seuils à respecter sont configurables. Pendant une promotion, la nouvelle version est taggée production avant que l'ancienne soit archivée. Si quelque chose rate au milieu, il reste toujours une version servie. Chaque déploiement est aussi noté dans un historique (artifacts/registry/deployments.json) qui sert de base au retour arrière. Enfin, le flow écrit son état (réussi ou non, durée, version produite) dans artifacts/pipeline/state.json, exposé par l'API et affiché par Grafana et on supervise les traitements, pas seulement le service.

L'API est en FastAPI. Elle valide strictement les entrées avec pydantic, garde une trace de chaque prédiction en SQLite, et expose des métriques Prometheus. Les tests de charge utilisent Locust avec un scénario qui garantit 90% de prédictions unitaires et 10% de lots de 20 étudiants, tirés au hasard à chaque requête. La supervision repose sur Prometheus et Grafana, déployés par docker compose, avec un dashboard installé automatiquement. Les trois niveaux d'observabilité vus en cours sont couverts, le service est-il up, comment se comporte l'API (débit, latences, erreurs), et que prédit-on (prédictions par classe, feedbacks) plus l'état des traitements Prefect.

## 3. La chaine complète en conditions normales

La chaine a vraiment été exécutée, make retrain avec 1, puis 3, puis 5 lots. A chaque fois, Prefect enchaine toutes les étapes et une nouvelle version apparait dans le registre. La version 1, entrainée sur le premier lot, atteint un F1 macro de 0.680, un rappel Dropout de 0.692, et 0.697 de F1 macro sur le jeu de validation. Elle passe donc en production. La version 2, entrainée sur les trois premiers lots, atteint 0.708 sur le même jeu de validation, mieux que la production (0.697). Elle est donc promue. La version 3, entrainée sur les cinq lots, atteint 0.707, moins bien que la production (0.708). Elle reste donc en release_candidate.

La règle de mise en production tient en deux conditions, d'abord dépasser les seuils configurés (F1 macro d'au moins 0.45 et rappel Dropout d'au moins 0.60, mesurés sur le jeu de test), ensuite battre le F1 macro de la production sur le jeu de validation commun. Ce jeu est un échantillon stratifié de 10% du dataset, jamais utilisé pour l'entrainement. Le candidat et la production sont donc comparés sur exactement les mêmes observations, quel que soit le nombre de lots déjà consommés. La décision complète (choix, justification, seuils, version comparée, F1 des deux modèles, identifiant de l'exécution Prefect) est notée dans le run MLflow du candidat, en paramètres et tags.

L'API sert ensuite la version taggée production, et la dit dans chaque réponse :

```json
{
	"prediction_id": 1,
	"classe_predite": "Graduate",
	"probabilites": {
		"Dropout": 0.068,
		"Enrolled": 0.2868,
		"Graduate": 0.6452
	},
	"model_version": "2",
	"latence_ms": 25.04
}
```

Le réentrainement se déclenche tout seul via le scheduler Prefect (make serve-pipeline, une fois par semaine). Le déploiement reentrainement_hebdomadaire a été créé et vérifié en écoute.

## 4. Fiabilité et incident simulé

La démo d'incident exécute le flow Prefect complète (retrain_flow) avec un point d'injection de corruption placé après la construction des lots (paramètre corrupt_data). Elle imite un bug de collecte en amont, que la reconstruction des lots ne peut pas écraser. Le lot injecté contient un texte dans une colonne numérique, une valeur interdite pour une variable binaire, et des classes cibles écrasées.

Voici le résultat réel de l'exécution (make demo-incident):

```
Production avant l'incident: version 2
Versions enregistrées avant: ['1', '2', '3']
Prédiction de production avant: version 2, classe Graduate

Injection d'un lot corrompu et exécution de la flow Prefect...

État final de la flow Prefect: Failed (échec: True)

=== VERIFICATIONS ===
1. Aucune nouvelle version créée: OK (3 versions avant et après)
2. Production inchangée: OK (version 2)
3. La production sert toujours: version 2, classe Graduate (identique à avant)
4. Alerte enregistrée: OK (artifacts/alerts/alerts.log)
5. État du traitement exposé à la supervision: echec

RESULTAT: incident maitrisé
Preuve sauvegardée: artifacts/demo_incident/preuve_incident.txt
```

Concrètement, les controles qualité ont bloqué sur trois règles (types_numeriques: texte dans Admission grade, domaines_categoriques: Gender=9 hors du domaine {0, 1}, representativite_classes: deux classes disparues), le flow est passée en Failed, l'alerte a été enregistrée (elle partirait par mail si un serveur SMTP était configuré), le rapport de non-conformité est gardé comme preuve, et le modèle en production a continué à servir exactement la même prédiction avant et après l'incident. L'échec est visible dans la supervision (dropout_pipeline_last_run_success à 0 sur le dashboard Grafana). Les retries de l'ingestion couvrent de leur coté le cas où la source UCI est indisponible quelques instants.

## 5. Tracabilité et retour arrière

Chaque run MLflow contient les paramètres (nombre de lots, empreinte SHA-256 des données, type de modèle), les métriques dont le F1 macro sur le jeu de validation commun, le rapport de controles qualité, la matrice de confusion en image, et la décision de mise en production (décision, seuils, version comparée, F1 des deux modèles, identifiant de l'exécution Prefect). Le registre garde l'historique des versions avec leurs tags, et chaque déploiement est noté dans l'historique (artifacts/registry/deployments.json). Le retour arrière s'appuie sur cet historique (pas sur le numéro de la version archivée) et a vraiment été démontré.

```
Versions enregistrées :
  v3 — stage=release_candidate
  v2 — stage=production
  v1 — stage=archive

Retour arrière effectué : production v2 -> v1
```

Un second retour arrière a ensuite remis la v2 en production. Le mécanisme restaure à chaque fois la production qui précédait le dernier déploiement noté, donc le second ramène la version que le premier venait d'archiver. L'API recharge le modèle au pire à l'expiration de son cache (300 secondes), et tout de suite via POST /admin/reload (changer de version ne demande aucun redéploiement).

## 6. Le service au quotidien

Les prédictions se font une par une (POST /predict) ou en lot (POST /predict/batch, 100 étudiants maximum par requête). Les entrées sont validées par pydantic, un champ manquant, une valeur hors plage ou interdite (par exemple Gender=9) renvoie une erreur 422 claire, et cette erreur est elle aussi tracée en SQLite et comptée dans /stats. Si le modèle ne charge pas, l'API renvoie une 500, tracée en SQLite et comptée par Prometheus, un flag interne évite d'écrire la trace deux fois. Après les tests de charge, la base contenait 36404 prédictions avec un taux d'erreur de 0.07%, dont 23 erreurs "database is locked" pendant la salve violente, gardées avec leur cause et visibles dans /stats.

L'utilisateur peut signaler ce qui s'est vraiment passé via POST /feedback, qui enregistre la classe réelle observée et la relie à la prédiction concernée. L'endpoint GET /stats résume volumes, distributions et taux d'erreur. C'est la matière première pour suivre la dérive et préparer les réentrainements futurs.

Pour la supervision, l'API expose ses métriques au format Prometheus et le dashboard Grafana affiche le débit, les latences p50, p95 et p99, le taux d'erreur, les prédictions par classe, les feedbacks, les rechargements de modèle, la disponibilité et l'état des traitements Prefect (dernier succès, durée, réussite du dernier run). Deux points ont été vérifiés en vrai, /ready répond 503 tant que le modèle de production n'est pas chargeable (le /health reste un simple "je suis vivant"), et les compteurs Prometheus sont bien additionnés entre workers grace au mode multiprocess, avec 2 workers, la somme api_predictions_total lue sur /metrics (8551) correspond au nombre de prédictions ajoutées en SQLite sur la même période. Enfin, /model/info interroge la version réellement en cache (pas le registre) pendant le TTL, l'API annonce vraiment ce qu'elle sert.

## 7. Tests de charge et taille recommandée

Le protocole utilise Locust en mode headless, 60 secondes par configuration, avec une charge mixte réaliste (90% de prédictions unitaires et 10% de lots de 20 étudiants). La répartition est tirée à chaque requête par le scénario. Elle porte donc sur les requêtes, indépendamment des temps d'attente et du nombre d'utilisateurs (un simple poids entre classes Locust ne jouerait que sur la répartition des utilisateurs). Elle a été mesurée à chaque exécution (90.9 / 9.1, 90.4 / 9.6, 90.4 / 9.6 et 92.5 / 7.5) pour les quatre configurations. Trois configurations de workers ont été comparées, puis une salve plus violente pour trouver la limite. Les CSV bruts sont conservés dans load_tests/results/. Les mesures ci-dessous en sont extraites directement.

Avec 1 worker et 50 utilisateurs, l'API a traité 2678 requêtes sans aucune erreur, à 45.3 requêtes par seconde, médiane à 80ms et p95 à 1500ms sur les prédictions unitaires. Avec 2 workers et les mêmes 50 utilisateurs, on a 2802 requêtes sans erreur, 47.4 requêtes par seconde, médiane à 53ms et p95 à 1400ms. Avec 4 workers, le gain devient faible, on a 2934 requêtes, 49.7 requêtes par seconde, médiane à 46ms, p95 à 1500ms. Enfin, avec 4 workers et 200 utilisateurs, la salve violente a atteint 62.5 requêtes par seconde pour 3687 requêtes, mais 17 erreurs sont apparues (0.46%), avec une médiane à 620ms et un p95 à 4800ms sur les prédictions unitaires.

Les médianes sont bonnes, les p95 beaucoup moins. Les lenteurs viennent des lots de 20 étudiants, dont le p95 va de 7.8s (4 workers, 50 utilisateurs) à 9.3s (1 worker), et monte à 30s pendant la salve violente. Pour dimensionner, il faut donc regarder les p95 autant que les médianes.

Ces mesures se lisent simplement. Le problème n'est pas le CPU mais l'écriture SQLite. Chaque prédiction est enregistrée, et les workers se battent sur le verrou de la base. Passer de 1 à 2 workers n'améliore le débit que de 5%, et 4 workers de 10% de plus. A 200 utilisateurs, les erreurs qui apparaissent sont des verrous SQLite sous trop forte concurrence d'écriture (OperationalError: database is locked). C'est la limite de SQLite, pas celle du modèle. Ces erreurs ont été retrouvées coté serveur, la base garde 23 traces "database is locked" sur la fenêtre de la salve, et les compteurs Prometheus api_requests_total {statut="500"} sont additionnés sur les 4 workers. Les deux comptages ne mesurent pas la même chose (requêtes coté client, traces coté base) et l'écart entre 17 et 23 n'a pas été rapproché requête par requête.

La taille recommandée, au vu de ces mesures, est un conteneur de 2 vCPU et 1 Go de RAM avec 2 workers uvicorn. C'est le point à partir du quel ajouter des workers n'apporte plus de débit mesurable, et cette configuration encaisse environ 47 requêtes par seconde en charge mixte sans aucune erreur, au prix d'un p95 autour de 1.4s sur les prédictions unitaires. Ce budget ne couvre que l'API. MLflow, Prefect et la supervision tournent dans des processus séparés, non comptés ici. Pour aller plus loin, la prochaine étape est de remplacer SQLite par PostgreSQL pour supprimer les lenteurs d'écriture, puis de mettre en place un autoscaling horizontal plafonné des pods, à revalider par une nouvelle salve de charge comme vu en cours.

## 8. Limites connues

SQLite convient parfaitement à la démonstration mais devient un problème au-delà d'environ 60 requêtes par seconde en écriture intensive. Passer à PostgreSQL est la première étape prévue. Le rejeu de lots découpe l'historique en lots statiques. Un vrai flux de données datées permettrait un suivi de dérive plus fin. Les alertes sont écrites dans un fichier de log. L'envoi par mail est implémenté mais n'a pas pu être activé, faute de serveur SMTP en local. La CI rejoue le lint et les tests, mais pas l'entrainement complet, car le serveur MLflow n'est pas encore instancié dans la CI. Enfin, l'agrégation Prometheus multi-workers utilise le mode multiprocess de prometheus_client, c'est à dire, les fichiers de compteurs (.prometheus_multiproc/) sont nettoyés au démarrage par le Makefile, mais un worker redémarré sans nettoyage garderait des compteurs obsolètes. Un exporter dédié serait plus propre au-delà de la démonstration.
