"""Métriques Prometheus exposées par l'API (niveau API + niveau métier).

Sont suivis : le débit et les latences par endpoint, le taux d'erreur, la distribution
des prédictions par classe, les rechargements de modèle et les feedbacks reçus.
"""

from prometheus_client import Counter, Histogram

REQUESTS_TOTAL = Counter(
    "api_requests_total",
    "Nombre total de requêtes",
    ["endpoint", "statut"],
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latence des requêtes",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PREDICTIONS_TOTAL = Counter(
    "api_predictions_total",
    "Prédictions servies par classe",
    ["classe"],
)
MODEL_RELOADS_TOTAL = Counter(
    "api_model_reloads_total",
    "Rechargements du modèle servi",
)
FEEDBACKS_TOTAL = Counter(
    "api_feedbacks_total",
    "Feedbacks enregistrés",
)
