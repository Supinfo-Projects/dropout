"""Test de charge Locust : prédictions en ligne et en lot sur l'API.

Scénarios :
- PredictionEnLigne : un étudiant par requête (usage interactif) ;
- PredictionEnLot : 20 étudiants par requête (usage par lots).

Lancement : make loadtest (interface web) ou make loadtest-headless (protocole chiffré).
"""

import random

from locust import HttpUser, between, task

# Un étudiant type, tiré du dataset UCI (valeurs réalistes)
ETUDIANT = {
    "Marital status": 1,
    "Application mode": 17,
    "Application order": 5,
    "Course": 171,
    "Daytime/evening attendance": 1,
    "Previous qualification": 1,
    "Previous qualification (grade)": 122.0,
    "Nacionality": 1,
    "Mother's qualification": 19,
    "Father's qualification": 12,
    "Mother's occupation": 5,
    "Father's occupation": 9,
    "Admission grade": 127.3,
    "Displaced": 1,
    "Educational special needs": 0,
    "Debtor": 0,
    "Tuition fees up to date": 1,
    "Gender": 1,
    "Scholarship holder": 0,
    "Age at enrollment": 20,
    "International": 0,
    "Curricular units 1st sem (credited)": 0,
    "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (evaluations)": 6,
    "Curricular units 1st sem (approved)": 5,
    "Curricular units 1st sem (grade)": 13.5,
    "Curricular units 1st sem (without evaluations)": 0,
    "Curricular units 2nd sem (credited)": 0,
    "Curricular units 2nd sem (enrolled)": 6,
    "Curricular units 2nd sem (evaluations)": 6,
    "Curricular units 2nd sem (approved)": 5,
    "Curricular units 2nd sem (grade)": 13.2,
    "Curricular units 2nd sem (without evaluations)": 0,
    "Unemployment rate": 10.8,
    # TODO(guillaume): arthur relis les bornes de Inflation rate avec le codebook UCI
    "Inflation rate": 1.4,
    "GDP": 1.74,
}


def etudiant_aleatoire() -> dict:
    """Fait varier quelques caractéristiques pour éviter un cache applicatif éventuel."""
    etudiant = dict(ETUDIANT)
    etudiant["Admission grade"] = round(random.uniform(80, 180), 1)
    etudiant["Age at enrollment"] = random.randint(17, 40)
    etudiant["Curricular units 1st sem (approved)"] = random.randint(0, 6)
    etudiant["Curricular units 2nd sem (approved)"] = random.randint(0, 6)
    return etudiant


class PredictionEnLigne(HttpUser):
    """Scénario interactif : une prédiction par requête, avec un temps de réflexion."""

    wait_time = between(0.1, 0.5)

    @task(9)
    def predire(self):
        self.client.post("/predict", json=etudiant_aleatoire())

    @task(1)
    def feedback(self):
        # Un utilisateur sur dix consulte les statistiques du service
        self.client.get("/stats")


class PredictionEnLot(HttpUser):
    """Scénario par lots : 20 étudiants par requête, sans temps de réflexion."""

    wait_time = between(0.0, 0.1)

    # TODO(guillaume): arthur prépare la migration PostgreSQL quand SQLite sature
    @task
    def predire_lot(self):
        students = [etudiant_aleatoire() for _ in range(20)]
        self.client.post("/predict/batch", json={"students": students})
