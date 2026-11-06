"""Schéma attendu du dataset et constantes partagées (colonnes, cible, plages de valeurs)."""

# Cible : 3 classes
TARGET_COLUMN = "Target"
TARGET_CLASSES = ["Dropout", "Enrolled", "Graduate"]

# Colonnes numériques continues : plages de valeurs attendues (bornes inclusives)
CONTINUOUS_RANGES = {
    "Previous qualification (grade)": (0.0, 200.0),
    "Admission grade": (0.0, 200.0),
    "Curricular units 1st sem (grade)": (0.0, 20.0),
    "Curricular units 2nd sem (grade)": (0.0, 20.0),
    "Unemployment rate": (0.0, 40.0),
    "Inflation rate": (-5.0, 20.0),
    "GDP": (-10.0, 10.0),
    "Age at enrollment": (15.0, 100.0),
}

# Colonnes numériques discrètes : valeurs entières >= 0
DISCRETE_NON_NEGATIVE = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (without evaluations)",
]

FEATURE_COLUMNS = DISCRETE_NON_NEGATIVE + list(CONTINUOUS_RANGES.keys())
assert len(FEATURE_COLUMNS) == 36

# Règles de qualité : seuils configurés pour un lot de données
MIN_ROWS_PER_BATCH = 300
MIN_CLASS_SHARE = 0.05  # chaque classe doit représenter au moins 5 % des lignes
MAX_MISSING_SHARE = 0.01  # tolérance de valeurs manquantes avant blocage
MAX_DUPLICATE_SHARE = 0.02  # tolérance de doublons avant blocage
