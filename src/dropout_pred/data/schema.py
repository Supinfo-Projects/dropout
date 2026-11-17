"""Schéma attendu du dataset et constantes partagées."""

TARGET_COLUMN = "Target"
TARGET_CLASSES = ["Dropout", "Enrolled", "Graduate"]

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

BINARY_COLUMNS = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

CATEGORICAL_RANGES = {
    "Marital status": (1, 6),
    "Application mode": (1, 57),
    "Application order": (0, 9),
    "Course": (1, 9991),
    "Previous qualification": (1, 44),
    "Nacionality": (1, 109),
    "Mother's qualification": (1, 44),
    "Father's qualification": (1, 44),
    "Mother's occupation": (0, 195),
    "Father's occupation": (0, 195),
}

FEATURE_COLUMNS = DISCRETE_NON_NEGATIVE + list(CONTINUOUS_RANGES.keys())
assert len(FEATURE_COLUMNS) == 36

MIN_ROWS_PER_BATCH = 300
MIN_CLASS_SHARE = 0.05  # chaque classe doit représenter au moins 5% des lignes
MAX_MISSING_SHARE = 0.01  # tolérance de valeurs manquantes avant blocage
MAX_DUPLICATE_SHARE = 0.02  # tolérance de doublons avant blocage
