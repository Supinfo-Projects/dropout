"""Schémas de validation des entrées et sorties de l'API."""

from pydantic import BaseModel, Field


class StudentFeatures(BaseModel):
    marital_status: int = Field(alias="Marital status", ge=0)
    application_mode: int = Field(alias="Application mode", ge=0)
    application_order: int = Field(alias="Application order", ge=0)
    course: int = Field(alias="Course", ge=0)
    daytime_evening_attendance: int = Field(alias="Daytime/evening attendance", ge=0, le=1)
    previous_qualification: int = Field(alias="Previous qualification", ge=0)
    previous_qualification_grade: float = Field(
        alias="Previous qualification (grade)", ge=0, le=200
    )
    nacionality: int = Field(alias="Nacionality", ge=0)
    mother_qualification: int = Field(alias="Mother's qualification", ge=0)
    father_qualification: int = Field(alias="Father's qualification", ge=0)
    mother_occupation: int = Field(alias="Mother's occupation", ge=0)
    father_occupation: int = Field(alias="Father's occupation", ge=0)
    admission_grade: float = Field(alias="Admission grade", ge=0, le=200)
    displaced: int = Field(alias="Displaced", ge=0, le=1)
    educational_special_needs: int = Field(alias="Educational special needs", ge=0, le=1)
    debtor: int = Field(alias="Debtor", ge=0, le=1)
    tuition_fees_up_to_date: int = Field(alias="Tuition fees up to date", ge=0, le=1)
    gender: int = Field(alias="Gender", ge=0, le=1)
    scholarship_holder: int = Field(alias="Scholarship holder", ge=0, le=1)
    age_at_enrollment: int = Field(alias="Age at enrollment", ge=15, le=100)
    international: int = Field(alias="International", ge=0, le=1)
    curricular_units_1st_credited: int = Field(alias="Curricular units 1st sem (credited)", ge=0)
    curricular_units_1st_enrolled: int = Field(alias="Curricular units 1st sem (enrolled)", ge=0)
    curricular_units_1st_evaluations: int = Field(alias="Curricular units 1st sem (evaluations)", ge=0)
    curricular_units_1st_approved: int = Field(alias="Curricular units 1st sem (approved)", ge=0)
    curricular_units_1st_grade: float = Field(alias="Curricular units 1st sem (grade)", ge=0, le=20)
    curricular_units_1st_without_evaluations: int = Field(
        alias="Curricular units 1st sem (without evaluations)", ge=0
    )
    curricular_units_2nd_credited: int = Field(alias="Curricular units 2nd sem (credited)", ge=0)
    curricular_units_2nd_enrolled: int = Field(alias="Curricular units 2nd sem (enrolled)", ge=0)
    curricular_units_2nd_evaluations: int = Field(alias="Curricular units 2nd sem (evaluations)", ge=0)
    curricular_units_2nd_approved: int = Field(alias="Curricular units 2nd sem (approved)", ge=0)
    curricular_units_2nd_grade: float = Field(alias="Curricular units 2nd sem (grade)", ge=0, le=20)
    curricular_units_2nd_without_evaluations: int = Field(
        alias="Curricular units 2nd sem (without evaluations)", ge=0
    )
    unemployment_rate: float = Field(alias="Unemployment rate", ge=0, le=40)
    inflation_rate: float = Field(alias="Inflation rate", ge=-5, le=20)
    gdp: float = Field(alias="GDP", ge=-10, le=10)

    model_config = {"populate_by_name": True}


class BatchPredictionRequest(BaseModel):
    students: list[StudentFeatures] = Field(min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    prediction_id: int
    classe_predite: str
    probabilites: dict[str, float]
    model_version: str
    latence_ms: float


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    model_version: str


class FeedbackRequest(BaseModel):
    classe_reelle: str


class ModelInfoResponse(BaseModel):
    nom_modele: str
    version: str
    stage: str
    run_id: str
    f1_macro: float | None = None


class StatsResponse(BaseModel):
    total_predictions: int
    total_feedbacks: int
    distribution_predictions: dict[str, int]
    distribution_classes_reelles: dict[str, int]
    taux_erreur: float
