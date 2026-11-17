import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import Settings, get_settings
from .schema import (
    BINARY_COLUMNS,
    CATEGORICAL_RANGES,
    CONTINUOUS_RANGES,
    DISCRETE_NON_NEGATIVE,
    FEATURE_COLUMNS,
    MAX_DUPLICATE_SHARE,
    MAX_MISSING_SHARE,
    MIN_CLASS_SHARE,
    MIN_ROWS_PER_BATCH,
    TARGET_CLASSES,
    TARGET_COLUMN,
)


class DataValidationError(Exception):
    """Erreur levée quand les données ne sont pas conformes, donc l'entrainement doit être interrompu."""


@dataclass
class RuleResult:
    name: str
    passed: bool
    details: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    passed: bool
    n_rows: int
    n_failed_rules: int
    results: list
    generated_at: str

    def failures(self) -> list[RuleResult]:
        return [result for result in self.results if not result.passed]

    def summary(self) -> str:
        lines = [
            f"Validation des données: {'CONFORME' if self.passed else 'NON CONFORME'}",
            f"  Lignes analysées: {self.n_rows}",
        ]
        for result in self.results:
            status = "OK " if result.passed else "ECHEC"
            lines.append(f"  [{status}] {result.name} {json.dumps(result.details, ensure_ascii=False)}")
        return "\n".join(lines)


def _check_schema(frame: pd.DataFrame) -> RuleResult:
    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in frame.columns]
    return RuleResult("schema", not missing, {"colonnes_absentes": missing})


def _check_volume(frame: pd.DataFrame) -> RuleResult:
    n_rows = len(frame)
    return RuleResult("volume", n_rows >= MIN_ROWS_PER_BATCH, {"lignes": n_rows, "minimum": MIN_ROWS_PER_BATCH})


def _check_missing(frame: pd.DataFrame) -> RuleResult:
    shares = frame.isna().mean().to_dict()
    worst = max(shares.items(), key=lambda item: item[1])
    passed = all(share <= MAX_MISSING_SHARE for share in shares.values())
    return RuleResult(
        "valeurs_manquantes",
        passed,
        {"part_max": round(worst[1], 4), "colonne_max": worst[0], "tolerance": MAX_MISSING_SHARE},
    )


def _check_duplicates(frame: pd.DataFrame) -> RuleResult:
    n_duplicates = int(frame.duplicated().sum())
    share = n_duplicates / max(len(frame), 1)
    return RuleResult(
        "doublons", share <= MAX_DUPLICATE_SHARE, {"doublons": n_duplicates, "part": round(share, 4)}
    )


def _check_types(frame: pd.DataFrame) -> RuleResult:
    violations = {}
    for column in FEATURE_COLUMNS:
        if column not in frame.columns:
            continue
        original = frame[column]
        converted = pd.to_numeric(original, errors="coerce")
        non_numeriques = int(((converted.isna()) & (~original.isna())).sum())
        non_finies = int((converted.abs() == float("inf")).sum())
        if non_numeriques or non_finies:
            violations[column] = {"non_numeriques": non_numeriques, "non_finies": non_finies}
    return RuleResult("types_numeriques", not violations, violations)


def _check_ranges(frame: pd.DataFrame) -> RuleResult:
    violations = {}
    for column, (low, high) in CONTINUOUS_RANGES.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        out_of_range = int(((values < low) | (values > high)).sum())
        if out_of_range:
            violations[column] = {"hors_plage": out_of_range, "plage_attendue": [low, high]}
    return RuleResult("plages_de_valeurs", not violations, violations)


def _check_discrete(frame: pd.DataFrame) -> RuleResult:
    violations = {}
    for column in DISCRETE_NON_NEGATIVE:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = int((values < 0).sum())
        non_entiers = int(((values.dropna() % 1) != 0).sum())
        if invalid or non_entiers:
            violations[column] = {"valeurs_negatives": invalid, "valeurs_non_entieres": non_entiers}
    return RuleResult("valeurs_discretes", not violations, violations)


def _check_domains(frame: pd.DataFrame) -> RuleResult:
    violations = {}
    for column in BINARY_COLUMNS:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = int((~values.isin([0, 1])).sum())
        if invalid:
            violations[column] = {"valeurs_hors_domaine": invalid, "domaine_attendu": [0, 1]}
    for column, (low, high) in CATEGORICAL_RANGES.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = int(((values < low) | (values > high)).sum())
        if invalid:
            violations[column] = {"valeurs_hors_domaine": invalid, "domaine_attendu": [low, high]}
    return RuleResult("domaines_categoriques", not violations, violations)


def _check_target(frame: pd.DataFrame) -> RuleResult:
    if TARGET_COLUMN not in frame.columns:
        return RuleResult("cible", False, {"colonne_absente": TARGET_COLUMN})
    unexpected = sorted(set(frame[TARGET_COLUMN].dropna()) - set(TARGET_CLASSES))
    return RuleResult("cible", not unexpected, {"valeurs_inattendues": unexpected})


def _check_class_distribution(frame: pd.DataFrame) -> RuleResult:
    if TARGET_COLUMN not in frame.columns:
        return RuleResult("representativite_classes", False, {"colonne_absente": TARGET_COLUMN})
    shares = frame[TARGET_COLUMN].value_counts(normalize=True).to_dict()
    missing = [cls for cls in TARGET_CLASSES if shares.get(cls, 0.0) < MIN_CLASS_SHARE]
    return RuleResult(
        "representativite_classes",
        not missing,
        {"parts": {cls: round(shares.get(cls, 0.0), 4) for cls in TARGET_CLASSES}, "sous_représentées": missing},
    )


def validate_frame(frame: pd.DataFrame) -> ValidationReport:
    checks = [
        _check_schema,
        _check_volume,
        _check_missing,
        _check_duplicates,
        _check_types,
        _check_ranges,
        _check_discrete,
        _check_domains,
        _check_target,
        _check_class_distribution,
    ]
    results = [check(frame) for check in checks]
    n_failed = sum(1 for result in results if not result.passed)
    return ValidationReport(
        passed=n_failed == 0,
        n_rows=len(frame),
        n_failed_rules=n_failed,
        results=results,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def validate_or_raise(frame: pd.DataFrame, settings: Settings | None = None) -> tuple[ValidationReport, Path]:
    settings = settings or get_settings()
    report = validate_frame(frame)

    reports_dir = settings.artifacts_dir / "validation"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"validation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(
        json.dumps(
            {
                "passed": report.passed,
                "n_rows": report.n_rows,
                "n_failed_rules": report.n_failed_rules,
                "generated_at": report.generated_at,
                "results": [
                    {"name": result.name, "passed": result.passed, "details": result.details}
                    for result in report.results
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if not report.passed:
        failures = "\n".join(
            f"  - {result.name} : {json.dumps(result.details, ensure_ascii=False)}"
            for result in report.failures()
        )
        raise DataValidationError(
            f"Controles qualité en échec ({report.n_failed_rules} règle(s)) :\n{failures}\n"
            f"Rapport complet: {report_path}"
        )
    return report, report_path
