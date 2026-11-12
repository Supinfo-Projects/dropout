"""Démo d'incident contrôlé : un lot de données corrompu doit bloquer l'entraînement.

Scénario (cf. rapport) : un bug en amont corrompt le lot de données du jour —
valeurs impossibles (notes > 200) et une classe surreprésentée. Les contrôles
qualité doivent interrompre le pipeline AVANT l'entraînement et déclencher une
alerte, sans toucher au modèle en production.

Le script restaure les données saines à la fin pour ne pas laisser le dépôt
dans un état dégradé.
"""

import shutil

from dropout_pred.alerting import send_alert
from dropout_pred.config import get_settings
from dropout_pred.data.batches import load_cumulative_batches
from dropout_pred.data.validate import DataValidationError, validate_or_raise

settings = get_settings()
batches_dir = settings.data_dir / "batches"
lot_path = batches_dir / "lot_005.csv"
sauvegarde = batches_dir / "lot_005.csv.sauvegarde"


def corrompre_le_lot() -> None:
    """Simule un bug de collecte : notes impossibles + classe cible écrasée."""
    import pandas as pd

    frame = pd.read_csv(lot_path)
    frame["Admission grade"] = 999.0  # hors plage : la note va de 0 à 200
    frame["Target"] = "Graduate"  # collecte partielle : deux classes ont disparu
    frame.to_csv(lot_path, index=False)
    print("Lot 005 corrompu (notes impossibles + classes manquantes).")


def main() -> None:
    if not lot_path.exists():
        raise SystemExit(
            "Les lots ne sont pas construits. Lancez d'abord : make retrain N=1"
        )

    shutil.copy(lot_path, sauvegarde)
    try:
        corrompre_le_lot()
        frame = load_cumulative_batches(5, settings)
        print(f"Chargement des 5 lots : {len(frame)} lignes")
        print("Lancement des contrôles qualité...\n")
        try:
            validate_or_raise(frame, settings)
            print("ERREUR DE LA DEMO : les données corrompues auraient dû être bloquées !")
        except DataValidationError as exc:
            print("Pipeline interrompue comme prévu :")
            print(exc)
            send_alert(
                "Données non conformes : entraînement interrompu",
                "Le lot 005 contient des valeurs hors plage et des classes manquantes. "
                "Le modèle en production n'a pas été modifié.",
                settings,
            )
            print("\nAlerte enregistrée dans artifacts/alerts/alerts.log")
            print("Le modèle en production est intact : aucune version n'a été créée.")
    finally:
        shutil.move(sauvegarde, lot_path)
        print("\nDonnées saines restaurées.")


if __name__ == "__main__":
    main()
