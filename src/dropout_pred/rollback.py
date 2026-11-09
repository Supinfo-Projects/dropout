"""Redéploye la version précédente du modèle en production.

Utilisation = python -m dropout_pred.rollback
L'ancienne version en production est archivée, la version archivée la plus récente
redevient production. L'API la sert au rechargement suivant du cache (ou via
POST /admin/reload pour un basculement immédiat).
"""

import sys

from .config import get_settings
from .registry import get_versions, rollback


def main() -> int:
    settings = get_settings()
    versions = get_versions(settings)
    print("Versions enregistrées:")
    for version in versions:
        print(f"  v{version.version} — stage={version.tags.get('stage', 'aucun')}")
    try:
        result = rollback(settings)
    except RuntimeError as exc:
        print(f"Impossible: {exc}")
        return 1
    print(
        f"\nRetour arrière effectué: production v{result['ancienne_production']} -> "
        f"v{result['nouvelle_production']}"
    )
    print("Relancez l'API ou appelez POST /admin/reload pour servir immédiatement cette version.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
