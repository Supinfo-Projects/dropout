"""Notification en cas d'échec de pipeline ou de non-conformité des données.

Sans configuration SMTP, les alertes sont écrites dans artifacts/alerts/alerts.log.
Le mécanisme est démontré localement et prêt à être branché sur un vrai serveur mail
en renseignant les variables SMTP_* du fichier .env.
"""

from datetime import datetime, timezone

from .config import Settings, get_settings


def send_alert(subject: str, message: str, settings: Settings | None = None) -> None:
    """Émet une alerte."""
    settings = settings or get_settings()
    alerts_dir = settings.artifacts_dir / "alerts"
    alerts_dir.mkdir(parents=True, exist_ok=True)
    log_path = alerts_dir / "alerts.log"

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = f"[{timestamp}] {subject}\n{message}\n{'-' * 60}\n"
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(entry)

    if settings.alert_email and settings.smtp_host:
        _send_email(settings, subject, message)


def _send_email(settings: Settings, subject: str, message: str) -> None:
    import smtplib
    from email.mime.text import MIMEText

    mail = MIMEText(message, "plain", "utf-8")
    mail["Subject"] = subject
    mail["From"] = settings.smtp_user
    mail["To"] = settings.alert_email
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
        server.starttls()
        if settings.smtp_user and settings.smtp_password:
            server.login(settings.smtp_user, settings.smtp_password)
        server.send_message(mail)
