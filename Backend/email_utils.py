"""Email utilities for sending emergency notifications.

This module supports two modes:
- FastAPI-Mail (`EMAIL_BACKEND=fastapi_mail`) — preferred async sender when `fastapi_mail` is installed.
- SMTP fallback (`EMAIL_BACKEND=smtp`) — compatible with the previous implementation using smtplib.

Environment variables (examples):
- EMAIL_BACKEND=fastapi_mail  # or 'smtp'

FastAPI-Mail specific (used when EMAIL_BACKEND=fastapi_mail):
- FASTAPI_MAIL_SERVER
- FASTAPI_MAIL_PORT
- FASTAPI_MAIL_USERNAME
- FASTAPI_MAIL_PASSWORD
- FASTAPI_MAIL_FROM
- FASTAPI_MAIL_TLS (true|false)
- FASTAPI_MAIL_SSL (true|false)

SMTP fallback variables (kept for backward compatibility):
- EMERGENCY_SMTP_HOST
- EMERGENCY_SMTP_PORT
- EMERGENCY_SMTP_USER
- EMERGENCY_SMTP_PASS
- EMERGENCY_FROM_EMAIL
"""
import os
import asyncio
from typing import Optional

# Backward-compatible SMTP constants
SMTP_HOST = os.getenv('EMERGENCY_SMTP_HOST')
SMTP_PORT = int(os.getenv('EMERGENCY_SMTP_PORT') or 587)
SMTP_USER = os.getenv('EMERGENCY_SMTP_USER')
SMTP_PASS = os.getenv('EMERGENCY_SMTP_PASS')
FROM_EMAIL = os.getenv('EMERGENCY_FROM_EMAIL') or SMTP_USER

# New FastAPI-Mail / backend selection
EMAIL_BACKEND = os.getenv('EMAIL_BACKEND', 'fastapi_mail')
FASTAPI_MAIL_SERVER = os.getenv('FASTAPI_MAIL_SERVER')
FASTAPI_MAIL_PORT = int(os.getenv('FASTAPI_MAIL_PORT') or 587)
FASTAPI_MAIL_USERNAME = os.getenv('FASTAPI_MAIL_USERNAME')
FASTAPI_MAIL_PASSWORD = os.getenv('FASTAPI_MAIL_PASSWORD')
FASTAPI_MAIL_FROM = os.getenv('FASTAPI_MAIL_FROM') or FASTAPI_MAIL_USERNAME
FASTAPI_MAIL_TLS = os.getenv('FASTAPI_MAIL_TLS', 'true').lower() in ('1', 'true', 'yes')
FASTAPI_MAIL_SSL = os.getenv('FASTAPI_MAIL_SSL', 'false').lower() in ('1', 'true', 'yes')

# Try to import fastapi_mail — optional dependency
_fastapi_mail_available = False
try:
    from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
    _fastapi_mail_available = True
except Exception:
    _fastapi_mail_available = False


def is_email_configured() -> bool:
    """Return True if at least one email backend is properly configured."""
    if EMAIL_BACKEND == 'fastapi_mail' and _fastapi_mail_available:
        return bool(FASTAPI_MAIL_SERVER and FASTAPI_MAIL_USERNAME and FASTAPI_MAIL_PASSWORD and FASTAPI_MAIL_FROM)
    # Fallback to SMTP
    return bool(SMTP_HOST and SMTP_USER and SMTP_PASS)


async def _async_send_fastmail(to_email: str, subject: str, body_text: str, body_html: Optional[str] = None) -> None:
    """Send email using FastMail async client."""
    if not _fastapi_mail_available:
        raise RuntimeError('fastapi_mail package not installed')

    config = ConnectionConfig(
        MAIL_USERNAME=FASTAPI_MAIL_USERNAME,
        MAIL_PASSWORD=FASTAPI_MAIL_PASSWORD,
        MAIL_FROM=FASTAPI_MAIL_FROM,
        MAIL_PORT=FASTAPI_MAIL_PORT,
        MAIL_SERVER=FASTAPI_MAIL_SERVER,
        MAIL_FROM_NAME=FASTAPI_MAIL_FROM,
        MAIL_STARTTLS=FASTAPI_MAIL_TLS,
        MAIL_SSL_TLS=FASTAPI_MAIL_SSL,
        USE_CREDENTIALS=True,
        TEMPLATE_FOLDER=None,
    )

    # FastMail MessageSchema accepts body + subtype
    body = body_html if body_html else body_text
    subtype = 'html' if body_html else 'plain'

    message = MessageSchema(subject=subject, recipients=[to_email], body=body, subtype=subtype)
    fm = FastMail(config)
    await fm.send_message(message)


def send_emergency_email(to_email: str, subject: str, body_text: str, body_html: Optional[str] = None) -> None:
    """Send an emergency email using the configured backend.

    This function is synchronous for compatibility with background tasks in `app.py`.
    If the FastAPI-Mail backend is selected it will run the async send in an event loop.
    """
    if EMAIL_BACKEND == 'fastapi_mail' and _fastapi_mail_available and FASTAPI_MAIL_SERVER:
        # Run the async send synchronously (safe inside a background thread/task)
        try:
            asyncio.run(_async_send_fastmail(to_email, subject, body_text, body_html))
            return
        except Exception as e:
            raise

    # Fallback to SMTP (legacy behavior)
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASS:
        raise RuntimeError('SMTP configuration missing (EMERGENCY_SMTP_HOST/USER/PASS)')

    # Use standard library for simple delivery
    from email.message import EmailMessage
    import smtplib

    msg = EmailMessage()
    msg['From'] = FROM_EMAIL
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.set_content(body_text)
    if body_html:
        msg.add_alternative(body_html, subtype='html')

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
        smtp.starttls()
        smtp.login(SMTP_USER, SMTP_PASS)
        smtp.send_message(msg)
