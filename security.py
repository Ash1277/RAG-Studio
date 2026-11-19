"""OS credential-store integration for the user's Groq API key."""
from __future__ import annotations
import os

SERVICE = "RAG Studio"
ACCOUNT = "groq-api-key"

_PLACEHOLDERS = ("put_your_", "your_", "paste_", "gsk_your", "<", ">")


class CredentialStoreError(RuntimeError):
    pass


def _is_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    if not lowered or lowered == "put_your_groq_api_key_here":
        return True
    return any(tok in lowered for tok in _PLACEHOLDERS)


def get_api_key_info() -> dict[str, str | bool | None]:
    """Return key configuration status and source without ever exposing the secret value."""
    try:
        import keyring
        value = keyring.get_password(SERVICE, ACCOUNT)
        if value and value.strip() and not _is_placeholder(value):
            return {"configured": True, "source": "OS credential store"}
    except Exception:
        pass

    env_val = os.getenv("GROQ_API_KEY", "").strip()
    if env_val and not _is_placeholder(env_val) and len(env_val) >= 20:
        return {"configured": True, "source": "developer environment (.env)"}
    return {"configured": False, "source": None}


def get_api_key() -> str:
    """Return the active Groq API key from OS keyring or backward-compatible environment fallback."""
    try:
        import keyring
        value = keyring.get_password(SERVICE, ACCOUNT)
        if value and value.strip() and not _is_placeholder(value):
            return value.strip()
    except Exception:
        pass

    value = os.getenv("GROQ_API_KEY", "").strip()
    if value and not _is_placeholder(value) and len(value) >= 20:
        return value
    return ""


def set_api_key(value: str) -> str:
    value = (value or "").strip()
    if len(value) < 20 or any(ch.isspace() for ch in value) or _is_placeholder(value):
        raise ValueError("Enter a valid-looking Groq API key (at least 20 characters, no spaces).")
    try:
        import keyring
        backend = keyring.get_keyring()
        backend_name = type(backend).__name__.lower()
        if "fail" in backend_name or "null" in backend_name:
            raise CredentialStoreError(
                "No active operating-system credential store backend is available."
            )
        keyring.set_password(SERVICE, ACCOUNT, value)
        stored = keyring.get_password(SERVICE, ACCOUNT)
        if stored != value:
            raise CredentialStoreError("The operating-system credential store did not confirm the saved key.")
        return type(backend).__name__
    except CredentialStoreError:
        raise
    except Exception as exc:
        raise CredentialStoreError(
            "Could not access the operating-system credential store. On Linux, install and unlock Secret Service (GNOME Keyring/KWallet), then retry."
        ) from exc


def delete_api_key() -> None:
    try:
        import keyring
        try:
            keyring.delete_password(SERVICE, ACCOUNT)
        except keyring.errors.PasswordDeleteError:
            pass
    except Exception as exc:
        raise CredentialStoreError("Could not access the operating-system credential store.") from exc


def validate_api_key(value: str) -> bool:
    """Validate the supplied key against Groq's models endpoint without logging the secret."""
    value = (value or "").strip()
    if len(value) < 20 or any(ch.isspace() for ch in value) or _is_placeholder(value):
        raise ValueError("Enter a valid-looking Groq API key (at least 20 characters, no spaces).")
    import requests
    try:
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {value}"},
            timeout=12,
        )
    except Exception as exc:
        raise ValueError("Could not reach Groq to validate the key. Check your internet connection and retry.") from exc
    if response.status_code in (401, 403):
        raise ValueError("Groq rejected this API key. Check that it is active and copied correctly.")
    if not getattr(response, "ok", False):
        raise ValueError(f"Groq key validation could not complete (HTTP {response.status_code}). Try again later.")
    return True
