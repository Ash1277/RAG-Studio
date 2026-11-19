"""OS credential-store and Groq API key validation tests."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _test_shims import install_fastapi_shims
install_fastapi_shims()

import security


class FakeKeyring:
    value = None
    class errors:
        class PasswordDeleteError(Exception):
            pass
    @classmethod
    def get_keyring(cls):
        return cls()


FakeKeyring.get_password = classmethod(lambda cls, s, a: cls.value)
FakeKeyring.set_password = classmethod(lambda cls, s, a, v: setattr(cls, "value", v))
FakeKeyring.delete_password = classmethod(lambda cls, s, a: setattr(cls, "value", None))


class SecurityTests(unittest.TestCase):
    def setUp(self):
        self.old_keyring = sys.modules.get("keyring")
        sys.modules["keyring"] = FakeKeyring
        FakeKeyring.value = None
        self.old_env = os.environ.get("GROQ_API_KEY")
        os.environ.pop("GROQ_API_KEY", None)

    def tearDown(self):
        if self.old_keyring is None:
            sys.modules.pop("keyring", None)
        else:
            sys.modules["keyring"] = self.old_keyring
        if self.old_env is None:
            os.environ.pop("GROQ_API_KEY", None)
        else:
            os.environ["GROQ_API_KEY"] = self.old_env

    def test_key_saved_and_retrieved_from_os_credential_store_adapter(self):
        backend = security.set_api_key("gsk_" + "x" * 40)
        self.assertEqual(backend, "FakeKeyring")
        self.assertEqual(security.get_api_key(), "gsk_" + "x" * 40)
        info = security.get_api_key_info()
        self.assertTrue(info["configured"])
        self.assertEqual(info["source"], "OS credential store")

    def test_env_fallback_used_when_keyring_empty(self):
        os.environ["GROQ_API_KEY"] = "gsk_" + "e" * 40
        self.assertEqual(security.get_api_key(), "gsk_" + "e" * 40)
        info = security.get_api_key_info()
        self.assertTrue(info["configured"])
        self.assertIn("developer environment", info["source"])

    def test_placeholder_env_key_ignored(self):
        os.environ["GROQ_API_KEY"] = "put_your_groq_api_key_here"
        self.assertEqual(security.get_api_key(), "")
        self.assertFalse(security.get_api_key_info()["configured"])

    def test_rejects_short_or_spaced_key(self):
        with self.assertRaises(ValueError):
            security.set_api_key("short")
        with self.assertRaises(ValueError):
            security.set_api_key("gsk_invalid key with spaces 1234567890")

    def test_delete_api_key_clears_credential_store(self):
        security.set_api_key("gsk_" + "z" * 40)
        security.delete_api_key()
        self.assertEqual(security.get_api_key(), "")

    def test_groq_validation_rejects_unauthorized_key(self):
        class Response:
            status_code = 401
            ok = False

        fake_requests = type("requests", (), {"get": staticmethod(lambda *a, **k: Response())})
        with patch.dict(sys.modules, {"requests": fake_requests}):
            with self.assertRaisesRegex(ValueError, "Groq rejected"):
                security.validate_api_key("gsk_" + "x" * 40)


if __name__ == "__main__":
    unittest.main()
