import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from fastapi.testclient import TestClient
from main import app


class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_and_validation_do_not_call_proxy(self):
        with patch("llm.claude.request") as request:
            self.assertEqual(self.client.get("/health").status_code, 200)
            self.assertEqual(self.client.post("/generate", json={"topic": "AI"}).status_code, 422)
            request.assert_not_called()

    def test_generation_contract_and_proxy_envelope(self):
        body = {"content": [{"type": "text", "text": "A mocked post."}],
                "model": "claude-sonnet-4-5", "usage": {"input_tokens": 12, "output_tokens": 5}}
        with patch("llm.claude.request", return_value=iter([{"type": "response", "body": body}])) as request:
            response = self.client.post("/generate", json={"topic": "Enterprise AI agents"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "A mocked post.", "model": body["model"],
                                           "input_tokens": 12, "output_tokens": 5})
        path, envelope = request.call_args.args
        self.assertEqual(path, "/Library/Application Support/CredentialProxy/run/api.sock")
        self.assertEqual(envelope["integration"], "linkedin-anthropic")
        self.assertEqual(envelope["operation"], "messages.create")
        self.assertIn("Enterprise AI agents", envelope["payload"]["messages"][0]["content"])

    def test_proxy_failures_are_bad_gateway_without_sensitive_details(self):
        for error in (RuntimeError("denied secret detail"), OSError("socket unavailable"),
                      ValueError("invalid JSON"), KeyError("content")):
            with self.subTest(error=type(error).__name__), patch("llm.claude.request", side_effect=error):
                response = self.client.post("/generate", json={"topic": "Enterprise AI agents"})
                self.assertEqual(response.status_code, 502)
                self.assertNotIn(str(error), response.text)

    def test_missing_proxy_response_is_bad_gateway(self):
        with patch("llm.claude.request", return_value=iter([])):
            self.assertEqual(self.client.post("/generate", json={"topic": "Enterprise AI agents"}).status_code, 502)


if __name__ == "__main__":
    unittest.main()
