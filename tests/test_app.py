import json
import os
import sys
import unittest
from unittest.mock import MagicMock


def _make_event(method="GET", path="/health", body=None, api_key=None):
    headers = {}
    if api_key is not None:
        headers["x-api-key"] = api_key
    return {
        "requestContext": {"http": {"method": method}},
        "rawPath": path,
        "headers": headers,
        "body": json.dumps(body) if body is not None else "",
    }


def _load_app(env_overrides=None):
    """Import src/app.py with patched env vars and a mocked boto3."""
    env = {
        "ALLOWED_MODELS": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "MAX_TOKENS_LIMIT": "2000",
        "ADAPTER_API_KEY": "test-key",
        "LOG_PROMPTS": "false",
        "BEDROCK_REGION": "us-east-1",
        "CORS_ENABLED": "false",
    }
    if env_overrides:
        env.update(env_overrides)

    # Remove cached module so env vars are re-read on import
    for key in list(sys.modules.keys()):
        if key == "app" or key.endswith(".app"):
            del sys.modules[key]

    mock_boto3 = MagicMock()
    src_path = os.path.join(os.path.dirname(__file__), "..", "src")

    import importlib
    from unittest.mock import patch

    with patch.dict(os.environ, env, clear=False):
        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            sys.path.insert(0, os.path.abspath(src_path))
            import app
            sys.path.pop(0)
    return app, mock_boto3


class TestHealthRoute(unittest.TestCase):
    def test_health_returns_ok(self):
        app, _ = _load_app()
        event = _make_event("GET", "/health")
        resp = app.lambda_handler(event, None)
        self.assertEqual(resp["statusCode"], 200)
        body = json.loads(resp["body"])
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["service"], "infer-bedrock")

    def test_unknown_route_returns_404(self):
        app, _ = _load_app()
        event = _make_event("GET", "/unknown")
        resp = app.lambda_handler(event, None)
        self.assertEqual(resp["statusCode"], 404)
        body = json.loads(resp["body"])
        self.assertEqual(body["error"]["code"], "not_found")


class TestApiKeyAuth(unittest.TestCase):
    def test_missing_api_key_returns_401(self):
        app, _ = _load_app()
        event = _make_event("POST", "/v1/chat/completions", body={"model": "x", "messages": []})
        resp = app.lambda_handler(event, None)
        self.assertEqual(resp["statusCode"], 401)
        body = json.loads(resp["body"])
        self.assertEqual(body["error"]["code"], "unauthorized")

    def test_wrong_api_key_returns_401(self):
        app, _ = _load_app()
        event = _make_event("POST", "/v1/chat/completions", body={}, api_key="wrong")
        resp = app.lambda_handler(event, None)
        self.assertEqual(resp["statusCode"], 401)

    def test_no_key_required_when_env_empty(self):
        app, mock_boto3 = _load_app({"ADAPTER_API_KEY": ""})
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 2},
        }
        body = {
            "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 100,
        }
        event = _make_event("POST", "/v1/chat/completions", body=body)
        resp = app.lambda_handler(event, None)
        self.assertEqual(resp["statusCode"], 200)


class TestChatCompletionsValidation(unittest.TestCase):
    def _post(self, body, api_key="test-key"):
        app, _ = _load_app()
        event = _make_event("POST", "/v1/chat/completions", body=body, api_key=api_key)
        return app.lambda_handler(event, None)

    def test_invalid_json_returns_400(self):
        app, _ = _load_app()
        event = {
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/v1/chat/completions",
            "headers": {"x-api-key": "test-key"},
            "body": "not-json{{{",
        }
        resp = app.lambda_handler(event, None)
        self.assertEqual(resp["statusCode"], 400)
        body = json.loads(resp["body"])
        self.assertEqual(body["error"]["code"], "invalid_json")

    def test_disallowed_model_returns_400(self):
        resp = self._post({
            "model": "openai.gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
        })
        self.assertEqual(resp["statusCode"], 400)
        body = json.loads(resp["body"])
        self.assertEqual(body["error"]["code"], "disallowed_model")

    def test_max_tokens_exceeded_returns_400(self):
        resp = self._post({
            "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 9999,
        })
        self.assertEqual(resp["statusCode"], 400)
        body = json.loads(resp["body"])
        self.assertEqual(body["error"]["code"], "max_tokens_exceeded")

    def test_missing_messages_returns_400(self):
        resp = self._post({"model": "anthropic.claude-3-5-sonnet-20240620-v1:0"})
        self.assertEqual(resp["statusCode"], 400)

    def test_non_string_content_returns_400(self):
        resp = self._post({
            "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "messages": [{"role": "user", "content": ["list", "content"]}],
        })
        self.assertEqual(resp["statusCode"], 400)
        body = json.loads(resp["body"])
        self.assertEqual(body["error"]["code"], "invalid_messages")


class TestMessageConversion(unittest.TestCase):
    def setUp(self):
        app, _ = _load_app()
        self.convert = app.convert_messages

    def test_user_message(self):
        sys_blocks, msgs = self.convert([{"role": "user", "content": "Hello"}])
        self.assertEqual(sys_blocks, [])
        self.assertEqual(msgs, [{"role": "user", "content": [{"text": "Hello"}]}])

    def test_system_message_extracted(self):
        sys_blocks, msgs = self.convert([
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ])
        self.assertEqual(sys_blocks, [{"text": "Be concise."}])
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")

    def test_assistant_message(self):
        _, msgs = self.convert([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello there"},
        ])
        self.assertEqual(msgs[1]["role"], "assistant")
        self.assertEqual(msgs[1]["content"][0]["text"], "Hello there")

    def test_non_string_content_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.convert([{"role": "user", "content": 42}])
        self.assertIn("must be a string", str(ctx.exception))

    def test_unknown_role_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.convert([{"role": "tool", "content": "result"}])
        self.assertIn("not supported", str(ctx.exception))


class TestHappyPath(unittest.TestCase):
    def test_successful_converse_call(self):
        app, mock_boto3 = _load_app()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "Hello, world!"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

        body = {
            "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Say hello."},
            ],
            "max_tokens": 100,
            "temperature": 0.2,
        }
        event = _make_event("POST", "/v1/chat/completions", body=body, api_key="test-key")
        resp = app.lambda_handler(event, None)

        self.assertEqual(resp["statusCode"], 200)
        result = json.loads(resp["body"])
        self.assertEqual(result["object"], "chat.completion")
        self.assertEqual(result["choices"][0]["message"]["content"], "Hello, world!")
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        self.assertEqual(result["usage"]["input_tokens"], 10)
        self.assertEqual(result["usage"]["output_tokens"], 5)
        self.assertEqual(result["usage"]["total_tokens"], 15)

        call_kwargs = mock_client.converse.call_args.kwargs
        self.assertIn("system", call_kwargs)
        self.assertEqual(call_kwargs["system"], [{"text": "Be concise."}])


if __name__ == "__main__":
    unittest.main()
