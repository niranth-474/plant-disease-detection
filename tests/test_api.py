import asyncio
import json
import unittest
from io import BytesIO

from PIL import Image

from backend.web import MAX_IMAGE_BYTES, app
from backend.services.prediction_service import PredictionService
from model.config import BEST_MODEL_PATH, CLASS_NAMES_PATH


def make_multipart(file_bytes: bytes | None = None, filename: str = "leaf.jpg"):
    boundary = "plant-test-boundary"
    if file_bytes is None:
        body = f"--{boundary}--\r\n".encode()
    else:
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: image/jpeg\r\n\r\n"
        ).encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()
    return boundary, body


async def asgi_request(path: str, method: str = "GET", body: bytes = b"", content_type: str = ""):
    headers = []
    if content_type:
        headers.append((b"content-type", content_type.encode()))
    if body:
        headers.append((b"content-length", str(len(body)).encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "server": ("test", 80),
        "client": ("test", 1),
    }
    received = False
    messages = []

    async def receive():
        nonlocal received
        if received:
            return {"type": "http.disconnect"}
        received = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    await app(scope, receive, send)
    status = next(message["status"] for message in messages if message["type"] == "http.response.start")
    response_body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    return status, json.loads(response_body) if response_body else None


class PredictionApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BEST_MODEL_PATH.is_file() or not CLASS_NAMES_PATH.is_file():
            raise unittest.SkipTest("Trained checkpoint files are not available.")
        app.state.prediction_service = PredictionService.from_files()

    def request(self, *args, **kwargs):
        return asyncio.run(asgi_request(*args, **kwargs))

    def test_health_reports_loaded_model(self):
        status, payload = self.request("/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok", "model_loaded": True})

    def test_missing_file_is_rejected(self):
        boundary, body = make_multipart()
        status, payload = self.request(
            "/api/v1/predict",
            method="POST",
            body=body,
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        self.assertEqual(status, 422)
        self.assertIn("detail", payload)

    def test_invalid_image_is_rejected(self):
        boundary, body = make_multipart(b"not an image")
        status, payload = self.request(
            "/api/v1/predict",
            method="POST",
            body=body,
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        self.assertEqual(status, 415)
        self.assertIn("supported image", payload["detail"])

    def test_oversized_image_is_rejected(self):
        boundary, body = make_multipart(b"x" * (MAX_IMAGE_BYTES + 1))
        status, payload = self.request(
            "/api/v1/predict",
            method="POST",
            body=body,
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        self.assertEqual(status, 413)
        self.assertIn("10 MB", payload["detail"])

    def test_valid_image_returns_top_three_predictions(self):
        image_data = BytesIO()
        Image.new("RGB", (224, 224), (70, 140, 55)).save(image_data, format="JPEG")
        boundary, body = make_multipart(image_data.getvalue())
        status, payload = self.request(
            "/api/v1/predict",
            method="POST",
            body=body,
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        self.assertEqual(status, 200)
        self.assertIn(payload["predicted_class"], app.state.prediction_service.class_names)
        self.assertEqual(len(payload["top_predictions"]), 3)
        self.assertEqual(
            payload["predicted_class"],
            payload["top_predictions"][0]["class_name"],
        )


if __name__ == "__main__":
    unittest.main()
