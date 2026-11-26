import pytest
from fastapi.testclient import TestClient

from traffic_classifier.api.app import create_app
from traffic_classifier.inference import Predictor
from traffic_classifier.ml.cnn import PacketCNN
from traffic_classifier.ml.model_bundle import ModelBundle
from traffic_classifier.ml.normalization import Normalizer
from traffic_classifier.vnat_groups import CLASSES

PACKETS = {"offsets_us": [0, 2000, 5250], "sizes_bytes": [100, 300, 180], "directions": [1, 0, 1]}


def test_api_loads_model_once_and_reports_prediction():
    def loader():
        nonlocal calls
        calls += 1
        normalizer = Normalizer(8, (0.0, 0.0), (1.0, 1.0), "i", "s")
        return Predictor(ModelBundle(PacketCNN().eval(), normalizer, "test-model"))

    calls = 0
    with TestClient(create_app(loader)) as client:
        assert client.get("/ready").json() == {"status": "ready", "model_id": "test-model", "n": 8}
        for _ in range(2):
            response = client.post("/api/v1/predict", json=PACKETS)
            assert response.status_code == 200
            data = response.json()
            assert (data["n"], data["observed_packets"], data["model_id"]) == (8, 3, "test-model")
            assert tuple(data["scores"]) == CLASSES
            assert sum(data["scores"].values()) == pytest.approx(1.0)
            assert data["category"] == max(data["scores"], key=data["scores"].get)
        redirect = client.get("/", follow_redirects=False)
        assert redirect.status_code == 307
        assert redirect.headers["location"] == "/docs"
        docs = client.get("/")
        assert docs.status_code == 200
        assert "Swagger UI" in docs.text
        assert "/api/v1/predict" in client.get("/openapi.json").json()["paths"]
        assert calls == 1


def test_api_rejects_invalid_packet_arrays():
    normalizer = Normalizer(8, (0.0, 0.0), (1.0, 1.0), "i", "s")
    predictor = Predictor(ModelBundle(PacketCNN().eval(), normalizer, "test"))
    with TestClient(create_app(lambda: predictor)) as client:
        response = client.post("/api/v1/predict", json={**PACKETS, "directions": [1]})
        assert response.status_code == 422
        assert "same length" in response.json()["detail"]
        assert (
            client.post(
                "/api/v1/predict", json={**PACKETS, "sizes_bytes": [100, "300", 180]}
            ).status_code
            == 422
        )


@pytest.mark.parametrize("error", [FileNotFoundError("missing.pt"), EOFError("truncated")])
def test_api_reports_unavailable_bundle(error):
    def missing():
        raise error

    with TestClient(create_app(missing)) as client:
        assert client.get("/ready").status_code == 503
        assert client.post("/api/v1/predict", json=PACKETS).status_code == 503


def test_frontend_mount_is_optional_and_preserves_api(tmp_path):
    dist = tmp_path / "dist"
    with TestClient(create_app(lambda: None, frontend_dist=dist)) as client:
        assert client.get("/app/").status_code == 404
        assert client.get("/docs").status_code == 200
        assert client.get("/ready").status_code == 503

    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text(
        '<link rel="stylesheet" href="/app/assets/index.css"><main>TrafficClassifier</main>',
        encoding="utf-8",
    )
    (assets / "index.css").write_text("body { color: white; }", encoding="utf-8")
    with TestClient(create_app(lambda: None, frontend_dist=dist)) as client:
        assert client.get("/app", follow_redirects=False).headers["location"].endswith("/app/")
        assert "TrafficClassifier" in client.get("/app/").text
        css = client.get("/app/assets/index.css")
        assert css.status_code == 200
        assert css.headers["content-type"].startswith("text/css")
        assert client.get("/docs").status_code == 200
        assert client.get("/ready").status_code == 503
        assert "/api/v1/runs" in client.get("/openapi.json").json()["paths"]
