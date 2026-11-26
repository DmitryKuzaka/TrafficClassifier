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
        assert client.get("/docs").status_code == 200
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
