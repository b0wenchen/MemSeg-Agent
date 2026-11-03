import base64
import io
import json
import zipfile

import numpy as np
import pytest
import torch

from memseg_agent.interactive import SegmentationSession
from memseg_agent.interactive.session import png_bytes


class FakeModel:
    def eval(self):
        return self

    def encode_observation(self, image, mask):
        return mask.float()[None, None], torch.zeros(1)

    def __call__(self, image, working_memory, use_static_memory):
        assert use_static_memory is False
        assert len(working_memory) > 0
        self.last_memories = working_memory
        return working_memory[-1][0] * 20 - 10, torch.ones(1)


class FakeEmbedder:
    def encode(self, image):
        return torch.tensor([float(image.mean()) + 0.01, 1.0])


@pytest.fixture
def session():
    return SegmentationSession(FakeModel(), FakeEmbedder(), "Test target")


def case(value=120):
    image = np.full((12, 16, 3), value, dtype=np.uint8)
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[3:8, 4:10] = 1
    return image, mask


def test_user_feedback_loop_without_static(session):
    image, mask = case()
    session.add_support(image, mask, "Support")
    query_id = session.set_query(case(170)[0], "Query")
    prediction, retrieved = session.predict(query_id)
    assert len(session.cases) == 1  # Predictions are never automatically saved.
    assert retrieved[0]["bank"] == "fewshot"
    np.testing.assert_array_equal(prediction, mask > 0)
    with pytest.raises(ValueError, match="Confirm"):
        session.confirm_correction(query_id, mask)
    # Empty masks are legitimate user-reviewed negative examples.
    session.confirm_correction(query_id, np.zeros_like(mask), confirmed=True)
    assert len(session.cases) == 2
    assert session.cases[-1].bank == "working"
    new_query = session.set_query(case(190)[0], "Later query")
    _, retrieved = session.predict(new_query)
    assert {c["bank"] for c in retrieved} == {"fewshot", "working"}
    with pytest.raises(ValueError, match="Query changed"):
        session.confirm_correction(query_id, mask, confirmed=True)


def test_no_memories_and_invalid_masks(session):
    image, mask = case()
    query = session.set_query(image)
    with pytest.raises(ValueError, match="few-shot"):
        session.predict(query)
    for bad in (mask[:4], mask * 2):
        with pytest.raises(ValueError):
            session.add_support(image, bad)
    with pytest.raises(ValueError, match="foreground"):
        session.add_support(image, np.zeros_like(mask))
    with pytest.raises(ValueError, match="uint8"):
        session.set_query(image.astype(np.float32))
    assert session.cases == []


def test_upsert_delete_and_capacity():
    session = SegmentationSession(FakeModel(), FakeEmbedder(), capacity=1)
    image, mask = case()
    first = session.add_support(image, mask)
    assert session.add_support(image, 1 - mask) == first
    assert len(session.cases) == 1
    with pytest.raises(ValueError, match="full"):
        session.add_support(case(90)[0], mask)
    session.remove(first)
    assert not session.cases


def test_archive_roundtrip_and_atomic_failure(session):
    image, mask = case()
    session.add_support(image, mask)
    query = session.set_query(case(150)[0])
    session.confirm_correction(query, mask, confirmed=True)
    archive = session.export_archive()
    restored = SegmentationSession(FakeModel(), FakeEmbedder())
    restored.import_archive(archive)
    assert restored.target == "Test target"
    assert [c.bank for c in restored.cases] == ["fewshot", "working"]
    np.testing.assert_array_equal(restored.cases[0].image, image)
    with pytest.raises(ValueError, match="empty"):
        restored.import_archive(archive)
    session.reset()
    invalid = io.BytesIO()
    with zipfile.ZipFile(invalid, "w") as z:
        z.writestr("manifest.json", '{"format":"unknown","cases":[]}')
    with pytest.raises(ValueError):
        session.import_archive(invalid.getvalue())
    assert session.cases == []


def test_sessions_do_not_share_memory(session):
    session.add_support(*case())
    another = SegmentationSession(FakeModel(), FakeEmbedder())
    assert another.summary()["memories"] == []


def test_archive_late_failure_does_not_partially_import(session):
    image, mask = case()
    manifest = {
        "format": "memseg-reviewed-cases-v1",
        "cases": [
            {"bank": "fewshot", "name": "Valid", "image": "image.png", "mask": "mask.png"},
            {"bank": "working", "name": "Invalid", "image": "image.png", "mask": "missing.png"},
        ],
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("image.png", png_bytes(image))
        archive.writestr("mask.png", png_bytes(mask))
    with pytest.raises(KeyError):
        session.import_archive(output.getvalue())
    assert session.cases == []
    assert session.target == "Test target"


def test_oversized_export_is_rejected(session, monkeypatch):
    session.add_support(*case())
    monkeypatch.setattr("memseg_agent.interactive.session.MAX_ARCHIVE_BYTES", 1024 * 1024 + 100)
    with pytest.raises(ValueError, match="Export exceeds"):
        session.export_archive()
    assert len(session.cases) == 1


def test_invalid_query_embedding_cannot_select_memories(session):
    image, mask = case()
    session.add_support(image, mask)
    query = session.set_query(case(190)[0])
    session.embedder.encode = lambda image: torch.tensor([float("nan"), 1.0])
    with pytest.raises(ValueError, match="Invalid retrieval embedding"):
        session.predict(query)
    assert session.prediction is None


@pytest.fixture
def client(session):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from memseg_agent.interactive.server import create_app

    with TestClient(
        create_app(session), base_url="http://127.0.0.1", headers={"X-MemSeg-Request": "1"}
    ) as client:
        yield client


def url(image):
    return "data:image/png;base64," + base64.b64encode(png_bytes(image)).decode()


def test_api_complete_workflow(client, session):
    image, mask = case()
    response = client.post(
        "/api/support", json={"image": url(image), "mask": url(mask * 255), "name": "example"}
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["memories"]) == 1
    query = client.post("/api/query", json={"image": url(case(180)[0]), "name": "query"}).json()
    prediction = client.post("/api/predict", json={"id": query["id"]})
    assert prediction.status_code == 200, prediction.text
    assert prediction.json()["retrieved"][0]["bank"] == "fewshot"
    correction = {"id": query["id"], "mask": url(mask * 255)}
    assert client.post("/api/correct", json=correction).status_code == 400
    correction["confirmed"] = True
    result = client.post("/api/correct", json=correction)
    assert result.status_code == 200, result.text
    assert len(session.cases) == 2
    memory_id = result.json()["memories"][1]["id"]
    detail = client.get(f"/api/memory/{memory_id}")
    assert detail.status_code == 200
    assert detail.headers["cache-control"] == "no-store"
    archive = client.get("/api/export")
    assert archive.status_code == 200
    assert client.post("/api/reset", json={"confirmed": False}).status_code == 400
    assert client.post("/api/reset", json={"confirmed": True}).status_code == 200
    assert client.post("/api/import", content=archive.content).status_code == 200
    assert len(session.cases) == 2


def test_api_security_and_bad_input(client):
    assert client.get("/api/state", headers={"X-MemSeg-Request": ""}).status_code == 403
    assert (
        client.get("/api/state", headers={"Origin": "https://untrusted.example"}).status_code == 403
    )
    assert client.get("/api/state", headers={"Host": "untrusted.example"}).status_code == 400
    assert client.post("/api/query", json={"image": "bad"}).status_code == 400
    assert client.post("/api/query", content="[]").status_code == 400
    assert client.post("/api/import", content=b"not a zip").status_code == 400
    assert client.get("/").status_code == 200
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200


def test_api_rejects_colored_transparent_and_high_bit_masks(client, session):
    image, mask = case()
    colored = np.zeros_like(image)
    colored[3:8, 4:10, 2] = 8  # Converts to gray value 1; must still be rejected.
    transparent = np.zeros((*mask.shape, 4), dtype=np.uint8)
    for invalid in (colored, transparent, mask.astype(np.uint16)):
        response = client.post("/api/support", json={"image": url(image), "mask": url(invalid)})
        assert response.status_code == 400, response.text
    assert session.cases == []


def test_slicer_slice_api_is_atomic_and_preserves_browser_query(client, session):
    from memseg_agent.interactive.wire import pack_array, unpack_array

    image, mask = case()
    assert client.get("/api/capabilities").json()["protocol"] == "memseg-slicer-v1"
    support = {"image": pack_array(image), "mask": pack_array(mask), "name": "Support K=0"}
    assert client.post("/api/slice-support", json=support).status_code == 200
    browser_query = session.set_query(case(160)[0], "Browser query")
    browser_prediction, _ = session.predict(browser_query)
    result = client.post("/api/slice-predict", json={"image": pack_array(case(200)[0])})
    assert result.status_code == 200, result.text
    np.testing.assert_array_equal(unpack_array(result.json()["mask"], mask=True), mask)
    assert session.current["id"] == browser_query
    np.testing.assert_array_equal(session.prediction, browser_prediction)
    assert len(session.cases) == 1
    correction = {"image": pack_array(case(200)[0]), "mask": pack_array(1 - mask)}
    assert client.post("/api/slice-correct", json=correction).status_code == 400
    correction["confirmed"] = True
    assert client.post("/api/slice-correct", json=correction).status_code == 200
    result = client.post("/api/slice-predict", json={"image": pack_array(case(220)[0])}).json()
    assert {entry["bank"] for entry in result["retrieved"]} == {"fewshot", "working"}


@pytest.mark.parametrize(
    "bad",
    [
        {"dtype": "uint16", "shape": [10, 10], "data": ""},
        {"dtype": "uint8", "shape": [10**9, 10**9], "data": ""},
        {"dtype": "uint8", "shape": [10, 10, 4], "data": ""},
        {"dtype": "uint8", "shape": [10, 10], "data": "bad"},
        {"dtype": "uint8", "shape": [True, 10], "data": ""},
    ],
)
def test_slice_api_rejects_invalid_dimensions_before_decode(client, bad):
    assert client.post("/api/slice-predict", json={"image": bad}).status_code == 400
