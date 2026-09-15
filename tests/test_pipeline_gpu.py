import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.smart import SmartPhotoPipeline


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def engine() -> SmartPhotoPipeline:
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    return SmartPhotoPipeline(ROOT, config)


def regression_cases() -> list[dict]:
    manifest = ROOT / "tests" / "regression" / "dataset.json"
    return json.loads(manifest.read_text(encoding="utf-8"))["cases"]


@pytest.mark.gpu
@pytest.mark.parametrize("case", regression_cases(), ids=lambda case: case["id"])
def test_regression_portrait_analysis_and_gating(engine: SmartPhotoPipeline, case: dict) -> None:
    path = ROOT / "tests" / "regression" / "images" / case["file"]
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    analysis = engine.analyze(rgb)
    _, decisions = engine.render(rgb, analysis, "balanced")
    expected = case["expect"]
    assert analysis.metrics["has_people"] is expected["has_people"]
    assert analysis.metrics["scene"] == expected["scene"]
    assert 0.0 <= analysis.metrics["scene_confidence"] <= 1.0
    assert analysis.metrics["scene_reasons"]
    assert analysis.metrics["person_count"] >= expected["min_people"]
    assert analysis.metrics["face_count"] >= expected.get("min_faces", 0)
    assert analysis.metrics["face_count"] <= expected.get("max_faces", 10_000)
    assert analysis.metrics["landmark_face_count"] >= expected["min_landmark_faces"]
    assert analysis.person_mask.mean() >= expected["min_person_area"]
    assert decisions["upscale_requested"] is expected["needs_upscale"]
    assert decisions["scene"] == expected["scene"]
    if expected["scene"] in {"portrait", "group"}:
        assert "subject_lift" in decisions
    else:
        assert "general_gamma" in decisions
    if expected["min_landmark_faces"]:
        assert analysis.eye_mask.mean() > 0.0
        assert analysis.hair_mask.mean() > 0.0
        assert decisions["eye_detail_amount"] > 0.0
        assert decisions["hair_detail_amount"] > 0.0
    else:
        assert decisions["eye_detail_amount"] == 0.0
        assert decisions["hair_detail_amount"] == 0.0


def test_primary_segmentation_provider_is_instance_aware(engine: SmartPhotoPipeline) -> None:
    assert engine.segmenter.identity.provider == "torchvision-maskrcnn-resnet50-fpn"


@pytest.mark.gpu
def test_portrait_region_strengths_follow_profiles(engine: SmartPhotoPipeline) -> None:
    path = ROOT / "tests" / "regression" / "images" / "couple_outdoors_public_domain.jpg"
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    analysis = engine.analyze(rgb)
    decisions = [engine.render(rgb, analysis, profile)[1] for profile in ("natural", "balanced", "strong")]
    assert [item["profile"] for item in decisions] == ["natural", "balanced", "strong"]
    assert decisions[0]["eye_detail_amount"] < decisions[1]["eye_detail_amount"] < decisions[2]["eye_detail_amount"]
    assert decisions[0]["hair_detail_amount"] < decisions[1]["hair_detail_amount"] < decisions[2]["hair_detail_amount"]
    assert all(item["teeth_protected"] is True for item in decisions)
    isolated, _ = engine._portrait_region_detail(rgb, analysis, engine.config["profiles"]["balanced"])
    region = analysis.eye_mask + analysis.hair_mask
    assert np.array_equal(isolated[region <= 1e-5], rgb[region <= 1e-5])


@pytest.mark.gpu
def test_no_person_fallback(engine: SmartPhotoPipeline) -> None:
    rgb = np.full((768, 1024, 3), 120, dtype=np.uint8)
    analysis = engine.analyze(rgb)
    result, decisions = engine.render(rgb, analysis, "balanced")
    assert analysis.metrics["has_people"] is False
    assert analysis.metrics["face_count"] == 0
    assert analysis.metrics["scene"] == "general"
    assert result.shape == rgb.shape
    assert "general_gamma" in decisions
