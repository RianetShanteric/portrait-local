import json
from pathlib import Path

from app.cli import unique_output_path
from core.recipe import model_identity_from_manifest, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_config_and_model_manifest_are_consistent() -> None:
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
    names = {item["filename"] for item in manifest["models"]}
    assert config["analysis"]["person_segmentation_model"] in names
    assert "yolo11s-seg.pt" not in names


def test_manifest_identity_uses_recipe_schema() -> None:
    identity = model_identity_from_manifest(ROOT, "RealESRGAN_x4plus.pth")

    assert identity["capability"] == "conditional-upscale"
    assert identity["provider"] == "real-esrgan"
    assert identity["artifact"] == "RealESRGAN_x4plus.pth"


def test_installed_model_hashes_match_manifest() -> None:
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
    for model in manifest["models"]:
        artifact = ROOT / model.get("install_path", f"models/{model['filename']}")
        if artifact.exists():
            assert sha256_file(artifact) == model["sha256"]


def regression_cases() -> list[dict]:
    path = ROOT / "tests" / "regression" / "dataset.json"
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def test_regression_source_hashes_are_frozen_and_metadata_free() -> None:
    from PIL import Image

    for case in regression_cases():
        source = ROOT / "tests" / "regression" / "images" / case["file"]
        assert sha256_file(source) == case["stored_sha256"]
        with Image.open(source) as image:
            assert len(image.getexif()) == 0


def test_baseline_recipe_records_non_destructive_input() -> None:
    for case in regression_cases():
        path = ROOT / "tests" / "regression" / "baseline" / "balanced" / case["id"] / "recipe.json"
        recipe = json.loads(path.read_text(encoding="utf-8"))
        assert recipe["regression_case"] == case["id"]
        assert recipe["input"]["sha256"] == case["stored_sha256"]
        assert recipe["input"]["sha256"] != recipe["output"]["sha256"]
        assert all("landmarks" not in face for face in recipe["metrics"]["faces"])
        assert all("bbox" not in item for item in recipe["metrics"]["detected_objects"])


def test_upscale_baseline_contains_the_full_before_after_evidence() -> None:
    from PIL import Image

    target = ROOT / "tests" / "regression" / "baseline" / "balanced" / "longfellow_group"
    recipe = json.loads((target / "recipe.json").read_text(encoding="utf-8"))
    with Image.open(target / "result.jpg") as result, Image.open(target / "before-after.jpg") as comparison:
        assert result.size == (1400, 928)
        assert comparison.size == (2800, 928)
    assert recipe["models"]["upscale"]["provider"] == "real-esrgan"


def test_batch_outputs_do_not_collide_on_shared_stem(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    first = output_dir / "photo__portrait.jpg"
    first.write_bytes(b"first")

    second = unique_output_path(output_dir, Path("photo.png"), "__portrait")

    assert second.name == "photo_2__portrait.jpg"
