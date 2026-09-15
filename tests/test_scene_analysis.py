import numpy as np

from core.contracts import ObjectDetection
from pipeline.scene import classify_scene


def detection(label: str, confidence: float = 0.95, area: float = 0.2) -> ObjectDetection:
    return ObjectDetection(label, confidence, (0, 0, 100, 100), area)


def test_people_take_priority_only_when_they_are_the_subject() -> None:
    rgb = np.full((600, 900, 3), 128, dtype=np.uint8)
    portrait = classify_scene(rgb, [detection("person")], 1, 1, 0.24)
    city = classify_scene(rgb, [detection("person", area=0.002), detection("car"), detection("traffic light")], 1, 0, 0.002)
    assert portrait.scene == "portrait"
    assert city.scene == "city"


def test_object_categories_are_explainable() -> None:
    rgb = np.full((600, 900, 3), 128, dtype=np.uint8)
    assert classify_scene(rgb, [detection("dog")], 0, 0, 0.0).scene == "animal"
    food = classify_scene(rgb, [detection("pizza"), detection("bowl")], 0, 0, 0.0)
    assert food.scene == "food"
    assert any(reason.startswith("food_objects=") for reason in food.reasons)


def test_low_light_is_a_scene_tag_without_overriding_a_portrait() -> None:
    rgb = np.full((600, 900, 3), 20, dtype=np.uint8)
    result = classify_scene(rgb, [detection("person")], 1, 1, 0.3)
    assert result.scene == "portrait"
    assert "night" in result.tags
