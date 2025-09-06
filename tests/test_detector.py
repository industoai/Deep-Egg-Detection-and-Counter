"""Test the detector module."""

from pathlib import Path
import pytest
from ultralytics import YOLO
from egg_detection_counter.detector import EggInference


@pytest.fixture(name="infer_function")
def fixture_infer_function() -> EggInference:
    """Fixture to create an EggInference instance before each test."""
    return EggInference(
        model_path=Path("./src/egg_detection_counter/model/egg_detector.pt"),
        result_path="",
    )


def test_load_model(infer_function: EggInference) -> None:
    """Test the EggInference class."""
    model = infer_function.load_model()
    assert isinstance(model, YOLO)


def test_inference(infer_function: EggInference) -> None:
    """Test the inference method of EggInference class."""
    result = infer_function.inference(data_path="./tests/test_data/sample1.png")
    assert result


def test_number_of_eggs(infer_function: EggInference) -> None:
    """Test the number of eggs detected."""
    result = infer_function.inference(data_path="./tests/test_data/sample1.png")
    counts = infer_function.number_of_eggs(result)
    if counts:
        for key, val in counts.items():
            assert sum(item["count"] for item in val) == 33
            assert key == "sample1.png"
            assert val == [
                {"class": "white-egg", "count": 20},
                {"class": "brown-egg", "count": 13},
            ]


def test_result_images(infer_function: EggInference) -> None:
    """Test the result_images method of EggInference class."""
    result = infer_function.inference(data_path="./tests/test_data/sample1.png")
    result_images = infer_function.result_images(result)
    assert result_images
