"""Package level tests"""

from click.testing import CliRunner

from egg_detection_counter import __version__
from egg_detection_counter.main import egg_detection_counter_cli


def test_version() -> None:
    """Unit test for checking the version of the code"""
    assert __version__ == "0.2.0"


def test_egg_detection_counter_cli() -> None:
    """Unit test for checking the CLI for egg detection counter"""
    runner = CliRunner()
    result = runner.invoke(egg_detection_counter_cli, ["--help"])
    assert result.exit_code == 0
    assert result


def test_train() -> None:
    """Unit test for training the YOLO model for egg detection"""
    runner = CliRunner()
    result = runner.invoke(egg_detection_counter_cli, ["train", "--help"])
    assert result.exit_code == 0
    assert result


def test_infer() -> None:
    """Unit test for testing the YOLO model for egg detection"""
    runner = CliRunner()
    result = runner.invoke(
        egg_detection_counter_cli,
        ["infer", "--data_path", "./tests/test_data/sample1.png", "--result_path", ""],
    )
    assert result.exit_code == 0
    assert result
