"""Run the main code for Deep-Egg-Detection-and-Counter"""

import logging
from pathlib import Path
import click

from egg_detection_counter import __version__
from egg_detection_counter.logging import config_logger
from egg_detection_counter.detector import EggTrainer, EggInference

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Shorthand for info/debug/warning/error loglevel (-v/-vv/-vvv/-vvvv)",
)
def egg_detection_counter_cli(verbose: int) -> None:
    """Detection and count eggs in an egg-shell with the help of deep learning models."""
    if verbose == 1:
        log_level = 10
    elif verbose == 2:
        log_level = 20
    elif verbose == 3:
        log_level = 30
    else:
        log_level = 40
    config_logger(log_level)


@egg_detection_counter_cli.command()
@click.option("--img_resize", type=int, default=640, help="Resize images to this size.")
@click.option(
    "--conf_path",
    type=str,
    default="src/egg_detection_counter/data/data.yaml",
    help="Path to the config file",
)
@click.option(
    "--epochs", type=int, default=100, help="Number of epochs used in training."
)
@click.option("--batch_size", type=int, default=16, help="Batch size used in training.")
@click.option(
    "--device", type=str, default="cuda", help="Use cuda or cpu for training."
)
def train(
    img_resize: int, conf_path: str, epochs: int, batch_size: int, device: str
) -> None:
    """This the CLI for training purposes"""
    detect = EggTrainer(
        conf=conf_path,
        img_size=img_resize,
        epochs=epochs,
        device=device,
        batch_size=batch_size,
    )
    detect.train()
    _ = detect.validation()
    detect.model_export()


@egg_detection_counter_cli.command()
@click.option(
    "--model_path",
    type=click.Path(),
    default=Path("./src/egg_detection_counter/model/egg_detector.pt"),
    help="Path to the pre-trained model.",
)
@click.option(
    "--data_path",
    type=click.Path(),
    default=Path("./tests/test_data"),
    help="Path to the test data.",
)
@click.option(
    "--result_path", type=str, default="./results", help="Path to the results."
)
def infer(model_path: Path, data_path: str, result_path: str) -> None:
    """This the CLI for testing purposes"""
    logger.info("Testing the YOLO model for egg detection...")
    inferer = EggInference(model_path=Path(model_path), result_path=result_path)
    detections = inferer.inference(data_path=data_path)
    counts = inferer.number_of_eggs(detections)
    if counts:
        for key, val in counts.items():
            logger.info(
                "%s eggs are detected in %s as: %s",
                sum(item["count"] for item in val),
                key,
                val,
            )
    res = inferer.result_images(detections)
    logger.info(
        "%s images are analyzed. The final images are in res variable.", len(res)
    )
