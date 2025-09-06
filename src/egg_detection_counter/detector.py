"""This is the code for training the YOLO model for egg detection."""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, List, Mapping

from collections import Counter
from ultralytics import YOLO
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class EggTrainer:
    """Class for training the YOLO model for egg detection."""

    conf: str = field(default="src/egg_detection_counter/data/data.yaml")
    epochs: int = field(default=100)
    img_size: int = field(default=640)
    batch_size: int = field(default=16)
    device: str = field(default="cuda")
    model: Any = field(init=False)

    def train(self) -> None:
        """Train the YOLO model for egg detection."""
        logger.info("Start training the YOLO model for egg detection and counter.")
        self.model = YOLO("yolov8n.pt")
        self.model.train(
            data=self.conf,
            epochs=self.epochs,
            imgsz=self.img_size,
            batch=self.batch_size,
            device=self.device,
        )

    def validation(self) -> Any:
        """Validate the YOLO model for egg detection."""
        logger.info("Validating the YOLO model for egg detection.")
        return self.model.val()

    def model_export(self) -> None:
        """Export the YOLO model for egg detection."""
        logger.info("Exporting the YOLO model for egg detection.")
        self.model.export(format="onnx")


@dataclass
class EggInference:
    """Class for testing the YOLO model for egg detection."""

    model_path: Optional[Any] = field(default=None)
    result_path: Optional[str] = field(default=None)

    def __post_init__(self) -> None:
        """Post-initialization method for EggInference."""
        if self.model_path is None or not self.model_path.exists():
            raise ValueError("Model does not exist or the path is not correct.")

    def load_model(self) -> Any:
        """Load the YOLO model for egg detection."""
        logger.info("Loading the trained model for egg detection.")
        return YOLO(self.model_path)

    def inference(self, data_path: str) -> Any:
        """Inference code for egg detection"""
        if not Path(data_path).exists():
            logger.error("Data path does not exist or the path is not correct.")
        model = self.load_model()
        results = model(
            data_path,
            save=False if not self.result_path else True,  # pylint: disable=R1719
            project=self.result_path,
            name="detections",
        )
        return results

    @staticmethod
    def number_of_eggs(detections: Any) -> Mapping[str, Any]:
        """Count the number of eggs detected."""
        counts = {}
        for result in detections:
            class_count = Counter(int(box.cls.item()) for box in result.boxes)
            temp = []
            for name, count in class_count.items():
                temp.append({"class": result.names[name], "count": count})
            file_name = Path(result.path).name
            counts[str(file_name)] = temp
        return counts

    @staticmethod
    def results_detail(detection: Any) -> Mapping[str, Any]:
        """Get the detailed results of the detected eggs such as bounding boxes, class names, and confidences."""
        results = {}
        for result in detection:
            temp = []
            for box in result.boxes:
                temp.append(
                    {
                        "class": result.names[int(box.cls.item())],
                        "confidence": box.conf[0].item(),
                        "bounding_box": box.xyxy[0].tolist(),
                    }
                )
            file_name = Path(result.path).name
            results[str(file_name)] = temp
        return results

    @staticmethod
    def result_images(detections: Any) -> List[Any]:
        """Make a list of the result images with detections."""
        images = []
        for result in detections:
            images.append(np.array(result.plot())[:, :, [2, 1, 0]])
        return images
