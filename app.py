"""This is a demo for running the egg detection and counter usng streamlit library"""

from dataclasses import dataclass, field
from pathlib import Path
import tempfile

import streamlit as st
import pandas as pd
from PIL import Image


from src.egg_detection_counter.detector import EggInference


@dataclass
class DemoEggDetectionCounter:
    """Class for running the egg detection and counter app using Streamlit."""

    image: str = field(init=False)

    def upload_image(self) -> None:
        """Upload an image from the streamlit page"""
        uploaded_file = st.file_uploader(
            "Upload an image...", type=["jpg", "png", "jpeg"]
        )
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
            self.image = tmp_file.name
        else:
            self.image = "tests/test_data/sample1.jpg"
        st.image(
            Image.open(self.image),
            caption="Original/Uploaded Image",
            use_container_width=True,
        )

    def process_image(self) -> None:
        """Process the image for the egg detection and counter"""
        if st.button("Detect/Count Eggs"):
            inferer = EggInference(
                model_path=Path("./src/egg_detection_counter/model/egg_detector.pt"),
                result_path="",
            )
            detections = inferer.inference(data_path=self.image)
            counts = inferer.number_of_eggs(detections)
            result_image = inferer.result_images(detections)

            st.markdown("<h3>Detected Results</h3>", unsafe_allow_html=True)
            st.image(result_image[0], caption="Detected Eggs", use_container_width=True)
            results = pd.DataFrame()
            if counts:
                for _, val in counts.items():
                    results = pd.DataFrame(
                        {
                            "Image": "Sample",
                            "Total No. Eggs": [sum(item["count"] for item in val)],
                            val[0]["class"]: [val[0]["count"]],
                            val[1]["class"] if len(val) > 1 else "other": [
                                val[1]["count"] if len(val) > 1 else 0
                            ],
                        }
                    )
            st.markdown('<div class="center-container">', unsafe_allow_html=True)
            st.markdown(
                "<h3>Detailed Information of Detections</h3>", unsafe_allow_html=True
            )
            st.markdown(
                """
                <style>
                table {
                    width: 100%;
                }
                th, td {
                    text-align: center !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.table(results)
            st.markdown("</div>", unsafe_allow_html=True)

    def design_page(self) -> None:
        """Design the streamlit page for eg detector and counter"""
        st.title("Egg detector and counter")
        self.upload_image()
        self.process_image()


demo = DemoEggDetectionCounter()
demo.design_page()
