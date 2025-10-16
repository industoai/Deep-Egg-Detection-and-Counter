---
title: Deep Egg Detection And Counter
emoji: 🥚🧮
colorFrom: green
colorTo: pink
sdk: streamlit
sdk_version: 1.43.2
app_file: app.py
pinned: false
license: mit
short_description: Detect and count white and brown eggs from images
---
# Deep-Egg-Detection-and-Counter
This repository allows you to detect and count eggs in an egg-shell with the help of deep learning models specifically YOLOv8.
A labeled dataset for training this model is collected manually.
A YOLO model is trained on this dataset and the trained model is provided and used to detect and count eggs in an egg-shell.

## Dataset
A dataset from real application is collected and labeled manually in TOLO-format in order to be used for training and validation purposes.
This dataset is available [here](https://huggingface.co/datasets/industoai/Egg-Detection).
The eggs are collected within different shell types and colours (the egg-shall might be transparent plastic, bright/dark carton, etc.) in order
to make the model robust to different background condition that might be encountered in real applications.
Right now, two classes exists in this dataset as:
- White eggs
- Brown eggs

The boundaries of the eggs are also specified in the dataset and the dataset is split into training and validation sets.
We try to make this dataset more diverse and rich by adding more classes and more images in the future.

### Dataset YOLO-format Structure
The [dataset](https://huggingface.co/datasets/industoai/Egg-Detection) is prepared in th YOLO-format make it easier to use.
This dataset is split into training and validation sets. The structure of the dataset to be used in YOLO models are as following:
```
data/
├── images/
│   ├── train/
│   │   ├── sample1.jpg
│   │   ├── ...
│   │   ├── sample49.jpg
│   ├── val/
│   │   ├── sample50.jpg
│   │   ├── sample51.jpg
├── labels/
│   ├── train/
│   │   ├── sample1.txt
│   │   ├── ...
│   │   ├── sample49.txt
│   ├── val/
│   │   ├── sample50.txt
│   │   ├── sample51.txt
├── data.yaml
├── train.txt
├── val.txt
```
It is good to mention that although the number of images are not too much at the moment, there are many different eggs in each image which increases the
number of samples used in the training and validation stages. The `txt` files in the labels folders contain the bounding boxes around each egg in the images as well as
the class of egg which is white/brown.

The `train.txt` and `val.txt` files contain the path to the images in the training and validation sets.
The `data.yaml` file contains the class names, the path to the training and validation sets, and also the path to the parent `data` directory.

**NOTE:** For training, it is important to change the path to **absolute path** of the main directory where the data is located (In the above tree-structure it should be the absolute path to `data` directory.)


## How to Use Locally
This repo can be used for fine-tuning the YOLOv8 model to detect and count eggs and also can be used for testing purposes with the current fine-tuned model.
In order to use the model for training/testing purposes locally, one can first create a virtual environment and then install the requirements
by running the `poetry install` command (Install poetry if you do not have it in your system from [here](https://python-poetry.org/docs/#installing-with-pipx).)

### Fine-Tuning
YOLO model is fine-tuned with the collected dataset. In order to find-tune the model with other egg classes or repeat the whole process,
one can clone this repo and download the dataset from [here](https://huggingface.co/datasets/industoai/Egg-Detection) and put in the `src/egg_detection_counter/data` directory.
Then for training or fine-tuning the model, one can run the following command:
```bash
egg_detection_counter -vv train --conf_path src/egg_detection_counter/data/data.yaml --img_resize 640 --batch_size 16 --epochs 100 --device cuda
```

### Inference
The fine-tuned model can be used for inference purposes. The model is provided in the `src/egg_detection_counter/models` directory.
By uploading and using the model, one can detect white/brown eggs and count them in an egg-shell. The model can be used with the following command:
```bash
egg_detection_counter -vv infer --model_path src/egg_detection_counter/model/egg_detector.pt --data_path ./tests/test_data --result_path ./results
```
It is good to mention that, the `data_path` could be a directory containing images or a single image. The `result_path` is the directory where the results are saved.

As an example, white and brown eggs are detected properly in the following image:
<p align="center">
    <img width="1000" src="./results/sample2.jpg" alt="Egg Detection">
</p>

## Hugging Face Deployment
The repository is also deployed in [hugging face](https://huggingface.co/spaces/industoai/Deep-Egg-Detection-and-Counter) in which one can upload images, 
and then the detected white/brown eggs and the number of them will be shown.

It is good to mention that you can also run the demo application locally by running the following command:
```shell
streamlit run app.py
```
and then open the browser and go to the address `http://localhost:8501`.

## How to load Trained Model from Hugging Face
The trained model is also uploaded to [hugging face](https://huggingface.co/industoai/Egg-Detection) from which one can use it as following:
```shell
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model_path = hf_hub_download(repo_id="industoai/Egg-Detection", filename="model/egg_detector.pt")
model = YOLO(model_path)
result = model("path/to/image")
```
Then, the uploaded model can be used for different purposes.


## Docker Container
To run the docker with ssh, do the following first and then based on your need select ,test, development, or production containers:
```shell
export DOCKER_BUILDKIT=1
export DOCKER_SSHAGENT="-v $SSH_AUTH_SOCK:$SSH_AUTH_SOCK -e SSH_AUTH_SOCK"
```
### Test Container
This container is used for testing purposes while it runs the test
```shell
docker build --progress plain --ssh default --target test -t egg_detection_docker:test .
docker run -it --rm -v "$(pwd):/app" $(echo $DOCKER_SSHAGENT) egg_detection_docker:test
```

### Development Container
This container can be used for development purposes:
```shell
docker build --progress plain --ssh default --target development -t egg_detection_docker:development .
docker run -it --rm -v "$(pwd):/app" -v /tmp:/tmp $(echo $DOCKER_SSHAGENT) egg_detection_docker:development
```

### Production Container
This container can be used for production purposes:
```shell
docker build --progress plain --ssh default --target production -t egg_detection_docker:production .
docker run -it --rm -v "$(pwd):/app" -v /tmp:/tmp $(echo $DOCKER_SSHAGENT) egg_detection_docker:production egg_detection_counter -vv infer --model_path src/egg_detection_counter/model/egg_detector.pt --data_path ./tests/test_data --result_path ./results
```


## How to Develop
Do the following only once after creating your project:
- Init the git repo with `git init`.
- Add files with `git add .`.
- Then `git commit -m 'initialize the project'`.
- Add remote url with `git remote add origin REPO_URL`.
- Then `git branch -M master`.
- `git push origin main`.
Then create a branch with `git checkout -b BRANCH_NAME` for further developments.
- Install poetry if you do not have it in your system from [here](https://python-poetry.org/docs/#installing-with-pipx).
- Create a virtual env preferably with virtualenv wrapper and `mkvirtualenv -p $(which python3.10) ENVNAME`.
- Then `git add poetry.lock`.
- Then `pre-commit install`.
- For applying changes use `pre-commit run --all-files`.
