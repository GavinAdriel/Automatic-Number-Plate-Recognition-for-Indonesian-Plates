# Automatic Number Plate Recognition for Indonesian Plates with YOLOV11 and PaddleOCR

This is a ANPR based on YOLOV11 pretrained model to detect license plate on a vehicle and PaddleOCR to handle the OCR.

based on this project, I have created a Research Paper on enhancing the performance of PaddleOCR for Indonesia ANPR with different image pre-processing method.
The paper is published at The 10th ICCSCI 2025 Conference in Binus Semarang.

Go ahead and check the [research paper](https://authors.elsevier.com/sd/article/S1877050925027383).


## Model

A licensed plate detector was used to detect license plates. The model was trained with Yolov11 using [this dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4).

**Model Training CLI**

* Download the dataset from roboflow and provide the path to the data.yaml
``` bash
yolo detect train data=path/to/dataset.yaml model=yolo11n.pt epochs=100 imgsz=640
```
**Model Validation CLI**

* Download the dataset from roboflow and provide the path to the data.yaml
``` bash
yolo val model=path/to/best.pt data=path/to/dataset.yaml
```

## Project Setup

* Make an environment with python3 using the following command 
``` bash
python3 -m venv virtualenvname
```
* Activate the environment
``` bash
source /path/to/venv/bin/activate
``` 

* Install the project dependencies using the following command 
```bash
pip install -r requirement.txt
```
* Run anpr.py with the webcam as the video input to generate the plates.csv file and detected folder (cropped image of the plates)
``` python
python anpr.py
```
