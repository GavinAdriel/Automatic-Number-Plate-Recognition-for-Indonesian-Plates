# Automatic-Number-Plate-Recognition-for-Indonesian-Plates

This is a ANPR based on YOLOV11 pretrained model to detect license plate on a vehicle and PaddleOCR to handle the OCR

## Model

A licensed plate detector was used to detect license plates. The model was trained with Yolov11 using [this dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4). 

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
