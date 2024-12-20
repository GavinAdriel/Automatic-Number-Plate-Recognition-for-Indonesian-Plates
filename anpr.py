#import lib
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem
from PyQt5.QtCore import QTimer, pyqtSignal, QThread

import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

import time
import os

import pandas as pd
import numpy as np
import re

#declare OCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")

class FrameGrabber(QThread):
    signal = pyqtSignal(QtGui.QImage, list)

    def __init__(self, parent=None):
        super(FrameGrabber, self).__init__(parent)
        self.running = True
        self.model = YOLO("best.pt") #load yolo model
        self.last_saved_time = time.time()
        self.save_interval = 5 
        self.saved_count = 0

        #make required file and folder
        if not os.path.exists("detected"):
            os.makedirs("detected")

        if not os.path.exists("plates.csv"):
            with open('plates.csv', 'w') as f:
                f.write('PlateNumber\n')

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        while self.running and cap.isOpened():
            success, frame = cap.read()
            if success:
                results = self.model.predict(frame, conf=0.5, iou=0.5)[0] #detect license plate
                detections = []

                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cropped_plate = frame[y1:y2, x1:x2] #crop ROI

                    self.read_save_plate(cropped_plate) #read and save plate

                    detections.append((self.model.names[int(box.cls[0])], box.conf[0], (x1, y1, x2, y2)))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = QtGui.QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QtGui.QImage.Format_RGB888)
                self.signal.emit(image, detections)
        cap.release()

    def stop(self):
        self.running = False
        self.wait()
    
    #preprocess cropped plate
    def preprocess_image(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image to increase readability 
        height, width = gray.shape
        scale_factor = 2  # Increase the scale factor for more aggressive resizing
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized_image = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply Bilateral filter to remove noise
        blurred_image = cv2.bilateralFilter(resized_image, 13, 15, 15)

        # Apply thresholding with Otsu's method
        _, binary_otsu = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Sharpen the image
        kernel_sharpening = np.array([[0, -1, 0], 
                                    [-1, 5, -1],
                                    [0, -1, 0]])
        binary = cv2.filter2D(binary_otsu, -1, kernel_sharpening)
        
        # Apply dilation and erosion to remove noise
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        binary = cv2.erode(binary, kernel, iterations=1)
        return binary
    
    #correct misread text
    def correct_ocr_errors(self, plate):
        # Correct '8' to 'B' and '0' to 'O' if misdetected
        plate_chars = list(plate)
        i = 0
        while i < len(plate_chars):
            if plate_chars[i] == '8':
                # Check if '8' is part of the numeric section (digits only)
                if i > 0 and plate_chars[i-1].isdigit():
                    i += 1
                    continue
                # Change '8' to 'B' if not in numeric section
                plate_chars[i] = 'B'
            elif plate_chars[i] == '0':
                # Replace '0' with 'O' in non-numeric contexts
                if i == 0 or not plate_chars[i-1].isdigit():
                    plate_chars[i] = 'O'
            i += 1

        # Join corrected characters back
        return ''.join(plate_chars)
    
    #read cropped plate
    def perform_ocr(self, image):
        results = ocr.ocr(image, rec=True) #read text
        detected_text = []

        #combine detected text seperated by spaces and perform regex
        if results and results[0]:
            for result in results[0]:
                text = re.sub(r'[^a-zA-Z0-9]', '', result[1][0])
                detected_text.append(text)
        combineOCR = ' '.join(detected_text)
        return self.validate_license_plate(combineOCR)
    
    #validate detected text to indonesia plate format
    def validate_license_plate(self,plate):
        plate = self.correct_ocr_errors(plate) #correct misdetected text

        pattern = r'^([A-Z]{1,2}\s?\d{1,4}\s?[A-Z0]{0,3})'
        match = re.match(pattern, plate)
        if match:
            return match.group(1) #return text that passed the format
        return None

    #read and save cropped plate
    def read_save_plate(self, cropped_plate):
        current_time = time.time()
        #do for every determined time interval(5s) if the model detect plate
        if current_time - self.last_saved_time >= self.save_interval: 
            self.last_saved_time = current_time
            plate_img = self.preprocess_image(cropped_plate) 
            filename = f"detected/license_plate_{self.saved_count}.jpg"
            cv2.imwrite(filename, plate_img) #save preprocessed plate
            result_text = self.perform_ocr(plate_img)
            if result_text != None:
                self.write_csv(result_text) #write detected plate to csv
            self.saved_count += 1 

    #append detected plate to csv
    def write_csv(self, plate_text):
        f = open('plates.csv', 'a') 
        f.write(f'{plate_text}\n')
        f.close()

class Ui_MainWindow(QMainWindow):
    def __init__(self, MainWindow):
        super().__init__()
        self.MainWindow = MainWindow
        self.setupUi(MainWindow)
        self.grabber = FrameGrabber()
        self.grabber.signal.connect(self.updateFrame)
        self.grabber.start()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(840, 480)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(640, 0, 200, 480))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(1)
        self.tableWidget.setRowCount(15)
        self.tableWidget.setHorizontalHeaderLabels(['Plate'])
        self.tableWidget.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.imgLabel = QtWidgets.QLabel(self.centralwidget)
        self.imgLabel.setGeometry(QtCore.QRect(0, 0, 640, 480))
        self.imgLabel.setText("")
        self.imgLabel.setObjectName("imgLabel")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Camera
        self.imgLabel = QtWidgets.QLabel(self.centralwidget)
        self.imgLabel.setGeometry(QtCore.QRect(0, 0, 640, 480)) #600,480
        self.imgLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.imgLabel.setText("")
        self.imgLabel.setObjectName("imgLabel")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ANAL"))
    
    @QtCore.pyqtSlot(QtGui.QImage, list)
    def updateFrame(self, image, _):
        self.imgLabel.setPixmap(QtGui.QPixmap.fromImage(image))

    def refresh_table(self):
        df = pd.read_csv('plates.csv')
        recent_data = df.tail(15).iloc[::-1]
        self.tableWidget.clearContents()

        for row, plate_number in enumerate(recent_data['PlateNumber']):
            self.tableWidget.setItem(row, 0, QTableWidgetItem(str(plate_number)))

    def closeEvent(self, event):
        self.grabber.stop()
        event.accept()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow(MainWindow)
    MainWindow.show()

    timer = QTimer()
    timer.timeout.connect(ui.refresh_table)
    timer.start(5000)

    sys.exit(app.exec_())
