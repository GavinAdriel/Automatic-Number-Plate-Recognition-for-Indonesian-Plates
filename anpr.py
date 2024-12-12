from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem
from PyQt5.QtCore import QTimer, pyqtSignal, QThread
import cv2
from ultralytics import YOLO
import time
import os
from paddleocr import PaddleOCR
import pandas as pd
import re

ocr = PaddleOCR(use_angle_cls=True, lang="en")

class FrameGrabber(QThread):
    signal = pyqtSignal(QtGui.QImage, list)

    def __init__(self, parent=None):
        super(FrameGrabber, self).__init__(parent)
        self.running = True
        self.model = YOLO("best.pt")
        self.last_saved_time = time.time()
        self.save_interval = 5 
        self.saved_count = 0

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
                results = self.model.predict(frame, conf=0.5, iou=0.5)[0]
                detections = []

                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cropped_plate = frame[y1:y2, x1:x2]

                    if cropped_plate.size > 0:
                        self.save_cropped_image(cropped_plate)

                    detections.append((self.model.names[int(box.cls[0])], box.conf[0], (x1, y1, x2, y2)))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = QtGui.QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QtGui.QImage.Format_RGB888)
                self.signal.emit(image, detections)
        cap.release()

    def stop(self):
        self.running = False
        self.wait()

    def save_cropped_image(self, cropped_plate):
        current_time = time.time()
        if current_time - self.last_saved_time >= self.save_interval:
            self.last_saved_time = current_time
            gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
            thresh_plate = cv2.adaptiveThreshold(gray_plate, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)
            filename = f"detected/license_plate_{self.saved_count}.jpg"
            cv2.imwrite(filename, thresh_plate)
            result_text = self.perform_ocr(thresh_plate)
            self.write_csv(result_text)
            self.saved_count += 1

    def perform_ocr(self, image):
        results = ocr.ocr(image, rec=True)
        detected_text = []

        if results and results[0]:
            for result in results[0]:
                text = re.sub(r'[^a-zA-Z0-9]', '', result[1][0])
                detected_text.append(text)
        combineOCR = ' '.join(detected_text)
        return self.validate_license_plate(combineOCR)
    
    def validate_license_plate(self, plate):
        pattern = r'^([A-Z]{1,2}\s?\d{1,4}\s?[A-Z0]{0,3})'
        match = re.match(pattern, plate)
        if match:
            return match.group(1)
        return None

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
        df = df.iloc[-15:].sort_index(ascending=False)
        for each_row in range(len(df)):
            self.tableWidget.setItem(each_row,0,QTableWidgetItem(df.iloc[each_row][0]))

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
