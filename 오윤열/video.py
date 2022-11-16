import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic, QtCore
import time
from multiprocessing import Process,Queue
import multiprocessing as mp
import datetime

img_num = 0
def road_condition_producer(q):
    proc_road = mp.current_process()
    print(proc_road.name)

    while True:
        if int(time.time() * 100) % 10 <5:
            road_condition = "Car Road"
        else:
            road_condition = "Bicycle Road"
        q.put(road_condition)
        time.sleep(1)
def image_producer(q):
    global img_num
    proc_img = mp.current_process()
    print(proc_img.name)

    while True:
        img_num += 1
        if img_num == 8:
            img_num = 0
        q.put(str(img_num))
        time.sleep(0.07)


class Consumer(QThread):
    poped = pyqtSignal(str)

    def __init__(self, q):
        super().__init__()
        self.q = q

    def run(self):
        while True:
            if not self.q.empty():
                data = self.q.get()
                self.poped.emit(data)

from_class = uic.loadUiType(r"C:\pyqt\video\video.ui")[0]
class WindowClass(QMainWindow, from_class):
    def __init__(self, q1, q2):
        super().__init__()
        self.setupUi(self)
        self.show()
        self.width = 1280
        self.height = 720
        self.initUI()

        self.consumer_road = Consumer(q1)
        self.consumer_road.poped.connect(self.setRoad)
        self.consumer_road.start()

        self.consumer_img_num = Consumer(q2)
        self.consumer_img_num.poped.connect(self.setimg)
        self.consumer_img_num.start()


    def setRoad(self,data):
        self.text_road.setPlainText(data)

    def initUI(self):
        self.setupUi(self)
        self.btn_video.clicked.connect(self.setimg)

    def setimg(self,data):
        self.label.setStyleSheet('border-image:url('+str(data)+'.JPG); border : 0px;')
        print(data)

if __name__ == '__main__':
    q1 = Queue()
    q2 = Queue()

    multiprocess_1 = Process(name="producer_road", target=road_condition_producer, args=(q1, ), daemon=True)
    multiprocess_2 = Process(name="producer_img_num", target=image_producer, args=(q2, ), daemon=True)

    multiprocess_1.start()
    multiprocess_2.start()


    app = QApplication(sys.argv)
    mainWindow = WindowClass(q1,q2) #q2 넣어주기
    mainWindow.show()
    app.exec_()