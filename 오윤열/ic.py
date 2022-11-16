import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic
import time
from multiprocessing import Process,Queue
import multiprocessing as mp
import datetime

def road_condition_producer(q):
    proc_road = mp.current_process()
    print(proc_road.name)

    while True:
        if int(time.time() * 100) % 10 <5:
            road_condition = "Car Road"
        else:
            road_condition = "Bicycle Road"
        q.put(road_condition)
        time.sleep(1.654)

def date_producer(q):
    proc_date = mp.current_process()
    print(proc_date.name)
    while True:
        now = datetime.datetime.now()
        data = str(now)
        q.put(data)
        time.sleep(1)

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

from_class = uic.loadUiType(r"C:\pyqt\practice\practice.ui")[0]
class WindowClass(QMainWindow, from_class):
    def __init__(self, q1,q2):
        super().__init__()
        self.setupUi(self)
        self.light = False

        self.consumer_road = Consumer(q1)
        self.consumer_road.poped.connect(self.setRoad)
        self.consumer_road.start()
        self.consumer_date = Consumer(q2)
        self.consumer_date.poped.connect(self.setDate)
        self.consumer_date.start()

        self.btn_light.clicked.connect(self.setLight)

    def setRoad(self,data):
        self.text_road.setPlainText(data)
    def setDate(self,data):
        self.text_date.setPlainText(data)

    def setLight(self):
        if self.light == True:
            self.light = False
            print("light on!")
        else:
            self.light = True
            print("light off!")

if __name__ == '__main__':
    q1 = Queue()
    q2 = Queue()

    multiprocess_1 = Process(name="producer_road", target=road_condition_producer, args=(q1, ), daemon=True)
    multiprocess_2 = Process(name="producer_date", target=date_producer, args=(q2,), daemon=True)

    multiprocess_1.start()
    multiprocess_2.start()

    app = QApplication(sys.argv)
    mainWindow = WindowClass(q1,q2)
    mainWindow.show()
    app.exec_()