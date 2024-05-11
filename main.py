import cv2
import cv2.aruco as aruco
import numpy as np
import os
from kivy.app import App
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock

# Функции для работы с Aruco маркерами
def loadAugImages(path='.'):
    myList = os.listdir(path)
    numOfMarkers = len(myList)
    augDics = {}
    for imgPath in myList:
        if 'jpg' in imgPath or 'png' in imgPath:
            key = int(os.path.splitext(imgPath)[0])
            imgAug = cv2.imread(f'{path}/{imgPath}')
            augDics[key] = imgAug
    return augDics

def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam)
    if draw:
        aruco.drawDetectedMarkers(img, bboxs)
    return [bboxs, ids]

def augmentAruco(bbox, id, img, imgAug, drawId=True):
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    h, w, c = imgAug.shape
    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix, _ = cv2.findHomography(pts2, pts1)
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))
    imgOut = img + imgOut
    return imgOut

# Класс Kivy для отображения камеры
class KivyCamera(Image):
    def __init__(self, capture, fps, findArucoMarkers, augDics, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        self.findArucoMarkers = findArucoMarkers
        self.augDics = augDics
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            arucoFound = self.findArucoMarkers(frame)
            if len(arucoFound[0]) != 0:
                for bbox, id in zip(arucoFound[0], arucoFound[1]):
                    frame = augmentAruco(bbox, id, frame, self.augDics[int(id)])
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = texture

# Приложение Kivy
class TestApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.augDics = loadAugImages('.')  # Укажите путь к вашим изображениям
        self.my_camera = KivyCamera(capture=self.capture, fps=30, findArucoMarkers=findArucoMarkers, augDics=self.augDics)
        return self.my_camera

    def on_stop(self):
        self.capture.release()

if __name__ == '__main__':
    TestApp().run()
