

"""PySide2 Multimedia Camera Example"""

import os, sys
from PyQt5.QtCore import QDate, QDir, QStandardPaths, Qt, QUrl
from PyQt5.QtGui import QClipboard, QGuiApplication, QDesktopServices, QIcon
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QAction, qApp, QApplication, QHBoxLayout, QLabel,
    QMainWindow, QPushButton, QTabWidget, QToolBar, QVBoxLayout, QWidget)
from PyQt5.QtMultimedia import QCamera, QCameraImageCapture, QCameraInfo
from PyQt5.QtMultimediaWidgets import QCameraViewfinder

import numpy as np

from keras.preprocessing import image
from keras.applications import resnet50

from gtts import gTTS

# This module is imported so that we can
# play the converted audio
import os

class ImageView(QWidget):

    def __init__(self, previewImage, fileName,audioFile):
        super(ImageView, self).__init__()

        self.fileName = fileName

        mainLayout = QVBoxLayout(self)
        self.imageLabel = QLabel()
        self.imageLabel.setPixmap(QPixmap.fromImage(previewImage))
        mainLayout.addWidget(self.imageLabel)

        topLayout = QHBoxLayout()
        self.fileNameLabel = QLabel(QDir.toNativeSeparators(fileName))
        self.fileNameLabel.setTextInteractionFlags(Qt.TextBrowserInteraction)

        topLayout.addWidget(self.fileNameLabel)
        topLayout.addStretch()
        copyButton = QPushButton("Copy")
        copyButton.setToolTip("Copy file name to clipboard")
        topLayout.addWidget(copyButton)
        copyButton.clicked.connect(self.copy)
        launchButton = QPushButton("Launch")
        launchButton.setToolTip("Launch image viewer")
        topLayout.addWidget(launchButton)
        launchButton.clicked.connect(self.launch)
        mainLayout.addLayout(topLayout)

        self.playSound(audioFile)



    def copy(self):
        QGuiApplication.clipboard().setText(self.fileNameLabel.text())

    def launch(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.fileName))

    def playSound(self,filename):


        os.system("afplay"+" "+filename)




class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.cameraInfo = QCameraInfo.defaultCamera()
        self.camera = QCamera(self.cameraInfo)
        self.camera.setCaptureMode(QCamera.CaptureStillImage)
        self.imageCapture = QCameraImageCapture(self.camera)
        self.imageCapture.imageCaptured.connect(self.imageCaptured)
        self.imageCapture.imageSaved.connect(self.imageSaved)
        self.currentPreview = QImage()

        toolBar = QToolBar()


        self.addToolBar(toolBar)

        fileMenu = self.menuBar().addMenu("&File")
        shutterIcon = QIcon("/Users/thethelafaltein/PycharmProjects/ResNetApplication/res/img/shutter.svg")
        self.takePictureAction = QAction(shutterIcon, "&Take Picture", self,
                                         shortcut="Ctrl+T",
                                         triggered=self.takePicture)
        self.takePictureAction.setToolTip("Take Picture")
        fileMenu.addAction(self.takePictureAction)
        toolBar.addAction(self.takePictureAction)

        exitAction = QAction(QIcon.fromTheme("application-exit"), "E&xit",
                             self, shortcut="Ctrl+Q", triggered=self.close)
        fileMenu.addAction(exitAction)

        aboutMenu = self.menuBar().addMenu("&About")
        aboutQtAction = QAction("About &Qt", self, triggered=qApp.aboutQt)
        aboutMenu.addAction(aboutQtAction)

        self.tabWidget = QTabWidget()
        self.setCentralWidget(self.tabWidget)


        self.cameraViewfinder = QCameraViewfinder()
        self.camera.setViewfinder(self.cameraViewfinder)
        self.tabWidget.addTab(self.cameraViewfinder, "Viewfinder")




        if self.camera.status() != QCamera.UnavailableStatus:
            name = self.cameraInfo.description()
            self.setWindowTitle("PySide2 Camera Example (" + name + ")")
            self.statusBar().showMessage("Starting: '" + name + "'", 5000)
            self.camera.start()
        else:
            self.setWindowTitle("Object classifier")
            self.takePictureAction.setEnabled(False)
            self.statusBar().showMessage("Camera unavailable", 5000)

    def nextImageFileName(self):
        picturesLocation = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)
        dateString = QDate.currentDate().toString("yyyyMMdd")
        pattern = picturesLocation + "/pyside2_camera_" + dateString + "_{:03d}.jpg"
        n = 1
        while True:
            result = pattern.format(n)
            if not os.path.exists(result):
                return result
            n = n + 1
        return None


    def predicition(self,filename):

        model = resnet50.ResNet50()

        img = image.load_img(filename,
                             target_size=(224, 224))


        x = image.img_to_array(img)


        x = np.expand_dims(x, axis=0)


        x = resnet50.preprocess_input(x)


        predictions = model.predict(x)

        predicted_classes = resnet50.decode_predictions(predictions, top=9)

        top_value = []

        for imagenet_id, name, likelihood in predicted_classes[0]:
            top_value.append(name)

        if len(top_value)>0:
            return top_value[0]
        else:
            return "Unknown"


    def createAudio(self,text):

        language = 'en'

        myobj = gTTS(text=text, lang=language, slow=False)

        myobj.save(text+".mp3")

        return text+".mp3"


    def takePicture(self):
        self.currentPreview = QImage()
        self.camera.searchAndLock()
        self.imageCapture.capture(self.nextImageFileName())
        self.camera.unlock()

    def imageCaptured(self, id, previewImage):
        self.currentPreview = previewImage

    def imageSaved(self, id, fileName):

        predicitionName = self.predicition(fileName)

        audiofile = self.createAudio(predicitionName)

        index = self.tabWidget.count()

        imageView = ImageView(self.currentPreview, fileName,audiofile)

        self.tabWidget.addTab(imageView, predicitionName)
        self.tabWidget.setCurrentIndex(index)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    availableGeometry = app.desktop().availableGeometry(mainWin)
    mainWin.resize(availableGeometry.width() / 3, availableGeometry.height() / 2)
    mainWin.show()
    sys.exit(app.exec_())
