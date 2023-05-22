# Demonstrating epipolar geometry
import cv2
import numpy as np
import sys

from PyQt5 import QtWidgets
from PyQt5 import QtCore

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# global helper variables
window_width = 1400
window_height = 600

# PyQt5 stuff based on https://pythonspot.com/pyqt5-matplotlib/
# Code adapted from https://docs.opencv2.org/4.5.5/da/de9/tutorial_py_epipolar_geometry.html
# and https://www.andreasjakl.com/understand-and-apply-stereo-rectification-for-depth-maps-part-2/


# Main application class where the GUI elements are defined
class App(QtWidgets.QDialog):

    def __init__(self):
        super().__init__()
        self.left = 100
        self.top = 100
        self.title = '3D Computer Vision SoSe2023'
        self.width = 1400
        self.height = 600
        self.initUI()

    def picture_dropped_left(self, left):
        print(left[0])
        self.m.image_name_left = left[0]
        self.m.loadImages()
        self.m.plot()

    def picture_dropped_right(self, right):
        print(right[0])
        self.m.image_name_right = right[0]
        self.m.loadImages()
        self.m.plot()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.createGridLayout()

        windowLayout = QtWidgets.QVBoxLayout()
        windowLayout.addWidget(self.horizontalGroupBox)
        self.setLayout(windowLayout)

        self.show()

    def index_changed(self, index):
        Computation.combo_index_changed(self, index)


    def createGridLayout(self):
        self.horizontalGroupBox = QtWidgets.QGroupBox("Epipolar")
        layout = QtWidgets.QGridLayout()

        # The canvas for two images side by side that can be replaced via drag'n'drop into
        # the zones on the right side
        self.m = PlotTwoImagesCanvas(self, width=11, height=15)
        layout.setRowMinimumHeight(0, 200)
        layout.addWidget(self.m, 0, 0, 9, 1)
        layout.setRowStretch(0, 2)
        layout.setColumnStretch(0, 2)

        # The canvas for a comparison image, either showing the found matches or all epilines
        singleCanvas = PlotSingleImageCanvas(self, width=11, height=15)
        layout.setRowMinimumHeight(9, 200)
        layout.addWidget(singleCanvas, 9, 0, 6, 1)
        layout.setRowStretch(9, 2)

        # Tow drop zones for left and right image
        button = CustomLabel('Drop zone: Left', self)
        button.setToolTip('Drop left image here')
        button.resize(140, 100)
        button.c = Communicate()
        button.c.newImageDropped.connect(self.picture_dropped_left)
        layout.addWidget(button, 0, 1, 1, 1)

        button = CustomLabel('Drop zone: Right', self)
        button.setToolTip('Drop right image here')
        button.resize(140, 100)
        button.c = Communicate()
        button.c.newImageDropped.connect(self.picture_dropped_right)
        layout.addWidget(button, 1, 1, 1, 1)

        # Combo Box for algorithm selection
        comboFundAlgo = QtWidgets.QComboBox()
        comboFundAlgo.addItem('FM_7POINT')
        comboFundAlgo.addItem('FM_8POINT')
        comboFundAlgo.addItem('FM_LMEDS')
        comboFundAlgo.addItem('FM_RANSAC')
        comboFundAlgo.setCurrentIndex(1)
        comboFundAlgo.currentIndexChanged.connect(self.index_changed)
        layout.addWidget(comboFundAlgo, 2, 1, 1, 1)

        # Various buttons
        button = QtWidgets.QPushButton('Recompute Fundamental matrix', self.m)
        button.clicked.connect(Computation.computeFundamentalMatrix)
        layout.addWidget(button, 3, 1, 1, 1)

        button = QtWidgets.QPushButton('Rectify images', self.m)
        button.clicked.connect(self.m.triggerRectification)
        layout.addWidget(button, 4, 1, 1, 1)

        button = QtWidgets.QPushButton('Clear images', self.m)
        button.clicked.connect(self.m.triggerClearImages)
        layout.addWidget(button, 5, 1, 1, 1)

        button = QtWidgets.QPushButton('Show corresponding features', self.m)
        button.clicked.connect(singleCanvas.triggerCorrespondingFeatures)
        button.resize(140, 100)
        layout.addWidget(button, 9, 1, 1, 1)

        button = QtWidgets.QPushButton('Show all epilines', self.m)
        button.clicked.connect(singleCanvas.triggerEpilines)
        layout.addWidget(button, 10, 1, 1, 1)

        radiobutton = QtWidgets.QRadioButton("left points")
        radiobutton.toggled.connect(singleCanvas.onSelectLeftPoints)
        layout.addWidget(radiobutton, 11, 1, 1, 1)

        radiobutton = QtWidgets.QRadioButton("right points")
        radiobutton.toggled.connect(singleCanvas.onSelectRightPoints)
        layout.addWidget(radiobutton, 12, 1, 1, 1)

        radiobutton = QtWidgets.QRadioButton("all epilines")
        radiobutton.setChecked(True)
        radiobutton.toggled.connect(singleCanvas.onSelectAllPoints)
        layout.addWidget(radiobutton, 13, 1, 1, 1)

        self.horizontalGroupBox.setLayout(layout)


class Communicate(QtCore.QObject):
    # Qt signal that indicates if a file has been dropped
    newImageDropped = QtCore.pyqtSignal(list)


# This class is meant for the computations like estimating the fundamental matrix and rectifying the images
class Computation:
    # define the images for computation here
    img = []
    img2 = []
    F = []
    fm_algo = 2
    matches = []
    matchesMask = []
    pts1 = []
    pts2 = []
    kp1 = []
    kp2 = []
    drawCorrespondingFeatures = True

    def combo_index_changed(self, index):
        Computation.fm_algo = np.power(2,index)
        # print(cv2.FM_7POINT) --> 1
        # print(cv2.FM_8POINT) --> 2
        # print(cv2.FM_LMEDS) --> 4
        # print(cv2.FM_RANSAC) --> 8


    def computeFundamentalMatrix(self):
        sift = cv2.SIFT_create()
        # find the keypoints and descriptors with SIFT
        Computation.kp1, des1 = sift.detectAndCompute(Computation.img, None)
        Computation.kp2, des2 = sift.detectAndCompute(Computation.img2, None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        Computation.matches = flann.knnMatch(des1, des2, k=2)
        Computation.matchesMask = [[0, 0] for i in range(len(Computation.matches))]
        pts1 = []
        pts2 = []

        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(Computation.matches):
            if m.distance < 0.8*n.distance:
                Computation.matchesMask[i] = [1, 0]
                pts2.append(Computation.kp2[m.trainIdx].pt)
                pts1.append(Computation.kp1[m.queryIdx].pt)

        # Now we have the list of best matches from both the images. Let's find the Fundamental Matrix.
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        Computation.F, mask = cv2.findFundamentalMat(pts1, pts2, Computation.fm_algo)
        print("Fundamental matrix found:")
        print(Computation.F)
        # We store only inlier points
        Computation.pts1 = pts1[mask.ravel() == 1]
        Computation.pts2 = pts2[mask.ravel() == 1]

    def rectify(self):
        # Stereo rectification (uncalibrated variant)
        # Adapted from: https://stackoverflow.com/a/62607343
        h1, w1 = Computation.img.shape
        h2, w2 = Computation.img2.shape
        _, H1, H2 = cv2.stereoRectifyUncalibrated(
            np.float32(Computation.pts1), np.float32(Computation.pts2), Computation.F, imgSize=(w1, h1)
            )

        # Undistort (rectify) the images and save them
        # Adapted from: https://stackoverflow.com/a/62607343
        # Rectify both the drawing images and the computation images
        PlotTwoImagesCanvas.drawImg1 = cv2.warpPerspective(PlotTwoImagesCanvas.drawImg1, H1, (w1, h1))
        PlotTwoImagesCanvas.drawImg2 = cv2.warpPerspective(PlotTwoImagesCanvas.drawImg2, H2, (w2, h2))
        Computation.img = cv2.warpPerspective(Computation.img, H1, (w1, h1))
        Computation.img2 = cv2.warpPerspective(Computation.img2, H2, (w2, h2))


class PlotTwoImagesCanvas(FigureCanvas):

    # define images for drawing here
    drawImg1 = []
    drawImg2 = []

    def __init__(self, parent=None, width=14, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(121)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.p = []
        self.q = []
        self.image_name_left = 'images/table_left.jpg'
        self.image_name_right = 'images/table_right.jpg'

        self.loadImages()

        Computation.computeFundamentalMatrix(self)

        self.plot()

    def loadImages(self):
        # Load the images, greyscale for computation, color for drawing.
        Computation.img = cv2.imread(self.image_name_left, cv2.IMREAD_GRAYSCALE)
        Computation.img2 = cv2.imread(self.image_name_right, cv2.IMREAD_GRAYSCALE)
        PlotTwoImagesCanvas.drawImg1 = cv2.imread(self.image_name_left, cv2.IMREAD_COLOR)
        PlotTwoImagesCanvas.drawImg2 = cv2.imread(self.image_name_right, cv2.IMREAD_COLOR)

    # trigger functions that are connected to the buttons
    def triggerFundamentalMatrix(self):
        Computation.computeFundamentalMatrix(self)

    def triggerRectification(self):
        Computation.rectify(self)
        self.plot()

    def triggerClearImages(self):
        PlotTwoImagesCanvas.loadImages(self)
        self.plot()

    # plotting the two images via subplot and add callback for clicks
    def plot(self):

        self.figure.clear()

        self.ax1 = self.figure.add_subplot(121)
        self.ax2 = self.figure.add_subplot(122)

        # show the original image
        plot_img1 = cv2.cvtColor(PlotTwoImagesCanvas.drawImg1, cv2.COLOR_BGR2RGB)
        self.ax1.imshow(plot_img1, interpolation='nearest')
        self.ax1.set_axis_off()
        self.ax1.set(title='Left image')

        plot_img2 = cv2.cvtColor(PlotTwoImagesCanvas.drawImg2, cv2.COLOR_BGR2RGB)
        self.ax2.imshow(plot_img2, interpolation='nearest')
        self.ax2.set_axis_off()
        self.ax2.set(title='Right image')

        self.figure.tight_layout()

        cid = self.figure.canvas.mpl_connect('button_press_event', self.onclick)

        self.draw()

    def onclick(self, event):
        # check if images have been clicked
        if event.inaxes != self.ax1 and event.inaxes != self.ax2:
            return
        print('pixel:', event.xdata, event.ydata)
        # draw the clicked point and corresponding epiline in other image
        if event.inaxes == self.ax1:
            self.p = np.int32([(event.xdata, event.ydata)])
            self.drawOneEpiline(PlotTwoImagesCanvas.drawImg1, PlotTwoImagesCanvas.drawImg2, self.p, 1)
        if event.inaxes == self.ax2:
            self.q = np.int32([(event.xdata, event.ydata)])
            self.drawOneEpiline(PlotTwoImagesCanvas.drawImg2, PlotTwoImagesCanvas.drawImg1, self.q, 2)
        if len(self.p) > 0 or len(self.q) > 0:
            self.plot()

    # drawing one epiline copmuted from the fundamental matrix
    def drawOneEpiline(self, img, img2, pt, idx):
        line1 = cv2.computeCorrespondEpilines(pt, idx, Computation.F)
        line1 = line1.reshape(-1, 3)[0]
        r, c, _ = img2.shape
        line_width = int(r / 100)
        circle_size = int(min(r, c)/50)
        color = tuple(np.random.randint(0, 200, 3).tolist())
        x0, y0 = map(int, [0, -line1[2]/line1[1]])
        x1, y1 = map(int, [c, -(line1[2]+line1[0]*c)/line1[1]])
        img2 = cv2.line(img2, (x0, y0), (x1, y1), color, line_width)
        img = cv2.circle(img, pt[0], circle_size, color, -1)


class PlotSingleImageCanvas(FigureCanvas):

    drawImg = np.array([])
    epilineDrawStatus = 2

    def __init__(self, parent=None, width=14, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.plot()

    def triggerCorrespondingFeatures(self):
        PlotSingleImageCanvas.getMatchesImage(self)
        PlotSingleImageCanvas.plot(self)

    def getMatchesImage(self):
        # Draw the keypoint matches between both pictures
        # Still based on: https://docs.opencv2.org/master/dc/dc3/tutorial_py_matcher.html
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=Computation.matchesMask,
                           flags=cv2.DrawMatchesFlags_DEFAULT)

        PlotSingleImageCanvas.drawImg = cv2.drawMatchesKnn(
            Computation.img, Computation.kp1, Computation.img2, Computation.kp2,
            Computation.matches, None, **draw_params)

    def triggerEpilines(self):
        PlotSingleImageCanvas.getEpilinesImage(self)
        PlotSingleImageCanvas.plot(self)

    def getEpilinesImage(self):
        # Next we find the epilines. Epilines corresponding to the points in first image is drawn on second image
        # So mentioning of correct images are important here. We get an array of lines. So we define a new function to
        # draw these lines on the images.
        def drawlines(img1, img2, lines, pts1, pts2):
            ''' img1 - image on which we draw the epilines for the points in img2
                lines - corresponding epilines '''
            r, c = img1.shape
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            for r, pt1, pt2 in zip(lines, pts1, pts2):
                color = tuple(np.random.randint(0, 255, 3).tolist())
                x0, y0 = map(int, [0, -r[2]/r[1]])
                x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
                img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
                img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
                img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
            return img1, img2

        # Now we find the epilines in both the images and draw them.
        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        lines1 = cv2.computeCorrespondEpilines(Computation.pts2.reshape(-1, 1, 2), 2, Computation.F)
        lines1 = lines1.reshape(-1, 3)
        img5, img6 = drawlines(Computation.img, Computation.img2, lines1, Computation.pts1, Computation.pts2)

        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(Computation.pts1.reshape(-1, 1, 2), 1, Computation.F)
        lines2 = lines2.reshape(-1, 3)
        img3, img4 = drawlines(Computation.img2, Computation.img, lines2, Computation.pts2, Computation.pts1)
        match PlotSingleImageCanvas.epilineDrawStatus:
            case 0:
                PlotSingleImageCanvas.drawImg = np.concatenate((img4, img3), axis=1)
            case 1:
                PlotSingleImageCanvas.drawImg = np.concatenate((img5, img6), axis=1)
            case 2:
                PlotSingleImageCanvas.drawImg = np.concatenate((img5, img3), axis=1)

    def onSelectLeftPoints(self):
        PlotSingleImageCanvas.epilineDrawStatus = 0
        PlotSingleImageCanvas.getEpilinesImage(self)
        PlotSingleImageCanvas.plot(self)

    def onSelectRightPoints(self):
        PlotSingleImageCanvas.epilineDrawStatus = 1
        PlotSingleImageCanvas.getEpilinesImage(self)
        PlotSingleImageCanvas.plot(self)

    def onSelectAllPoints(self):
        PlotSingleImageCanvas.epilineDrawStatus = 2
        PlotSingleImageCanvas.getEpilinesImage(self)
        PlotSingleImageCanvas.plot(self)

    def plot(self):

        self.figure.clear()
        if PlotSingleImageCanvas.drawImg.size > 0:
            self.ax1 = self.figure.add_subplot(111)

            # show the original image
            plot_img1 = cv2.cvtColor(PlotSingleImageCanvas.drawImg, cv2.COLOR_BGR2RGB)
            self.ax1.imshow(plot_img1, interpolation='nearest')
            self.ax1.set_axis_off()
            self.ax1.set(title='Comparison image')

            self.figure.tight_layout()

            self.draw()


class CustomLabel(QtWidgets.QLabel):

    def __init__(self, title, parent):
        super().__init__(title, parent)
        self.setAcceptDrops(True)
        self.image_name = 'images/table_02.jpg'
        self.setAlignment(QtCore.Qt.AlignCenter)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        file_name = []
        for url in e.mimeData().urls():
            file_name.append(str(url.toLocalFile()))
        print(file_name)
        self.image_name = file_name
        self.c.newImageDropped.emit(file_name)

    def getImageName(self):
        return self.image_name


if (__name__ == '__main__'):
    app = QtWidgets.QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
