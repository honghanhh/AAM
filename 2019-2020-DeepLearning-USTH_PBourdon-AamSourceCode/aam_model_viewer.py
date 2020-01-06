#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Active Appearance Model (AAM) viewer based on PyQt5"""

import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
import argparse
import sys
import os

import itertools

from PyQt5.QtWidgets import QMainWindow, QApplication, QDockWidget, QWidget, QGridLayout, QSlider
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton, QGroupBox, QHBoxLayout, QDialog
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSlot

import matplotlib.pyplot
import matplotlib.backends.backend_qt5agg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvasQTAgg
from matplotlib.figure import Figure

class AamModelCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=4, height=6, dpi=100):
        self._fig = Figure(figsize=(width, height), dpi=100, facecolor='black', tight_layout=True)
        self._fig.patch.set_alpha(0.0)

        FigureCanvasQTAgg.__init__(self, self._fig)
        self.setParent(parent)
 
        FigureCanvasQTAgg.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)
 
        self._ax = self._fig.add_subplot(111)
        self._ax.set_title('AAM Model')
 
    def displayImg(self, img):
        self._ax.cla()
        self._ax.imshow(img, cmap='gray')
        self.draw()

class AamModelViewer(QDialog):
    def __init__(self, model=None):
        assert model is not None, 'An AAM model is needed'

        super().__init__()
        
        self.initRenderingData(model)

        self.resize(250, 150)
        self.move(300, 300)
        self.setWindowTitle('AAM model viewer')
    
        self._canvas = AamModelCanvas(self, width=4, height=6)

        windowLayout = QHBoxLayout()

        windowLayout.addWidget(self._canvas)

        param_layout = QVBoxLayout()

        reset_btn = QPushButton('Reset')
        reset_btn.clicked.connect(self.on_reset)
        param_layout.addWidget(reset_btn) 

        groupbox = QGroupBox("Shape Parameters")
        layout = QVBoxLayout()
        self._shape_sliders = [QSlider(Qt.Horizontal) for k in range(self._n_shape_params)]
        for slider in self._shape_sliders:
            slider.valueChanged.connect(self.onShapeParamUpdate)
            layout.addWidget(slider)
        groupbox.setLayout(layout)
        param_layout.addWidget(groupbox) 
            
        groupbox = QGroupBox("Texture Parameters")
        layout = QVBoxLayout()
        self._texture_sliders = [QSlider(Qt.Horizontal) for k in range(self._n_texture_params)]
        for slider in self._texture_sliders:
            slider.valueChanged.connect(self.onTextureParamUpdate)
            layout.addWidget(slider)
        groupbox.setLayout(layout)
        param_layout.addWidget(groupbox) 

        windowLayout.addLayout(param_layout)
        self.setLayout(windowLayout)

        self.initSliderRanges()
        self.setGeometry(50, 50, 480, 640)
        self.show()

        self.updateShapeParams()
        self.updateTextureParams()
        self.updateModel()

    def initSliderRanges(self):
        for k, slider in enumerate(self._shape_sliders):
            if type(self._model)==AamModel:
                valrange = self._model.pca_shape.explained_variance_[k]
            else:
                valrange = 1
            valrange = 3*np.sqrt(valrange)
            slider.setMinimum(-valrange)
            slider.setMaximum(valrange)
        for k, slider in enumerate(self._texture_sliders):
            if type(self._model)==AamModel:
                valrange = self._model.pca_texture.explained_variance_[k]
            else:
                valrange = 1
            valrange = 3*np.sqrt(valrange)
            slider.setMinimum(-valrange)
            slider.setMaximum(valrange)

    def initRenderingData(self, model):
        self._model = model
        self._n_shape_params = min(model.getNumShapeParams(), 10)
        self._n_texture_params = min(model.getNumTextureParams(), 10)
        self._shape_params = np.zeros(model.getNumShapeParams())
        self._texture_params = np.zeros(self._model.getNumTextureParams())
        _, self._shape_rec, self._texture_rec = self._model.render(self._shape_params, self._texture_params)

    def updateModel(self):
        img_rec, self._shape_rec, self._texture_rec = self._model.render(self._shape_params, self._texture_params, texture_rec=self._texture_rec, shape_rec=self._shape_rec)
        self._canvas.displayImg(img_rec)

    def updateShapeParams(self):
        self._shape_params[:] = 0
        self._shape_params[:self._n_shape_params] += np.array([slider.value() for slider in self._shape_sliders])

    def updateTextureParams(self):
        self._texture_params[:] = 0
        self._texture_params[:self._n_texture_params] += np.array([slider.value() for slider in self._texture_sliders])

    @pyqtSlot()
    def on_reset(self):
        try:
            QApplication.setOverrideCursor(QtGui.QCursor(Qt.WaitCursor))
            for slider in itertools.chain(self._shape_sliders, self._texture_sliders):
                slider.setValue(0)
            self.updateShapeParams()
            self.updateTextureParams()
            self.updateModel()
            QApplication.restoreOverrideCursor()
        except:
            sys.stderr.write('Unexpected error: {}'.format(sys.exc_info()))

    @pyqtSlot()
    def onShapeParamUpdate(self):
        try:
            self.updateShapeParams()
            self.updateModel()
        except:
            sys.stderr.write('Unexpected error: {}'.format(sys.exc_info()))

    @pyqtSlot()
    def onTextureParamUpdate(self):
        try:
            self.updateTextureParams()
            self.updateModel()
        except:
            sys.stderr.write('Unexpected error: {}'.format(sys.exc_info()))
        

if __name__ == '__main__':
    plt.close('all')
    
    parser = argparse.ArgumentParser(description='AAM model viewer')
    parser.add_argument('input_model', help='input AAM model file')
    parser.add_argument('--is_deep', nargs='?', const=False, default=True, help='AAM model is a deep model')

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    input_model = os.path.abspath(args.input_model)
    is_deep = args.is_deep

    print('*** Viewing AAM model ***')
    print('... input_model: "{}"'.format(input_model))

    if is_deep is False:
        from aam_pca import AamModel
        model = AamModel.load(input_model)
    else:
        from aam_deep import AamDeepModel
        model = AamDeepModel.load(input_model)

    if QtCore.QCoreApplication.instance() != None:
        app = QtCore.QCoreApplication.instance()
    else:
        app = QApplication(sys.argv)

    window = AamModelViewer(model)
    window.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    window.show()
    
    if app:
        app.exec_()


