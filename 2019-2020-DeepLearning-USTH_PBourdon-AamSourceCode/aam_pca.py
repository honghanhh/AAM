# -*- coding: utf-8 -*-

""" Original, PCA-based Active Appearance Model (AAM) """

from sklearn.decomposition import PCA

import sys
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import skimage
import cv2

from aam_base import *

class AamModel(AamModelBase):
    """ The original, PCA-based AAM model"""
    
    def __init__(self, n_components_shape=None, n_components_texture=None):
        super().__init__()

        self.pca_shape = PCA(whiten=False, n_components=n_components_shape)         #: shape principal components analysis object
        self.pca_texture = PCA(whiten=False, n_components=n_components_texture)     #: texture principal components analysis object

    def _buildShapeModel(self):
        """Overload. Perform Principal Component Analysis on shape data"""
        print('... performing PCA shape analysis on {num_shapes} shapes'.format(num_shapes=len(self)))

        shape_data = self._retrieveShapeDataVector()
        self.pca_shape.fit(shape_data)

        print('AAM PCA shape model: found {} components'.format(self.getNumShapeParams()))

    def shapeDataVecToParams(self, shape_data):
        return self.pca_shape.transform(shape_data.reshape(1,-1)).flatten()

    def shapeParamsToShape(self, p_params, dst):
        """Back-project :math:`p _i` shape parameters to shape"""
        dst[:] = self.pca_shape.inverse_transform(p_params).reshape(-1,2)
        
    def textureParamsToTexture(self, lambd_params, dst):
        """Back-project :math:`\lambda _i` texture parameters to texture"""
        if dst is None:
            dst = np.zeros(self.a0_mask.shape)
        dst[self.a0_mask] = self.pca_texture.inverse_transform(lambd_params).reshape(-1,)
        return dst

    def getMeanShape(self, map_to_a0=False):
        if map_to_a0:
            return self.pca_shape.mean_.reshape(-1,2) + self.a0_offset
        else:
            return self.pca_shape.mean_.reshape(-1,2)

    def getShapeComponent(self, idx):
        return self.pca_shape.components_[idx].reshape(-1,2)

    def pcaShapeGetS(self, idx, map_to_a0=False):
        """Return :math:`s_i` shape PCA component"""
        
        if idx == 0:
            if not map_to_a0:
                return self.pca_shape.mean_.reshape(-1,2)
            else:
                return self.pca_shape.mean_.reshape(-1,2) + self.a0_offset

        else:
            return self.pca_shape.components_[idx-1].reshape(-1,2)

    def getMeanTexture(self, dst=None):
        return self.pcaTextureGetA(0, dst=dst)

    def getTextureComponent(self, idx, dst=None):
        return self.pcaTextureGetA(idx+1, dst=dst)

    def pcaTextureGetA(self, idx, dst=None):
        if dst is None:
            dst = np.ma.masked_array(np.zeros(self.a0_mask.shape), mask=~self.a0_mask)
        else:
            assert dst.shape==self.a0_mask.shape, 'Texture size mismatch'
        if idx == 0:
            dst[self.a0_mask] = self.pca_texture.mean_
        else:
            dst[self.a0_mask] = self.pca_texture.components_[idx-1]
        return dst

    def displayShapeModel(self):
        """Show shape model components"""
        s0_normalized = self.getMeanShape().copy()
        s0_normalized[:,0] /= 2*np.max(np.abs(s0_normalized[:,0]))
        s0_normalized[:,1] /= 2*np.max(np.abs(s0_normalized[:,1]))

        nrows, ncols = 2, 3
        fig, ax =  plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
        ax[0][0].plot(s0_normalized[:,0], s0_normalized[:,1], 'ko', linewidth=0.5)
        ax[0][0].set_title('Mean shape')
        for idx in np.arange(0, self.getNumShapeParams()):
            i = int((idx+1) / ncols)
            j = (idx+1) % ncols
            if i<nrows and j<ncols:
                ax[i][j].plot(s0_normalized[:,0], s0_normalized[:,1], 'k.', linewidth=0.5)
                shape_component = self.getShapeComponent(idx)
                ax[i][j].quiver(s0_normalized[:,0], s0_normalized[:,1], shape_component[:,0], shape_component[:,1], color='r', units='xy', pivot='mid', width=5e-3, scale=2)
                ax[i][j].set_title('Component {idx}'.format(idx=idx))
                ax[i][j].axis([-.65, .65, .65, -.65])

    def displayTextureModel(self):
        nrows, ncols = 2, 3
        fig, ax =  plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
        ax[0][0].imshow(self.getMeanTexture(), cmap='gray')
        ax[0][0].set_title('Mean texture')
        for idx in range(self.getNumTextureParams()):
            img = self.getTextureComponent(idx)
            i = int((idx+1) / ncols)
            j = (idx+1) % ncols
            if i<nrows and j<ncols:
                ax[i][j].imshow(img, cmap='gray')
                ax[i][j].set_title('Texture component {idx}'.format(idx=idx))

