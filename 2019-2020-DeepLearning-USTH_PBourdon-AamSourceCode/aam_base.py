# -*- coding: utf-8 -*-

""" Active Appearance Model (AAM). Base abstract class model """

import sys
import argparse
import os

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import skimage
import cv2

from shape_list import *

class AamModelBase(ShapeList):
    """Active Appearance Model (AAM) base class"""

    def __init__(self):
        ShapeList.__init__(self)

        self.s0_hull = None                         #: delaunay tesselation of normalized mean shape s0
        self.a0_mask = None                         #: mask of mean texture A0
        self.a0_coords_xy = None                    #: (x,y) coordinates map of base shape s0 within mean texture A0 image matrix
        self.a0_simplices = None                    #: Triangle index for each coordinate
        self._a0_coords_xy_warped = None             #: Warped (x,y) coordinates of A0

    @classmethod
    def load(cls, filename, input_dir=None):
        """Overloading"""
        obj = super().load(filename=filename)
        if input_dir is not None:
            obj.setDataDir(input_dir)
        print('...  num shape params: {}'.format(obj.getNumShapeParams()))
        print('...  num texture params: {}'.format(obj.getNumTextureParams()))
        return obj

    def _createCoordMaps(self):
        """Pre-allocate (x,y) coordinate maps for fast warping"""

#        s0 = self.pcaShapeGetS(0)
        s0 = self.getMeanShape()
        self.s0_hull = sp.spatial.Delaunay(s0)
        self.a0_offset = [-np.min(s0[:,0]), -np.min(s0[:,1])]

        x_min, x_max = np.round(s0[:,0].min()), np.round(s0[:,0].max())
        y_min, y_max = np.round(s0[:,1].min()), np.round(s0[:,1].max())
        self.a0_coords_xy = np.array(np.meshgrid(np.arange(x_min,x_max+1), np.arange(y_min,y_max+1)))
        self.a0_mask = self.s0_hull.find_simplex(np.transpose(self.a0_coords_xy, axes=[1,2,0])-0.5) != -1

        self.a0_coords_xy = np.array(np.transpose(self.a0_coords_xy, axes=[1,2,0]).reshape(-1,2), dtype=np.float32, order='C')

        self._a0_coords_xy_warped = np.empty_like(self.a0_coords_xy)
        
        self.a0_simplices = self.s0_hull.find_simplex(self.a0_coords_xy-0.5)    # triangle index for each coordinate

        self._a0_coords_xy_warped[self.a0_simplices == -1, :] = -1   # coordinates outside of mesh

    def _warpImageToA0(self, src, dst, W, N, accelerate=True):
        """Warp image area into mean texture matrix A0 according to global transform N(x,q) and local transform W(x,p)"""
        
        assert src.dtype==dst.dtype, 'Image format mismatch between src and dst ({0}!={1})'.format(src.dtype, dst.dtype)

        self._a0_coords_xy_warped[:] = W(self.a0_coords_xy)
        self._a0_coords_xy_warped[:] = N(self._a0_coords_xy_warped)

        return cv2.remap(src=src, dst=dst, map1=self._a0_coords_xy_warped.reshape(self.a0_mask.shape + (2,)), map2=None, interpolation=cv2.INTER_LINEAR)

    def _retrieveShapeDataVector(self):
        return np.array([shape.getAlignedPoints(dst_shape=self.mean_shape_centered)[0].flatten() for shape in self])

    def _computeMeanShapeAlignedTexture(self, idx, dst=None):
        if dst is None:
            dst = np.ma.masked_array(np.zeros(self.a0_mask.shape, dtype=np.float), mask=~self.a0_mask)
        s0 = self.getMeanShape()
        tform_global = skimage.transform.estimate_transform('affine', src=s0, dst=self[idx].points)
        tform_piecewise = skimage.transform.estimate_transform('piecewise-affine', src=s0, dst=tform_global.inverse(self[idx].points))
        img = self.loadImg(idx)
        self._warpImageToA0(src=img, dst=dst.data, W=tform_piecewise, N=tform_global, accelerate=False)
        return dst

    def _retrieveTextureDataVector(self):
        aligned_texture = np.ma.masked_array(np.zeros(self.a0_mask.shape, dtype=np.float), mask=~self.a0_mask)

        texture_data = []
        for idx in range(len(self)):
            self._computeMeanShapeAlignedTexture(idx, dst=aligned_texture)
            texture_data.append(aligned_texture.compressed())
            
        return np.array(texture_data)

    def getMeanShape(self, map_to_a0=False):
        if map_to_a0 is True:
            return self.mean_shape_centered + self.a0_offset
        else:
            return self.mean_shape_centered
        
    def getShapeComponent(self, idx):
        raise NotImplementedError

    def getMeanTexture(self, dst=None):
        raise NotImplementedError

    def getTextureComponent(self, idx, dst=None):
        raise NotImplementedError

    def buildModel(self):
        """The main method for AAM building"""

        ## Perform shape analysis
        self._buildShapeModel()

        ## Compute (x,y) coordinates map of base shape s0 within mean texture A0
        self._createCoordMaps()

        ## Perform texture analysis
        self._buildTextureModel()

    def _buildShapeModel(self):
        raise NotImplementedError

    def _buildTextureModel(self):
        raise NotImplementedError

    def getNumShapeParams(self):
        raise NotImplementedError

    def shapeDataVecToParams(self, shape_data):
        raise NotImplementedError

    def shapeParamsToShape(self, shape_params, dst):
        raise NotImplementedError

    def getNumTextureParams(self):
        raise NotImplementedError

    def textureDataVecToParams(self, texture_data):
        raise NotImplementedError

    def textureParamsToTexture(self, lambd_params, dst):
        raise NotImplementedError
        
    def render(self, p_params, lambd_params, texture_rec=None, shape_rec=None):
        if texture_rec is None:
            texture_rec = np.ma.masked_array(np.zeros(self.a0_mask.shape), mask=~self.a0_mask)
        else:
            assert texture_rec.shape == self.a0_mask.shape, 'Texture size does not match'

        if shape_rec is None:
            shape_rec = np.zeros(self.mean_shape.shape)
        else:
            assert shape_rec.shape == self.mean_shape.shape, 'Shape size does not match'

        s0 = self.getMeanShape(map_to_a0=True)
        self.shapeParamsToShape(p_params, dst=shape_rec)
        self.textureParamsToTexture(lambd_params, dst=texture_rec.data)
        
        texture_rec = np.minimum(np.maximum(texture_rec,0),1)
        shape_rec[:,0] -= np.min(shape_rec[:,0])
        shape_rec[:,1] -= np.min(shape_rec[:,1])
        bbox = (int(round(shape_rec[:,1].max()-shape_rec[:,1].min())), int(round(shape_rec[:,0].max()-shape_rec[:,0].min())))
        tform_piecewise = skimage.transform.estimate_transform('piecewise-affine', src=shape_rec, dst=s0)
        return Shape.warpImage(texture_rec, tform_piecewise, output_shape=bbox), shape_rec, texture_rec

    def retrieveShapeData(self, idx):
        aligned_points, tform = self.getAlignedPoints(idx, dst_shape=self.mean_shape_centered)
        return aligned_points.flatten()

    def retrieveTextureData(self, idx):
        aligned_texture = self._computeMeanShapeAlignedTexture(idx)
        return aligned_texture.compressed()

    def testReconstruction(self, idx, n_components_shape=None, n_components_texture=None):
        img = self.loadImg(idx)
        shape_data = self.retrieveShapeData(idx)
        texture_data = self.retrieveTextureData(idx)
                
        shape_params = self.shapeDataVecToParams(shape_data)
        texture_params = self.textureDataVecToParams(texture_data)
        
        if n_components_shape is not None:
            shape_params[n_components_shape:] = 0
        if n_components_texture is not None:
            texture_params[n_components_texture:] = 0
        
        img_rec, shape_rec, texture_rec = self.render(shape_params, texture_params)

        org_pts = self.getPoints(idx)
        h, H, w, W = int(org_pts[:,1].min()), int(org_pts[:,1].max()), int(org_pts[:,0].min()), int(org_pts[:,0].max())
        fig, ax = plt.subplots(2, 2)
        fig.suptitle('AAM reconstruction')
        ax[0][0].imshow(img[h:H,w:W], cmap='gray')
        ax[0][0].set_title('Original image')
        ax[1][0].scatter(shape_rec[:,0], shape_rec[:,1], color='r')
        ax[1][0].invert_yaxis()
        ax[1][0].set_title('Reconstructed shape')
        ax[1][1].imshow(texture_rec, cmap='gray')
        ax[1][1].set_title('Reconstructed texture')
        ax[0][1].imshow(img_rec, cmap='gray')
        ax[0][1].set_title('Fully reconstructed model')

