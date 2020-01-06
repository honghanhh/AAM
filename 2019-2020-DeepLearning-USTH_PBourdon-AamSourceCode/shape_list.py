# -*- coding: utf-8 -*-

"""Shape list"""

import argparse
import glob
import re
import sys
import os
import numpy as np
import scipy

import cv2
import skimage
import skimage.transform
import sklearn
import matplotlib.pyplot as plt
import pickle

from procrustes import *
from shape import *

class ShapeList(list):
    """
    A list of Shape objects with database browsing methods
    """

    verbose = False

    def __init__(self):
        self.filename = None                                #: Serialization file name
        self.img_shape = None                               #: Image dimensions. .. note:: Use as a hint only! May differ from image to image (e.g. Cohn-Kanade)
        self.mean_shape = None                              #: Mean shape
        self._datadir = None

    def setDataDir(self, datadir):
        self._datadir = datadir

    def save(self, filename=None, compress=3):
        """Save instance to file"""
        print('... saving {filename}'.format(filename=os.path.abspath(filename)))
        if filename is None:
            filename = self.filename
        else:
            self.filename = filename
        sklearn.externals.joblib.dump(self, filename, compress=compress)
    
    @classmethod
    def load(cls, filename):
        """Load instance from file"""
        print('... loading {filename}'.format(filename=os.path.abspath(filename)))
        obj = sklearn.externals.joblib.load(filename)
        assert not isinstance(obj, cls.__class__), '{objclass} object is not an instance of {classname}'.format(objclass=obj.__class__.__name__, classname=cls.__name__)
        print('... found {num_shapes} shapes'.format(num_shapes=len(obj)))
        obj.filename = os.path.abspath(filename)
        return obj

    def loadImg(self, idx):
        img_path = list.__getitem__(self, idx).img_path
        if not os.path.exists(img_path):
            img_path = list.__getitem__(self, idx).img_path = os.path.join(self._datadir, os.path.basename(img_path))
        return list.__getitem__(self, idx).loadImg()

    def getPoints(self, idx):
        return list.__getitem__(self, idx).points

    def getAlignedPoints(self, idx, dst_shape):
        return list.__getitem__(self, idx).getAlignedPoints(dst_shape=dst_shape)

    def getNumPoints(self):
        return self.mean_shape.shape[0]

    def computeGlobalProcrustesAnalysis(self):
        """Perform Generalized Procrustes analysis (GPA) on shapes"""
        
        print('... performing global procrustes analysis on {num_shapes} shapes'.format(num_shapes=len(self)))
        shapes = np.array([shape.points for shape in self])
        reference_shape = shapes[0]
        num_iter_max = 10
        k = 0
        while k<num_iter_max:
            for idx, shape in enumerate(shapes):
                shape, _ = ComputeProcrustesAnalysis(reference_shape, shape)
                shapes[idx] = shape
            self.mean_shape = np.mean(shapes, axis=0)
            if np.max(np.abs(self.mean_shape - reference_shape)) < 1:
                break
            reference_shape = self.mean_shape
            k += 1
            
        points_x = [shape[:,0] for shape in shapes]
        points_y = [shape[:,1] for shape in shapes]

        self.mean_shape_centered = np.copy(self.mean_shape)
        self.mean_shape_centered[:,0] -= np.mean(self.mean_shape_centered[:,0])
        self.mean_shape_centered[:,1] -= np.mean(self.mean_shape_centered[:,1])

    @classmethod
    def createTextureMask(cls, points_xy):
        assert np.min(points_xy)>=0.0, 'Only positive coordinates are allowed'
        tesselation = scipy.spatial.Delaunay(points_xy)
        width = int(np.ceil(np.max(points_xy[:,0])))
        height = int(np.ceil(np.max(points_xy[:,1])))
        grid = np.mgrid[0:width+1,0:height+1]
        return (tesselation.find_simplex(grid.T) != -1)

    def display(self, aligned=False):
        plt.figure()
        for shapes in self:
            if aligned:
                points, tform = shapes.getAlignedPoints(self.mean_shape)
            else:
                points = shapes.points
            plt.plot(points[:,0], points[:,1], 'kx', linewidth=0.5)
        if aligned:
            plt.plot(self.mean_shape[:,0], self.mean_shape[:,1], 'ro', linewidth=2, label='Mean shape')
            plt.legend()
            plt.title('Shapes (aligned)')
        else:
            plt.title('Shapes (unaligned)')
        plt.axis('scaled')
        plt.gca().invert_yaxis()

    def loadDirCohnKanade(self, input_dir, num_shapes=None):
        print('... reading Cohn-kanade database')
        self.setDataDir(input_dir)
        
        self.clear()
        
        search_pattern = os.path.join(input_dir, 'S[0-9][0-9][0-9]_[0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_landmarks.txt')
        files = glob.glob(search_pattern)
        assert len(files)>0, 'No file found in input directory'
        if num_shapes is not None:
            files = np.array(files)[np.random.randint(0,len(files),num_shapes)]

        n_landmarks = None
        for file in files:
            shape = Shape.loadCohnKanade(file)

            if n_landmarks is None:
                n_landmarks = shape.getNumPoints()
            else:
                assert n_landmarks==shape.getNumPoints(), 'Mismatch in number of landmarks'
            self.append(shape)

        print('... found {num_shapes} shapes'.format(num_shapes=len(self)))
        self.computeGlobalProcrustesAnalysis()
        
def main(argv):
    plt.close('all')
    
    parser = argparse.ArgumentParser(description='Test: read, process and display N randomly selected and aligned shapes from the\
                                     Cohn-Kanade re-organized dataset')
    parser.add_argument('input_dir', help='input directory')
    parser.add_argument('-N', '--num_shapes', type=int, default=None, help='number of shapes to read and process (leave empty to use all shapes)')

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    input_dir = os.path.abspath(args.input_dir)
    num_shapes = args.num_shapes

    print('*** Testing shape list analyzer reader ***')
    print('... input_dir: "{}"'.format(input_dir))
    print('... num_shapes: {}'.format('ALL' if num_shapes is None else num_shapes))

    shape_list = ShapeList()
    shape_list.loadDirCohnKanade(input_dir=input_dir, num_shapes=num_shapes)
    shape_list.display(aligned=False)
    shape_list.display(aligned=True)
    
    plt.show()

if __name__ == '__main__':
    main(sys.argv)
