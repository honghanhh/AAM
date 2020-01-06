# -*- coding: utf-8 -*-

""" (2D) shape / fiducial points. """

import sys
import os
import argparse
import glob
import collections
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import re

import cv2
import skimage
import skimage.transform

from procrustes import *

Rect = collections.namedtuple('Rect', 'x_min x_max y_min y_max')

class Shape(object):
    """
    Fiducial points for image annotations

    :ivar name:             name of shape
    :ivar subject_id:       subject identifier
    :ivar session_id:       session identifier
    :ivar frame_id:         frame identifier
    :ivar emotion_label:    emotion label
    :ivar facs:             facial action coding system
    :ivar points:           (x,y) positions of fiducial points

    .. note::
        (x,y) positions of fiducial points should follow top-left origin image convention. Each (x,y) point maps to pixel Image[y,x]

    """

    _cohn_kanade_emotion_dict = {None:'???', 0:'Neutral', 1:'Anger', 2:'???', 3:'Disgust', 4:'Fear', 5:'Happiness', 6:'Sadness', 7:'Surprise'}
    _emotion_labels = ['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise']

    def __init__(self, points=None):
        self.name = None            #: name of shape
        self.subject_id = None      #: subject identifier
        self.session_id = None      #: session identifier
        self.frame_id = None        #: frame identifier
        self.emotion_label = None   #: emotion label
        self.facs = []              #: facial action coding system
    
        self.points = points        #: (ndarray of float, shape (npoints, ndim)) coordinates of fiducial points
        self.img_shape = tuple
        self.img_path = str()
        self.points_filename = None     #: path to .pts file
        self.emotion_filename = None
        self.facs_filename = None
        
    def getNumPoints(self):
        return self.points.shape[0]

    def loadImg(self):
        """Load image file linked to shape instance (if available)"""
        return cv2.imread(self.img_path, flags=cv2.IMREAD_GRAYSCALE).astype(np.float)/255.0
       
    def display(self, background_img=None, fig=None, pointcolor='r.', textcolor='red', title=None):
        if fig is None:
            fig = plt.figure()
            if background_img is None:
                fig.gca().invert_yaxis()
        if background_img is not None:
            fig.gca().imshow(background_img, cmap='gray')
        fig.gca().plot(self.points[:,0], self.points[:,1], pointcolor)
        if textcolor is not None:
            for idx, point in enumerate(self.points):
                fig.gca().text(point[0], point[1], str(idx), color=textcolor)
        if title is not None:
            fig.set_title(title)
        return fig
            

    def getAlignedPoints(self, dst_shape):
        """Return aligned points and similarity (Procrustes) transform to dst_shape"""
        return ComputeProcrustesAnalysis(dst_shape, self.points)

    @classmethod
    def warpImage(cls, img, tform, output_shape=None):
        warped_img = skimage.transform.warp(img, tform, output_shape=output_shape)
        if img.dtype==np.uint8:
            print('warning: perf pitfall : using skimage warp on uint8 image...')
            return np.round(255*warped_img).astype(np.uint8)
        else:
            return warped_img

    @classmethod
    def loadCohnKanade(cls, path):
        """Load data from Cohn-Kanade *_landmarks.txt annotation file"""

        basename = os.path.basename(path)
        r = re.compile('S(\d+)_(\d+)_(\d+)_landmarks.txt')
        res = r.findall(basename)
        assert len(res)==1, 'File does not match Cohn-Kanade pattern'
        subject_id, session_id, _ = res[0]

        assert os.path.exists(path), 'Landmarks file {} does not exist'.format(path)
        img_path = path.replace('_landmarks.txt', '.png')
        assert os.path.exists(img_path), 'Image file {} does not exist'.format(img_path)
        emotion_path = path.replace('_landmarks.txt', '_emotion.txt')
        assert os.path.exists(emotion_path), 'Emotion label file {} does not exist'.format(emotion_path)

        shape = cls()

        shape.subject_id = int(subject_id)
        shape.session_id = int(session_id)
        shape.name = basename.replace('_landmarks.txt', '')
        shape.img_path = img_path

        with open(path, 'r') as f:
            shape.points = np.array([[float(x) for x in line.split()] for line in f])
        assert shape.points.shape[0]!=0, 'No landmark found in file {}'.format(path)

        with open(emotion_path, 'r') as f:
            shape.emotion_label = shape._cohn_kanade_emotion_dict.get(float(f.readline()))
            
        return shape

def main(argv):
    plt.close('all')

    parser = argparse.ArgumentParser(description='Test: read and display a random shape from the Cohn-Kanade re-organized dataset')
    parser.add_argument('input_dir', help='input directory')

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    input_dir = os.path.abspath(args.input_dir)

    print('*** Testing shape reader ***')
    print('... input_dir: {}'.format(input_dir))

    search_pattern = os.path.join(input_dir, 'S[0-9][0-9][0-9]_[0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_landmarks.txt')
    files = glob.glob(search_pattern)
    assert len(files)>0, 'No file found in input directory'
    landmark_path = files[np.random.randint(0,len(files))]
    shape = Shape.loadCohnKanade(landmark_path)
    img = shape.loadImg()

    shape.display(background_img=img)

    print('*********** DONE! **********')
    
    plt.show()

if __name__ == '__main__':
    main(sys.argv)
