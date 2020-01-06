# -*- coding: utf-8 -*-

""" AAM-based face data builder for Orange Data Mining """

import sys
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import skimage
import cv2

import Orange
from Orange.data import Domain, Table
from Orange.data import Table, Instance

import itertools

def AamCreateOrangeData(model, out_tabfile):
    sd  = ['s{:02d}'.format(k) for k in range(model.getNumShapeParams())]
    td = ['t{:02d}'.format(k) for k in range(model.getNumTextureParams())]
    
    with open(out_tabfile, 'w') as f:
        f.write('name\t' + '\t'.join(map('{}'.format, itertools.chain(sd, td))) + '\tsubject' + '\temotion' + '\n')
        for k, shape in enumerate(model):
            aligned_points, tform = model.getAlignedPoints(k, dst_shape=model.mean_shape_centered)
            shape_data = aligned_points.flatten()

            aligned_texture = model._computeMeanShapeAlignedTexture(k)
            texture_data = aligned_texture.compressed()
    
            sp = shape_params = model.shapeDataVecToParams(shape_data)
            tp = texture_params = model.textureDataVecToParams(texture_data)
            
            f.write(model[k].name + '\t'\
                    + '\t'.join(map('{}'.format, itertools.chain(sp, tp)))\
                    + '\t' + 'Subject{:02d}'.format(model[k].subject_id)\
                    + '\t' + model[k].emotion_label\
                    + '\n')

def main(argv):
    plt.close('all')
    
    parser = argparse.ArgumentParser(description='Build learning data file from AAM model for Orange data mining')
    parser.add_argument('in_model', help='input AAM model')
    parser.add_argument('out_datafile', help='output .tab data file')
    parser.add_argument('--is_deep', nargs='?', const=False, default=True, help='AAM model is a deep model')

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    in_model = os.path.abspath(args.in_model)
    out_datafile = os.path.abspath(args.out_datafile)
    is_deep = args.is_deep

    print('*** Building Orange data ***')
    print('... in_model: "{}"'.format(in_model))
    print('... out_datafile: "{}"'.format(out_datafile))

    if is_deep is False:
        from aam_pca import AamModel
        model = AamModel.load(in_model)
    else:
        from aam_deep import AamDeepModel
        model = AamDeepModel.load(in_model)

    model.testReconstruction(15)
    AamCreateOrangeData(model, out_datafile)

if __name__ == '__main__':
    main(sys.argv)
