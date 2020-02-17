# -*- coding: utf-8 -*-

""" Original, PCA-based Active Appearance Model (AAM) - Testing application"""

import sys
import os
import argparse
import matplotlib.pyplot as plt

from aam_pca import AamModel

def main(argv):
    plt.close('all')
    
    parser = argparse.ArgumentParser(description='Load an AAM model, fit and display reconstruction tests')
    parser.add_argument('aam_model', type=str, default=None, help='AAM model to load')
    parser.add_argument('--input_dir', help='input directory for training/testing data')
    parser.add_argument('--n', default=16, type=int, help='shape index to test (default: 6)')

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    aam_model = os.path.abspath(args.aam_model)
    input_dir = os.path.abspath(args.input_dir) if args.input_dir is not None else None
    n = args.n

    print('*** Testing AAM model ***')
    print('... aam_model: "{}"'.format(aam_model))
    print('... input_dir: "{}"'.format(input_dir))
    print('... n: {}'.format(n))

    model = AamModel.load(aam_model, input_dir)
    model.displayTextureModel()
    model.displayShapeModel()
    model.testReconstruction(n, None, None)
    model.testReconstruction(n, 3, None)
    model.testReconstruction(n, None, 3)
    model.testReconstruction(n, 3, 3)
    plt.show()


if __name__ == '__main__':
    main(sys.argv)
