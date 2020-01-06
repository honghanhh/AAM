# -*- coding: utf-8 -*-

""" Autoencoder-based Active Appearance Model (AAM) - Testing application"""

import sys
import os
import argparse
import matplotlib.pyplot as plt

from aam_deep import AamDeepModel

def main(argv):
    plt.close('all')
    
    parser = argparse.ArgumentParser(description='Load an AAM deep model, fit and display reconstruction tests')
    parser.add_argument('aam_deep_model', type=str, default=None, help='AAM deep model to load')
    parser.add_argument('--input_dir', help='input directory for training/testing data')
    parser.add_argument('--n', default=6, type=int, help='shape index to test (default: 6)')

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    aam_deep_model = os.path.abspath(args.aam_deep_model)
    input_dir = os.path.abspath(args.input_dir) if args.input_dir is not None else None
    n = args.n
    
    print('*** Testing AAM deep model ***')
    print('... aam_deep_model: "{}"'.format(aam_deep_model))
    print('... input_dir: "{}"'.format(input_dir))
    print('... n: {}'.format(n))

    model = AamDeepModel.load(aam_deep_model, input_dir)
    model.testReconstruction(n, None, None)
    model.testReconstruction(n, 0, None)
    model.testReconstruction(n, None, 0)
    model.testReconstruction(n, 0, 0)

if __name__ == '__main__':
    main(sys.argv)
