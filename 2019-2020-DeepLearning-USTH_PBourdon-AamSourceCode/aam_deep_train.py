# -*- coding: utf-8 -*-

""" Autoencoder-based Active Appearance Model (AAM) - Training application"""

import sys
import os
import argparse
import matplotlib.pyplot as plt

from aam_deep import AamDeepModel

def main(argv):
    plt.close('all')
    
    parser = argparse.ArgumentParser(description='Train an AAM deel model using all or N randomly selected aligned shapes from the\
                                     Cohn-Kanade re-organized dataset')
    parser.add_argument('input_dir', help='input directory')
    parser.add_argument('-N', '--num_shapes', type=int, default=None, help='number of shapes to read and process (leave empty to use all shapes)')
    parser.add_argument('-O', '--out_model', type=str, default=None, help='save AAM model to output file')
    parser.add_argument('--n_components_shape', type=int, default=32, help='number of shape components for model (default 32)')
    parser.add_argument('--n_components_texture', type=int, default=128, help='number of texture components for model (default 128)')
    parser.add_argument('--n_epochs_shape', type=int, default=500, help='number of epochs to train shape model (default 500)')
    parser.add_argument('--n_epochs_texture', type=int, default=500, help='number of epochs to train texture model (default 500)')

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    args = parser.parse_args()
    input_dir = os.path.abspath(args.input_dir)
    num_shapes = args.num_shapes
    out_model = args.out_model
    n_components_shape = args.n_components_shape
    n_components_texture = args.n_components_texture
    n_epochs_shape = args.n_epochs_shape
    n_epochs_texture = args.n_epochs_texture

    print('*** Training AAM deep model ***')
    print('*** WARNING: SERIALIZATION NEEDS DOWNGRADE TO KERAS 2.1 !!! ***')
    print('... input_dir: "{}"'.format(input_dir))
    print('... num_shapes: {}'.format('ALL' if num_shapes is None else num_shapes))
    print('... out_model: {}'.format('DON\'T SAVE' if out_model is None else out_model))
    print('... n_components_shape: {}'.format(n_components_shape))
    print('... n_components_texture: {}'.format(n_components_texture))
    print('... n_epochs_shape: {}'.format(n_epochs_shape))
    print('... n_epochs_texture: {}'.format(n_epochs_texture))

    model = AamDeepModel(n_components_shape=n_components_shape, n_components_texture=n_components_texture,\
                         n_epochs_shape=n_epochs_shape, n_epochs_texture=n_epochs_texture)
    model.loadDirCohnKanade(input_dir=input_dir, num_shapes=num_shapes)
    model.buildModel()
    if out_model is not None:
        model.save(out_model)

if __name__ == '__main__':
    main(sys.argv)
