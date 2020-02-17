# -*- coding: utf-8 -*-

""" Autoencoder-based Active Appearance Model (AAM) """

import sys
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import skimage
import cv2

import sklearn
from sklearn.model_selection import train_test_split

import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K

import types
import tempfile
import keras.models

from aam_base import *

#assert keras.__version__[:3]=='2.1', 'at this time model serialization requires downgrading to Keras 2.1'
def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__

make_keras_picklable()

class AamDeepModel(AamModelBase):
    """ A new, autoencoder-based AAM model"""
    
    def __init__(self, n_components_shape=10, n_components_texture=20, n_epochs_shape=500, n_epochs_texture=500):
        super().__init__()
        
        self._n_components_shape = n_components_shape
        self._n_components_texture = n_components_texture

        self._shape_scaler = None
        self._shape_autoencoder = None
        self._shape_encoder = None
        self._shape_decoder = None
        self._shape_batch_size = 32
        self._shape_epochs = n_epochs_shape

        self._texture_autoencoder = None
        self._texture_encoder = None
        self._texture_decoder = None
        self._texture_batch_size = 32
        self._texture_epochs = n_epochs_texture
        self._texture_patch_size = 128
        self._a0_mask_patch = None

    @classmethod
    def load(cls, filename, input_dir=None):
        """Overloading to load Keras models"""

        obj = super().load(filename=filename, input_dir=input_dir)
        print('...  shape num epochs: {}'.format(obj._shape_epochs))
        print('...  shape batch size: {}'.format(obj._shape_batch_size))
        print('...  texture num epochs: {}'.format(obj._texture_epochs))
        print('...  texture batch size: {}'.format(obj._texture_batch_size))
        print('...  texture patch dimension: {size}x{size}'.format(size=obj._texture_patch_size))
        return obj

    def _createEncoderSingleLayer(self, input_dim, code_size):
        encoder = keras.models.Sequential()
        encoder.add(keras.layers.InputLayer((input_dim,)))
        encoder.add(keras.layers.Dense(code_size, activation='relu'))

        decoder = keras.models.Sequential()
        decoder.add(keras.layers.InputLayer((code_size,)))
        decoder.add(keras.layers.Dense(input_dim, activation='sigmoid'))
        
        return encoder, decoder

    def _learnAutoEncoder(self, data_scaled, code_size, batch_size=128, epochs=5000):
        input_dim = data_scaled.shape[1]
        encoder, decoder = self._createEncoderSingleLayer(input_dim, code_size)
        inp = keras.layers.Input(shape=(input_dim,))
        autoencoder = keras.models.Model(inp, decoder(encoder(inp)))
        autoencoder.compile(loss='mse', optimizer='adam')
        history = autoencoder.fit(data_scaled, data_scaled, batch_size=batch_size, epochs=epochs,\
                                  shuffle=True, verbose=1, validation_split=0.3)
        print(autoencoder.summary())

        return autoencoder, encoder, decoder
    
    def _buildTextureModel(self):
        """Perform autoencoding on texture data"""
        print('... performing auto-encoder texture analysis on {num_textures} textures'.format(num_textures=len(self)))
        
        assert type(self._n_components_texture) is int, 'n_components_texture type mismatch : found {}, should be int'.format(type(self._n_components_texture))

        texture_data = self._retrieveTextureDataVector()
        self._texture_scaler = sklearn.preprocessing.MinMaxScaler()
        texture_data_scaled = self._texture_scaler.fit_transform(texture_data)

        input_dim = texture_data_scaled.shape[1]
        self._texture_encoder, self._texture_decoder = self._createEncoderSingleLayer(input_dim=input_dim, code_size=self._n_components_texture)
        inp = keras.layers.Input(shape=(input_dim,))
        self._texture_autoencoder = keras.models.Model(inp, self._texture_decoder(self._texture_encoder(inp)))
        self._texture_autoencoder.compile(loss='mse', optimizer='adam')

        self._texture_autoencoder.fit(texture_data_scaled, texture_data_scaled,\
                                    batch_size=self._texture_batch_size, epochs=self._texture_epochs,\
                                    shuffle=True, verbose=1, validation_split=0.3)

        print(self._texture_autoencoder.summary())

    def _buildShapeModel(self):
        """Perform autoencoding on shape data"""
        print('... performing auto-encoder shape analysis on {num_shapes} shapes'.format(num_shapes=len(self)))
        
        assert type(self._n_components_shape) is int, 'n_components_shape type mismatch : found {}, should be int'.format(type(self._n_components_shape))

        shape_data = self._retrieveShapeDataVector()
        self._shape_scaler = sklearn.preprocessing.MinMaxScaler()
        shape_data_scaled = self._shape_scaler.fit_transform(shape_data)

        input_dim = shape_data_scaled.shape[1]
        self._shape_encoder, self._shape_decoder = self._createEncoderSingleLayer(input_dim=input_dim, code_size=self._n_components_shape)
        inp = keras.layers.Input(shape=(input_dim,))
        self._shape_autoencoder = keras.models.Model(inp, self._shape_decoder(self._shape_encoder(inp)))
        self._shape_autoencoder.compile(loss='mse', optimizer='adam')

        self._shape_autoencoder.fit(shape_data_scaled, shape_data_scaled,\
                                    batch_size=self._shape_batch_size, epochs=self._shape_epochs,\
                                    shuffle=True, verbose=1, validation_split=0.3)

        print(self._shape_autoencoder.summary())

    def _testShapeModel(self):
        shape_data = self._retrieveShapeDataVector()
        test_data = shape_data[np.random.randint(0,shape_data.shape[0])].reshape(1,-1)
        test_data_scaled = self._shape_scaler.transform(test_data)

        encoded_data = self._shape_encoder.predict(test_data_scaled)
        decoded_data = self._shape_decoder.predict(encoded_data)
        shape_test_org = Shape()
        shape_test_org.points = self._shape_scaler.inverse_transform(test_data_scaled).reshape(-1,2)
        fig = shape_test_org.display()
        shape_test_rec = Shape()
        shape_test_rec.points = self._shape_scaler.inverse_transform(decoded_data).reshape(-1,2)
        shape_test_rec.display(fig=fig, pointcolor='g.', textcolor='green')

    def _createCoordMaps(self):
        """Overload"""

        super()._createCoordMaps()
        w = h = self._texture_patch_size
        self._a0_mask_patch = cv2.resize(255*self.a0_mask.astype(np.uint8), dsize=(w, h), interpolation=cv2.INTER_CUBIC)
        self._a0_mask_patch = (self._a0_mask_patch > 128).astype(np.bool)

    # def _retrieveTextureDataVector(self):
    #     """Overload. Quit using individual retrieveTextureData() method for performance improval?"""

    #     texture_data = []
    #     for idx in range(len(self)):
    #         texture_data.append(self.retrieveTextureData(idx))

    #     texture_data = np.array(texture_data)
    #     if len(texture_data.shape)<4:   # adjust ndim for scalar/grayscale images
    #         texture_data = texture_data.reshape(texture_data.shape+(1,))
    #     assert 0<=texture_data.min() and texture_data.max()<=1.0, 'texture data should be normalized'

    #     return texture_data

    def _testTextureModel(self):
        texture_data = self._retrieveTextureDataVector()
        test_data = texture_data[np.random.randint(0,texture_data.shape[0])]
        img_shape = test_data.shape

        encoded_data = self._texture_encoder.predict(np.array([test_data]))
        decoded_data = self._texture_decoder.predict(encoded_data).reshape(img_shape)

        test_data = test_data[:,:,0] if len(test_data.shape)==3 and test_data.shape[2]==1 else test_data
        decoded_data = decoded_data[:,:,0] if len(decoded_data.shape)==3 and decoded_data.shape[2]==1 else decoded_data
        texture_test_org = np.ma.masked_array(test_data, mask=~self._a0_mask_patch)
        texture_test_rec = np.ma.masked_array(decoded_data, mask=~self._a0_mask_patch)
        
        fig, ax =  plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
        ax[0].imshow(texture_test_org, cmap='gray')
        ax[1].imshow(texture_test_rec, cmap='gray')

    def getNumShapeParams(self):
        return self._n_components_shape

    def shapeDataVecToParams(self, shape_data):
        shape_data_scaled = self._shape_scaler.transform(shape_data.reshape(1,-1))
        return self._shape_encoder.predict(shape_data_scaled).flatten()

    def shapeParamsToShape(self, shape_params, dst):
        """Back-project shape parameters to shape"""
        decoded_data = self._shape_decoder.predict(shape_params.reshape(1,-1))
        dst[:] = self._shape_scaler.inverse_transform(decoded_data).reshape(-1,2)
        
    def getNumTextureParams(self):
        return self._n_components_texture

    def textureDataVecToParams(self, texture_data):
        return self._texture_encoder.predict(np.array([texture_data])).flatten()

    def textureParamsToTexture(self, texture_params, dst):
        if dst is None:
            dst = np.zeros(self.a0_mask.shape)
        texture_rec = self._texture_decoder.predict(texture_params.reshape(1,-1))[0]
        texture_rec = texture_rec[:,:,0] if len(texture_rec.shape)==3 and texture_rec.shape[2]==1 else texture_rec
        h, w = self.a0_mask.shape
        dst[:] = cv2.resize(src=texture_rec, dsize=(w, h))
        return dst

    def retrieveTextureData(self, idx):
        """Override"""
        w = h = self._texture_patch_size
        aligned_texture = self._computeMeanShapeAlignedTexture(idx)
        aligned_texture_resized = np.ma.masked_array(np.zeros(self._a0_mask_patch.shape, dtype=np.float), mask=~self._a0_mask_patch)
        cv2.resize(src=aligned_texture.data, dst=aligned_texture_resized.data, dsize=(w, h))
        return aligned_texture_resized.reshape(aligned_texture_resized.shape+(1,))


