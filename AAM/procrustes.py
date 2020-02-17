# -*- coding: utf-8 -*-

""" Procrustes analysis """

import numpy as np
import skimage
import skimage.transform

def ComputeProcrustesAnalysis(X, Y):
    """
    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion::
    
        Z, [tform] = ComputeProcrustesAnalysis(X, Y)
    
    :param (X,Y): Matrices of target and input coordinates. they must have equal
                  numbers of  points (rows), but Y may have fewer dimensions
                  (columns) than X.

    :return: (Z, tform) tuple

        - Z: (n_points, ndim) array: The matrix of transformed Y-values
        
        - tform: the affine transform matrix specifying the rotation, translation and scaling that maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY
    
    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    rotation = np.dot(V, U.T)

    traceTA = s.sum()

    # optimum scaling of Y
    scale = traceTA * normX / normY

    # transformed coords
    Z = normX*traceTA*np.dot(Y0, rotation) + muX

    # transformation matrix
    if my < m:
        rotation = rotation[:my,:]
    translation = muX - scale*np.dot(muY, rotation)

    affine_matrix = np.zeros((3,3))
    affine_matrix[0:2,0:2] = np.transpose(rotation)
    affine_matrix *= scale
    affine_matrix[:2,2:] = np.transpose(translation.reshape(1,-1))
    affine_matrix[2,2] = 1

    return Z, skimage.transform.AffineTransform(affine_matrix)

