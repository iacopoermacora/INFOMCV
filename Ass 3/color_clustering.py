import glm
import numpy as np

from tqdm import tqdm
import camera_calibration as cc
import settings as settings
import cv2 as cv
import assignment as ass

def color_model():
    '''
    Creates the color models for all the subjects for all the cameras

    NOTE: Keep into consideration the clustered voxels and in particular 
    their intersection with the colored voxels. Also, only consider the 
    voxels of the shirt (use body proportions to exclude the voxels of 
    legs and head)
    '''
    # TO IMPLEMENT
    return #color_models where color_models is a list of n_camera lists of n_people color models

def cluster_voxels():
    ''''
    Creates the cluster of voxels for each subject
    '''
    #TO IMPLEMENT
    return # labels, centres

def remove_outliers_and_ghosts():
    '''
    Remove the outliers and the ghost voxels to improve the clustering
    '''
    return # labels, centres (of the new voxels)

