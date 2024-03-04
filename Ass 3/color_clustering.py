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
    '''

    # NOTE: visible_voxels_per_cam, visible_voxels_colors_per_cam
    # NOTE: change the code to other names and shape it differently

    labels, centers = cluster_voxels(voxels)
    labels = np.ravel(labels)

    voxels = np.float32(voxels)

    cam_color_models = [[], [], [], []]

    # Loop over all cameras
    for n_camera in range(1, settings.NUM_CAMERAS+1):

        color_models = []
        for label in range(settings.NUMBER_OF_CLUSTERS):

            voxels_person = voxels[labels == label]  # save voxel if the label is same NOTE: Not sure of the shape of stuff happening here
            pixelCluster, colorCluster = [], []

            # Take only above the belt and cut the head
            tshirt = np.mean(voxels_person[:, 1], dtype=np.int_)
            voxel_roi = voxels_person[:, 1] > tshirt
            voxels_person_roi = voxels_person[voxel_roi]

            head = np.max(voxels_person_roi[:, 1])
            voxel_roi = voxels_person_roi[:, 1] < 3 / 4 * head
            voxels_person_roi = voxels_person_roi[voxel_roi]

            # NOTE: Only keep voxels that are part of intersection between visible_voxels_per_cam and voxels_person_roi. Assign them the color from visible_voxels_colors_per_cam
            # Create a numpy array of visible_voxels_colors_per_cam if the intersection between visible_voxels_per_cam and voxels_person_roi
            roi = np.array([visible_voxels_colors_per_cam[n_camera][tuple(v)] for v in voxels_person_roi if visible_voxels_per_cam[n_camera][tuple(v)]])
            roi = np.float32(roi)

            # Create a GMM model
            model = cv.ml.EM_create()
            model.setClustersNumber(3)

            # Create model per person (cluster)
            model.trainEM(roi)

            color_models.append(model)

        cam_color_models[n_camera] = color_models

    return cam_color_models # color_models is a list of n_camera lists of n_people color models

def cluster_voxels(voxels):
    ''''
    Creates the cluster of voxels for each subject
    '''
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.1) # number of iterations, epsilon value (either when epsilon or number of iteration is reached)
    
    voxels = np.float32(voxels)[:, [0, 2]] # Remove first and third column

    _, labels_def, centers = cv.kmeans(voxels, 4, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS) # 4 is number of clusters, 20 is number of attempts, centres randomly chosen

    # TODO: IMPLEMENT REMOVAL OF OUTLIERS AND GHOSTS HERE
    # TODO: change the code a bit (variables, order and stuff)

    return labels_def, centers

def remove_outliers_and_ghosts():
    '''
    Remove the outliers and the ghost voxels to improve the clustering
    '''
    return # labels, centres (of the new voxels)

