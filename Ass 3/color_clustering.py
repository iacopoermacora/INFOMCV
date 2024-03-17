import glm
import numpy as np

from tqdm import tqdm
import camera_calibration as cc
import settings as settings
import cv2 as cv
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import pickle
import os

import matplotlib.pyplot as plt

block_size = settings.BLOCK_SIZE

def color_model(total_voxels, total_labels_in, total_visible_voxels_per_cam, total_visible_voxels_colors_per_cam, idx = 0, offline = False):
    
    '''
    Creates the color models for all the subjects for all the cameras

    :param total_voxels: 3D coordinates of the voxels
    :param total_labels_in: labels of the voxels
    :param total_visible_voxels_per_cam: visible voxels for each camera
    :param total_visible_voxels_colors_per_cam: visible voxels colors for each camera
    :param idx: index of the frame for online color model

    :return: cam_color_models
    '''

    cam_color_models = [[], [], [], []]
    cam_rois = [[], [], [], []]

    total_labels = total_labels_in.copy()

    for n_camera in range(1, settings.NUM_CAMERAS+1):
        if offline:
            idx = n_camera - 1
            for label in total_labels[idx]:
                if n_camera == 1:
                    if label == 0:
                        label = 1
                    elif label == 1:
                        label = 2
                    elif label == 2:
                        label = 0
                    elif label == 3:
                        label = 3
                elif n_camera == 2:
                    if label == 0:
                        label = 0
                    elif label == 1:
                        label = 2
                    elif label == 2:
                        label = 3
                    elif label == 3:
                        label = 1
                elif n_camera == 3:
                    if label == 0:
                        label = 0
                    elif label == 1:
                        label = 1
                    elif label == 2:
                        label = 3
                    elif label == 3:
                        label = 2
                elif n_camera == 4:
                    if label == 0:
                        label = 0
                    elif label == 1:
                        label = 2
                    elif label == 2:
                        label = 3
                    elif label == 3:
                        label = 1
                        
            labels = np.ravel(total_labels[idx])
        else:
            labels = np.ravel(total_labels[idx])
        voxels = np.float32(total_voxels[idx])
        color_models = [[], [], [], []]
        rois = []
        for label in range(settings.NUMBER_OF_CLUSTERS):

            voxels_subject = voxels[labels == label]
            # Reshape voxels_subject to x, y, z coordinates (and not voxel shaped)
            voxels_subject[:, 0] = (voxels_subject[:, 0] + settings.WIDTH/2) / settings.BLOCK_SIZE
            voxels_subject[:, 1] = voxels_subject[:, 1] / settings.BLOCK_SIZE
            voxels_subject[:, 2] = (voxels_subject[:, 2] + settings.DEPTH/2) / settings.BLOCK_SIZE

            # Take into consideration only the voxels that are at the same height of the tshirt
            legs_level = np.mean(voxels_subject[:, 1], dtype=np.int_)
            voxels_subject_roi = voxels_subject[voxels_subject[:, 1] > legs_level]
            head_level = np.max(voxels_subject_roi[:, 1])
            voxels_subject_roi = voxels_subject_roi[voxels_subject_roi[:, 1] < 3 / 4 * head_level]

            # Swap y and z in voxels_subject_roi
            voxels_subject_roi[:, 1], voxels_subject_roi[:, 2] = voxels_subject_roi[:, 2], voxels_subject_roi[:, 1].copy()

            # Only take the visible voxels for the roi
            roi = np.array([total_visible_voxels_colors_per_cam[idx][n_camera-1][tuple(v.astype(int))] for v in voxels_subject_roi if total_visible_voxels_per_cam[idx][n_camera-1][tuple(v.astype(int))]])
            roi = np.float32(roi)

            # Only create the model if there are at least 3 voxels with colors
            if len(roi) >= 3:
                # Create the color model
                model = cv.ml.EM_create()

                # Set the number of clusters
                model.setClustersNumber(3)

                # Train the color model
                model.trainEM(roi)

                color_models[label] = model

            rois.append(roi)

        cam_color_models[n_camera-1] = color_models
        cam_rois[n_camera-1] = rois

    return cam_color_models, cam_rois

def remove_outliers(labels_def, centers, voxels, voxels_no_height):
    '''
    Remove the outliers and the ghost voxels to improve the clustering

    :param labels_def: labels of the voxels
    :param centers: centers of the clusters
    :param voxels: 3D coordinates of the voxels
    :param voxels_no_height: 2D coordinates of the voxels

    :return: labels_def_no_outliers, centers_no_outliers, voxels_no_outliers
    '''
    # Calculate the distance of each voxel to the center of its cluster
    distance = []
    for i in range(len(voxels_no_height)):
        center = centers[labels_def[i]]
        distance.append(np.linalg.norm(voxels_no_height[i] - center))

    # Calculate the 0.05 percentile of the distances
    threshold = np.percentile(distance, 90)

    # Remove the outliers based on the threshold
    no_outliers_mask = distance < threshold
    voxels_no_outliers = np.float32(voxels)[no_outliers_mask]
    voxels_no_outliers_no_height = voxels_no_height[no_outliers_mask]

    # Cluster voxels in 3d space based on x/y information
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.1)

    # Seed to make the results reproducible
    np.random.seed(0)

    # Kmeans clustering of the voxels
    _, labels_def_no_outliers, centers_no_outliers = cv.kmeans(voxels_no_outliers_no_height, 4, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)

    voxels_no_outliers = [[float(row[0]), float(row[1]), float(row[2])] for row in voxels_no_outliers]

    return labels_def_no_outliers, centers_no_outliers, voxels_no_outliers

def remove_ghost_voxels(total_labels, total_voxels, total_visible_voxels_per_cam):
    '''
    Removes the ghost voxels

    :param total_labels: The labels of the voxels
    :param total_voxels: The voxel coordinates
    :param total_visible_voxels_per_cam: The visible voxels per camera

    :return: new_total_labels, new_total_centers, new_total_voxels, new_total_voxel_volume_cleaned
    '''


    if os.path.exists(f'remove_ghost.pkl'):
        with open(f'remove_ghost.pkl', 'rb') as f:
            return pickle.load(f)
        
    print("\n\nSTEP 5 - Removing Ghost Voxels")
    new_total_labels = []
    new_total_centers = []
    new_total_voxels = []
    new_total_voxel_volume_cleaned = []
    
    for idx, (frame_voxels, frame_labels) in tqdm(enumerate(zip(total_voxels, total_labels)), desc = "Removing Ghost Voxels - Frame Iteration"):
        active_voxels = []
        count_active = np.zeros(settings.NUMBER_OF_CLUSTERS)
        for label_check in range(settings.NUMBER_OF_CLUSTERS):
            for voxel, label in zip(frame_voxels, frame_labels):
                # Check if the voxel is visible from at least one camera
                if label == label_check:
                    if total_visible_voxels_per_cam[idx][0][int((voxel[0]+settings.WIDTH/2)/block_size)][int((voxel[2]+settings.DEPTH/2)/block_size)][int(voxel[1]/block_size)] or total_visible_voxels_per_cam[idx][1][int((voxel[0]+settings.WIDTH/2)/block_size)][int((voxel[2]+settings.DEPTH/2)/block_size)][int(voxel[1]/block_size)] or total_visible_voxels_per_cam[idx][2][int((voxel[0]+settings.WIDTH/2)/block_size)][int((voxel[2]+settings.DEPTH/2)/block_size)][int(voxel[1]/block_size)] or total_visible_voxels_per_cam[idx][3][int((voxel[0]+settings.WIDTH/2)/block_size)][int((voxel[2]+settings.DEPTH/2)/block_size)][int(voxel[1]/block_size)]:
                        count_active[label_check] += 1
        sorted_count = sorted(count_active)
        if sorted_count[0] < sorted_count[1]*0.1:
            print("Ghost cluster of voxels found in frame ", idx)
            label_not_keep = np.where(count_active == sorted_count[0])[0][0]
            active_voxels.extend([voxel for voxel, label in zip(frame_voxels, frame_labels) if label != label_not_keep])
        else:
            active_voxels = frame_voxels.copy()

        labels, centers, voxels = cluster_voxels(active_voxels, remove_outliers = False)

        voxel_volume_cleaned = np.zeros((settings.WIDTH, settings.DEPTH, settings.HEIGHT), dtype=bool)
        for voxel in voxels:
            voxel_volume_cleaned[int((voxel[0]+settings.WIDTH/2)/block_size)][int((voxel[2]+settings.DEPTH/2)/block_size)][int(voxel[1]/block_size)] = True

        new_total_labels.append(labels)
        new_total_centers.append(centers)
        new_total_voxels.append(voxels)
        new_total_voxel_volume_cleaned.append(voxel_volume_cleaned)
    
    with open(f'remove_ghost.pkl', 'wb') as f:
        data_to_save = (new_total_labels, new_total_centers, new_total_voxels, new_total_voxel_volume_cleaned)
        pickle.dump(data_to_save, f, protocol=4)

    return new_total_labels, new_total_centers, new_total_voxels, new_total_voxel_volume_cleaned

def cluster_voxels(voxels, remove_outliers = True):
    ''''
    Creates the cluster of voxels for each subject
    '''

    # Kmeans clustering of the voxels
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.1) # number of iterations, epsilon value (either when epsilon or number of iteration is reached)
    
    # Cluster voxels in 3d space based on x/y information
    voxels_no_height = np.float32(voxels)[:, [0, 2]] # Keep first and third column
    
    # Seed to make the results reproducible
    np.random.seed(0)

    # Kmeans clustering of the voxels
    _, labels_def, centers = cv.kmeans(voxels_no_height, 4, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS) # 4 is number of clusters, 20 is number of attempts, centres randomly chosen

    # If we don't want to remove outliers, return the labels, centers and voxels
    if not remove_outliers:
        return labels_def, centers, voxels

    labels_def_no_outliers, centers_no_outliers, voxels_no_outliers = remove_outliers(labels_def, centers, voxels, voxels_no_height)

    return labels_def_no_outliers, centers_no_outliers, voxels_no_outliers

def online_phase(total_labels, total_voxels, total_visible_voxels_colors_per_cam, total_visible_voxels_per_cam, cam_color_models_offline, idx):
    """
    Online phase of the color clustering algorithm: it uses the color models to assign the labels to the voxels

    :param total_labels: labels of the voxels
    :param total_voxels: 3D coordinates of the voxels
    :param total_visible_voxels_colors_per_cam: visible voxels colors for each camera
    :param total_visible_voxels_per_cam: visible voxels for each camera
    :param cam_color_models_offline: offline color models for each camera
    :param idx: index of the frame for online color model

    :return: predictions
    """
    
    # Create the color models for the online phase
    _, cam_rois_online = color_model(total_voxels, total_labels, total_visible_voxels_per_cam, total_visible_voxels_colors_per_cam, idx = idx)
    
    # Create the cost matrix for the Hungarian algorithm
    total_cost_matrix = np.zeros((settings.NUMBER_OF_CLUSTERS, settings.NUMBER_OF_CLUSTERS), dtype=np.float32)
    
    for n_camera in range(1, settings.NUM_CAMERAS+1):
        cost_matrix = []
        max_log_likelihood = 0.0
        # Loop over all the ROIs (regions of interest) for the current camera
        for roi in cam_rois_online[n_camera - 1]:
            cost_row = []
            # Loop over all the color models for the current camera
            for label_offline, color_model_offline in enumerate(cam_color_models_offline[n_camera-1]):
                if len(roi) == 0:
                    # Append logarithm of 0 to the cost row (it has 0 probability to be in the cluster)
                    cost_row.append(-np.log(0))
                else:
                    log_likelihood = 0.0
                    # Loop over all the colors in the ROI
                    for color in roi:
                        # Calculate the log likelihood of the color to be in the cluster
                        (single_log, _), _ = color_model_offline.predict2(color)
                        log_likelihood += single_log

                    # Append the negative log likelihood to the cost row
                    cost_row.append(-(log_likelihood))
                    # Update the max log likelihood
                    if -(log_likelihood) > max_log_likelihood:
                        max_log_likelihood = -(log_likelihood)
                    
            # Add the complete row to the cost matrix  
            cost_matrix.append(cost_row)

        # If a value is "inf" set it to "10*max_log_likelihood" (to avoid infinite values in the cost matrix)
        for i in range(len(cost_matrix)):
            for j in range(len(cost_matrix[i])):
                if cost_matrix[i][j] == -np.log(0):
                    cost_matrix[i][j] = 10*max_log_likelihood
        
        # Convert the cost matrix to a numpy array
        cost_matrix = np.array(cost_matrix)

        print("Cost matrix:\n", cost_matrix)

        total_cost_matrix += cost_matrix

    print("Total cost matrix:\n", total_cost_matrix)

    # Run the Hungarian algorithm
    _, predictions = linear_sum_assignment(total_cost_matrix)

    print("\nOptimal assignments:", predictions)
        
    return predictions
            