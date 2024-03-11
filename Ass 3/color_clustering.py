import glm
import numpy as np

from tqdm import tqdm
import camera_calibration as cc
import settings as settings
import cv2 as cv
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt


def color_model(total_voxels, total_labels, total_visible_voxels_per_cam, total_visible_voxels_colors_per_cam, idx = 0, offline = False):
    
    '''
    Creates the color models for all the subjects for all the cameras
    '''
    
    # TODO: change the code to other names and shape it differently

    cam_color_models = [[], [], [], []]
    cam_rois = [[], [], [], []]

    # Loop over all cameras
    for n_camera in range(1, settings.NUM_CAMERAS+1):
        if offline:
            idx = n_camera - 1
            print(f"idx: {idx}")
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
        color_models = []
        rois = []
        for label in range(settings.NUMBER_OF_CLUSTERS):

            voxels_person = voxels[labels == label]  # save voxel if the label is same TODO: Not sure of the shape of stuff happening here
            # Reshape voxels_person to x, y, z coordinates (and not voxel shaped)
            voxels_person[:, 0] = (voxels_person[:, 0] + settings.WIDTH/2) / settings.BLOCK_SIZE
            voxels_person[:, 1] = voxels_person[:, 1] / settings.BLOCK_SIZE
            voxels_person[:, 2] = (voxels_person[:, 2] + settings.DEPTH/2) / settings.BLOCK_SIZE

            # Take only above the belt and cut the head
            tshirt = np.mean(voxels_person[:, 1], dtype=np.int_)
            voxel_roi = voxels_person[:, 1] > tshirt
            voxels_person_roi = voxels_person[voxel_roi]

            head = np.max(voxels_person_roi[:, 1])
            voxel_roi = voxels_person_roi[:, 1] < 3 / 4 * head
            voxels_person_roi = voxels_person_roi[voxel_roi]

            # Swap y and z in voxels_person_roi
            voxels_person_roi[:, 1], voxels_person_roi[:, 2] = voxels_person_roi[:, 2], voxels_person_roi[:, 1].copy()

            # Only take the visible voxels for the roi
            roi = np.array([total_visible_voxels_colors_per_cam[idx][n_camera-1][tuple(v.astype(int))] for v in voxels_person_roi if total_visible_voxels_per_cam[idx][n_camera-1][tuple(v.astype(int))]])
            roi = np.float32(roi)

            # Create a GMM model
            model = cv.ml.EM_create()
            model.setClustersNumber(3)

            # Create model per person (cluster)
            model.trainEM(roi)

            color_models.append(model)
            rois.append(roi)

        cam_color_models[n_camera-1] = color_models
        cam_rois[n_camera-1] = rois

    return cam_color_models, cam_rois # color_models is a list of n_camera lists of n_people color models

'''def match_center(centers, centers_no_outliers):
    
    Match the centers of the clusters with the centers of the clusters without outliers
    
    
    # Calculate all pairwise distances between original and filtered centers
    distances = cdist(centers, centers_no_outliers)

    # Find the closest new center for each original center
    min_indices = np.argmin(distances, axis=1)

    # Order the centers_filtered based on the match with original centers
    centers_no_outliers_matched = centers_no_outliers[min_indices]

    return centers_no_outliers_matched'''

def remove_outliers_and_ghosts(labels_def, centers, voxels, voxels_no_height):
    '''
    Remove the outliers and the ghost voxels to improve the clustering
    '''
    distance = []
    for i in range(len(voxels_no_height)):
        center = centers[labels_def[i]]
        distance.append(np.linalg.norm(voxels_no_height[i] - center))

    # Calculate the 0.05 percentile of the distances
    threshold = np.percentile(distance, 90)

    no_outliers_mask = distance < threshold
    voxels_no_outliers = np.float32(voxels)[no_outliers_mask]
    voxels_no_outliers_no_height = voxels_no_height[no_outliers_mask]
    # labels_def_no_outliers = labels_def[no_outliers_mask]

    # Remove the ghost voxels
    '''    for i in range(len(voxels_no_outliers)):
        for j in range(i+1, len(voxels_no_outliers)):
            if labels_def_no_outliers[i] == labels_def_no_outliers[j]:
                if np.linalg.norm(voxels_no_outliers[i] - voxels_no_outliers[j]) < 10:
                    labels_def_no_outliers[j] = -1'''

    # Cluster voxels in 3d space based on x/y information
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.1)

    np.random.seed(0)

    _, labels_def_no_outliers, centers_no_outliers = cv.kmeans(voxels_no_outliers_no_height, 4, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)

    # centers_no_outliers_matched = match_center(centers, centers_no_outliers)

    voxels_no_outliers = [[float(row[0]), float(row[1]), float(row[2])] for row in voxels_no_outliers]

    return labels_def_no_outliers, centers_no_outliers, voxels_no_outliers

def cluster_voxels(voxels):
    ''''
    Creates the cluster of voxels for each subject
    '''
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.1) # number of iterations, epsilon value (either when epsilon or number of iteration is reached)
    
    voxels_no_height = np.float32(voxels)[:, [0, 2]] # Keep first and third column

    np.random.seed(0)

    _, labels_def, centers = cv.kmeans(voxels_no_height, 4, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS) # 4 is number of clusters, 20 is number of attempts, centres randomly chosen

    # TODO: IMPLEMENT REMOVAL OF OUTLIERS AND GHOSTS HERE
    labels_def_no_outliers, centers_no_outliers, voxels_no_outliers = remove_outliers_and_ghosts(labels_def, centers, voxels, voxels_no_height)
    # TODO: change the code a bit (variables, order and stuff)

    return labels_def_no_outliers, centers_no_outliers, voxels_no_outliers

def majority_labeling(all_predictions, camera_preference = None):
    '''
    Aggregate predictions from all cameras and determine the final label for each cluster
    based on a majority vote. In case of ties, preferences 
    '''
    # Initialize a structure to hold vote counts for each cluster's label
    vote_counts = [{} for _ in range(max(map(len, all_predictions)))]  # Create a list of dictionaries, one for each cluster
    

    for camera_index, predictions in enumerate(all_predictions): 
        for cluster_index, label in enumerate(predictions):
            if label not in vote_counts[cluster_index]:
                # Initialize the vote count for this label
                vote_counts[cluster_index][label] = 0 
            # Increment the vote count for this label; consider camera preference if specified
            vote_increment = 1 if camera_preference is None else camera_preference[camera_index]
            vote_counts[cluster_index][label] += vote_increment

    # Determine the final label for each cluster based on majority vote
    final_labels = []
    for cluster_votes in vote_counts:
        # Sort labels by vote count (and by camera preference in case of a tie)
        sorted_votes = sorted(cluster_votes.items(), key=lambda item: (-item[1], camera_preference.index(item[0]) if camera_preference else 0))
        final_labels.append(sorted_votes[0][0] if sorted_votes else None)

    return final_labels

def online_phase(total_labels, total_voxels, total_visible_voxels_colors_per_cam, total_visible_voxels_per_cam, idx):
    """
    Contains a comparison of the offline color models with the online ones, a
    label matching to obtain the final labelling of each person, and initiates 2D path tracking on the floor.
    """
    
    # TODO: We have to pay attention here to the order of the labeling, by choosing a different idx for each camera the labels are going to be all fucked. 
    # I'll try to figure this out but i need to think about it a bit.
    # POSSIBLE SOLUTION: Because we know every frame (in settings.OFFLINE_IDX there is the idx of the frame) and the centres of the clustering, 
    # we could map for each camera, the label to the cluster we want, so that we make sure that for all cameras, the models are created for the guys we want.
    cam_color_models_offline, _ = color_model(total_voxels, total_labels, total_visible_voxels_per_cam, total_visible_voxels_colors_per_cam, offline = True)
    _, cam_rois_online = color_model(total_voxels, total_labels, total_visible_voxels_per_cam, total_visible_voxels_colors_per_cam, idx = idx)
    
    # Total cost matrix initiation, shaped number of clusters x number of clusters
    total_cost_matrix = np.zeros((settings.NUMBER_OF_CLUSTERS, settings.NUMBER_OF_CLUSTERS), dtype=np.float32)
    # Loop over all cameras
    for n_camera in range(1, settings.NUM_CAMERAS+1):
        cost_matrix = [] # TODO: Check if we need and where
        for roi in cam_rois_online[n_camera - 1]:
            cost_row = []
            for label_offline, color_model_offline in enumerate(cam_color_models_offline[n_camera-1]):
                
                log_likelihood = 0.0
                for color in roi:
                    (single_log, _), _ = color_model_offline.predict2(color)
                    log_likelihood += single_log

                cost_row.append(-log_likelihood)
                    
            # Add the complete row to the cost matrix  
            cost_matrix.append(cost_row)

        # Convert the cost matrix to a numpy array
        cost_matrix = np.array(cost_matrix)

        print("Cost matrix:\n", cost_matrix)

        total_cost_matrix += cost_matrix

    print("Total cost matrix:\n", total_cost_matrix)
    # Run the Hungarian algorithm
    _, predictions = linear_sum_assignment(total_cost_matrix)

    print("\nOptimal assignments:", predictions) # NOTE: Predictions are, in the order of the labels of the frame (random ones), which label (ACTUAL, CORRECT LABELS) is the best for the current camera based on the color models
        
    return predictions # NOTE: I will assume that this final labels are the new labels in the order of the old labels (example in next line)
    # EXAMPLE: If final_labels = [3, 1, 2, 4] then the labels will be changed like this [1, 2, 3, 4] -> [3, 1, 2, 4] (the first label is changed to 3, the second to 1, the third to 2 and the fourth to 4)
            