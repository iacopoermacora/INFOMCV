import glm
import numpy as np

from tqdm import tqdm
import camera_calibration as cc
import settings as settings
import cv2 as cv
import assignment as ass
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment



'''def color_model():
    
    Creates the color models for all the subjects for all the cameras
    

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

    return cam_color_models # color_models is a list of n_camera lists of n_people color models'''

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

    _, labels_def_no_outliers, centers_no_outliers = cv.kmeans(voxels_no_outliers_no_height, 4, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)

    # centers_no_outliers_matched = match_center(centers, centers_no_outliers)

    return labels_def_no_outliers, centers_no_outliers, voxels_no_outliers

def cluster_voxels(voxels):
    ''''
    Creates the cluster of voxels for each subject
    '''
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.1) # number of iterations, epsilon value (either when epsilon or number of iteration is reached)
    
    voxels_no_height = np.float32(voxels)[:, [0, 2]] # Keep first and third column

    _, labels_def, centers = cv.kmeans(voxels_no_height, 4, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS) # 4 is number of clusters, 20 is number of attempts, centres randomly chosen

    # TODO: IMPLEMENT REMOVAL OF OUTLIERS AND GHOSTS HERE
    labels_def_no_outliers, centers_no_outliers, voxels_no_outliers = remove_outliers_and_ghosts(labels_def, centers, voxels, voxels_no_height)
    # TODO: change the code a bit (variables, order and stuff)

    return labels_def_no_outliers, centers_no_outliers, voxels_no_outliers

def final_labeling(cam_color_models, visible_voxels_colors_per_cam, visible_voxels_per_cam, roi):
    return


def online_phase(cam_color_models, voxels, visible_voxels_colors_per_cam, visible_voxels_per_cam):
    """
    online is a function dedicated to develop the online phase of assignment 3. It is composed of a
    K-means clustering, a comparison of the offline color models to the online GMMs probabilities, a
    label matching to obtain the final labelling of each person and a 2D path tracking on the floor.
    """
    
    cam_color_models_offline = color_model(visible_voxels_colors_per_cam, visible_voxels_per_cam)
    cam_color_models_online = color_model(visible_voxels_colors_per_cam, visible_voxels_per_cam)
    
    predictions = []
    log_likelihood_offline = []
    log_likelihood_online = []

    cost_matrix = []
    for label_offline, color_model_offline in enumerate(cam_color_models_offline[n_camera]):
        cost_row = []
        for label_online, color_model_online in enumerate(cam_color_models_online[n_camera]):
            
            distance = calculate_distance(color_model_offline, color_model_online) # NOTE: fix function/implement it
            cost_row.append(distance)
            # PSEUDO:
            # 1. Calculate the distance between the offline and the online model
            # 2. Insert the distance in the cost matrix
        cost_matrix = np.array(cost_matrix)
    # 3. Call the hungarian algorithm
    # 4. Save the values in the cost matrix
    # Run the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # The `row_ind` and `col_ind` arrays represent the optimal assignment
    # You can use these indices to understand which offline model best matches with which online model

    # For handling ties or preferences over cameras, you would need to implement additional logic
    # This could involve comparing distances or having predefined rules for tiebreakers
    
    # 5. Save based on majority label over the different cameras (hope for no ties or implement a preference system over the cameras)

    predictions.append(col_ind) # Store the optimal assignment for the current camera

    # Get the final labeling
    final_labeling(cam_color_models_offline, visible_voxels_colors_per_cam, visible_voxels_per_cam, roi) # to check and implement

    return predictions


        
            
        
            
           
