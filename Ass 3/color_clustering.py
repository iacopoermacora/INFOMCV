import glm
import numpy as np

from tqdm import tqdm
import camera_calibration as cc
import settings as settings
import cv2 as cv
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def color_model(total_voxels, total_labels, total_visible_voxels_per_cam, total_visible_voxels_colors_per_cam, idx = 0, offline = False):
    
    '''
    Creates the color models for all the subjects for all the cameras
    '''
    
    # TODO: change the code to other names and shape it differently

    cam_color_models = [[], [], [], []]

    # Loop over all cameras
    for n_camera in range(1, settings.NUM_CAMERAS+1):
        if offline:
            idx == settings.OFFLINE_IDX[n_camera-1]
            labels = np.ravel(total_labels[idx])
            # TODO: Implement code that, based on "known" (to find) centres of the clusters for each camera, 
            # assigns the labels to the clusters in the order of the known centres so that for all the cameras 
            # the labels are the same (for instance, guy with stripes tshirt always have label 1, guy with blue 
            # tshirt always have label 2, etc.)
        else:
            labels = np.ravel(total_labels[idx])
        voxels = np.float32(total_voxels[idx])
        color_models = []
        for label in range(settings.NUMBER_OF_CLUSTERS):

            voxels_person = voxels[labels == label]  # save voxel if the label is same TODO: Not sure of the shape of stuff happening here
            pixelCluster, colorCluster = [], []

            # Take only above the belt and cut the head
            tshirt = np.mean(voxels_person[:, 1], dtype=np.int_)
            voxel_roi = voxels_person[:, 1] > tshirt
            voxels_person_roi = voxels_person[voxel_roi]

            head = np.max(voxels_person_roi[:, 1])
            voxel_roi = voxels_person_roi[:, 1] < 3 / 4 * head
            voxels_person_roi = voxels_person_roi[voxel_roi]

            # TODO/NOTE: Only keep voxels that are part of intersection between visible_voxels_per_cam and voxels_person_roi. Assign them the color from visible_voxels_colors_per_cam
            # Create a numpy array of visible_voxels_colors_per_cam if the intersection between visible_voxels_per_cam and voxels_person_roi
            roi = np.array([total_visible_voxels_colors_per_cam[idx][n_camera][tuple(v)] for v in voxels_person_roi if total_visible_voxels_per_cam[idx][n_camera][tuple(v)]])
            roi = np.float32(roi)

            # Create a GMM model
            model = cv.ml.EM_create()
            model.setClustersNumber(3)

            # Create model per person (cluster)
            model.trainEM(roi)

            color_models.append(model)

        cam_color_models[n_camera] = color_models

    return cam_color_models # color_models is a list of n_camera lists of n_people color models

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

    voxels_no_outliers = [[float(row[0]), float(row[1]), float(row[2])] for row in voxels_no_outliers]

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

def online_phase(total_centers, total_labels, total_voxels, total_visible_voxels_colors_per_cam, total_visible_voxels_per_cam, idx):
    """
    Contains a comparison of the offline color models with the online ones, a
    label matching to obtain the final labelling of each person, and initiates 2D path tracking on the floor.
    """
    
    # TODO: We have to pay attention here to the order of the labeling, by choosing a different idx for each camera the labels are going to be all fucked. 
    # I'll try to figure this out but i need to think about it a bit.
    # POSSIBLE SOLUTION: Because we know every frame (in settings.OFFLINE_IDX there is the idx of the frame) and the centres of the clustering, 
    # we could map for each camera, the label to the cluster we want, so that we make sure that for all cameras, the models are created for the guys we want.
    cam_color_models_offline = color_model(total_centers, total_voxels, total_labels, total_visible_voxels_colors_per_cam, total_visible_voxels_per_cam, offline = True)
    cam_color_models_online = color_model(total_centers, total_voxels, total_labels, total_visible_voxels_colors_per_cam, total_visible_voxels_per_cam, idx = idx)
    
    all_predictions = []
    
    # Loop over all cameras
    for n_camera in range(1, settings.NUM_CAMERAS+1):
        cost_matrix = []
        for label_offline, color_model_offline in enumerate(cam_color_models_offline[n_camera]):
            cost_row = []
            for label_online, color_model_online in enumerate(cam_color_models_online[n_camera]):
                log_likelihood_offline = color_model_offline.score_samples(total_visible_voxels_colors_per_cam[idx][n_camera]) # TODO: I do not understand what is happening here and what is being passed, I think these are not the values that we want to give or I am understanding something wrong
                log_likelihood_online = color_model_online.score_samples(total_visible_voxels_colors_per_cam[idx][n_camera])
                
                # a higher likelihood (less negative) indicates a better match between models
                distance = -(log_likelihood_offline + log_likelihood_online) 
                cost_row.append(distance)
                
            # Add the complete row to the cost matrix  
            cost_matrix.append(cost_row)

        # Convert the cost matrix to a numpy array
        cost_matrix = np.array(cost_matrix)

        # Run the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Store the optimal assignment for the current camera
        predictions = col_ind # TODO: How do we know that they are the predictions? And are the predictions of what and in what order?
        all_predictions.append(predictions)  # Store prediction for all cameras

        # Print the cost matrix
        print("Cost matrix:\n", cost_matrix)
        
        # Print the optimal assignments
        print("\nOptimal assignments:")
        for row, col in zip(row_ind, col_ind):
            print(f"Offline model {row} matched with online model {col} with cost (distance): {cost_matrix[row, col]}")
        
        # Get the final labeling
        camera_preference = None
        final_labels = majority_labeling(all_predictions, camera_preference) # TODO: These final labels, what are they? And in what order are they given?

    return final_labels # NOTE: I will assume that this final labels are the new labels in the order of the old labels (example in next line)
    # EXAMPLE: If final_labels = [3, 1, 2, 4] then the labels will be changed like this [1, 2, 3, 4] -> [3, 1, 2, 4] (the first label is changed to 3, the second to 1, the third to 2 and the fourth to 4)
                
                
                # PSEUDO:
                # 1. Calculate the distance between the offline and the online model
                # 2. Insert the distance in the cost matrix
            # cost_matrix = np.array(cost_matrix)
        # 3. Call the hungarian algorithm
        # 4. Save the values in the cost matrix
        # Run the Hungarian algorithm
        # row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # The `row_ind` and `col_ind` arrays represent the optimal assignment
        # You can use these indices to understand which offline model best matches with which online model

        # For handling ties or preferences over cameras, you would need to implement additional logic
        # This could involve comparing distances or having predefined rules for tiebreakers
        
        # 5. Save based on majority label over the different cameras (hope for no ties or implement a preference system over the cameras)

        # predictions.append(col_ind) # Store the optimal assignment for the current camera

        # Get the final labeling
        # final_labeling(cam_color_models_offline, visible_voxels_colors_per_cam, visible_voxels_per_cam, roi) # to check and implement