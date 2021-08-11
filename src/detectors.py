"""This module includes the detection functions of different tags.

Pameters:    
    img_rgb : Color image for the visualization
    img_gray : Gray image for detection
    calib_mtx : Calibration matrix in opencv format
    dist_coef : Distortion coefficients in opencv format
    size_of_tag : Size of the tags in meters
    draw (boolean) : True for visualization
    cube_color (tuple) : RGB color of the cube to be visualized, by default red
    tag_family (string) : Tag family that is used for the AprilTag 

Return:
    img_rgb : Color image (with the visualization if selected)
    list_tag (Dictionary): List of dictionary containing the information about the tag 
                            tag = {'id': id, 'corners': corners, 'rvec': rvec, 'tvec': tvec, 'size_of_tag': size_of_tag}
                            id (np.array) : numpy array of the ids
                            img_corners (np.array) : numpy array of corners of the tags in the image
                            rvec (np.array) : numpy array of rotational vectors (rodrigues rotation vector)
                            tvec (np.array) : numpy array of translation vector
                            size_of_tag : size of tag in meters
"""

import cv2
import utils
import numpy as np

def april_detector(img_rgb, img_gray, calib_mtx, dist_coef, tag_size = 1, visualize = False,
                    cube_color = (255, 0, 0), tag_family = 'tag36h11'):
    """Detects the AprilTags in the image.

    Parameters:
    img_rgb : Color image for the visualization
    img_gray : Gray image for detection
    calib_mtx : Calibration matrix in opencv format
    dist_coef : Distortion coefficients in opencv format
    size_of_tag : Size of the tags in meters
    draw (boolean) : True for visualization
    cube_color (tuple) : RGB color of the cube to be visualized, by default red
    tag_family (string) : Tag family that is used for the AprilTag 

    Returns:
    img_rgb : Color image (with the visualization if selected)
    list_tag (Dictionary): List of dictionary containing the information about the tag 
                            tag = {'id': id, 'corners': corners, 'rvec': rvec, 'tvec': tvec, 'size_of_tag': size_of_tag}
                            id (np.array) : numpy array of the ids
                            img_corners (np.array) : numpy array of corners of the tags in the image
                            rvec (np.array) : numpy array of rotational vectors (rodrigues rotation vector)
                            tvec (np.array) : numpy array of translation vector
                            size_of_tag : size of tag in meters

    """
    from apriltag import apriltag

    detector = apriltag(tag_family)
    detections = detector.detect(img_gray) # Detect markers on the gray image
    n_detections = len(detections)
        
    list_tag = []

    id = np.empty((n_detections))
    img_corners = np.empty((n_detections, 4, 2))
    rvec = np.empty((n_detections, 3))
    tvec = np.empty((n_detections, 3))
    
    for i, apriltag in enumerate(detections):

        cur_hamming = apriltag['hamming']
        cur_margin = apriltag['margin']
        cur_id = apriltag["id"]
        cur_center = apriltag['center']
        cur_corners = apriltag['lb-rb-rt-lt']

        cur_corners = np.array([cur_corners[::-1]])
        
        # Estimate pose of the respective marker
        cur_rvec, cur_tvec, objPoints = cv2.aruco.estimatePoseSingleMarkers(cur_corners, tag_size, calib_mtx, dist_coef)

        if visualize == True:
            img_rgb = utils.drawCube(img_rgb, cur_rvec, cur_tvec, calib_mtx, dist_coef, cube_color = cube_color, tag_size = tag_size)
        
        id[i] = cur_id
        img_corners[i] = cur_corners
        rvec[i] = cur_rvec
        tvec[i] = cur_tvec

    id = id.astype(int)
    detections = {'id': id, 'img_corners': img_corners, 'rvec': rvec, 'tvec': tvec, 'size_of_tag': tag_size}
    
    return img_rgb, detections



def aruco_detector(img_rgb, img_gray, calib_mtx, dist_coef, tag_size = 1, visualize = False,
                cube_color = (0, 255, 0), tag_family = cv2.aruco.DICT_4X4_250):
    """Detects the ArucoTags in the image.

    Parameters:
    img_rgb : Color image for the visualization
    img_gray : Gray image for detection
    calib_mtx : Calibration matrix in opencv format
    dist_coef : Distortion coefficients in opencv format
    size_of_tag : Size of the tags in meters
    draw (boolean) : Visualizes the
    cube_color (tuple) : RGB color of the cube to be visualized, by default green
    tag_family : Tag family that is used for the ArucoTag 

    Returns:
    img : Image (with the visualization if selected)
    list_tag (Dictionary): List of dictionary containing the information about the tag 
                            tag = {'id': id, 'corners': corners, 'rvec': rvec, 'tvec': tvec, 'size_of_tag': size_of_tag}

    """
    aruco_corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img_gray, cv2.aruco.getPredefinedDictionary(tag_family))
    n_detections = len(aruco_corners)
    img_corners = np.array(aruco_corners).squeeze()

    rvec = np.empty((n_detections, 3))
    tvec = np.empty((n_detections, 3))

    for i in range(len(aruco_corners)):
        # Estimate pose of the respective marker
        cur_rvec, cur_tvec, objPoints = cv2.aruco.estimatePoseSingleMarkers(aruco_corners[i], tag_size, calib_mtx, dist_coef)
        
        if visualize == True:
            img_rgb = utils.drawCube(img_rgb, cur_rvec, cur_tvec, calib_mtx, dist_coef, cube_color)
        
        rvec[i] = cur_rvec
        tvec[i] = cur_tvec

    detections = {'id': ids, 'img_corners': img_corners, 'rvec': rvec, 'tvec': tvec, 'size_of_tag': tag_size}
    
    return img_rgb, detections



def charuco_detector(img_rgb, img_gray, calib_mtx, dist_coef, tag_size = 1, visualize = False,
                cube_color = (255, 255, 255), tag_family = cv2.aruco.DICT_6X6_250, charuco_board = None
                ,threshold_dist_multiplier = 1.55):
    """Detects the ArucoTags in the image.

    Parameters:
    img_rgb : Color image for the visualization
    img_gray : Gray image for detection
    calib_mtx : Calibration matrix in opencv format
    dist_coef : Distortion coefficients in opencv format
    size_of_tag : Size of the tags in meters
    draw (boolean) : Visualizes the
    cube_color (tuple) : RGB color of the cube to be visualized, by default white
    tag_family : Tag family that is used for the ArucoTag in the Charuco 
    charuco_board :
    threshold_dist_multiplier : For filtering out the aruco tags that are multiplier time the id2 and id5 away from the center


    Returns:
    img : Image (with the visualization if selected)
    list_tag (Dictionary): List of dictionary containing the information about the tag 
                            tag = {'id': id, 'corners': corners, 'rvec': rvec, 'tvec': tvec, 'size_of_tag': size_of_tag}

    """

    def charuco_single_detector( img_rgb, img_gray ,corners, ids, charUcoBoard):

        ret1, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, img_gray, charUcoBoard)
        ret2, p_rvec, p_tvec = cv2.aruco.estimatePoseCharucoBoard(c_corners, c_ids, charUcoBoard, calib_mtx, dist_coef, rvec = None, tvec = None, useExtrinsicGuess=False)

        if (p_rvec is not None and p_tvec is not None) and (not np.isnan(p_rvec).any() and not np.isnan(p_tvec).any()) :
            if visualize == True:
                img_rgb = utils.drawCube(img_rgb, p_rvec, p_tvec, calib_mtx, dist_coef, cube_color, tag_size = 0.18, is_centered=False)
                # aruco.drawAxis(img_rgb, calib_mtx, dist_coef, p_rvec, p_tvec, 1)
        return img_rgb, p_rvec, p_tvec
              
    if charuco_board == None:
        charuco_board = charUcoBoard = cv2.aruco.CharucoBoard_create(4, 4, .045, .0225, cv2.aruco.Dictionary_get(tag_family))

    corners, ids, rejected_points = cv2.aruco.detectMarkers(img_gray, cv2.aruco.getPredefinedDictionary(tag_family))
    
    if (corners is not None and ids is not None) and (len(corners) == len(ids) and len(corners) != 0) and (list(ids).count(2) > 0 and list(ids).count(5) > 0 ):
        
        ids_l = list(ids)
        corners_np = np.array(corners).squeeze()
        ids = ids.squeeze()
        
        aruco_centers = corners_np.mean(axis=1)

        if (ids_l.count(2) > 1 and ids_l.count(5) > 1 ): # in the case of multiple markers

            distance_matrix_2_5 = utils.distance_matrix(aruco_centers[np.where(ids == 2)], aruco_centers[np.where(ids == 5)], squared=True)
            id_5_match_index = np.argmin(distance_matrix_2_5,axis=1)
            center_of_closest_id5 = aruco_centers[np.where(ids == 5)][id_5_match_index]
            charuco_centers = ( aruco_centers[np.where(ids == 2)] +  center_of_closest_id5 ) / 2
            distance_2_5 = utils.elementwise_distance(aruco_centers[np.where(ids == 2)], center_of_closest_id5, squared = True)

            distance_matrix = utils.distance_matrix(aruco_centers, charuco_centers).squeeze()
            center_index = np.argmin(distance_matrix, axis=1)  # Closest charuco center to the aruco tags

            rvec, tvec = [], []

            for i in range(distance_matrix.shape[1]):
                cur_aruco_centers = aruco_centers[np.where(center_index == i)]
                cur_corners = corners_np[np.where(center_index == i)]
                cur_ids = ids[np.where(center_index == i)]
                filtered_index = (np.where(utils.distance_matrix(cur_aruco_centers, charuco_centers[i].reshape(1,2), squared = True) < threshold_dist_multiplier**2 * distance_2_5[i]))[0]
                filtered_corners = cur_corners[filtered_index]
                filtered_ids = cur_ids[filtered_index]
                print('The number of filtered elements : ', cur_ids.shape[0] - filtered_ids.shape[0])
                print('Curent ids : ',  filtered_ids)
                img_rgb, cur_rvec, cur_tvec = charuco_single_detector(img_rgb, img_gray, filtered_corners, filtered_ids, charUcoBoard)
                rvec.append(cur_rvec)
                tvec.append(cur_tvec)

        else: # Single tag
            img_rgb, rvec, tvec = charuco_single_detector(img_rgb, img_gray ,corners, ids, charUcoBoard)

    return img_rgb


def stag_detector(img_rgb, img_gray, calib_mtx, dist_coef, tag_size = 1, visualize = False,
                cube_color = (0, 0, 255)):
    """Detects the ArucoTags in the image.

    Parameters:
    img_rgb : Color image for the visualization
    img_gray : Gray image for detection
    calib_mtx : Calibration matrix in opencv format
    dist_coef : Distortion coefficients in opencv format
    size_of_tag : Size of the tags in meters
    draw (boolean) : Visualizes the
    cube_color (tuple) : RGB color of the cube to be visualized, by default blue
    tag_family : Tag family that is used for the ArucoTag 

    Returns:
    img : Image (with the visualization if selected)
    list_tag (Dictionary): List of dictionary containing the information about the tag 
                            tag = {'id': id, 'corners': corners, 'rvec': rvec, 'tvec': tvec, 'size_of_tag': size_of_tag}

    """
    import pyStag as stag

    stagDetector = stag.Detector(11, 7, False)
    numberofMarkers = stagDetector.detect(img_gray)
    ids = stagDetector.getIds()
    detections = stagDetector.getContours()

    n_detections = len(detections)

    list_tag = []

    for i in range(n_detections):
        corners2 = detections[i]
        corners2 = np.array([corners2[::1]])
        # Estimate pose of the respective marker, with matrix size 1x1
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners2, tag_size, calib_mtx, dist_coef)

        if visualize == True:
            img_rgb = utils.drawCube(img_rgb, rvec, tvec, calib_mtx, dist_coef, cube_color)

        id = ids[i]
        corners = corners2
        
        tag = {'id': id, 'corners': corners, 'rvec': rvec, 'tvec': tvec, 'size_of_tag': tag_size}
        list_tag.append(tag)

    return img_rgb, list_tag, n_detections