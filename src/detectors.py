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
from numpy.core.numeric import indices
from numpy.lib.arraysetops import unique
import utils
import numpy as np

def detections_writer(ids, img_corners, rvecs, tvecs, tag_size, tag_name):
    
    return {'ids': ids, 'img_corners': img_corners, 'rvecs': rvecs, 'tvec': tvecs, 'tag_size': tag_size, 'tag_name': tag_name}


def april_detector(img_rgb, img_gray, calib_mtx, dist_coef, tag_size = 1, visualize = False,
                    cube_color = (255, 0, 0), tag_family = 'tag36h11', just_img_corners = False):
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
    just_img_corners(bool) : Returns just the ids and the img corners

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
    
    if (n_detections == 0) : return img_rgb, None # No detections

    id = np.empty((n_detections))
    img_corners = np.empty((n_detections, 4, 2))
    rvec = np.empty((n_detections, 3))
    tvec = np.empty((n_detections, 3))

    if just_img_corners == True:
        for i, apriltag in enumerate(detections):
            cur_id = apriltag["id"]
            cur_corners = apriltag['lb-rb-rt-lt']
            id[i] = cur_id
            img_corners[i] = cur_corners
        return id, img_corners
    
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
    detections = detections_writer(id, img_corners, rvec, tvec, tag_size, 'april_tag')
    
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

    if (n_detections == 0) : return img_rgb, None # No detections

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

    
    detections = detections_writer(ids, img_corners, rvec, tvec, tag_size, 'aruco_tag')
       
    return img_rgb, detections



def charuco_detector(img_rgb, img_gray, calib_mtx, dist_coef, tag_size = 1, visualize = False,
                cube_color = (255, 255, 255), tag_family = cv2.aruco.DICT_APRILTAG_36h10, charuco_board = None, use_april_detecotor = False):
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
              
    if use_april_detecotor == True:
        tag_ids, corners = april_detector(img_rgb, img_gray,calib_mtx,dist_coef,tag_size, tag_family = 'tag36h10', just_img_corners = True)
    else :
        corners, tag_ids, rejected_points = cv2.aruco.detectMarkers(img_gray, cv2.aruco.getPredefinedDictionary(tag_family))

    if (len(corners) == 0) : return img_rgb, None # No detections

    if (corners is not None and tag_ids is not None) and (len(corners) == len(tag_ids) and len(corners) != 0):
        
        corners = np.array(corners).squeeze()
        tag_ids = tag_ids.squeeze()
        number_of_charuco = int(np.ceil((np.max(tag_ids) + 1) / 18))

        id = np.empty((number_of_charuco))
        img_corners = np.empty((number_of_charuco, 23, 2))
        rvec = np.empty((number_of_charuco, 3))
        tvec = np.empty((number_of_charuco, 3))

        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h10)
        charUcoBoard = cv2.aruco.CharucoBoard_create(6, 6, tag_size / 6, tag_size / 6 * 0.8, dictionary)

        for i in range(number_of_charuco):
            
            indices = (tag_ids >= 18 * i) & (tag_ids < (18 * (i + 1)))

            if np.any(indices == True): 
                cur_corners = corners[indices]
                cur_ids = tag_ids[indices] - 18 * i

                ret1, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(cur_corners, cur_ids, img_gray, charUcoBoard)
                ret2, p_rvec, p_tvec = cv2.aruco.estimatePoseCharucoBoard(c_corners, c_ids, charUcoBoard, calib_mtx, dist_coef, rvec = None, tvec = None, useExtrinsicGuess=False)

                if (p_rvec is not None and p_tvec is not None) and (not np.isnan(p_rvec).any() and not np.isnan(p_tvec).any()):
                    
                    id[i] = i
                    # img_corners[i] = c_corners.squeeze()
                    rvec[i] = p_rvec.squeeze()
                    tvec[i] = p_tvec.squeeze()
                    if visualize == True:
                        img_rgb = utils.drawCube(img_rgb, p_rvec, p_tvec, calib_mtx, dist_coef, cube_color, tag_size, is_centered=False)
                        # aruco.drawAxis(img_rgb, calib_mtx, dist_coef, p_rvec, p_tvec, 1)

        detections = detections_writer(id, img_corners, rvec, tvec, tag_size, 'charuco_tag')
        return img_rgb, detections

    return img_rgb, None

def stag_detector(img_rgb, img_gray, calib_mtx, dist_coef, tag_size = 1, visualize = False,
                cube_color = (0, 0, 255), tag_family = 17, take_mean = True):
    """Detects the STag in the image.

    Parameters:
    img_rgb : Color image for the visualization
    img_gray : Gray image for detection
    calib_mtx : Calibration matrix in opencv format
    dist_coef : Distortion coefficients in opencv format
    size_of_tag : Size of the tags in meters
    draw (boolean) : Visualizes the
    cube_color (tuple) : RGB color of the cube to be visualized, by default blue
    tag_family(int) : Tag family that is used for the STag
    take_mean (boolean) : If true, takes the mean for the corners of the same ids else takes the first one

    Returns:
    img : Image (with the visualization if selected)
    list_tag (Dictionary): List of dictionary containing the information about the tag 
                            tag = {'id': id, 'corners': corners, 'rvec': rvec, 'tvec': tvec, 'size_of_tag': size_of_tag}

    """
    import pyStag as stag

    stagDetector = stag.Detector(tag_family, 7, False)
    numberofMarkers = stagDetector.detect(img_gray)
    ids = stagDetector.getIds()
    img_corners = stagDetector.getContours()

    if numberofMarkers == 0 : return img_rgb, None # No detection

    ids_np = np.array(ids)
    img_corners_np = np.array(img_corners)

    # Because there are multiple detections for same id it has to be filtered
    unique_ids = np.unique(ids)
    n_detections = unique_ids.shape[0]

    rvec = np.empty((n_detections, 3))
    tvec = np.empty((n_detections, 3))
    filtered_img_corners = np.empty((n_detections, 4, 2))

    for i, cur_id in enumerate (unique_ids):
        if take_mean == True:
            cur_img_corners = np.mean(img_corners_np[ids == cur_id],axis = 0)
        else:
            cur_img_corners = img_corners_np[ids == cur_id][0]

        cur_img_corners = np.array([cur_img_corners[::1]])
            
        # Estimate pose of the respective marker, with matrix size 1x1
        cur_rvec, cur_tvec, _ = cv2.aruco.estimatePoseSingleMarkers(cur_img_corners, tag_size, calib_mtx, dist_coef)

        if visualize == True:
            img_rgb = utils.drawCube(img_rgb, cur_rvec, cur_tvec, calib_mtx, dist_coef, cube_color)

        filtered_img_corners[i] = cur_img_corners
        rvec [i] = cur_rvec
        tvec [i] = cur_tvec

    detections = detections_writer(unique_ids, filtered_img_corners, rvec, tvec, tag_size, 'stag')
    return img_rgb, detections