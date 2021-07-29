import cv2
import utils
import numpy as np

def april_detector(img_rgb, img_gray, calib_mtx, dist_coef, tag_size = 1, visualize = False,
                    cube_color = (255, 0, 0), tag_family = 'tagStandard41h12'):
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
    img : Image (with the visualization if selected)
    list_tag (Dictionary): List of dictionary containing the information about the tag 
                            tag = {'id': id, 'corners': corners, 'rvec': rvec, 'tvec': tvec, 'size_of_tag': size_of_tag}

    """
    from apriltag import apriltag

    detector = apriltag(tag_family)
    detections = detector.detect(img_gray) # Detect markers on the gray image
    n_detections = len(detections)
        
    list_tag = []
    
    for apriltag in detections:

        hamming = apriltag['hamming']
        margin = apriltag['margin']
        id = apriltag["id"]
        center = apriltag['center']
        corners = apriltag['lb-rb-rt-lt']

        corners = np.array([corners[::-1]])
        
        # Estimate pose of the respective marker
        rvec, tvec, objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, tag_size, calib_mtx, dist_coef)

        if visualize == True:
            img_rgb = utils.drawCube(img_rgb, rvec, tvec, calib_mtx, dist_coef, cube_color = cube_color, tag_size = tag_size)

        tag = {'id': id, 'corners': corners, 'rvec': rvec, 'tvec': tvec, 'size_of_tag': tag_size}
        list_tag.append(tag)

    return img_rgb, list_tag, n_detections



def aruco_detector(img_rgb, img_gray, calib_mtx, dist_coef, tag_size = 1, visualize = False,
                cube_color = (0, 255, 0), tag_family = cv2.aruco.DICT_6X6_250):
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

    list_tag = []

    for i in range(len(aruco_corners)):
        # Estimate pose of the respective marker
        rvec, tvec, objPoints = cv2.aruco.estimatePoseSingleMarkers(aruco_corners[i], tag_size, calib_mtx, dist_coef)
        
        if visualize == True:
            img_rgb = utils.drawCube(img_rgb, rvec, tvec, calib_mtx, dist_coef, cube_color)

        id = ids[i]
        corners = aruco_corners[i]
        
        tag = {'id': id, 'corners': corners, 'rvec': rvec, 'tvec': tvec, 'size_of_tag': tag_size}
        list_tag.append(tag)

    return img_rgb, list_tag, n_detections

def charuco_detector(img_rgb, img_gray, calib_mtx, dist_coef, tag_size = 1, visualize = False,
                cube_color = (255, 255, 255), tag_family = cv2.aruco.DICT_6X6_250, charuco_board = None):
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

    Returns:
    img : Image (with the visualization if selected)
    list_tag (Dictionary): List of dictionary containing the information about the tag 
                            tag = {'id': id, 'corners': corners, 'rvec': rvec, 'tvec': tvec, 'size_of_tag': size_of_tag}

    """
    if charuco_board == None:
        charuco_board = charUcoBoard = cv2.aruco.CharucoBoard_create(4, 4, .045, .0225, cv2.aruco.Dictionary_get(tag_family))

    corners, ids, rejected_points = cv2.aruco.detectMarkers(img_gray, cv2.aruco.getPredefinedDictionary(tag_family))
    if (corners is not None and ids is not None) and (len(corners) == len(ids) and len(corners) != 0):
        # try:
        ret1, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, img_gray, charUcoBoard)

        # rvecN = None 
        # tvecN = None
        ret2, p_rvec, p_tvec = cv2.aruco.estimatePoseCharucoBoard(c_corners, c_ids, charUcoBoard, calib_mtx, dist_coef, rvec = None, tvec = None, useExtrinsicGuess=False)
        if (p_rvec is not None and p_tvec is not None) and (not np.isnan(p_rvec).any() and not np.isnan(p_tvec).any()) :
            # R , _ = cv2.Rodrigues(p_rvec)
            # p_tvec = p_tvec + ( R @ np.array([0, 0.09 , -0.09])).reshape(3,1)
            if visualize == True:
                img_rgb = utils.drawCube(img_rgb, p_rvec, p_tvec, calib_mtx, dist_coef, (255, 255, 255), tag_size = 0.18, is_centered=False)
                #img_rgb = utils.drawCube(img_rgb, p_rvec, p_tvec*5.53, calib_mtx, dist_coef, (255, 255, 255), is_centered=False)
                # aruco.drawAxis(img_rgb, calib_mtx, dist_coef, p_rvec, p_tvec, 1)

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