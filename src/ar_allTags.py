import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
from apriltag import apriltag
import pyStag as stag
import utils
import time  #For calculating the time it takes to calculate

# Defines the path of the calibration file and the dictonary used
calibration_path = "../calibration/calibfiles/realsense_d415_010721_2.npz"

# Load calibration from file
mtx = None
dist = None
with np.load(calibration_path) as X:
	mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]


# Initialize communication with intel realsense
pipeline = rs.pipeline()
realsense_cfg = rs.config()
realsense_cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 6)
pipeline.start(realsense_cfg)

# Check communication
print("Test data source...")
try:
	np.asanyarray(pipeline.wait_for_frames().get_color_frame().get_data())
except:
	raise Exception("Can't get rgb frame from data source")

print("Connection is succesful!")


arUcoDictionary = aruco.DICT_6X6_250
charUcoDictionary = aruco.DICT_6X6_250
# Define what the calibration board looks like (same as the pdf)
charUcoBoard = cv2.aruco.CharucoBoard_create(4, 4, .045, .0225, aruco.Dictionary_get(charUcoDictionary))

print("Press [ESC] to close the application")
while True:

	# Get frame from realsense and convert to grayscale image
	frames = pipeline.wait_for_frames()
	img_rgb = np.asanyarray(frames.get_color_frame().get_data())
	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
	
	" --- AprilTag --- "
	# Detect markers on the gray image
	aprilDetector = apriltag("tagStandard41h12")
	detections = aprilDetector.detect(img_gray)
		
	# Draw each marker 
	for tag in detections:
		corners2 = tag["lb-rb-rt-lt"]
		corners2 = np.array([corners2[::-1]])
		# Estimate pose of the respective marker, with matrix size 1x1
		rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners2, 1, mtx, dist)

		print('April shape of rvec is' ,rvecs, '\n April shape of tvec is', tvecs)
		img_rgb = utils.drawCube(img_rgb, rvecs, tvecs, mtx, dist)
		aruco.drawAxis(img_rgb, mtx, dist, rvecs, tvecs, 0.1)

	" --- End of AprilTag --- "

	" --- ArUco --- "
	corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco.getPredefinedDictionary(arUcoDictionary))
	
	# Draw each marker 
	for i in range(len(corners)):
		# Estimate pose of the respective marker, with matrix size 1x1
		rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners[i], 1, mtx, dist)
		img_rgb = utils.drawCube(img_rgb, rvecs, tvecs, mtx, dist, (0, 255, 0))

	" --- End of AruCo --- "

	" --- CharUco --- "
	corners, ids, rejected_points = cv2.aruco.detectMarkers(img_gray, aruco.getPredefinedDictionary(charUcoDictionary))

	if (corners is not None and ids is not None) and (len(corners) == len(ids) and len(corners) != 0):
		print('Passed the 1st')
		# try:
		ret1, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(corners,
																	ids,
																	img_gray,
																	charUcoBoard)

		rvecN = None 
		tvecN = None
		ret2, p_rvec, p_tvec = cv2.aruco.estimatePoseCharucoBoard(c_corners,
																c_ids,
																charUcoBoard,
																mtx,
																dist,
																rvecN, tvecN, useExtrinsicGuess=False)
		print(ret1,ret2)
		if (p_rvec is not None and p_tvec is not None) and (not np.isnan(p_rvec).any() and not np.isnan(p_tvec).any()) :
			print("Passed the 2nd", p_rvec, p_tvec)
			print('Char shape of rvec is' ,rvecs, '\n Char Shape of tvec is', tvecs)
			# img_rgb = utils.drawCube(img_rgb, p_rvec, p_tvec, mtx, dist, (255, 255, 255))
			aruco.drawAxis(img_rgb, mtx, dist, p_rvec, p_tvec/10, 0.1)
        # cv2.aruco.drawDetectedCornersCharuco(frame, c_corners, c_ids)
        # cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        # cv2.aruco.drawDetectedMarkers(frame, rejected_points, borderColor=(100, 0, 240))

		# except cv2.error:
		# 	pass
        

	
	# if len(corners) == 8:
	# 	retval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, img_gray, charUcoBoard)
		
	# 	# Draw each marker 
	# 	for i in range(int(len(charucoIds)/9)):
	# 		# Estimate pose of the respective marker, with matrix size 1x1
	# 		rvecN = np.zeros((3,3))
	# 		rvecN = np.array([rvecN])
	# 		tvecN = np.zeros(3)
	# 		tvecN = np.array([[tvecN]])
	# 		retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, charUcoBoard, mtx, dist, rvecN, tvecN, useExtrinsicGuess=False)
	# 		rvec = rvec.reshape(1,1,3)
	# 		tvec = tvec.reshape(1,1,3)

	# 		img_rgb = utils.drawCube(img_rgb, rvec, tvec, mtx, dist)
	# 		aruco.drawAxis(img_rgb, mtx, dist, rvec, tvec, 0.1)

	" --- End of CharUco --- "

	" --- STag --- "
	stagDetector = stag.Detector(11, 7, False)
	numberofMarkers = stagDetector.detect(img_gray)
	markerIDs = stagDetector.getIds()
	detections = stagDetector.getContours()

	for tag in detections:
		corners2 = tag
		corners2 = np.array([corners2[::1]])
		# Estimate pose of the respective marker, with matrix size 1x1
		rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners2, 1, mtx, dist)
	
		img_rgb = utils.drawCube(img_rgb, rvecs, tvecs, mtx, dist, (0,0,255))
		
	" --- End of Stag --- "

	# Display the result
	cv2.imshow("AR-Example", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
	
	# If [ESC] pressed, close the application
	if cv2.waitKey(100) == 27:
		print("Application closed")
		break
# Close all cv2 windows
cv2.destroyAllWindows()
