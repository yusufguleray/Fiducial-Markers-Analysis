import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
from apriltag import apriltag
import pyStag as stag
import utils
import time  #For calculating the time it takes to calculate

isApril = False
isAruCo = False
isCharuco = True
isStag = False

isTest = False

mtx, dist = utils.getCalibData("realsense_d415_010721_2.npz")

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

"--- Data for testing ---"
nOfFrames = 0
averageCalculationTime = 0
averageDetectionNumber = 0
previousTags = None

while True:
	time_start = time.perf_counter()

	# Get frame from realsense and convert to grayscale image
	frames = pipeline.wait_for_frames()
	img_rgb = np.asanyarray(frames.get_color_frame().get_data())
	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
	
	if isApril == True:
		" --- AprilTag --- "
		# Detect markers on the gray image
		aprilDetector = apriltag("tagStandard41h12")
		detections = aprilDetector.detect(img_gray)
		nOfDetections = len(detections)
			
		# Draw each marker 
		for tag in detections:
			corners2 = tag["lb-rb-rt-lt"]
			corners2 = np.array([corners2[::-1]])
			# Estimate pose of the respective marker, with matrix size 1x1
			rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners2, 1, mtx, dist)

			img_rgb = utils.drawCube(img_rgb, rvecs, tvecs, mtx, dist)

		" --- End of AprilTag --- "

	if isAruCo == True:
		" --- ArUco --- "
		corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco.getPredefinedDictionary(arUcoDictionary))
		
		nOfDetections = len(corners)
		# Draw each marker 
		for i in range(len(corners)):
			# Estimate pose of the respective marker, with matrix size 1x1
			rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners[i], 1, mtx, dist)
			img_rgb = utils.drawCube(img_rgb, rvecs, tvecs, mtx, dist, (0, 255, 0))

		" --- End of AruCo --- "

	if isCharuco == True:
		" --- CharUco --- "
		corners, ids, rejected_points = cv2.aruco.detectMarkers(img_gray, aruco.getPredefinedDictionary(charUcoDictionary))
		print(ids)
		if (corners is not None and ids is not None) and (len(corners) == len(ids) and len(corners) != 0):
			# try:
			ret1, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, img_gray, charUcoBoard)

			rvecN = None 
			tvecN = None
			ret2, p_rvec, p_tvec = cv2.aruco.estimatePoseCharucoBoard(c_corners, c_ids, charUcoBoard, mtx, dist, rvecN, tvecN, useExtrinsicGuess=False)
			if (p_rvec is not None and p_tvec is not None) and (not np.isnan(p_rvec).any() and not np.isnan(p_tvec).any()) :
				R , _ = cv2.Rodrigues(p_rvec)
				# p_tvec = p_tvec + ( R @ np.array([0, 0.09 , -0.09])).reshape(3,1)
				img_rgb = utils.drawCubeChar(img_rgb, p_rvec, p_tvec*5.53, mtx, dist, (255, 255, 255))
				# aruco.drawAxis(img_rgb, mtx, dist, p_rvec, p_tvec, 1)
				

			# The problem is that charuco translation vector is too close
		" --- End of CharUco --- "

	if isStag == True:
		" --- STag --- "
		stagDetector = stag.Detector(11, 7, False)
		numberofMarkers = stagDetector.detect(img_gray)
		markerIDs = stagDetector.getIds()
		detections = stagDetector.getContours()

		nOfDetections = len(detections)

		for tag in detections:
			corners2 = tag
			corners2 = np.array([corners2[::1]])
			# Estimate pose of the respective marker, with matrix size 1x1
			rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners2, 1, mtx, dist)
		
			img_rgb = utils.drawCube(img_rgb, rvecs, tvecs, mtx, dist, (0,0,255))
			
		" --- End of Stag --- "

	# Display the result
	cv2.imshow("AR-Example", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
	
	if isTest == True:
		"--- Testing Stuff ---"
		nOfFrames+=1
		time_elapsed = (time.perf_counter() - time_start)
		averageCalculationTime = averageCalculationTime + 1/nOfFrames*(time_elapsed - averageCalculationTime)
		averageDetectionNumber = averageDetectionNumber + 1/nOfFrames*(nOfDetections - averageDetectionNumber)
		positional_error, orrientational_error  = utils.pose_error()


	# If [ESC] pressed, close the application
	if cv2.waitKey(100) == 27:
		print("Application closed")
		if isTest == True:
			print("   ---   The results of the test   ---")
			print("Number of the frames is : ", nOfFrames)
			print("Average computation time is : ", averageCalculationTime, "seconds")
			print("Average number of detections is : ", averageDetectionNumber)
		break
# Close all cv2 windows
cv2.destroyAllWindows()
