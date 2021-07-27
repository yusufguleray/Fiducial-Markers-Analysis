import cv2
import numpy as np	

def drawCube(img_rgb, rvecs, tvecs, mtx, dist, cubColor=(255, 0, 0)):
	# Define the ar cube
	# Since we previously set a matrix size of 1x1 for the marker and we want the cube to be the same size, it is also defined with a size of 1x1x1
	# It is important to note that the center of the marker corresponds to the origin and we must therefore move 0.5 away from the origin 
	axis = np.float32([[-0.5, -0.5, 0], [-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0],
						[-0.5, -0.5, 1], [-0.5, 0.5, 1], [0.5, 0.5, 1],[0.5, -0.5, 1]])
	# Now we transform the cube to the marker position and project the resulting points into 2d
	imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
	imgpts = np.int32(imgpts).reshape(-1, 2)

	# Now comes the drawing. 
	# In this example, I would like to draw the cube so that the walls also get a painted
	# First create six copies of the original picture (for each side of the cube one)
	side1 = img_rgb.copy()
	side2 = img_rgb.copy()
	side3 = img_rgb.copy()
	side4 = img_rgb.copy()
	side5 = img_rgb.copy()
	side6 = img_rgb.copy()
	
	# Draw the bottom side (over the marker)
	side1 = cv2.drawContours(side1, [imgpts[:4]], -1, cubColor, -2)
	# Draw the top side (opposite of the marker)
	side2 = cv2.drawContours(side2, [imgpts[4:]], -1, cubColor, -2)
	# Draw the right side vertical to the marker
	side3 = cv2.drawContours(side3, [np.array(
		[imgpts[0], imgpts[1], imgpts[5],
			imgpts[4]])], -1, cubColor, -2)
	# Draw the left side vertical to the marker
	side4 = cv2.drawContours(side4, [np.array(
		[imgpts[2], imgpts[3], imgpts[7],
			imgpts[6]])], -1, cubColor, -2)
	# Draw the front side vertical to the marker
	side5 = cv2.drawContours(side5, [np.array(
		[imgpts[1], imgpts[2], imgpts[6],
			imgpts[5]])], -1, cubColor, -2)
	# Draw the back side vertical to the marker
	side6 = cv2.drawContours(side6, [np.array(
		[imgpts[0], imgpts[3], imgpts[7],
			imgpts[4]])], -1, cubColor, -2)
	
	# Until here the walls of the cube are drawn in and can be merged
	img_rgb = cv2.addWeighted(side1, 0.1, img_rgb, 0.9, 0)
	img_rgb = cv2.addWeighted(side2, 0.1, img_rgb, 0.9, 0)
	img_rgb = cv2.addWeighted(side3, 0.1, img_rgb, 0.9, 0)
	img_rgb = cv2.addWeighted(side4, 0.1, img_rgb, 0.9, 0)
	img_rgb = cv2.addWeighted(side5, 0.1, img_rgb, 0.9, 0)
	img_rgb = cv2.addWeighted(side6, 0.1, img_rgb, 0.9, 0)
	
	# Now the edges of the cube are drawn thicker and stronger
	img_rgb = cv2.drawContours(img_rgb, [imgpts[:4]], -1, cubColor, 2)
	for i, j in zip(range(4), range(4, 8)):
		img_rgb = cv2.line(img_rgb, tuple(imgpts[i]), tuple(imgpts[j]), cubColor, 2)
	img_rgb = cv2.drawContours(img_rgb, [imgpts[4:]], -1, cubColor, 2)

	return img_rgb

def drawCubeChar(img_rgb, rvecs, tvecs, mtx, dist, cubColor=(255, 0, 0)):
	# Define the ar cube
	# Since we previously set a matrix size of 1x1 for the marker and we want the cube to be the same size, it is also defined with a size of 1x1x1
	# It is important to note that the center of the marker corresponds to the origin and we must therefore move 0.5 away from the origin 
	axis = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
						[0, 0, 1], [0, 1, 1], [1, 1, 1],[1, 0, 1]])
	# Now we transform the cube to the marker position and project the resulting points into 2d
	imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
	imgpts = np.int32(imgpts).reshape(-1, 2)

	# Now comes the drawing. 
	# In this example, I would like to draw the cube so that the walls also get a painted
	# First create six copies of the original picture (for each side of the cube one)
	side1 = img_rgb.copy()
	side2 = img_rgb.copy()
	side3 = img_rgb.copy()
	side4 = img_rgb.copy()
	side5 = img_rgb.copy()
	side6 = img_rgb.copy()
	
	# Draw the bottom side (over the marker)
	side1 = cv2.drawContours(side1, [imgpts[:4]], -1, cubColor, -2)
	# Draw the top side (opposite of the marker)
	side2 = cv2.drawContours(side2, [imgpts[4:]], -1, cubColor, -2)
	# Draw the right side vertical to the marker
	side3 = cv2.drawContours(side3, [np.array(
		[imgpts[0], imgpts[1], imgpts[5],
			imgpts[4]])], -1, cubColor, -2)
	# Draw the left side vertical to the marker
	side4 = cv2.drawContours(side4, [np.array(
		[imgpts[2], imgpts[3], imgpts[7],
			imgpts[6]])], -1, cubColor, -2)
	# Draw the front side vertical to the marker
	side5 = cv2.drawContours(side5, [np.array(
		[imgpts[1], imgpts[2], imgpts[6],
			imgpts[5]])], -1, cubColor, -2)
	# Draw the back side vertical to the marker
	side6 = cv2.drawContours(side6, [np.array(
		[imgpts[0], imgpts[3], imgpts[7],
			imgpts[4]])], -1, cubColor, -2)
	
	# Until here the walls of the cube are drawn in and can be merged
	img_rgb = cv2.addWeighted(side1, 0.1, img_rgb, 0.9, 0)
	img_rgb = cv2.addWeighted(side2, 0.1, img_rgb, 0.9, 0)
	img_rgb = cv2.addWeighted(side3, 0.1, img_rgb, 0.9, 0)
	img_rgb = cv2.addWeighted(side4, 0.1, img_rgb, 0.9, 0)
	img_rgb = cv2.addWeighted(side5, 0.1, img_rgb, 0.9, 0)
	img_rgb = cv2.addWeighted(side6, 0.1, img_rgb, 0.9, 0)
	
	# Now the edges of the cube are drawn thicker and stronger
	img_rgb = cv2.drawContours(img_rgb, [imgpts[:4]], -1, cubColor, 2)
	for i, j in zip(range(4), range(4, 8)):
		img_rgb = cv2.line(img_rgb, tuple(imgpts[i]), tuple(imgpts[j]), cubColor, 2)
	img_rgb = cv2.drawContours(img_rgb, [imgpts[4:]], -1, cubColor, 2)

	return img_rgb

def calibrator():
	import cv2
	import cv2.aruco as aruco
	import numpy as np
	import pyrealsense2 as rs

	# Defines the path to save the calibration file and the dictonary used
	name = "realsense_d415_010721_2.npz"
	dictionary = aruco.DICT_6X6_250

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

	# Define what the calibration board looks like (same as the pdf)
	board = cv2.aruco.CharucoBoard_create(4,4, .045, .0225, aruco.Dictionary_get(dictionary))
	record_count = 0
	# Create two arrays to store the recorded corners and ids
	all_corners = []
	all_ids = []

	print("Start recording [1/4]")
	print("1. Move the grid from calibration directory a little bit in front of the camera and press [r] to make a record (if auto record is not set to True)")
	print("2. Finish this task and start calculation press [c]")
	print("3. Interrupt application [ESC]")
	while True:
		# Get frame from realsense and convert to grayscale image
		frames = pipeline.wait_for_frames()
		img_rgb = np.asanyarray(frames.get_color_frame().get_data())
		img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
		
		# Detect markers on the gray image
		res = aruco.detectMarkers(img_gray, aruco.getPredefinedDictionary(aruco.DICT_6X6_250))
		# Draw the detected markers
		aruco.drawDetectedMarkers(img_rgb, res[0], res[1])
		# Display the result
		cv2.imshow("AR-Example", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
		
		key = cv2.waitKey(10)
		# If eight markers have been found, allow recording
		if len(res[0]) == 8:
			# Interpolate the corners of the markers
			res2 = aruco.interpolateCornersCharuco(res[0], res[1], img_gray, board)
			# Add the detected interpolated corners and the marker ids to the arrays if the user press [r] and the interpolation is valid
			if key == ord('r') and res2[1] is not None and res2[2] is not None and len(res2[1]) > 8:
				all_corners.append(res2[1])
				all_ids.append(res2[2])
				record_count += 1
				print("Record: " + str(record_count))
		# If [c] pressed, start the calculation
		if key == ord('c'):
			# Close all cv2 windows
			cv2.destroyAllWindows()
			# Check if recordings have been made
			if(record_count != 0):
				print("Calculate calibration [2/4] --> Use "+str(record_count)+" records"),
				# Calculate the camera calibration
				ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(all_corners, all_ids, board, img_gray.shape, None, None)
				print("Save calibration [3/4]")
				# Save the calibration information into a file
				np.savez_compressed(name, ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
				print("Done [4/4]")
			else:
				print("Interrupted since there are no records...")
			break
		# If [ESC] pressed, close the application
		if key == 27:
			print("Application closed without calculation")
			# Close all cv2 windows
			cv2.destroyAllWindows()
			break

def getCalibPath(filename):
	import os

	# Defines the path of the calibration file and the dictonary used
	return os.path.join(os.path.split(os.path.dirname(__file__))[0], 'calibration','calibfiles', filename)


def displayCalibMat(filename, showmtx= True, showDist=True):
	import os

	# Defines the path of the calibration file and the dictonary used
	calibration_path = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'calibration','calibfiles', filename)

	# Load calibration from file
	mtx = None
	dist = None
	with np.load(calibration_path) as X:
		mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

	if showmtx==True:print('Calibration Matrix : \n' , mtx,'\n')
	if showDist==True:print('Distortion Parameters : \n' ,dist,'\n')

def getCalibData(filename):
	import os

	# Defines the path of the calibration file and the dictonary used
	calibration_path = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'calibration','calibfiles', filename)

	# Load calibration from file
	mtx = None
	dist = None
	with np.load(calibration_path) as X:
		mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

	return mtx, dist

def angle_error(v1, v2):
	# v1 is your firsr vector
	# v2 is your second vector
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# def pose_error(prevPose, curPose):