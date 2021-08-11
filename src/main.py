import cv2
import numpy as np
import pyrealsense2 as rs
import utils, detectors
import time  #For calculating the time it takes to calculate

is_april = 1
isAruCo = 1
isCharuco = 0
isStag = 0

is_visualize = True

isTest = False

calib_file_name = "realsense_d415_010721_2.npz"
calib_mtx, dist_coef = utils.getCalibData(calib_file_name)

# Initialize communication with intel realsense
pipeline = rs.pipeline()
realsense_cfg = rs.config()
realsense_cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 6)
pipeline.start(realsense_cfg)

# Check communication
print("Testing the connection with the camera...")
try:
	np.asanyarray(pipeline.wait_for_frames().get_color_frame().get_data())
except:
	raise Exception("Can't get rgb frame from camera")

print("Connection with the camera is succesful!")
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
	
	" --- AprilTag --- "
	if is_april == True:
		img_rgb, april_list_tag = detectors.april_detector(img_rgb, img_gray,
												 calib_mtx, dist_coef, visualize = is_visualize, cube_color = (255,0,0))
		
		#img_rgb = utils.draw_cube_list(img_rgb, april_list_tag, calib_mtx, dist_coef)
	" --- ArUco --- "
	if isAruCo == True:
		img_rgb, aruco_list_tag = detectors.aruco_detector(img_rgb, img_gray,
										calib_mtx, dist_coef, visualize=True, cube_color = (0,0,0))

	" --- CharUco --- "
	if isCharuco == True:
		img_rgb = detectors.charuco_detector(img_rgb, img_gray,
								calib_mtx, dist_coef, visualize=True, cube_color = (0,0,0))

	" --- STag --- "
	if isStag == True:
		img_rgb, aruco_list_tag, aruco_n_detections = detectors.stag_detector(img_rgb, img_gray,
										calib_mtx, dist_coef, visualize=True, cube_color = (255,255,255))


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
