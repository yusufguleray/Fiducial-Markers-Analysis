import cv2
import numpy as np
import pyrealsense2 as rs
import utils, detectors, test

is_april = 1     # Visualization color : RED 
isAruCo = 0      # Visualization color : BLACK
isCharuco = 0	 # Visualization color : WHITE
isStag = 0       # Visualization color : BLUE

is_visualize = True
tag_size = 0.04 # in meters

tester = test.Test(is_time = True, is_n_of_detections = True, is_jitter = True, tag_size=tag_size, is_accuracy = True)

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


while True:

	# Get frame from realsense and convert to grayscale image
	frames = pipeline.wait_for_frames()
	img_rgb = np.asanyarray(frames.get_color_frame().get_data())
	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

	tester.start()
	detection_list = []
	
	" --- AprilTag --- "
	if is_april == True:
		img_rgb, april_detections = detectors.april_detector(img_rgb, img_gray,
												 calib_mtx, dist_coef, tag_size = tag_size, visualize = is_visualize, cube_color = (255,0,0))
		if april_detections is not None : detection_list.append(april_detections)

	" --- ArUco --- "
	if isAruCo == True:
		img_rgb, aruco_detections = detectors.aruco_detector(img_rgb, img_gray,
										calib_mtx, dist_coef, tag_size = tag_size, visualize=True, cube_color = (0,0,0))
		if aruco_detections is not None : detection_list.append(aruco_detections)

	" --- CharUco --- "
	if isCharuco == True:
		img_rgb, charuco_detections = detectors.charuco_detector(img_rgb, img_gray,
								calib_mtx, dist_coef,  tag_size = tag_size, visualize=True, cube_color = (255,255,255), use_april_detecotor = False)
		if charuco_detections is not None : detection_list.append(charuco_detections)

	" --- STag --- "
	if isStag == True:
		img_rgb, stag_detections = detectors.stag_detector(img_rgb, img_gray,
										calib_mtx, dist_coef, tag_size = tag_size, visualize=True, cube_color = (0,0,255), take_mean = True)
		if stag_detections is not None : detection_list.append(stag_detections)

	tester.stop(detection_list)
	# Display the result
	cv2.imshow("AR-Example", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
	

	# If [ESC] pressed, close the application
	if cv2.waitKey(100) == 27:
		print("Application closed")
		tester.finalize()
		break
# Close all cv2 windows
cv2.destroyAllWindows()
