import cv2
import numpy as np
import pyrealsense2 as rs
import utils, detectors, test

is_april = 0     # Visualization color : RED 
isAruCo = 0      # Visualization color : BLACK
isCharuco = 0	 # Visualization color : WHITE
isStag = 1       # Visualization color : BLUE
is_topo = 0      # Visualization color : TURQUOISE

is_visualize = True
tag_size = 0.08 # in meters

tester = test.Test(is_time = True, is_memory = True, is_jitter = True, is_accuracy = True, tag_size = tag_size, is_n_of_detections = True)

calib_file_name = "D41517082021_192037.npz"
calib_mtx, dist_coef = utils.getCalibData(calib_file_name)

# Initialize communication with intel realsense
pipeline = rs.pipeline()
realsense_cfg = rs.config()
realsense_cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)
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
	if is_april:
		img_rgb, april_detections = detectors.april_detector(img_rgb, img_gray,
												 calib_mtx, dist_coef, tag_size = tag_size, visualize = is_visualize, cube_color = (255,0,0))
		if april_detections is not None : detection_list.append(april_detections)

	" --- ArUco --- "
	if isAruCo:
		img_rgb, aruco_detections = detectors.aruco_detector(img_rgb, img_gray,
										calib_mtx, dist_coef, tag_size = tag_size, visualize=True, cube_color = (0,0,0))
		if aruco_detections is not None : detection_list.append(aruco_detections)

	" --- CharUco --- "
	if isCharuco:
		img_rgb, charuco_detections = detectors.charuco_detector(img_rgb, img_gray,
								calib_mtx, dist_coef,  tag_size = tag_size, visualize=True, cube_color = (255,255,255), use_april_detecotor = False)
		if charuco_detections is not None : detection_list.append(charuco_detections)

	" --- STag --- "
	if isStag:
		img_rgb, stag_detections = detectors.stag_detector(img_rgb, img_gray,
										calib_mtx, dist_coef, tag_size = tag_size, visualize=True, cube_color = (0,0,255), take_mean = True)
		if stag_detections is not None : detection_list.append(stag_detections)

	if is_topo:
		img_rgb, topo_detections = detectors.topo_detector(img_rgb, img_gray, calib_mtx, dist_coef, tag_size=tag_size, visualize=True)

	tester.stop(detection_list)
	# Display the result
	display_image = cv2.resize(img_rgb, (960, 540))
	cv2.imshow("AR-Example", cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR))
	

	# If [ESC] pressed, close the application
	if cv2.waitKey(100) == 27:
		print("Application closed")
		break
# Close all cv2 windows
cv2.destroyAllWindows()
tester.finalize()
