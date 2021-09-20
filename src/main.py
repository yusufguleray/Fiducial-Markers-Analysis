# Resolution : 1920 x 1080

import cv2
import utils, detectors, test

is_april = 0     # Visualization color : RED 
isAruCo = 1      # Visualization color : BLACK
isCharuco = 0	 # Visualization color : WHITE
isStag = 0       # Visualization color : BLUE
is_topo = 0      # Visualization color : TURQUOISE   dont forget to change the size in yml 

is_visualize = True
tag_size = 0.005 # in meters

#tester = test.Test(is_time = True, is_memory = True, is_jitter = True, is_accuracy = True, tag_size = tag_size, is_n_of_detections = True)  #Linux
tester = test.Test(tag_size = tag_size, is_memory = False)  #Windows

calib_file_name = "d415_200921_matlab.npz"
calib_mtx, dist_coef = utils.getCalibData(calib_file_name)

get_image = utils.GetImages(is_camera=False, dataset_name="aruco_05cm")

print("Press [ESC] to close the application")

"--- Data for testing ---"

is_stop = False
while is_stop == False:

	# Get frame from realsense and convert to grayscale image
	img_rgb = get_image.get_image()
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

	" --- TopoTag --- "
	if is_topo:
		img_rgb, topo_detections = detectors.topo_detector(img_rgb, img_gray, calib_mtx, dist_coef, tag_size=tag_size, visualize=True)
		if topo_detections is not None : detection_list.append(topo_detections)

	is_stop = tester.stop(detection_list)
	# Display the result
	display_image = cv2.resize(img_rgb, (960, 540))
	cv2.imshow("Fiducial Markers", cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR))
	

	# If [ESC] pressed, close the application
	if cv2.waitKey(100) == 27:
		print("Application closed")
		break
# Close all cv2 windows
cv2.destroyAllWindows()
tester.finalize()
