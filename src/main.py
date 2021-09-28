import cv2, platform
import utils, detectors, test


def fiducial_markers(detector = "apriltag", tag_size = 0.16, is_visualize = True, is_record = True, is_test = True, use_camera = False,
					dataset_name = 'apriltag_16cm', calib_file_name = "d415_200921_matlab.npz"):
	
	# Resolution : 1920 x 1080

	is_april, is_aruco, is_charuco, is_stag, is_topo = False, False, False, False, False 

	if detector == "apriltag" : is_april = 1 # Visualization color : RED 
	if detector == "aruco" : is_aruco = 1 # Visualization color : BLACK
	if detector == "charuco" : is_charuco = 1 # Visualization color : WHITE
	if detector == "stag" : is_stag = 1 # Visualization color : BLUE
	if detector == "topotag": is_topo = 1 # Visualization color : TURQUOISE  !!! dont forget to change the size in yml 

	if tag_size == 0.16 : is_accuracy = False 
	else: is_accuracy = True

	if is_test:
		if platform.system() == 'Linux': tester = test.Test(tag_size = tag_size, is_record = is_record, record_name=dataset_name, is_accuracy=is_accuracy)  #Linux
		else: tester = test.Test(tag_size = tag_size, is_memory = False, is_record=is_record,record_name=dataset_name, is_accuracy=is_accuracy)  #Windows

	calib_mtx, dist_coef = utils.getCalibData(calib_file_name)

	get_image = utils.GetImages(is_camera=use_camera, dataset_name=dataset_name)

	print("Press [ESC] to close the application")

	"--- Data for testing ---"

	is_stop = False
	while is_stop == False:

		# Get frame from realsense and convert to grayscale image
		img_rgb = get_image.get_image()
		img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

		if is_test: tester.start()
		detection_list = []
		
		" --- AprilTag --- "
		if is_april:
			img_rgb, april_detections = detectors.april_detector(img_rgb, img_gray,
													calib_mtx, dist_coef, tag_size = tag_size, visualize = is_visualize, cube_color = (255,0,0))
			if april_detections is not None : detection_list.append(april_detections)

		" --- ArUco --- "
		if is_aruco:
			img_rgb, aruco_detections = detectors.aruco_detector(img_rgb, img_gray,
											calib_mtx, dist_coef, tag_size = tag_size, visualize=True, cube_color = (0,0,0))
			if aruco_detections is not None : detection_list.append(aruco_detections)

		" --- CharUco --- "
		if is_charuco:
			img_rgb, charuco_detections = detectors.charuco_detector(img_rgb, img_gray,
									calib_mtx, dist_coef,  tag_size = tag_size, visualize=True, cube_color = (255,255,255), use_april_detecotor = True)
			if charuco_detections is not None : detection_list.append(charuco_detections)

		" --- STag --- "
		if is_stag:
			img_rgb, stag_detections = detectors.stag_detector(img_rgb, img_gray,
											calib_mtx, dist_coef, tag_size = tag_size, visualize=True, cube_color = (0,0,255), take_mean = True)
			if stag_detections is not None : detection_list.append(stag_detections)

		" --- TopoTag --- "
		if is_topo:
			img_rgb, topo_detections = detectors.topo_detector(img_rgb, img_gray, calib_mtx, dist_coef, tag_size=tag_size, visualize=True)
			if topo_detections is not None : detection_list.append(topo_detections)

		
		if is_test: is_stop = tester.stop(detection_list)
		
		# Display the result
		display_image = cv2.resize(img_rgb, (960, 540))
		cv2.imshow("Fiducial Markers", cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR))

		# If [ESC] pressed, close the application
		if cv2.waitKey(100) == 27:
			print("Application closed")
			break

	# Close all cv2 windows
	cv2.destroyAllWindows()
	if is_test: tester.finalize()

if __name__ == "__main__":
	# fiducial_markers(detector = "charuco", tag_size = 0.16, is_visualize = True, is_test = True, use_camera = False,dataset_name = 'charuco_16cm', calib_file_name = "d415_200921_matlab.npz")
	# fiducial_markers(detector = "charuco", tag_size = 0.08, is_visualize = True, is_test = True, use_camera = False,dataset_name = 'charuco_8cm', calib_file_name = "d415_200921_matlab.npz")
	# fiducial_markers(detector = "charuco", tag_size = 0.04, is_visualize = True, is_test = True, use_camera = False,dataset_name = 'charuco_4cm', calib_file_name = "d415_200921_matlab.npz")
	# fiducial_markers(detector = "charuco", tag_size = 0.02, is_visualize = True, is_test = True, use_camera = False,dataset_name = 'charuco_2cm', calib_file_name = "d415_200921_matlab.npz")
	# fiducial_markers(detector = "charuco", tag_size = 0.01, is_visualize = True, is_test = True, use_camera = False,dataset_name = 'charuco_1cm', calib_file_name = "d415_200921_matlab.npz")
	# fiducial_markers(detector = "charuco", tag_size = 0.005, is_visualize = True, is_test = True, use_camera = False,dataset_name = 'charuco_05cm', calib_file_name = "d415_200921_matlab.npz")
	
	fiducial_markers(detector = "aruco", tag_size = 0.16, is_visualize = True, is_test = True, use_camera = False,dataset_name = 'aruco_16cm', calib_file_name = "d415_200921_matlab.npz")
	fiducial_markers(detector = "aruco", tag_size = 0.08, is_visualize = True, is_test = True, use_camera = False,dataset_name = 'aruco_8cm', calib_file_name = "d415_200921_matlab.npz")
	fiducial_markers(detector = "aruco", tag_size = 0.04, is_visualize = True, is_test = True, use_camera = False,dataset_name = 'aruco_4cm', calib_file_name = "d415_200921_matlab.npz")
	fiducial_markers(detector = "aruco", tag_size = 0.02, is_visualize = True, is_test = True, use_camera = False,dataset_name = 'aruco_2cm', calib_file_name = "d415_200921_matlab.npz")
	fiducial_markers(detector = "aruco", tag_size = 0.01, is_visualize = True, is_test = True, use_camera = False,dataset_name = 'aruco_1cm', calib_file_name = "d415_200921_matlab.npz")
	fiducial_markers(detector = "aruco", tag_size = 0.005, is_visualize = True, is_test = True, use_camera = False,dataset_name = 'aruco_05cm', calib_file_name = "d415_200921_matlab.npz")

	fiducial_markers(detector = "stag", tag_size = 0.16, is_visualize = True, is_test = True, use_camera = False,dataset_name = 'stag_16cm', calib_file_name = "d415_200921_matlab.npz")
	fiducial_markers(detector = "stag", tag_size = 0.08, is_visualize = True, is_test = True, use_camera = False,dataset_name = 'stag_8cm', calib_file_name = "d415_200921_matlab.npz")
	fiducial_markers(detector = "stag", tag_size = 0.04, is_visualize = True, is_test = True, use_camera = False,dataset_name = 'stag_4cm', calib_file_name = "d415_200921_matlab.npz")
	fiducial_markers(detector = "stag", tag_size = 0.02, is_visualize = True, is_test = True, use_camera = False,dataset_name = 'stag_2cm', calib_file_name = "d415_200921_matlab.npz")
	fiducial_markers(detector = "stag", tag_size = 0.01, is_visualize = True, is_test = True, use_camera = False,dataset_name = 'stag_1cm', calib_file_name = "d415_200921_matlab.npz")
	fiducial_markers(detector = "stag", tag_size = 0.005, is_visualize = True, is_test = True, use_camera = False,dataset_name = 'stag_05cm', calib_file_name = "d415_200921_matlab.npz")

	fiducial_markers(detector = "apriltag", tag_size = 0.16, is_visualize = True, is_test = True, use_camera = False,dataset_name = 'apriltag_16cm', calib_file_name = "d415_200921_matlab.npz")
	fiducial_markers(detector = "apriltag", tag_size = 0.08, is_visualize = True, is_test = True, use_camera = False,dataset_name = 'apriltag_8cm', calib_file_name = "d415_200921_matlab.npz")
	fiducial_markers(detector = "apriltag", tag_size = 0.04, is_visualize = True, is_test = True, use_camera = False,dataset_name = 'apriltag_4cm', calib_file_name = "d415_200921_matlab.npz")
	fiducial_markers(detector = "apriltag", tag_size = 0.02, is_visualize = True, is_test = True, use_camera = False,dataset_name = 'apriltag_2cm', calib_file_name = "d415_200921_matlab.npz")
	fiducial_markers(detector = "apriltag", tag_size = 0.01, is_visualize = True, is_test = True, use_camera = False,dataset_name = 'apriltag_1cm', calib_file_name = "d415_200921_matlab.npz")
	fiducial_markers(detector = "apriltag", tag_size = 0.005, is_visualize = True, is_test = True, use_camera = False,dataset_name = 'apriltag_05cm', calib_file_name = "d415_200921_matlab.npz")