import cv2, os
import numpy as np
from numpy.lib.function_base import unwrap	

def drawCube(img_rgb, rvecs, tvecs, mtx, dist, cube_color = (255, 0, 0), tag_size = 1 ,is_centered = True):
	# Define the ar cube

	if is_centered == True:
		# It is important to note that the center of the marker corresponds to the origin and we must therefore move 0.5 away from the origin 
		axis = np.float32([[-0.5, -0.5, 0], [-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0],
							[-0.5, -0.5, 1], [-0.5, 0.5, 1], [0.5, 0.5, 1],[0.5, -0.5, 1]])
	else:
		axis = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
						[0, 0, 1], [0, 1, 1], [1, 1, 1],[1, 0, 1]])

	axis = axis * tag_size # scale the axis to match the tag size	

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
	side1 = cv2.drawContours(side1, [imgpts[:4]], -1, cube_color, -2)
	# Draw the top side (opposite of the marker)
	side2 = cv2.drawContours(side2, [imgpts[4:]], -1, cube_color, -2)
	# Draw the right side vertical to the marker
	side3 = cv2.drawContours(side3, [np.array(
		[imgpts[0], imgpts[1], imgpts[5],
			imgpts[4]])], -1, cube_color, -2)
	# Draw the left side vertical to the marker
	side4 = cv2.drawContours(side4, [np.array(
		[imgpts[2], imgpts[3], imgpts[7],
			imgpts[6]])], -1, cube_color, -2)
	# Draw the front side vertical to the marker
	side5 = cv2.drawContours(side5, [np.array(
		[imgpts[1], imgpts[2], imgpts[6],
			imgpts[5]])], -1, cube_color, -2)
	# Draw the back side vertical to the marker
	side6 = cv2.drawContours(side6, [np.array(
		[imgpts[0], imgpts[3], imgpts[7],
			imgpts[4]])], -1, cube_color, -2)
	
	# Until here the walls of the cube are drawn in and can be merged
	img_rgb = cv2.addWeighted(side1, 0.1, img_rgb, 0.9, 0)
	img_rgb = cv2.addWeighted(side2, 0.1, img_rgb, 0.9, 0)
	img_rgb = cv2.addWeighted(side3, 0.1, img_rgb, 0.9, 0)
	img_rgb = cv2.addWeighted(side4, 0.1, img_rgb, 0.9, 0)
	img_rgb = cv2.addWeighted(side5, 0.1, img_rgb, 0.9, 0)
	img_rgb = cv2.addWeighted(side6, 0.1, img_rgb, 0.9, 0)
	
	# Now the edges of the cube are drawn thicker and stronger
	img_rgb = cv2.drawContours(img_rgb, [imgpts[:4]], -1, cube_color, 2)
	for i, j in zip(range(4), range(4, 8)):
		img_rgb = cv2.line(img_rgb, tuple(imgpts[i]), tuple(imgpts[j]), cube_color, 2)
	img_rgb = cv2.drawContours(img_rgb, [imgpts[4:]], -1, cube_color, 2)

	return img_rgb

def draw_cube_list(img_rgb, detection_list, calib_mtx, dist_coef, tag_size =1, cube_color = (125,125,125), is_centered = True):
	
	for detection in detection_list:
		img_rgb = drawCube(img_rgb, detection['rvec'], detection['tvec'], calib_mtx, dist_coef, cube_color, tag_size = tag_size, is_centered = True)
	
	return img_rgb

def calibrator():
	import cv2
	import numpy as np
	import pyrealsense2 as rs

	# Defines the path to save the calibration file and the dictonary used
	device_name = input("Please enter the device name :")
	name = device_name + "_" + get_time() + ".npz"
	dictionary = cv2.aruco.DICT_APRILTAG_36h10

	# Initialize communication with intel realsense
	pipeline = rs.pipeline()
	realsense_cfg = rs.config()
	realsense_cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)
	pipeline.start(realsense_cfg)

	# Check communication
	print("Test data source...")
	try:
		np.asanyarray(pipeline.wait_for_frames().get_color_frame().get_data())
	except:
		raise Exception("Can't get rgb frame from data source")

	# Define what the calibration board looks like (same as the pdf)
	board = cv2.aruco.CharucoBoard_create(6, 6, 0.16/6, 0.16/6*0.8, cv2.aruco.Dictionary_get(dictionary))
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
		res = cv2.aruco.detectMarkers(img_gray, cv2.aruco.getPredefinedDictionary(dictionary))
		# Draw the detected markers
		cv2.aruco.drawDetectedMarkers(img_rgb, res[0], res[1])
		# Display the result
		display_image = cv2.resize(img_rgb, (960, 540))
		cv2.imshow("AR-Example", cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR))
		
		key = cv2.waitKey(10)
		# If 18 markers have been found, allow recording
		if len(res[0]) == 18:
			# Interpolate the corners of the markers
			res2 = cv2.aruco.interpolateCornersCharuco(res[0], res[1], img_gray, board)
			# Add the detected interpolated corners and the marker ids to the arrays if the user press [r] and the interpolation is valid
			if key == ord('r') and res2[1] is not None and res2[2] is not None and len(res2[1]) > 18:
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
				ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_corners, all_ids, board, img_gray.shape, None, None)
				print("Save calibration [3/4]")
				# Save the calibration information into a file
				np.savez_compressed(getCalibPath(name), ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
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
	# Defines the path of the calibration file and the dictonary used
	return os.path.join(os.path.split(os.path.dirname(__file__))[0], 'calibration','calibfiles', filename)

def get_test_path(filename):
	# Defines the path of the calibration file and the dictonary used
	return os.path.join(os.path.split(os.path.dirname(__file__))[0], 'testresults', filename)


def displayCalibMat(filename, showmtx= True, showDist=True):
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
	# Defines the path of the calibration file and the dictonary used
	calibration_path = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'calibration','calibfiles', filename)

	# Load calibration from file
	mtx = None
	dist = None
	with np.load(calibration_path) as X:
		mtx, dist = [X[i] for i in ('mtx', 'dist')]

	print("\nThe file used for the camera calibration :", calibration_path)

	return mtx, dist

def angle_error(v1, v2):
	# v1 is your firsr vector
	# v2 is your second vector
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def angle_error_rowwise(A, B):
	A, B = unwrap(A), unwrap(B)
	p1 = np.einsum('ij,ij->i',A,B)
	p2 = np.einsum('ij,ij->i',A,A)
	p3 = np.einsum('ij,ij->i',B,B)
	p4 = p1 / np.sqrt(p2*p3)
	return np.arccos(np.clip(p4,-1.0,1.0))


def distance_matrix(A, B, squared=False):
    """
    Compute all pairwise distances between vectors in A and B.

    Parameters
    ----------
    A : np.array
        shape should be (M, K)
    B : np.array
        shape should be (N, K)

    Returns
    -------
    D : np.array
        A matrix D of shape (M, N).  Each entry in D i,j represnets the
        distance between row i in A and row j in B.

    See also
    --------
    A more generalized version of the distance matrix is available from
    scipy (https://www.scipy.org) using scipy.spatial.distance_matrix,
    which also gives a choice for p-norm.
    """
    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1], f"The number of components for vectors in A \
        {A.shape[1]} does not match that of B {B.shape[1]}!"

    A_dots = (A*A).sum(axis=1).reshape((M,1))*np.ones(shape=(1,N))
    B_dots = (B*B).sum(axis=1)*np.ones(shape=(M,1))
    D_squared =  A_dots + B_dots -2*A.dot(B.T)

    if squared == False:
        zero_mask = np.less(D_squared, 0.0)
        D_squared[zero_mask] = 0.0
        return np.sqrt(D_squared)

    return D_squared

def elementwise_distance(A, B, squared = False):
	"""
	Compute all elementwise distances between vectors in A and B.

	Parameters
	----------
	A : np.array
		shape should be (M, K)
	B : np.array
		shape should be (M, K)

	Returns
	-------
	D : np.array
		A matrix D of shape (M, 1).  Each entry in D i represnets the
		distance between row i in A and row i in B.

	"""

	M = A.shape[0]
	N = B.shape[0]

	assert A.shape[0] == B.shape[0], f"The number of components for vectors in A \
		{A.shape[1]} does not match that of B {B.shape[1]}!"

	try:
		D_squared = np.sum(np.square(A - B), axis = 1)
	except:
		D_squared = np.sum(np.square(A - B), axis = 0)
	
	if squared == False:
		return np.sqrt(D_squared)

	return D_squared

def moving_average(old_mean, new_data, n_of_frames):
	
	return old_mean + 1 / n_of_frames * (new_data - old_mean)

def get_time():
	import time
	return time.strftime("%d%m%Y_%H%M%S")

def get_process_memory():
	import psutil, os

	process = psutil.Process(os.getpid())
	mem_info = process.memory_info()   
	return mem_info.rss + mem_info.vms + mem_info.shared  # in bytes

def user_prompt(question: str) -> bool:
    """ Prompt the yes/no-*question* to the user. """
    from distutils.util import strtobool

    while True:
        user_input = input(question + " [y/n]: ")
        try:
            return bool(strtobool(user_input))
        except ValueError:
            print("Please use y/n or yes/no.\n")

class GetImages():
	def __init__(self, is_camera = True, dataset_name = None):
		if is_camera:
			import pyrealsense2 as rs

			self.is_camera = True

			# Initialize communication with intel realsense
			self.pipeline = rs.pipeline()
			realsense_cfg = rs.config()
			realsense_cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)
			self.pipeline.start(realsense_cfg)

			# Check communication
			print("Testing the connection with the camera...")
			try:
				np.asanyarray(self.pipeline.wait_for_frames().get_color_frame().get_data())
			except:
				raise Exception("Can't get rgb frame from camera")

			print("Connection with the camera is succesful!")

		else:
			self.is_camera = False   #use dataset
			self.dataset_folder_path = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'datasets', dataset_name)
			file_list = os.listdir(self.dataset_folder_path)
			print("Path of the dataset to be used :", self.dataset_folder_path)
			self.filtered_file_list = [k for k in file_list if ".png" in k]
			self.i = 0

	def get_image(self):
		if self.is_camera:
			frames = self.pipeline.wait_for_frames()
			img_rgb = np.asanyarray(frames.get_color_frame().get_data())

			return img_rgb
		
		else:
			if self.i >= len(self.filtered_file_list):
				raise Exception("No more image left in the dataset!")
			else:
				img_rgb = cv2.imread(os.path.join(self.dataset_folder_path, self.filtered_file_list[self.i]))
				self.i += 1
				return img_rgb 

def record_dataset(folder_name, n_of_image = 100, file_name = None):
	import pathlib

	if file_name == None : file_name = folder_name
	file_id_padding = 5

	dataset_folder_path = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'datasets', folder_name)
	pathlib.Path(dataset_folder_path).mkdir(parents=True, exist_ok=True)

	get_image = GetImages(is_camera = True)

	for i in range(n_of_image):
		img_rgb = get_image.get_image()
		cv2.imwrite(os.path.join(dataset_folder_path, file_name + "_{:>0{}}.png".format(i, file_id_padding)), img_rgb)

def display_dataset(folder_name, n):
	get_image = GetImages(False, folder_name)

	for i in range(n):
		cv2.imshow(folder_name, cv2.cvtColor(get_image.get_image(), cv2.COLOR_RGB2BGR))
		cv2.waitKey(0) 
	cv2.destroyAllWindows() 

def image_saver(folder_name = None, wait_time = 5):
	import time, pathlib
	
	if folder_name == None: folder_name = input("What should the folder name for the images to be saved ? : ")
	get_image = GetImages(is_camera=True)
		
	image_folder_path = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'calibration', 'calibration_images', folder_name)
	pathlib.Path(image_folder_path).mkdir(parents=True, exist_ok=True)
	print("Images will be saved to :", image_folder_path)
	
	img_counter = 0
	start_time = time.time()

	print("Press [ESC] to close the application")

	while True:
		img_rgb = get_image.get_image()
		display_image = cv2.resize(img_rgb, (960, 540))

		remaining_time = str(wait_time - (time.time() - start_time))[0] + ' second(s) left'
		info = str(img_counter) + ' image(s) captured'

		#--- Position the time at (10, 70) coordinate with certain font style, size and color ---
		cv2.putText(display_image, remaining_time + " | " + info , (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(15,200,15),2,cv2.LINE_AA)
		cv2.imshow("Image Saver", cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR))

		if time.time() - start_time >= wait_time: # Check if 5 wait time is passed
			img_name = "image_{:>0{}}.png".format(img_counter, 3)

			cv2.imwrite(os.path.join(image_folder_path, img_name), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
			print("{} written!".format(img_name))
			start_time = time.time()
			img_counter += 1
		
		# If [ESC] pressed, close the application
		if cv2.waitKey(100) == 27:
			print("Application closed")
			break
	cv2.destroyAllWindows()

def unwrap(rvecs):
	"""Make sures that angels are in range of -pi and pi"""
	return (rvecs + np.pi) % (2 * np.pi) - np.pi

def find_ambiguity(ids, rvecs, tvecs):
	
	if ids is None: return []

	n = ids.size
	flipped_ids = []
	
	for i in range(n):
		R = cv2.Rodrigues(rvecs[i])[0]
		
		if R[1,2] > 0: 
			flipped_ids.append(int(ids[i]))

	return flipped_ids

def boxplot_data(data):
	if data.size == 0 : return None
	data_dict = {}
	data_dict['median'] = np.median(data)
	data_dict['average'] = np.average(data)
	data_dict['upper_quartile'] = np.percentile(data, 75)
	data_dict['lower_quartile'] = np.percentile(data, 25)
	data_dict['max'] = np.max(data)
	data_dict['min'] = np.min(data)

	return data_dict