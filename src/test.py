import time
import cv2  #For calculating the time it takes to calculate
import numpy as np
import utils

class Test:
    def __init__(self, is_time = False, is_memory = False, is_jitter = False, is_accuracy = False, tag_size = 1, padding_ratio = 0.5, is_n_of_detections = False) -> None:
        
        self.n_of_frames = 0

        self.is_time, self.is_memory , self.is_jitter , self.is_accuracy , self.is_n_of_detections = False, False, False, False, False
        
        if is_time:
            self.is_time = True
            self.av_time, self.cur_time = 0, 0
            self.duration_list = []

        if is_memory:
            self.is_memory = True
            self.av_memory_consumption = 0

        if is_jitter:
            self.is_jitter = True

            self.prev_ids, self.prev_rvecs, self.prev_tvecs = None, None, None
            self.av_position_jitter, self.av_orientation_jitter = 0, 0
            self.position_jitter_list, self.orientation_jitter_list = [], []

        if is_accuracy:
            self.is_accuracy = True
            self.av_position_accuracy, self.av_orientation_accuracy = 0, 0
            self.position_accuracy_list, self.orientation_accuracy = [], []

            array_size = list(map(int, input('Please enter the number of tags (Longitudinal, Lateral) :').split(",")))
            padding = tag_size * padding_ratio

            self.map = np.zeros((array_size[0] * array_size[1], 3)) # tvecs wrt Tag 1 (Top left)

            for j in range(array_size[1]):
                for i in range(array_size[0]):
                    self.map[j * array_size[0] + i,0] = i * (tag_size + padding)
                    self.map[j * array_size[0] + i,1] = - j * (tag_size + padding)
                    pass
                    

        if is_n_of_detections:
            self.is_n_of_detections = True
            self.av_n_of_detections = 0
            self.av_n_of_detections_list = []

    def start(self):

        if self.is_time == True:
            self.cur_time = time.perf_counter()

    def stop(self, detections):
        
        self.n_of_frames += 1 

        if self.is_time :
            time_elapsed = (time.perf_counter() - self.cur_time)
            self.duration_list.append(time_elapsed)
            self.av_time = utils.moving_average(self.av_time, time_elapsed, self.n_of_frames)

        if self.is_n_of_detections :
            n_of_detection = 0

            for detection in detections:
                if detection['ids'] is not None :
                    n_of_detection = n_of_detection + len(detection['ids'])

            self.av_n_of_detections = self.av_n_of_detections + 1/self.n_of_frames*(n_of_detection - self.av_n_of_detections)

        ids, rvecs, tvecs = None, None, None
        
        if self.is_jitter:
            #TODO : make ids, rvecs, tvecs dict such that it wil work with multiple detectors
            
            for detection in detections:
                ids, rvecs, tvecs = detection['ids'], detection['rvecs'], detection['tvecs']

            if self.prev_ids is not None and ids is not None:

                id_dist = utils.distance_matrix(ids.reshape(-1, 1), self.prev_ids.reshape(-1, 1))
                indices = (np.argwhere(id_dist == 0)).T

                cur_id_match_indices = indices[0]
                prev_id_match_indices = indices[1]
                
                positional_jitter = utils.elementwise_distance(self.prev_tvecs[prev_id_match_indices], tvecs[cur_id_match_indices])
                cur_av_positional_jitter = np.mean(positional_jitter)  # Average of all the tags
                self.av_position_jitter = utils.moving_average(self.av_position_jitter, cur_av_positional_jitter, self.n_of_frames)
                self.position_jitter_list.append(cur_av_positional_jitter)

                orientation_jitter = utils.angle_error_rowwise(self.prev_rvecs[prev_id_match_indices], rvecs[cur_id_match_indices])
                cur_av_orientational_jitter = np.mean(orientation_jitter)
                self.av_orientation_jitter = utils.moving_average(self.av_orientation_jitter, cur_av_orientational_jitter, self.n_of_frames)
                self.orientation_jitter_list.append(cur_av_orientational_jitter)

            self.prev_ids, self.prev_rvecs, self.prev_tvecs  = ids, rvecs, tvecs  

        if self.is_accuracy:

            if (ids is None) or (rvecs is None) or (tvecs is None):
                for detection in detections:
                    ids, rvecs, tvecs = detection['ids'], detection['rvecs'], detection['tvecs']

            mean_rvec = np.mean(rvecs, axis=0)
            R, _ = cv2.Rodrigues(mean_rvec)
            tvecs_origin_id0 = tvecs - tvecs[np.argwhere(ids == 0).squeeze()]
            tvecs_wrt_B = tvecs_origin_id0 @ R

            id_dist = utils.distance_matrix(ids.reshape(-1, 1), np.arange(self.map.shape[0]).reshape(-1, 1))
            indices = (np.argwhere(id_dist == 0)).T
            cur_id_match_indices = indices[0]
            map_id_match_indices = indices[1]

            bias = np.mean(tvecs_wrt_B[cur_id_match_indices]-self.map[map_id_match_indices], axis=0)
            distances = utils.elementwise_distance(tvecs_wrt_B[cur_id_match_indices]-bias, self.map[map_id_match_indices])
            cur_av_positional_accuracy = np.mean(distances)
            self.av_position_accuracy = utils.moving_average(self.av_position_accuracy, cur_av_positional_accuracy, self.n_of_frames)
            self.position_accuracy_list.append(cur_av_positional_accuracy)

            orientation_accuracy = utils.angle_error_rowwise(rvecs, np.ones((rvecs.shape[0], 1)) @ mean_rvec.reshape(1, -1))
            cur_av_orientation_accuracy = np.mean(orientation_accuracy)
            self.av_orientation_accuracy = utils.moving_average(self.av_orientation_accuracy, cur_av_orientation_accuracy, self.n_of_frames)
            self.orientation_accuracy.append(cur_av_orientation_accuracy)




    def finalize(self):

        print("   ---   The results of the test   ---")

        if self.is_time:
            print('Average proccesing time :', self.av_time,'seconds')
            av_frame_rate = 1/self.av_time
            print('Average frame rate :', av_frame_rate, 'frames/seconds')

        if self.is_n_of_detections:
            print('Average number of detection :', self.av_n_of_detections, 'detections/frame')

        if self.is_jitter:
            print('Average positional jitter :', self.av_position_jitter, 'meters/(tag*frame)')
            print('Average orientational jitter :', self.av_orientation_jitter, 'rad/(tag*frame)')

        if self.is_accuracy:
            print('Average positional accuracy :', self.av_position_accuracy, 'meters/(tag*frame)')
            print('Average orientational accuracy :', self.av_orientation_accuracy, 'rad/(tag*frame)')