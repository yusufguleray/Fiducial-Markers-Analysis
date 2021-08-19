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
            self.cur_memory = 0
            self.memory_consumption_list = []

        if is_jitter:
            self.is_jitter = True

            self.prev_ids, self.prev_rvecs, self.prev_tvecs = None, None, None
            self.av_position_jitter, self.av_orientation_jitter = 0, 0
            self.position_jitter_list, self.orientation_jitter_list = [], []

        if is_accuracy:
            self.is_accuracy = True
            self.av_position_accuracy, self.av_orientation_accuracy = 0, 0
            self.position_accuracy_list, self.orientation_accuracy_list = [], []

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

        if self.is_time:
            self.cur_time = time.perf_counter()

        if self.is_memory:
            self.cur_memory = utils.get_process_memory()

    def stop(self, detections):
        
        self.n_of_frames += 1 

        if self.is_time :
            time_elapsed = (time.perf_counter() - self.cur_time)
            self.duration_list.append(time_elapsed)

        if self.is_memory:
            memory_usage = utils.get_process_memory() - self.cur_memory
            self.memory_consumption_list.append(memory_usage)

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
                self.position_jitter_list.append(cur_av_positional_jitter)

                orientation_jitter = utils.angle_error_rowwise(self.prev_rvecs[prev_id_match_indices], rvecs[cur_id_match_indices])
                cur_av_orientational_jitter = np.mean(orientation_jitter)
                self.orientation_jitter_list.append(cur_av_orientational_jitter)

            self.prev_ids, self.prev_rvecs, self.prev_tvecs  = ids, rvecs, tvecs  

        if self.is_accuracy:

            if (ids is None) or (rvecs is None) or (tvecs is None):
                for detection in detections:
                    ids, rvecs, tvecs = detection['ids'], detection['rvecs'], detection['tvecs']

            if (ids is not None) and (rvecs is not None) and (tvecs is not None) and np.argwhere(ids == 0).size != 0:
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
                self.position_accuracy_list.append(cur_av_positional_accuracy)

                orientation_accuracy = utils.angle_error_rowwise(rvecs, np.ones((rvecs.shape[0], 1)) @ mean_rvec.reshape(1, -1))
                cur_av_orientation_accuracy = np.mean(orientation_accuracy)
                self.orientation_accuracy_list.append(cur_av_orientation_accuracy)


    def finalize(self):

        write_dict={}

        print("   ---   The results of the test   ---")

        if self.is_time:
            time_np = np.array(self.duration_list)
            self.av_time = np.average(time_np)
            print('Average proccesing time :', self.av_time,'seconds')
            self.av_frame_rate = 1/self.av_time
            print('Average frame rate :', self.av_frame_rate, 'frames/seconds')

            write_dict['Average Process Time per Frame [seconds]'] = self.av_time
            write_dict['Average Frame Rate [frames/seconds]'] = self.av_frame_rate

        if self.is_memory:
            memory_consumption_np = np.array(self.memory_consumption_list)
            average_memory_consumption = np.average(memory_consumption_np)
            self.average_memory_consumption_MB = average_memory_consumption/1048576
            print('Average memory consumption :', average_memory_consumption, 'bytes/frame |', average_memory_consumption/1048576, 'MB/frame')

            write_dict['Average Memory Consumption per Frame [MB/frame]'] = self.average_memory_consumption_MB

        if self.is_n_of_detections:
            print('Average number of detection :', self.av_n_of_detections, 'detections/frame')

            write_dict['Average Number of Detection per Frame'] = self.av_n_of_detections

        if self.is_jitter:
            position_jitters_np = np.array(self.position_jitter_list)
            self.av_position_jitter = np.average(position_jitters_np)

            orientation_jitters_np = np.array(self.orientation_jitter_list)
            self.av_orientation_jitter= np.average(orientation_jitters_np)

            print('Average positional jitter :', self.av_position_jitter, 'meters/(tag*frame) |',self.av_position_jitter*1000, 'milimeters/(tag*frame) |')
            print('Average orientational jitter :', self.av_orientation_jitter, 'rad/(tag*frame) |', self.av_orientation_jitter*180/np.pi, 'degrees/(tag*frame)')

            write_dict['Average Positional Jitter per Frame and Tag [mm/(frame*tag)]'] = self.av_position_jitter*1000
            write_dict['Average Orientational Jitter per Frame and Tag [degrees/(frame*tag)]'] = self.av_orientation_jitter*180/np.pi

        if self.is_accuracy:
            position_accuracy_np = np.array(self.position_accuracy_list)
            self.av_position_accuracy = np.average(position_accuracy_np)

            orientation_accuracy_np = np.array(self.orientation_accuracy_list)
            self.av_orientation_accuracy= np.average(orientation_accuracy_np)

            print('Average positional accuracy :', self.av_position_accuracy, 'meters/(tag*frame) |',self.av_position_accuracy*1000, 'milimeters/(tag*frame)')
            print('Average orientational accuracy :', self.av_orientation_accuracy, 'rad/(tag*frame) |', self.av_orientation_accuracy*180/np.pi, 'degrees/(tag*frame)')

            write_dict['Average Positional Accuracy per Frame and Tag [mm/(frame*tag)]'] = self.av_position_accuracy*1000
            write_dict['Average Orientational Accuracy per Frame and Tag [degrees/(frame*tag)]'] = self.av_orientation_accuracy*180/np.pi

        if utils.user_prompt("Should the data be recorded recorded?"):
            import csv
            from pathlib import Path

            filename = 'test2.csv'
            filepath = utils.get_test_path(filename)
            print('Results will be written in to :', filepath)

            test_name = input("What should be the name of these records? : ")

            write_dict['Test Name'] = test_name
            write_dict['Total Number of Frames'] = self.n_of_frames
                        
            atributes = ['Test Name', 
                        'Total Number of Frames', 
                        'Average Process Time per Frame [seconds]', 
                        'Average Frame Rate [frames/seconds]',
                        'Average Memory Consumption per Frame [MB/frame]' ,
                        'Average Number of Detection per Frame', 
                        'Average Positional Jitter per Frame and Tag [mm/(frame*tag)]', 
                        'Average Orientational Jitter per Frame and Tag [degrees/(frame*tag)]', 
                        'Average Positional Accuracy per Frame and Tag [mm/(frame*tag)]', 
                        'Average Orientational Accuracy per Frame and Tag [degrees/(frame*tag)]']

            if not Path(filepath).exists():
                print('New csv file created :', filepath)
                with open(filepath, mode='w') as results_file:
                    attributes_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                    attributes_writer.writerow(atributes)

            with open(filepath, mode='a') as csv_file:
                fieldnames = atributes
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                writer.writerow(write_dict)