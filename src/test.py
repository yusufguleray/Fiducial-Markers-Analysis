import time
import cv2  #For calculating the time it takes to calculate
import numpy as np
import utils

class Test:
    def __init__(self, tag_size = 1, max_iter = 100, padding_ratio = 0.5, is_time = True, is_memory = True, is_jitter = True, 
                is_accuracy = True,  is_n_of_detections = True, is_ambiguity=True, is_record=True, record_file = 'test_results.csv', record_name = None ) -> None:
        
        self.n_of_frames = 0
        self.tag_size = tag_size
        self.max_iter = max_iter

        self.is_time, self.is_memory , self.is_jitter , self.is_accuracy , self.is_n_of_detections, self.is_ambiguity, self.is_record = False, False, False, False, False, False, False
        
        if is_time:
            self.is_time = True
            self.cur_time = 0
            self.duration_list = []

        if is_memory:
            self.is_memory = True
            self.memory_consumption_list = []

        if is_jitter:
            self.is_jitter = True

            self.prev_ids, self.prev_rvecs, self.prev_tvecs = None, None, None
            self.position_jitter_list, self.orientation_jitter_list = [], []

        padding = tag_size * padding_ratio

        if tag_size == 0.16 : self.array_size = [1,1]
        if tag_size == 0.08 : self.array_size = [2,1]
        if tag_size == 0.04 : self.array_size = [4,3]
        if tag_size == 0.02 : self.array_size = [9,6]
        if tag_size == 0.01 : self.array_size = [16,8]
        if tag_size == 0.005 : self.array_size = [16,8]
        
        if is_accuracy:
            self.is_accuracy = True
            self.position_accuracy_list, self.position_accuracy_bias_corrected_l, self.orientation_accuracy_list = [], [], []
    
            self.map = np.zeros(( self.array_size[0] *  self.array_size[1], 3)) # tvecs wrt Tag 1 (Top left)

            for j in range( self.array_size[1]):
                for i in range( self.array_size[0]):
                    self.map[j *  self.array_size[0] + i,0] = i * (tag_size + padding)
                    self.map[j *  self.array_size[0] + i,1] = - j * (tag_size + padding)
                    
        if is_n_of_detections:
            self.is_n_of_detections = True
            self.n_of_detections_list = []

        if is_ambiguity: 
            self.is_ambiguity = True
            self.n_of_ambiguity = []

        if is_record:
            self.is_record = True
            self.record_file = record_file
            self.record_name = record_name

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
                if detection['ids'] is not None:
                    n_of_detection = n_of_detection + detection['ids'].size

            self.n_of_detections_list.append(n_of_detection)

        ids, rvecs, tvecs = None, None, None
        
        if self.is_jitter:
            
            for detection in detections:
                ids, rvecs, tvecs = detection['ids'], detection['rvecs'], detection['tvecs']

            if self.prev_ids is not None and ids is not None:

                id_dist = utils.distance_matrix(ids.reshape(-1, 1), self.prev_ids.reshape(-1, 1))
                indices = (np.argwhere(id_dist == 0)).T

                cur_id_match_indices = indices[0]
                prev_id_match_indices = indices[1]
                
                positional_jitter = utils.elementwise_distance(self.prev_tvecs[prev_id_match_indices], tvecs[cur_id_match_indices])
                self.position_jitter_list.extend(positional_jitter.tolist())

                orientation_jitter = utils.angle_error_rowwise(self.prev_rvecs[prev_id_match_indices], rvecs[cur_id_match_indices])
                self.orientation_jitter_list.extend(orientation_jitter.tolist())

            self.prev_ids, self.prev_rvecs, self.prev_tvecs  = ids, rvecs, tvecs  


        if self.is_accuracy:

            if (ids is None) or (rvecs is None) or (tvecs is None):
                for detection in detections:
                    ids, rvecs, tvecs = detection['ids'], detection['rvecs'], detection['tvecs']

            if (ids is not None) and (rvecs is not None) and (tvecs is not None):
                
                flipped_ids = utils.find_ambiguity(ids, rvecs, tvecs)
                mean_rvec = np.mean(rvecs[np.bitwise_not(np.in1d(ids, flipped_ids))], axis=0) #consider only non flipped ids
                R, _ = cv2.Rodrigues(mean_rvec)

                # Finds the matching indices between the map and the tvecs
                if self.record_name == 'charuco_4cm' : 
                    map_ids = np.array([0,1,10,11,2,3,4,5,6,7,8,9])
                    id_dist = utils.distance_matrix(ids.reshape(-1, 1), map_ids.reshape(-1, 1))
                else :id_dist = utils.distance_matrix(ids.reshape(-1, 1), np.arange(self.map.shape[0]).reshape(-1, 1))
                indices = (np.argwhere(id_dist == 0)).T
                cur_id_match_indices = indices[0]
                map_id_match_indices = indices[1]

                distances_list = []

                for origin_id in map_id_match_indices:
                    origin_tvec = tvecs[np.argwhere(ids == origin_id).squeeze()]
                    if len(origin_tvec.shape) == 2 : 
                        origin_tvec = origin_tvec[0] # Take the first row if multiple of same id is detected 
                    tvecs_wrt_origin_id = tvecs - origin_tvec
                    tvecs_wrt_B = tvecs_wrt_origin_id @ R
                    
                    if self.record_name == 'charuco_4cm':
                        map_wrt_B = self.map - self.map[map_ids == origin_id]
                    else : map_wrt_B = self.map - self.map[origin_id]

                    naive_position_accuracy = utils.elementwise_distance(tvecs_wrt_B[cur_id_match_indices], map_wrt_B[map_id_match_indices])
                    distances_list.extend(naive_position_accuracy.tolist())

                    # bias = np.mean(tvecs_wrt_B[cur_id_match_indices]-map_wrt_B[map_id_match_indices], axis=0)
                    # position_accuracy_bias_corrected = utils.elementwise_distance(tvecs_wrt_B[cur_id_match_indices]-bias, self.map[map_id_match_indices])
                    # self.position_accuracy_bias_corrected_l.append(np.mean(position_accuracy_bias_corrected))
                
                self.position_accuracy_list.extend(distances_list)

                orientation_accuracy = utils.angle_error_rowwise(rvecs, np.ones((rvecs.shape[0], 1)) @ mean_rvec.reshape(1, -1))
                self.orientation_accuracy_list.extend(orientation_accuracy.tolist())

        flipped_ids = None

        if self.is_ambiguity:
            if (ids is None) or (rvecs is None) or (tvecs is None):
                for detection in detections:
                    ids, rvecs, tvecs = detection['ids'], detection['rvecs'], detection['tvecs']
            if flipped_ids is None:
                flipped_ids = utils.find_ambiguity(ids, rvecs, tvecs)

            self.n_of_ambiguity.append(len(flipped_ids))

        if self.n_of_frames >= self.max_iter: return True
        else : return False


    def finalize(self):

        write_dict={}

        print("   ---   The results of the test   ---")
        print("Number of frames :", self.n_of_frames)

        if self.is_time:
            time_np = np.array(self.duration_list)
            av_time = np.average(time_np)
            print('Average proccesing time :', av_time,'seconds')
            self.av_frame_rate = 1/av_time
            print('Average frame rate :', self.av_frame_rate, 'frames/seconds')

            write_dict['Average Process Time per Frame [seconds]'] = av_time
            write_dict['Average Frame Rate [frames/seconds]'] = self.av_frame_rate

            data_dict = utils.boxplot_data(time_np)
            write_dict['Median Process Time per Frame [seconds]'] = data_dict['median']
            write_dict['Upper Quartile Process Time per Frame [seconds]'] = data_dict['upper_quartile']
            write_dict['Lower Quartile Process Time per Frame [seconds]'] = data_dict['lower_quartile']
            write_dict['Max Process Time per Frame [seconds]'] = data_dict['max']
            write_dict['Min Quartile Process Time per Frame [seconds]'] = data_dict['min']

        if self.is_memory:
            memory_consumption_np = np.array(self.memory_consumption_list)
            average_memory_consumption = np.average(memory_consumption_np)
            self.average_memory_consumption_MB = average_memory_consumption/1048576
            print('Average memory consumption :', average_memory_consumption, 'bytes/frame |', average_memory_consumption/1048576, 'MB/frame')

            write_dict['Average Memory Consumption per Frame [MB/frame]'] = self.average_memory_consumption_MB

        if self.is_n_of_detections:
            detection_percents = np.array(self.n_of_detections_list) / (self.array_size[0] * self.array_size[1]) * 100
            data_dict = utils.boxplot_data(detection_percents)

            print('Average detection percent:', data_dict['average'], '%')

            write_dict['Average Detection Percent'] = data_dict['average']
            write_dict['Median Detection Percent'] = data_dict['median']
            write_dict['Upper Quartile Detection Percent'] = data_dict['upper_quartile']
            write_dict['Lower Quartile Detection Percent'] = data_dict['lower_quartile']
            write_dict['Max Detection Percent'] = data_dict['max']
            write_dict['Min Detection Percent'] = data_dict['min']

        if self.is_jitter:
            position_jitters_np = np.array(self.position_jitter_list) * 1000 # Converted to mm
            positional_data_dict = utils.boxplot_data(position_jitters_np)

            orientation_jitters_np = np.array(self.orientation_jitter_list) * 180 / np.pi
            orientational_data_dict = utils.boxplot_data(orientation_jitters_np)

            if positional_data_dict is not None and orientational_data_dict is not None:
                print('Average positional jitter :', positional_data_dict['average'], 'milimeters/(tag*frame) |')
                print('Average orientational jitter :', orientational_data_dict['average'], 'degrees/(tag*frame)')

                write_dict['Average Positional Jitter per Frame and Tag [mm/(frame*tag)]'] = positional_data_dict['average']
                write_dict['Average Orientational Jitter per Frame and Tag [degrees/(frame*tag)]'] = orientational_data_dict['average']

                write_dict['Median Positional Jitter per Frame and Tag [mm/(frame*tag)]'] = positional_data_dict['median']
                write_dict['Upper Quartile Positional Jitter per Frame and Tag [mm/(frame*tag)]'] = positional_data_dict['upper_quartile']
                write_dict['Lower Quartile Positional Jitter per Frame and Tag [mm/(frame*tag)]'] = positional_data_dict['lower_quartile']
                write_dict['Max Positional Jitter per Frame and Tag [mm/(frame*tag)]'] = positional_data_dict['max']
                write_dict['Min Positional Jitter per Frame and Tag [mm/(frame*tag)]'] = positional_data_dict['min']

                write_dict['Median Orientational Jitter per Frame and Tag [degrees/(frame*tag)]'] = orientational_data_dict['median']
                write_dict['Upper Orientational Jitter per Frame and Tag [degrees/(frame*tag)]'] = orientational_data_dict['upper_quartile']
                write_dict['Lower Orientational Jitter per Frame and Tag [degrees/(frame*tag)]'] = orientational_data_dict['lower_quartile']
                write_dict['Max Orientational Jitter per Frame and Tag [degrees/(frame*tag)]'] = orientational_data_dict['max']
                write_dict['Min Orientational Jitter per Frame and Tag [degrees/(frame*tag)]'] = orientational_data_dict['min']

        if self.is_accuracy:

            position_accuracy_np = np.array(self.position_accuracy_list) * 1000 # in mm
            positional_acc_data_dict = utils.boxplot_data(position_accuracy_np)

            # position_accuracy_bias_corrected_np = np.array(self.position_accuracy_bias_corrected_l)
            # self.av_position_accuracy_bias_corrected = np.average(position_accuracy_bias_corrected_np)

            orientation_accuracy_np = np.array(self.orientation_accuracy_list) * 180 / np.pi
            orientation_acc_data_dict = utils.boxplot_data(orientation_accuracy_np)

            if positional_acc_data_dict is not None and orientation_acc_data_dict is not None:
                print('Average positional accuracy :', positional_acc_data_dict['average'], 'milimeters/(tag*frame)')
                print('Average orientational accuracy :', orientation_acc_data_dict['average'], 'degrees/(tag*frame)')

                write_dict['Average Positional Accuracy per Frame and Tag [mm/(frame*tag)]'] = positional_acc_data_dict['average']
                write_dict['Median Positional Accuracy per Frame and Tag [mm/(frame*tag)]'] = positional_acc_data_dict['median']
                write_dict['Upper Quartile Positional Accuracy per Frame and Tag [mm/(frame*tag)]'] = positional_acc_data_dict['upper_quartile']
                write_dict['Lower Quartile Positional Accuracy per Frame and Tag [mm/(frame*tag)]'] = positional_acc_data_dict['lower_quartile']
                write_dict['Max Positional Accuracy per Frame and Tag [mm/(frame*tag)]'] = positional_acc_data_dict['max']
                write_dict['Min Positional Accuracy per Frame and Tag [mm/(frame*tag)]'] = positional_acc_data_dict['min']

                write_dict['Average Orientational Accuracy per Frame and Tag [degrees/(frame*tag)]'] = orientation_acc_data_dict['average']
                write_dict['Median Orientational Accuracy per Frame and Tag [degrees/(frame*tag)]'] = orientation_acc_data_dict['median']
                write_dict['Upper Orientational Accuracy per Frame and Tag [degrees/(frame*tag)]'] = orientation_acc_data_dict['upper_quartile']
                write_dict['Lower Orientational Accuracy per Frame and Tag [degrees/(frame*tag)]'] = orientation_acc_data_dict['lower_quartile']
                write_dict['Max Orientational Accuracy per Frame and Tag [degrees/(frame*tag)]'] = orientation_acc_data_dict['max']
                write_dict['Min Orientational Accuracy per Frame and Tag [degrees/(frame*tag)]'] = orientation_acc_data_dict['min']
                # write_dict['Average Biased Corrected Positional Accuracy per Frame and Tag [mm/(frame*tag)]'] = self.av_position_accuracy_bias_corrected*1000

        if self.is_ambiguity:
            n_of_ambiguity_np = np.array(self.n_of_ambiguity)
            ambiguity_percent = (n_of_ambiguity_np / self.n_of_detections_list) * 100
            data_dict = utils.boxplot_data(ambiguity_percent)

            if data_dict is not None:
                print('Ambiguety Percent(Number of Flipped Detections/Total Number of Detections) :', data_dict['average'])
                
                write_dict['Average Ambiguety Percent(Number of Flipped Detections/Number of Detections)'] = data_dict['average']
                write_dict['Median Ambiguety Percent'] = data_dict['median']
                write_dict['Upper Ambiguety Percent'] = data_dict['upper_quartile']
                write_dict['Lower Ambiguety Percent'] = data_dict['lower_quartile']
                write_dict['Max Ambiguety Percent'] = data_dict['max']
                write_dict['Min Ambiguety Percent'] = data_dict['min']
            

        if self.is_record or utils.user_prompt("Should the data be recorded recorded?"):
            import csv
            from pathlib import Path

            filepath = utils.get_test_path(self.record_file)
            print('Results will be written in to :', filepath)

            if self.record_name is None : self.record_name = input("What should be the name of these records? : ")

            write_dict['Test Name'] = self.record_name
            write_dict['Tag Size [cm]'] = self.tag_size*100
            write_dict['Array Size'] = self.array_size
            write_dict['Total Number of Frames'] = self.n_of_frames
                        
            atributes = ['Test Name',
                        'Tag Size [cm]', 
                        'Array Size',
                        'Total Number of Frames',

                        'Average Process Time per Frame [seconds]', 
                        'Median Process Time per Frame [seconds]',
                        'Upper Quartile Process Time per Frame [seconds]',
                        'Lower Quartile Process Time per Frame [seconds]',
                        'Max Process Time per Frame [seconds]',
                        'Min Quartile Process Time per Frame [seconds]',

                        'Average Frame Rate [frames/seconds]',

                        'Average Memory Consumption per Frame [MB/frame]' ,

                        'Average Detection Percent',
                        'Median Detection Percent',
                        'Upper Quartile Detection Percent',
                        'Lower Quartile Detection Percent',
                        'Max Detection Percent',
                        'Min Detection Percent',

                        'Average Positional Jitter per Frame and Tag [mm/(frame*tag)]', 
                        'Median Positional Jitter per Frame and Tag [mm/(frame*tag)]',
                        'Upper Quartile Positional Jitter per Frame and Tag [mm/(frame*tag)]',
                        'Lower Quartile Positional Jitter per Frame and Tag [mm/(frame*tag)]',
                        'Max Positional Jitter per Frame and Tag [mm/(frame*tag)]',
                        'Min Positional Jitter per Frame and Tag [mm/(frame*tag)]',

                        'Average Orientational Jitter per Frame and Tag [degrees/(frame*tag)]',
                        'Median Orientational Jitter per Frame and Tag [degrees/(frame*tag)]',
                        'Upper Orientational Jitter per Frame and Tag [degrees/(frame*tag)]',
                        'Lower Orientational Jitter per Frame and Tag [degrees/(frame*tag)]',
                        'Max Orientational Jitter per Frame and Tag [degrees/(frame*tag)]',
                        'Min Orientational Jitter per Frame and Tag [degrees/(frame*tag)]',
                        
                        'Average Positional Accuracy per Frame and Tag [mm/(frame*tag)]',
                        'Median Positional Accuracy per Frame and Tag [mm/(frame*tag)]',
                        'Upper Quartile Positional Accuracy per Frame and Tag [mm/(frame*tag)]',
                        'Lower Quartile Positional Accuracy per Frame and Tag [mm/(frame*tag)]',
                        'Max Positional Accuracy per Frame and Tag [mm/(frame*tag)]',
                        'Min Positional Accuracy per Frame and Tag [mm/(frame*tag)]',

                        'Average Orientational Accuracy per Frame and Tag [degrees/(frame*tag)]',
                        'Median Orientational Accuracy per Frame and Tag [degrees/(frame*tag)]',
                        'Upper Orientational Accuracy per Frame and Tag [degrees/(frame*tag)]',
                        'Lower Orientational Accuracy per Frame and Tag [degrees/(frame*tag)]',
                        'Max Orientational Accuracy per Frame and Tag [degrees/(frame*tag)]',
                        'Min Orientational Accuracy per Frame and Tag [degrees/(frame*tag)]',

                        # 'Average Biased Corrected Positional Accuracy per Frame and Tag [mm/(frame*tag)]', 
                        'Average Ambiguety Percent(Number of Flipped Detections/Number of Detections)',
                        'Median Ambiguety Percent',
                        'Upper Ambiguety Percent',
                        'Lower Ambiguety Percent',
                        'Max Ambiguety Percent',
                        'Min Ambiguety Percent']

            if not Path(filepath).exists():
                print('New csv file created :', filepath)
                with open(filepath, mode='w') as results_file:
                    attributes_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                    attributes_writer.writerow(atributes)

            with open(filepath, mode='a') as csv_file:
                fieldnames = atributes
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                writer.writerow(write_dict)

            print(self.record_name,'is succesfully written into', filepath)