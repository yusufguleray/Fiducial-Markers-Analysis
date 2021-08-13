import time  #For calculating the time it takes to calculate

class Test:
    def __init__(self, is_time = False, is_memory = False, is_jitter = False, is_accuracy = False, is_n_of_detections = False) -> None:
        
        self.n_of_frames = 0

        if is_time == True:
            self.is_time = True
            self.av_time = 0
            self.cur_time = 0
            self.duration_list = []

        if is_memory == True:
            self.is_memory = True
            self.av_memory_consumption = 0

        if is_jitter == True:
            self.is_jitter == True
            self.av_posisiton_jitter = 0
            self.av_orientation_jitter = 0

        if is_accuracy == True:
            self.is_accuracy = True
            self.av_position_accuracy = 0

        if is_n_of_detections == True:
            self.is_n_of_detections = True
            self.av_n_of_detections = 0
            self.av_n_of_detections_list = []

    def start(self):

        self.n_of_frames += 1 

        if self.is_time == True:
            self.cur_time = time.perf_counter()

    def stop(self, detections):

        if self.is_time :
            time_elapsed = (time.perf_counter() - self.cur_time)
            self.duration_list.append(time_elapsed)
            self.av_time = self.av_time + 1/self.n_of_frames*(time_elapsed - self.av_time) # running average

        if self.is_n_of_detections :
            n_of_detection = 0

            for detection in detections:
                if detection['ids'] is not None :
                    n_of_detection = n_of_detection + len(detection['ids'])

            self.av_n_of_detections = self.av_n_of_detections + 1/self.n_of_frames*(n_of_detection - self.av_n_of_detections)

    def finalize(self):

        print("   ---   The results of the test   ---")

        if self.is_time :
            print('Average proccesing time =', self.av_time,'ms')

        if self.is_n_of_detections :
            print('Average number of detection =', self.av_n_of_detections)