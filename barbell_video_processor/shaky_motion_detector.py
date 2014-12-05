import sys
import cv2
import numpy as np
from collections import deque

IMAGE_WIDTH = 500
# FIXME if this doesnt equal the value in the other file it's going to break
# stuff, need to consolidate
MIN_PREVIOUS_FRAME_COUNT = 6
MAX_PREVIOUS_FRAME_COUNT = 20


def grayscale(frame):
    im = cv2.cv.fromarray(frame)
    gray = cv2.cv.CreateImage((im.width, im.height), 8, 1)
    cv2.cv.CvtColor(im, gray, cv2.cv.CV_BGR2GRAY)
    return np.asarray(gray[:, :])


def resized_frame(frame):
    height, width = frame.shape[0: 2]
    desired_width = IMAGE_WIDTH
    desired_to_actual = float(desired_width) / width
    new_width = int(width * desired_to_actual)
    new_height = int(height * desired_to_actual)
    return cv2.resize(frame, (new_width, new_height))


class ShakyMotionDetector(object):

    X_PIXEL_RANGE = 3
    Y_PIXEL_RANGE = 0

    def __init__(self, file_to_read):
        self.file_to_read = file_to_read
        self.capture = cv2.VideoCapture(self.file_to_read)
        self.resultant_frame = None

        self.video_writer = None
        self.frames_per_sec = 25
        self.codec = cv2.cv.FOURCC('M', 'J', 'P', 'G')

        self.frame_number = 0
        video_filename = (file_to_read.split("/")[-1]).split(".")[0]
        self.output_filename = "output_%s.avi" % video_filename

    def _generate_working_frames(self):
        while True:
            success, frame_from_video = self.capture.read()
            if not success:
                break
            frame_from_video = resized_frame(frame_from_video)
            yield frame_from_video

    def _generate_motion_detection_frames(self):
        previous_frame = None
        previous_previous_frame = None
        for frame in self._generate_working_frames():
            motion_detection_frame = None
            if previous_previous_frame is not None:
                motion_detection_frame = self._get_motion_detection_frame(previous_previous_frame, previous_frame, frame)
            previous_previous_frame = previous_frame
            previous_frame = frame
            if motion_detection_frame is not None:
                yield motion_detection_frame

    def _get_motion_detection_frame(self, previous_previous_frame, previous_frame, frame):
        d1 = cv2.absdiff(frame, previous_frame)
        d2 = cv2.absdiff(previous_frame, previous_previous_frame)
        motion_detection_frame = cv2.bitwise_xor(d1, d2)
        return motion_detection_frame

    def _get_clean_frame(self, frame, possible_previous_frames):
        previous_frames = self._get_previous_frames(possible_previous_frames)
        cumulative_motion = self._get_max_array(previous_frames)
        final_frame = frame.astype(int) - cumulative_motion.astype(int)
        final_frame[final_frame < 0] = 0
        return final_frame.astype(np.uint8)

    def _remove_shakiness_generator(self, frame_generator):
        initial_frames = []
        index = 0
        previous_frame_queue = deque()
        for frame in frame_generator:
            if index >= MAX_PREVIOUS_FRAME_COUNT:
                clean_frame = self._get_clean_frame(frame, list(previous_frame_queue))
                print "yielding %s" % index
                yield clean_frame
            elif index < MAX_PREVIOUS_FRAME_COUNT:
                initial_frames.append(frame)
            previous_frame_queue.append(frame)
            while len(previous_frame_queue) > MAX_PREVIOUS_FRAME_COUNT:
                previous_frame_queue.popleft()
            index += 1

        for frame in initial_frames:
            clean_frame = self._get_clean_frame(frame, list(previous_frame_queue))
            yield clean_frame
            previous_frame_queue.append(frame)
            while len(previous_frame_queue) > MAX_PREVIOUS_FRAME_COUNT:
                previous_frame_queue.popleft()

    def _get_previous_frames(self, trailing_frames):
        previous_frames = trailing_frames[:MAX_PREVIOUS_FRAME_COUNT - MIN_PREVIOUS_FRAME_COUNT]
        return previous_frames

    def _get_max_array(self, array_list):
        resultant_array = np.zeros(array_list[0].shape)
        for array in array_list:
            resultant_array = np.maximum(resultant_array, array)

        for y_offset in xrange(-self.Y_PIXEL_RANGE, self.Y_PIXEL_RANGE + 1):
            for x_offset in xrange(-self.X_PIXEL_RANGE, self.X_PIXEL_RANGE + 1):
                offset_array = np.roll(resultant_array, x_offset, axis=1)
                offset_array = np.roll(offset_array, y_offset, axis=0)
                resultant_array = np.maximum(resultant_array, offset_array)
        return resultant_array

    def generate_frames(self):
        all_frame_generator = self._generate_motion_detection_frames()
        all_frames = self._remove_shakiness_generator(all_frame_generator)
        return all_frames

    def get_union_frame(self):
        return self._create_union_frame(self.generate_frames())

    def _create_union_frame(self, filtered_motion_detection_frames):
        for frame in filtered_motion_detection_frames:
            grayscale_frame = grayscale(frame)
            # grayscale_frame[grayscale_frame < self.MOTION_THRESHOLD] = 0
            self.resultant_frame = self._get_resultant_frame(grayscale_frame)
        return self.resultant_frame.copy()

    def create(self):
        for motion_detection_frame in self.generate_frames():
            height, width = motion_detection_frame.shape[0: 2]
            self.video_writer = self.video_writer or cv2.VideoWriter(self.output_filename, self.codec, self.frames_per_sec, (width, height))
            self.video_writer.write(motion_detection_frame)
            self.frame_number += 1
            print "Writing %s" % self.frame_number
        if self.video_writer is not None:
            self.video_writer.release()


if __name__ == "__main__":
    file_to_read = sys.argv[1]
    ShakyMotionDetector(file_to_read).create()
