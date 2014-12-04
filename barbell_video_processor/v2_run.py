import sys

from barbell_video_processor.shaky_motion_detector import ShakyMotionDetector
from barbell_video_processor.barbell_width_finder import BarbellWidthFinder

# temp
import cv2


def resized_frame(frame):
    height, width = frame.shape[0: 2]
    desired_width = 500
    desired_to_actual = float(desired_width) / width
    new_width = int(width * desired_to_actual)
    new_height = int(height * desired_to_actual)
    return cv2.resize(frame, (new_width, new_height))


def temp_func(file_to_read):
    capture = cv2.VideoCapture(file_to_read)
    success, frame = capture.read()
    frame = resized_frame(frame)
    cv2.imshow("original", frame)


if __name__ == "__main__":
    file_to_read = sys.argv[1]
    temp_func(file_to_read)

    motion_detection_frames = ShakyMotionDetector(file_to_read).get_frames()
    BarbellWidthFinder(motion_detection_frames).find_barbell_width()
