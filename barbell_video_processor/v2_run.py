import sys

from barbell_video_processor.shaky_motion_detector import ShakyMotionDetector
from barbell_video_processor.barbell_width_finder import BarbellWidthFinder

if __name__ == "__main__":
    file_to_read = sys.argv[1]

    motion_detection_frames = ShakyMotionDetector(file_to_read).get_frames()
    x_offset, barbell_width = BarbellWidthFinder(motion_detection_frames).find_barbell_width()
