from collections import defaultdict
import sys
import datetime
import json
import cv2
import math
import numpy as np
import random

from barbell_video_processor.shaky_motion_detector import ShakyMotionDetector
from barbell_video_processor.barbell_width_finder import BarbellWidthFinder


LINE_DETECTION_THRESHOLD = 25  # 50
FRAMES_PER_SEC_KEY = 5
MOTION_THRESHOLD = 25
MIN_BAR_WIDTH_AS_PERCENT_OF_SCREEN = 0.50
MAX_PERCENT_DIFFERENCE_BETWEEN_BARS = 0.10
MAX_ROM_IN_INCHES = 40
METERS_PER_INCH = 0.0254
TOP_PERCENTILE_THRESHOLD_FOR_GOOD_DETECTION = 0.25
MAX_ROM_IN_METERS = MAX_ROM_IN_INCHES * METERS_PER_INCH
ACTUAL_FRAME_OFFSET = 2
RESIZE_WIDTH = 500
MINIMUM_BAR_DETECTIONS = 15
LOCAL = False


class Orientation(object):
    NONE = 1
    ROTATE_90 = 2
    ROTATE_180 = 3
    ROTATE_270 = 4


def get_avg_velocity_between_points(point_pairs):
    frame_to_avg_velocity = {}
    for start_point, end_point in point_pairs:
        delta_y = end_point[1] - start_point[1]
        delta_x = end_point[0] - start_point[0]
        avg_velocity = float(delta_y) / delta_x
        for x in xrange(int(start_point[0]), int(end_point[0])):
            frame_to_avg_velocity[x] = avg_velocity
    return frame_to_avg_velocity


def convert_ppf_to_mps(barbell_width_pixels, frame_to_avg_velocity_pixels_per_frame, frames_per_sec):
    barbell_width_meters = 2.2
    pixels_per_meter = barbell_width_pixels / barbell_width_meters
    frame_to_avg_velocity_meters_per_second = {}
    for frame, velocity_ppf in frame_to_avg_velocity_pixels_per_frame.items():
        velocity_mps = frames_per_sec * velocity_ppf / pixels_per_meter
        frame_to_avg_velocity_meters_per_second[frame] = velocity_mps
    return frame_to_avg_velocity_meters_per_second


def magic_1rm_formula(frame_to_avg_velocity_meters_per_second):
    '''
    Taken from
    https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0CB4QFjAA&url=http%3A%2F%2Fwww.researchgate.net%2Fpublication%2F40453550_Using_the_load-velocity_relationship_for_1RM_prediction%2Flinks%2F09e4150f761dd62684000000&ei=79ksVMCSJ8exogSol4DACw&usg=AFQjCNGd8w9aaM1zoZW3XuJe5xPXrqlQ1g&sig2=itLy-9QTfTlAxwpxxYgXHA
    '''
    one_rep_max_per_frame = {}
    for frame_number, meters_per_second in frame_to_avg_velocity_meters_per_second.items():
        if meters_per_second > 0:
            one_rep_max_per_frame[frame_number] = -1.0
            continue
        meters_per_second *= -1.0  # flip sign for this formula
        # percent_1rm = (meters_per_second - 1.7035) / -0.0146
        percent_1rm = (meters_per_second - 1.7022) / -0.015622
        scalar_coefficient = 100.0 / percent_1rm
        if scalar_coefficient < 1.0:
            scalar_coefficient = 1.0
        one_rep_max_per_frame[frame_number] = scalar_coefficient
    return one_rep_max_per_frame


def get_frame_to_1rm(detected_barbells, frames_per_sec):
    # this should already be done, but just being cautious
    detected_barbells.sort(key=lambda barbell: barbell.frame_number)
    x_values = np.asarray([detection.frame_number for detection in detected_barbells]).astype(np.float)
    y_values = np.asarray([detection.offset_y for detection in detected_barbells]).astype(np.float)

    print "Finding the local maxima..."
    min_maxima, max_maxima = find_maxima(x_values, y_values)
    print "Calculating basic velocity..."
    point_pairs = establish_point_pairs(min_maxima, max_maxima, x_values, y_values)
    frame_to_avg_velocity_pixels_per_frame = get_avg_velocity_between_points(point_pairs)

    barbell_width_pixels = detected_barbells[0].barbell_width
    frame_to_avg_velocity_meters_per_second = convert_ppf_to_mps(barbell_width_pixels, frame_to_avg_velocity_pixels_per_frame, frames_per_sec)
    frame_to_1rm = magic_1rm_formula(frame_to_avg_velocity_meters_per_second)
    return frame_to_1rm


def get_instantaneous_acceleration_from_points(point_pairs, x_values, y_values):
    frame_number_to_instantaneous_acceleration = {}
    for start_point, end_point in point_pairs:
        x_min = start_point[0]
        x_max = end_point[0]
        indexes = [index for index, x in enumerate(x_values) if x >= x_min and x <= x_max]
        x_range = [x_values[index] for index in indexes]
        y_range = [y_values[index] for index in indexes]
        if len(x_range) == 0:
            continue

        velocity = np.diff(y_range) / np.diff(x_range)
        velocity_list = [v for v in velocity]
        velocity_list = [0] + velocity_list
        velocity = np.asarray(velocity_list)

        acceleration = np.diff(velocity)
        acceleration_list = [a for a in acceleration]
        acceleration_list.append(0)
        acceleration = np.asarray(acceleration_list)
        for index, acceleration_value in enumerate(acceleration):
            frame_number = x_range[index]
            end_frame_number = frame_number + 1
            if index + 1 < len(x_range):
                end_frame_number = x_range[index + 1]
            for frame in xrange(int(frame_number), int(end_frame_number)):
                frame_number_to_instantaneous_acceleration[frame] = acceleration_value
    return frame_number_to_instantaneous_acceleration


def smooth_list_gaussian(input_list, degree=5):
    window = degree * 2 - 1
    weight = np.array([1.0] * window)
    weightGauss = []
    for i in range(window):
        i = i - degree + 1
        frac = i / float(window)
        gauss = 1 / (np.exp((4 * (frac)) ** 2))
        weightGauss.append(gauss)
    weight = np.array(weightGauss) * weight
    smoothed = [0.0] * (len(input_list) - window)
    for i in range(len(smoothed)):
        smoothed[i] = sum(np.array(input_list[i: i + window]) * weight) / sum(weight)
    return smoothed


def get_max_accelerations_from_points(point_pairs, frame_to_instantaneous_acceleration_in_gs):
    frame_to_max_acceleration = {}
    for start_point, end_point in point_pairs:
        delta_y = end_point[1] - start_point[1]
        if delta_y >= 0:
            continue  # this is a negative rep
        start_x = int(start_point[0])
        end_x = int(end_point[0])

        possible_accelerations = []
        for frame_number in xrange(start_x, end_x + 1):
            if frame_number in frame_to_instantaneous_acceleration_in_gs:
                acceleration = frame_to_instantaneous_acceleration_in_gs[frame_number]
                possible_accelerations.append(acceleration)
        if len(possible_accelerations) > 0:
            max_acceleration = max(possible_accelerations)
            possible_accelerations.remove(max_acceleration)  # remove 1 outlier from beginning
            if len(possible_accelerations) > 0:
                max_acceleration = max(possible_accelerations)
                for frame_number in xrange(start_x, end_x + 1):
                    frame_to_max_acceleration[frame_number] = max_acceleration
    return frame_to_max_acceleration


def _eliminate_0_derivatives(x_values, y_values):
    first_derivative = np.diff(y_values) / np.diff(x_values)
    max_iterations = len(first_derivative)
    counter = 0
    while len(first_derivative[first_derivative == 0] > 0):
        counter += 1
        if counter >= max_iterations:
            break
        for index, item in enumerate(first_derivative):
            if item == 0:
                y_values[index] = (y_values[index - 1] + y_values[index + 1]) / 2.0
        first_derivative = np.diff(y_values) / np.diff(x_values)
    return first_derivative


def _get_standard_maxima(x_values, y_values, cleaned_first_derivative):
    first_derivative = cleaned_first_derivative
    max_maxima = []
    min_maxima = []
    for index in xrange(1, len(first_derivative)):
        previous_velocity = first_derivative[index - 1]
        current_velocity = first_derivative[index]

        if previous_velocity * current_velocity < 0:
            # signs between the two have changed
            if previous_velocity > 0:  # max maxima
                max_maxima.append((x_values[index], y_values[index]))
            else:
                min_maxima.append((x_values[index], y_values[index]))
    return min_maxima, max_maxima


def _append_first_last_points_to_maxima(x_values, y_values, min_maxima, max_maxima):
    # SBL TODO if there's an index error here that probably indicates that no
    # detections were ever found
    first_maxima_is_min = min_maxima[0][0] < max_maxima[0][0]
    last_maxima_is_min = min_maxima[-1][0] > max_maxima[-1][0]
    if first_maxima_is_min:
        # make the very first point a max
        max_maxima.append((x_values[0], y_values[0]))
    else:
        # make the very first point a min
        min_maxima.append((x_values[0], y_values[0]))

    if last_maxima_is_min:
        # make very last point a max
        max_maxima.append((x_values[-1], y_values[-1]))
    else:
        # make very last point a min
        min_maxima.append((x_values[-1], y_values[-1]))


def find_maxima(x_values, y_values):
    y_values = y_values.astype(np.float)
    first_derivative = _eliminate_0_derivatives(x_values, y_values)
    min_maxima, max_maxima = _get_standard_maxima(x_values, y_values, first_derivative)
    _append_first_last_points_to_maxima(x_values, y_values, min_maxima, max_maxima)

    min_maxima.sort(key=lambda t: t[0])
    max_maxima.sort(key=lambda t: t[0])

    return min_maxima, max_maxima


def _get_distance_between_points(pt1, pt2):
    delta_x = pt1[0] - pt2[0]
    delta_y = pt1[1] - pt2[1]
    distance = np.sqrt(delta_x ** 2 + delta_y ** 2)
    return distance


def _get_distance_to_point(start_point, end_point, x_values, y_values):
    ''' Pre-condition: x and y values are sorted '''
    start_x = start_point[0]
    end_x = end_point[0]
    indexes_to_traverse = [index for index, x in enumerate(x_values) if x >= start_x and x <= end_x]
    points_to_traverse = [(x_values[index], y_values[index]) for index in indexes_to_traverse]
    total_distance = 0.0
    for index in xrange(1, len(points_to_traverse)):
        prev_point = points_to_traverse[index - 1]
        current_point = points_to_traverse[index]
        total_distance += _get_distance_between_points(prev_point, current_point)
    return total_distance


def establish_point_pairs(initial_min_maxima, initial_max_maxima, x_values, y_values):
    min_maxima = list(initial_min_maxima)
    max_maxima = list(initial_max_maxima)
    point_pairs = []
    while True:
        try:
            if min_maxima[0][0] < max_maxima[0][0]:
                starting_maxima = min_maxima
                maxima_to_search = max_maxima
            else:
                starting_maxima = max_maxima
                maxima_to_search = min_maxima
        except IndexError:
            break

        start_point = starting_maxima[0]

        max_score = 0
        for point in maxima_to_search:
            if starting_maxima == max_maxima and point[1] > start_point[1]:
                continue
            elif starting_maxima == min_maxima and point[1] < start_point[1]:
                continue
            actual_distance_to_point = _get_distance_to_point(start_point, point, x_values, y_values)
            delta_y = abs(point[1] - start_point[1])
            score = delta_y ** 2 / actual_distance_to_point
            if score > max_score:
                max_score = score
                best_endpoint = point

        for possible_better_start in starting_maxima:
            if not (possible_better_start[0] > start_point[0] and possible_better_start[0] < best_endpoint[0]):
                continue
            actual_distance_to_point = _get_distance_to_point(possible_better_start, best_endpoint, x_values, y_values)
            delta_y = abs(best_endpoint[1] - possible_better_start[1])
            score = delta_y ** 2 / actual_distance_to_point
            if score > max_score:
                max_score = score
                start_point = possible_better_start
        point_pairs.append((start_point, best_endpoint))
        # need to see about finding a better start point...this can happen in
        # between sets or something

        min_maxima = [t for t in min_maxima if t[0] >= best_endpoint[0]]
        max_maxima = [t for t in max_maxima if t[0] >= best_endpoint[0]]

    continuous_paths = _get_continuous_paths(point_pairs)
    non_continuous_paths = _get_non_continuous_paths(continuous_paths, x_values, y_values)
    for start_point, end_point in non_continuous_paths:
        indexes_to_keep = [index for index, x in enumerate(x_values) if start_point[0] <= x <= end_point[0]]
        new_x = [x_values[index] for index in indexes_to_keep]
        new_y = [y_values[index] for index in indexes_to_keep]
        min_maxima = [t for t in initial_min_maxima if start_point[0] <= t[0] <= end_point[0]]
        max_maxima = [t for t in initial_max_maxima if start_point[0] <= t[0] <= end_point[0]]

        recursive_point_pairs = establish_point_pairs(min_maxima, max_maxima, new_x, new_y)
        point_pairs += recursive_point_pairs

    point_pairs.sort(key=lambda pair: pair[0][0])
    return point_pairs


def _get_non_continuous_paths(continuous_paths, x_values, y_values):
    if not continuous_paths:
        return []
    non_continuous = []
    start_point = (x_values[0], y_values[0])
    non_continuous.append((start_point, continuous_paths[0][0]))

    if len(continuous_paths) > 1:
        for index in xrange(len(continuous_paths) - 1):
            start1, end1 = continuous_paths[index]
            start2, end2 = continuous_paths[index + 1]
            non_continuous.append((end1, start2))

    end_point = (x_values[-1], y_values[-1])
    non_continuous.append((continuous_paths[-1][1], end_point))

    final_non_continuous = []
    for start_point, end_point in non_continuous:
        if start_point != end_point:
            final_non_continuous.append((start_point, end_point))
    return final_non_continuous


def _get_continuous_paths(point_pairs):
    continuous_paths = []
    if len(point_pairs) == 1:
        return list(point_pairs)
    for index in xrange(len(point_pairs) - 1):
        start1, end1 = point_pairs[index]
        start_x = start1[0]
        x_pairs = [(st[0], en[0]) for st, en in continuous_paths]
        accounted_for = False
        for st, en in x_pairs:
            if st <= start_x <= en:
                accounted_for = True
        if accounted_for:
            continue
        points_to_add = None
        end = end1
        for index2 in xrange(index + 1, len(point_pairs)):
            start2, end2 = point_pairs[index2]
            if end == start2:
                points_to_add = (start1, end2)
                end = end2
        if points_to_add:
            continuous_paths.append(points_to_add)
    return continuous_paths


def has_fluctuations(detected_barbells):
    x_values = np.asarray([barbell.frame_number for barbell in detected_barbells]).astype(np.float)
    y_values = np.asarray([barbell.offset_y for barbell in detected_barbells]).astype(np.float)

    first_derivative = np.diff(y_values.astype(np.float)) / np.diff(x_values.astype(np.float))
    for index, item in enumerate(first_derivative):
        if index == 0 or index == len(first_derivative) - 1:
            continue
        previous_item = first_derivative[index - 1]
        next_item = first_derivative[index + 1]
        if previous_item * next_item > 0:
            # this item had better have the same sign
            if item * previous_item < 0:
                return True
    return False


def fix_one_fluctuation(detected_barbells):
    x_values = np.asarray([barbell.frame_number for barbell in detected_barbells]).astype(np.float)
    y_values = np.asarray([barbell.offset_y for barbell in detected_barbells]).astype(np.float)
    first_derivative = np.diff(y_values.astype(np.float)) / np.diff(x_values.astype(np.float))
    y_derivatives = []
    bad_indexes = []
    for index, item in enumerate(first_derivative):
        if index == 0 or index == len(first_derivative) - 1:
            continue
        previous_item = first_derivative[index - 1]
        next_item = first_derivative[index + 1]
        if previous_item * next_item > 0:
            # this item had better have the same sign
            if item * previous_item < 0:
                items = (previous_item, item, next_item)
                smallest_derivative = min(abs(item) for item in items)
                y_derivatives.append(smallest_derivative)
                bad_indexes.append(index)

    index_to_change = y_derivatives.index(min(y_derivatives))
    corresponding_index = bad_indexes[index_to_change]
    detected_barbells[corresponding_index].offset_y = (y_values[corresponding_index - 1] + y_values[corresponding_index + 1]) / 2


class CouldNotDetectException(Exception):
    pass


class OlympicBarbell(object):
    # 50 millimeters diameter at end
    # 70 millimeter diameter at the part that holds the plates
    # 28 millimeters diameter in middle

    # 2200 millimeters wide
    # 1310 mm wide on inside
    # 415 mm wide at ends
    # notch to hold plates is 30 mm

    # transparency means that the 4th byte in the array is 0
    TRANSPARENT_PIXEL = (0, 0, 0, 0)
    WHITE_PIXEL = (255, 255, 255, 255)
    BLACK_PIXEL = (0, 0, 0, 255)

    OPAQUE_GREEN = (0, 255, 0, 75)
    OPAQUE_YELLOW = (0, 255, 255, 75)
    OPAQUE_BLACK = (0, 0, 0, 100)

    BRUSH_SIZE = 1

    def __init__(self, for_display=False, for_negative=False, force_fill=False):
        shape = (70, 2200, 4)
        self.for_display = for_display
        self.make_notches_transparent = True
        if for_display:
            self.make_notches_transparent = False
        self.for_negative = for_negative

        self.canvas = np.zeros(shape, dtype=np.uint8)
        self.canvas[::] = self.TRANSPARENT_PIXEL
        self.resize_factor = 1.0
        self.force_fill = force_fill
        self.cached_canvases = {}

    def _shrink_args_by_resize_factor(self, arg_list):
        new_args = [item for item in arg_list]
        for index in xrange(len(arg_list)):
            item = arg_list[index]
            if isinstance(item, tuple) and len(item) == 2:
                new_tuple = tuple([int(self.resize_factor * val) for val in item])
                new_args[index] = new_tuple
        return new_args

    def with_width(self, width):
        width = int(width)
        if width in self.cached_canvases:
            return self.cached_canvases[width]

        rows, cols = 70, 2200
        resize_factor = float(width) / cols

        shape = (int(70 * resize_factor), int(width), 4)
        self.canvas = np.zeros(shape, dtype=np.uint8)
        self.canvas[::] = self.TRANSPARENT_PIXEL
        # canvas_copy = cv2.resize(self.canvas, (int(resize_factor * 2200), int(resize_factor * 70)))
        self.resize_factor = resize_factor

        self._fill_with_black()
        self._draw_ends()
        if not self.make_notches_transparent:
            self._draw_notches()
        self._draw_bar()
        if self.make_notches_transparent:
            self._erase_first_plate()
        self.cached_canvases[width] = self.canvas

        # original_opacity = self.canvas[:, :, 3].copy()
        # threshold_pixel = 50
        # self.canvas[self.canvas > threshold_pixel] = 255
        # self.canvas[self.canvas < threshold_pixel] = 0
        # if not self.for_display and np.sum(canvas_copy[:, :, 0]) / 255.0 < canvas_copy.shape[1]:
        #     return None
        # canvas_copy[:, :, 3] = original_opacity
        return self.canvas  # canvas_copy

    def _draw_ends(self):
        pixel_color = self.WHITE_PIXEL
        if self.for_display:
            pixel_color = self.OPAQUE_BLACK

        list_of_args = [
            (self.canvas, (0, 10), (415, 10), pixel_color, self.BRUSH_SIZE),
            (self.canvas, (0, 60), (415, 60), pixel_color, self.BRUSH_SIZE),

            (self.canvas, (2200, 10), (2200 - 415, 10), pixel_color, self.BRUSH_SIZE),
            (self.canvas, (2200, 60), (2200 - 415, 60), pixel_color, self.BRUSH_SIZE),
        ]
        if self.for_display:
            list_of_args.extend([
                (self.canvas, (0, 10), (0, 60), pixel_color, self.BRUSH_SIZE),
                (self.canvas, (2200, 10), (2200, 60), pixel_color, self.BRUSH_SIZE)
            ])
        for arg_list in list_of_args:
            arg_list = self._shrink_args_by_resize_factor(arg_list)
            cv2.line(*arg_list)

    def _draw_notches(self):
        pixel_color = self.WHITE_PIXEL
        if self.for_display:
            pixel_color = self.OPAQUE_BLACK

        list_of_args = [
            (self.canvas, (415, 10), (415, 0), pixel_color, self.BRUSH_SIZE),
            (self.canvas, (415, 0), (415 + 30, 0), pixel_color, self.BRUSH_SIZE),
            (self.canvas, (415 + 30, 0), (415 + 30, 21), pixel_color, self.BRUSH_SIZE),

            (self.canvas, (415, 60), (415, 60 + 10), pixel_color, self.BRUSH_SIZE),
            (self.canvas, (415, 60 + 10), (415 + 30, 60 + 10), pixel_color, self.BRUSH_SIZE),
            (self.canvas, (415 + 30, 60 + 10), (415 + 30, 70 - 21), pixel_color, self.BRUSH_SIZE),

            (self.canvas, (2200 - 415, 10), (2200 - 415, 0), pixel_color, self.BRUSH_SIZE),
            (self.canvas, (2200 - 415, 0), (2200 - 415 - 30, 0), pixel_color, self.BRUSH_SIZE),
            (self.canvas, (2200 - 415 - 30, 0), (2200 - 415 - 30, 21), pixel_color, self.BRUSH_SIZE),

            (self.canvas, (2200 - 415, 60), (2200 - 415, 60 + 10), pixel_color, self.BRUSH_SIZE),
            (self.canvas, (2200 - 415, 60 + 10), (2200 - 415 - 30, 60 + 10), pixel_color, self.BRUSH_SIZE),
            (self.canvas, (2200 - 415 - 30, 60 + 10), (2200 - 415 - 30, 70 - 21), pixel_color, self.BRUSH_SIZE),
        ]
        for arg_list in list_of_args:
            arg_list = self._shrink_args_by_resize_factor(arg_list)
            cv2.line(*arg_list)

    def _draw_bar(self):
        pixel_color = self.WHITE_PIXEL
        if self.for_display:
            pixel_color = self.OPAQUE_BLACK
        list_of_args = [
            (self.canvas, (415 + 30, 21), (2200 - 415 - 30, 21), pixel_color, self.BRUSH_SIZE),
            (self.canvas, (415 + 30, 70 - 21), (2200 - 415 - 30, 70 - 21), pixel_color, self.BRUSH_SIZE),
        ]
        for arg_list in list_of_args:
            arg_list = self._shrink_args_by_resize_factor(arg_list)
            cv2.line(*arg_list)

    def _fill_with_black(self):
        pixel_color = self.TRANSPARENT_PIXEL  # BLACK_PIXEL
        if self.force_fill:
            pixel_color = self.BLACK_PIXEL
        if self.for_display:
            pixel_color = self.OPAQUE_GREEN if not self.for_negative else self.OPAQUE_YELLOW

        list_of_args = [
            (self.canvas, (0 + self.BRUSH_SIZE, 21 + self.BRUSH_SIZE), (2200 - self.BRUSH_SIZE, 70 - 21 - self.BRUSH_SIZE), pixel_color, -1),

            (self.canvas, (0 + self.BRUSH_SIZE, 10 + self.BRUSH_SIZE), (415, 70 - 10 - self.BRUSH_SIZE), pixel_color, -1),
            (self.canvas, (2200 - 415 - self.BRUSH_SIZE, 10 + self.BRUSH_SIZE), (2200 - self.BRUSH_SIZE, 70 - 10 - self.BRUSH_SIZE), pixel_color, -1),
        ]
        if not self.make_notches_transparent:
            list_of_args.extend([
                (self.canvas, (415 + self.BRUSH_SIZE, self.BRUSH_SIZE), (415 + 30 - self.BRUSH_SIZE, 70 - self.BRUSH_SIZE), pixel_color, -1),
                (self.canvas, (2200 - 415 - 30 + self.BRUSH_SIZE, self.BRUSH_SIZE), (2200 - 415 - self.BRUSH_SIZE, 70 - self.BRUSH_SIZE), pixel_color, -1)
            ])
        for arg_list in list_of_args:
            arg_list = self._shrink_args_by_resize_factor(arg_list)
            cv2.rectangle(*arg_list)

    def _erase_first_plate(self):
        plate_width = 30
        list_of_args = [
            (self.canvas, (415 - plate_width, 0), (415, 70), self.TRANSPARENT_PIXEL, -1),
            (self.canvas, (2200 - 415 - plate_width, 0), (2200 - 415, 70), self.TRANSPARENT_PIXEL, -1)
        ]
        for arg_list in list_of_args:
            arg_list = self._shrink_args_by_resize_factor(arg_list)
            cv2.rectangle(*arg_list)


def convert_transparency_to_noise(png_img):
    print "Starting conversion..."
    rows, cols, channels = png_img.shape
    if channels != 4:
        raise ValueError("image must have 4 channels")
    three_channel_img = np.zeros((rows, cols, 3), np.uint8)
    for row_index in xrange(rows):
        for col_index in xrange(cols):
            color_bytes = png_img[row_index][col_index]
            transparency_byte = color_bytes[3]
            bgr_pixel = color_bytes[:3]
            is_transparent = transparency_byte == 0
            if is_transparent:
                three_channel_img[row_index][col_index][0] = int(random.random() * 255)
                three_channel_img[row_index][col_index][1] = int(random.random() * 255)
                three_channel_img[row_index][col_index][2] = int(random.random() * 255)
            else:
                three_channel_img[row_index][col_index] = bgr_pixel
    print "Finished"
    return three_channel_img


def get_left_and_right_limits(edges):
    cols = edges.shape[1]
    for col in xrange(cols):
        if np.sum(edges[:, col]) != 0:
            left_x = col
            break
    for col in reversed(xrange(cols)):
        if np.sum(edges[:, col]) != 0:
            right_x = col
            break
    # this is kind of a hack because IRL case seems to be a little to past
    # these limits
    left_x *= 0.9
    right_x *= 1.1
    return left_x, right_x


def grayscale(frame):
    im = cv2.cv.fromarray(frame)
    gray = cv2.cv.CreateImage((im.width, im.height), 8, 1)
    cv2.cv.CvtColor(im, gray, cv2.cv.CV_BGR2GRAY)
    return np.asarray(gray[:, :])


def grayscale_to_color(grayscale_frame):
    color_img = cv2.cvtColor(grayscale_frame, cv2.COLOR_GRAY2BGR)
    return color_img


def get_line_angle(x1, y1, x2, y2):
    delta_y = float(y2 - y1)
    delta_x = float(x2 - x1)
    angle_degrees = math.atan2(delta_y, delta_x) * 180.0 / cv2.cv.CV_PI
    return angle_degrees


def filter_noise_from_motion_detected_frame(grayscale_frame):
    threshold_pixel = LINE_DETECTION_THRESHOLD
    frame_copy = grayscale_frame.copy()
    frame_copy[frame_copy < threshold_pixel] = 0
    return frame_copy


def get_average_angle_from_lines(lines):
    # get average angle of all the lines
    if len(lines) == 0:
        return 0.0
    angle_sum = 0.0
    for x1, y1, x2, y2 in lines:
        angle_degrees = get_line_angle(x1, y2, x2, y2)
        angle_sum += angle_degrees
    average_angle = angle_sum / len(lines)
    return average_angle


def rotate_about_axis_by_angle(lines, angle_degrees):
    if len(lines) == 0:
        return []
    angle_radians = angle_degrees * math.pi / 180.0
    cosa = math.cos(angle_radians)
    sina = math.sin(angle_radians)

    new_lines = []
    for x1, y1, x2, y2 in lines:
        new_x1 = int(x1 * cosa - y1 * sina)
        new_x2 = int(x2 * cosa - y2 * sina)
        new_y1 = int(x1 * sina + y1 * cosa)
        new_y2 = int(x2 * sina + y2 * cosa)
        new_lines.append((new_x1, new_y1, new_x2, new_y2))
    return new_lines


def rotate_point_about_axis_by_angle(point, angle_degrees):
    angle_radians = angle_degrees * math.pi / 180.0
    cosa = math.cos(angle_radians)
    sina = math.sin(angle_radians)
    new_x = int(point[0] * cosa - point[1] * sina)
    new_y = int(point[0] * sina + point[1] * cosa)
    return new_x, new_y


def filter_above_standard_deviation(lines):
    times_above_std_dev_to_filter = 2
    # we only need to filter out the vertical component
    y_component1 = [tuple_obj[1] for tuple_obj in lines]
    y_component2 = [tuple_obj[3] for tuple_obj in lines]
    all_y = y_component1 + y_component2
    np_array = np.asarray(all_y)
    std_dev = np.std(np_array)
    filtered_lines = []
    mean = np.mean(all_y)
    for x1, y1, x2, y2 in lines:
        if abs(y1 - mean) > times_above_std_dev_to_filter * std_dev or abs(y2 - mean) > times_above_std_dev_to_filter * std_dev:
            continue
        filtered_lines.append((x1, y1, x2, y2))
    return filtered_lines


def get_width_from_points(top_left, top_right, bottom_right, bottom_left):
    left_x = min(top_left[0], bottom_left[0])
    right_x = min(bottom_right[0], top_right[0])
    return right_x - left_x


def get_center_from_points(top_left, top_right, bottom_right, bottom_left):
    x1 = top_left[0]
    y1 = top_left[1]
    x2 = bottom_right[0]
    y2 = bottom_right[1]
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def rotate_perpendicular(img, orientation_id):
    orientation_to_angle = {
        Orientation.NONE: 0,
        Orientation.ROTATE_90: 270,  # FIXME kind of a hack for now, I dunno what I did wrong
        Orientation.ROTATE_180: 180,
        Orientation.ROTATE_270: 90
    }
    angle = orientation_to_angle[orientation_id]

    rows, cols = img.shape[:2]
    expanded_rows = max(rows, cols)
    expanded_cols = max(rows, cols)
    temp_img = np.zeros((expanded_rows, expanded_cols, img.shape[2]), dtype=np.uint8)
    offset_rows = (expanded_rows - rows) / 2
    offset_cols = (expanded_cols - cols) / 2
    temp_img[offset_rows: offset_rows + rows, offset_cols: offset_cols + cols] = img

    rotation_matrix = cv2.getRotationMatrix2D((expanded_cols / 2, expanded_rows / 2), angle, 1)
    square_img = cv2.warpAffine(temp_img, rotation_matrix, (expanded_cols, expanded_rows))

    if orientation_id in (Orientation.ROTATE_90, Orientation.ROTATE_270):
        temp = offset_rows
        offset_rows = offset_cols
        offset_cols = temp

    length = square_img.shape[0]

    return square_img[offset_rows: length - offset_rows, offset_cols: length - offset_cols]


def rotate_image_to_scaled_canvas(img, angle):
    angle_radians = angle * math.pi / 180.0
    sina = abs(math.sin(angle_radians))
    cosa = abs(math.cos(angle_radians))

    rows, cols = img.shape[0:2]

    new_cols = int(cosa * cols + rows * sina)
    new_rows = int(cosa * rows + sina * cols)
    shape = list(img.shape)
    shape[0] = max(new_rows, rows)
    shape[1] = max(new_cols, cols)

    temp_img = np.zeros(shape, dtype=np.uint8)
    offset_rows = max(0, (new_rows - rows) / 2)
    offset_cols = max(0, (new_cols - cols) / 2)

    center = (temp_img.shape[1] / 2, temp_img.shape[0] / 2)
    rotated_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    temp_img[offset_rows: offset_rows + img.shape[0], offset_cols: offset_cols + img.shape[1]] = img
    return cv2.warpAffine(temp_img, rotated_matrix, (new_cols, new_rows), flags=cv2.INTER_LINEAR)


def _get_impact(x_offset, y_offset, edges, overlay_height, overlay_width, angled_overlay, show_output):
    num_pixels_for_motion = 10  # totally arbitrary right now
    pixel_sum_before_overlay = np.sum(edges[y_offset: y_offset + overlay_height, x_offset: x_offset + overlay_width])
    if pixel_sum_before_overlay < num_pixels_for_motion:
        return None
    grayscale_with_overlay = edges.copy()

    # this works because white and black have identical BGR
    # channels, so index 0 works fine
    masked_overlay = (angled_overlay[:, :, 0] * (angled_overlay[:, :, 3] / 255.0)
        + edges[y_offset: y_offset
        + angled_overlay.shape[0], x_offset: x_offset
        + angled_overlay.shape[1]]
        * (1.0 - angled_overlay[:, :, 3] / 255.0))
    grayscale_with_overlay[y_offset: y_offset + angled_overlay.shape[0], x_offset: x_offset + angled_overlay.shape[1]] = masked_overlay

    mat_before_overlay = edges[y_offset: y_offset + overlay_height, x_offset: x_offset + overlay_width]
    mat_after_overlay = grayscale_with_overlay[y_offset: y_offset + overlay_height, x_offset: x_offset + overlay_width]

    # we want the LEAST impact...

    # difference is nothing but an array of 255s each case
    total_before_overlay = np.sum(mat_before_overlay)
    if total_before_overlay == 0:
        return None

    impact = float(np.sum(mat_after_overlay)) / total_before_overlay
    num_non_transparent_pixels = np.sum(angled_overlay[:, :, 3]) / 255.0
    impact = impact / num_non_transparent_pixels
    '''
    impact = float(impact) / 255.0
    '''
    # impact = impact / (overlay_width + overlay_height)
    # initial impact represents the total number of affected
    # pixels
    if show_output and LOCAL:
        cv2.imshow("Barbell Finder", grayscale_with_overlay)
        cv2.waitKey(1)
    return impact


def _out_of_bounds(center_motion_detected_bar, x_offset, y_offset, overlay_width, overlay_height):
    center_x = center_motion_detected_bar[0]
    center_y = center_motion_detected_bar[1]
    if center_x < x_offset or center_x > x_offset + overlay_width:
        return True
    if center_y < y_offset or center_y > y_offset + overlay_height:
        return True
    return False


def get_best_bar_size_position_angle(olympic_bar,
                                     edges,
                                     min_bar_size,
                                     center_motion_detected_bar,
                                     left_x_limit,
                                     right_x_limit,
                                     show_output=False,
                                     max_bar_size=None):
    best_bar_width_to_frames_held = defaultdict(int)
    deleteme_counter = 0
    height, width = edges.shape[0: 2]
    best_impact = sys.maxint
    min_width_threshold = int(width * MIN_BAR_WIDTH_AS_PERCENT_OF_SCREEN)
    min_width = int(max(min_bar_size, min_width_threshold))
    max_bar_size = max_bar_size or width
    for bar_width in reversed(xrange(min_width, max_bar_size + 1)):
        bar_overlay = olympic_bar.with_width(bar_width)
        if bar_overlay is None:
            continue
        for angle in xrange(0, 1):
            angled_overlay = bar_overlay
            # angled_overlay = rotate_image_to_scaled_canvas(bar_overlay, angle)
            overlay_height, overlay_width = angled_overlay.shape[0: 2]
            lower_x = 0
            upper_x = width - overlay_width
            center = (upper_x + lower_x) / 2
            percent_offset = 0.80  # this is actually useless now, revisit
            for x_offset in xrange(int(center - center * percent_offset), int(center + center * percent_offset)):
                if x_offset < left_x_limit:
                    continue
                if x_offset + overlay_width > right_x_limit:
                    continue
                for y_offset in xrange(0, height - overlay_height):
                    deleteme_counter += 1
                    if _out_of_bounds(center_motion_detected_bar, x_offset, y_offset, overlay_width, overlay_height):
                        continue

                    impact = _get_impact(x_offset, y_offset, edges, overlay_height, overlay_width, angled_overlay, show_output)
                    if impact is None:
                        continue

                    if impact < best_impact:  # and bar_width != min_width_threshold:
                        best_impact = impact
                        best_x_offset = x_offset
                        best_y_offset = y_offset
                        best_angle = angle
                        best_bar_width = bar_width
                    else:
                        if "best_bar_width" in locals().keys():
                            key = (best_impact, best_bar_width, best_x_offset, best_y_offset, best_angle)
                            best_bar_width_to_frames_held[key] += 1
    if max_bar_size == min_bar_size or True:  # SBL attempting this!
        # don't care about frames held, just get best result, so find the
        # minimum bar size
        return best_impact, best_bar_width, best_x_offset, best_y_offset, best_angle, 0
    else:
        max_frames_held = 0
        for return_values, frames_held in best_bar_width_to_frames_held.items():
            if frames_held > max_frames_held:
                max_frames_held = frames_held
                best_return_value = return_values + (frames_held,)

    print "Returning %s with %s frames held" % (best_return_value[1], max_frames_held)
    return best_return_value


def resized_frame(frame):
    height, width = frame.shape[0: 2]
    desired_width = RESIZE_WIDTH
    desired_to_actual = float(desired_width) / width
    new_width = int(width * desired_to_actual)
    new_height = int(height * desired_to_actual)
    return cv2.resize(frame, (new_width, new_height))


class BarbellDetection(object):

    def __init__(self, frame_number, impact_score, barbell_width, offset_x, offset_y, angle, frames_held=None):
        self.frame_number = frame_number
        self.impact_score = impact_score
        self.barbell_width = barbell_width
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.angle = angle
        self.frames_held = frames_held or 0

        self.barbell_pixels = None
        self.symmetry_score = 9999
        self.likeness_score = 9999
        self.original_bar_width = barbell_width

    @property
    def total_score(self):
        return self.symmetry_score * self.likeness_score * self.impact_score

    def to_json(self):
        return {
            "frame_number": self.frame_number,
            "impact_score":  self.impact_score,
            "barbell_width": self.barbell_width,
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "angle": self.angle,
            "frames_held": self.frames_held or "",
        }

    @classmethod
    def from_json(self, json_dict):
        detection = BarbellDetection(json_dict["frame_number"],
                                     json_dict["impact_score"],
                                     json_dict["barbell_width"],
                                     json_dict["offset_x"],
                                     json_dict["offset_y"],
                                     json_dict["angle"],
                                     frames_held=json_dict["frames_held"] or 0)
        return detection


class BarbellDetector(object):

    olympic_bar = OlympicBarbell()
    show_output = True
    orientation_id = Orientation.NONE

    start_seconds = None
    stop_seconds = None

    def frame_number_to_seconds(self, frame_number):
        return float(frame_number) / self.frames_per_sec

    def __init__(self, capture):
        self.capture = capture
        self.barbell_detections = []
        self.frame_number = 0
        self.frames_per_sec = capture.get(FRAMES_PER_SEC_KEY)
        self.min_bar_size = None
        self.max_bar_size = None

        self.min_x = None
        self.max_x = None

        self.frame_set = None

    def with_bar_size(self, bar_size):
        self.min_bar_size = bar_size
        self.max_bar_size = bar_size
        return self

    def with_frame_num_limits(self, frame_set):
        self.frame_set = frame_set
        return self

    def with_min_and_max_x(self, min_x, max_x):
        self.min_x = min_x
        self.max_x = max_x
        return self

    def get_barbell_frame_data(self):
        self.frame_number = 0
        self.process_capture()
        return self.barbell_detections

    def process_capture(self):
        previous_frame = None
        previous_previous_frame = None
        while True:
            success, haystack = self.capture.read()
            if not success:
                break
            self.frame_number += 1
            if self.frame_number % 10 == 0:
                print "Processing frame %s" % self.frame_number

            current_seconds = self.frame_number_to_seconds(self.frame_number)
            if self.start_seconds is not None and current_seconds < self.start_seconds:
                continue
            if self.stop_seconds is not None and current_seconds > self.stop_seconds:
                break

            haystack = resized_frame(haystack)
            haystack = rotate_perpendicular(haystack, self.orientation_id)

            frame = haystack.copy()
            motion_detection_frame = None
            if previous_previous_frame is not None:
                motion_detection_frame = self._get_motion_detection_frame(previous_previous_frame, previous_frame, frame)
            previous_previous_frame = previous_frame
            previous_frame = frame

            if self.frame_set is not None and self.frame_number not in self.frame_set:
                continue

            if motion_detection_frame is not None:
                self.process_motion_detection_frame(motion_detection_frame, previous_previous_frame.copy())

            if 0xFF & cv2.waitKey(5) == 27:
                break

    def _get_motion_detection_frame(self, previous_previous_frame, previous_frame, frame):
        d1 = cv2.absdiff(frame, previous_frame)
        d2 = cv2.absdiff(previous_frame, previous_previous_frame)
        motion_detection_frame = cv2.bitwise_xor(d1, d2)
        return motion_detection_frame

    def process_motion_detection_frame(self, motion_detection_frame, draw_img):
        edges, lines = self._get_edge_frame_and_lines_from_motion_detection(motion_detection_frame)

        grayscale_motion_detection = grayscale(motion_detection_frame)

        # note that raise this too high might erase the image and create bugs

        # best_row = self._get_most_populated_row(grayscale_motion_detection)
        threshold_pixel = MOTION_THRESHOLD
        grayscale_motion_detection[grayscale_motion_detection < threshold_pixel] = 0
        grayscale_motion_detection[grayscale_motion_detection >= threshold_pixel] = 255

        best_row = self._get_most_populated_row(grayscale_motion_detection)
        # best_row2 = self._get_best_object_motion_row(grayscale_motion_detection)
        _, width = grayscale_motion_detection.shape[0: 2]
        cv2.circle(grayscale_motion_detection, (width / 2, best_row), 50, 200)

        if lines is not None:
            left_x, right_x = get_left_and_right_limits(edges)
            lines = lines[0]
            lines = self._get_filtered_lines_from_raw_lines(lines)

            # SBL TODO MAKE A BETTER THRESHOLD THAN THIS
            if len(lines) > 8:
                points = self.get_corners_from_lines(lines)
                points = self.make_top_lines_parallel(*points)
                self.draw_points_on_draw_img(draw_img, points)
                # center_point = get_center_from_points(*points)
                center_point = (width / 2, best_row)
                # min_width = get_width_from_points(*points)
                min_width = int(width * MIN_BAR_WIDTH_AS_PERCENT_OF_SCREEN)
                try:
                    self.find_bar_from_edges_and_line_heuristics(grayscale_motion_detection, center_point, min_width, left_x, right_x)
                except UnboundLocalError as e:
                    print e
                    print "Bypassing frame"
                    # dirty, dirty hack to just ignore bad frames
                    pass

        if self.show_output and LOCAL:
            cv2.imshow('some', draw_img)

    def _get_most_populated_row(self, grayscale_motion_detection_frame):
        rows, cols = grayscale_motion_detection_frame.shape[0: 2]
        best_row = 0
        max_pixels = 0
        for row in xrange(rows):
            white_pixels_this_row = np.sum(grayscale_motion_detection_frame[row, :])
            if white_pixels_this_row > max_pixels:
                best_row = row
                max_pixels = white_pixels_this_row
        return best_row

    def _get_best_object_motion_row(self, motion_matrix):
        '''
        Sum each row, square that value
        '''
        rows, cols = motion_matrix.shape[0: 2]
        row_scores = np.zeros(rows)
        for row_index in xrange(rows):
            row_array = motion_matrix[row_index, :]
            motion_score = np.sum(row_array) ** 2
            differential_score = np.std(np.abs(np.diff(row_array.astype(int))))
            row_scores[row_index] = motion_score / differential_score
        return row_scores.argmax()

    def _get_edge_frame_and_lines_from_motion_detection(self, motion_detection_frame):
        gray_frame = grayscale(motion_detection_frame)
        gray_frame = filter_noise_from_motion_detected_frame(gray_frame)
        edges = cv2.Canny(gray_frame, threshold1=80, threshold2=80, apertureSize=3)
        min_line_length = 15
        max_line_gap = 3
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 1, None, min_line_length, max_line_gap)  # minLineLength, maxLineGap)
        return edges, lines

    def _get_filtered_lines_from_raw_lines(self, lines):
        angle_filtered_lines = []
        for x1, y1, x2, y2 in lines:
            angle_degrees = get_line_angle(x1, y1, x2, y2)
            if abs(angle_degrees) <= 5:
                angle_filtered_lines.append((x1, y1, x2, y2))

        average_angle = get_average_angle_from_lines(angle_filtered_lines)
        rotated_lines = rotate_about_axis_by_angle(angle_filtered_lines, average_angle)
        within_std_dev_lines = filter_above_standard_deviation(rotated_lines)
        lines = rotate_about_axis_by_angle(within_std_dev_lines, -1 * average_angle)
        return lines

    def get_corners_from_lines(self, lines):
        if len(lines) == 0:
            return []

        top_left = (999999999, 99999999)
        top_right = (-9999999, 0)
        bottom_left = (99999999, 999999)
        bottom_right = (-999999, 0)

        # left most point gives top left
        # right most point gives bottom right
        rotated_upward = rotate_about_axis_by_angle(lines, -45)
        for index, line in enumerate(rotated_upward):
            pt1 = (line[0], line[1])
            pt2 = (line[2], line[3])
            for point in (pt1, pt2):
                if point[0] < top_left[0]:
                    top_left = point
                    top_left_ret = rotate_point_about_axis_by_angle(point, 45)
                if point[0] > bottom_right[0]:
                    bottom_right = point
                    bottom_right_ret = rotate_point_about_axis_by_angle(point, 45)

        # left most point gives bottom left
        # right most point gives top right
        rotated_downward = rotate_about_axis_by_angle(lines, 45)
        for index, line in enumerate(rotated_downward):
            pt1 = (line[0], line[1])
            pt2 = (line[2], line[3])
            for point in (pt1, pt2):
                if point[0] < bottom_left[0]:
                    bottom_left = point
                    bottom_left_ret = rotate_point_about_axis_by_angle(point, -45)
                if point[0] > top_right[0]:
                    top_right = point
                    top_right_ret = rotate_point_about_axis_by_angle(point, -45)
        return top_left_ret, top_right_ret, bottom_right_ret, bottom_left_ret

    def make_top_lines_parallel(self, top_left, top_right, bottom_right, bottom_left):
        delta_y_top = top_right[1] - top_left[1]
        delta_x_top = top_right[0] - top_left[0]
        top_slope = float(delta_y_top) / delta_x_top

        delta_y_bottom = bottom_right[1] - bottom_left[1]
        delta_x_bottom = bottom_right[0] - bottom_left[0]
        bottom_slope = float(delta_y_bottom) / delta_x_bottom

        better_slope = min(top_slope, bottom_slope)
        delta_x = max(delta_x_bottom, delta_x_top)
        new_top_right = (int(top_left[0] + delta_x), int(top_left[1] + better_slope * delta_x))
        new_bottom_right = (int(bottom_left[0] + delta_x), int(bottom_left[1] + better_slope * delta_x))
        return top_left, new_top_right, new_bottom_right, bottom_left

    def draw_points_on_draw_img(self, draw_img, points):
        for index in xrange(len(points)):
            next_i = index + 1
            if next_i == len(points):
                next_i = 0
            pt1 = points[index]
            pt2 = points[next_i]
            cv2.line(draw_img, pt1, pt2, (0, 255, 0), 1)

    def find_bar_from_edges_and_line_heuristics(self, edges, center_point, min_width, left_x, right_x):
        min_width = self.min_bar_size or min_width
        left_x = self.min_x or left_x
        right_x = self.max_x or right_x
        best_impact, barbell_width, best_x_offset, best_y_offset, best_angle, frames_held = get_best_bar_size_position_angle(self.olympic_bar,
                                                                                                                            edges,
                                                                                                                            min_width,
                                                                                                                            center_point,
                                                                                                                            left_x,
                                                                                                                            right_x,
                                                                                                                            show_output=self.show_output,
                                                                                                                            max_bar_size=self.max_bar_size)
        frame_number_for_previous_previous_frame = self.frame_number - ACTUAL_FRAME_OFFSET
        barbell_detection = BarbellDetection(frame_number_for_previous_previous_frame, best_impact, barbell_width, best_x_offset, best_y_offset, best_angle, frames_held=frames_held)
        self.barbell_detections.append(barbell_detection)


class BarbellDisplayer(object):

    olympic_bar = OlympicBarbell(for_display=True)
    negative_olympic_bar = OlympicBarbell(for_display=True, for_negative=True)
    orientation_id = Orientation.NONE

    start_seconds = None
    stop_seconds = None

    def __init__(self,
                 capture,
                 detected_barbells,
                 frame_to_1rm):
        self.capture = capture
        self.detected_barbells = detected_barbells
        self.frame_number = 0
        self.frame_to_detected_barbell = {barbell.frame_number: barbell for barbell in detected_barbells}
        self.video_writer = None

        self.frames_per_sec = capture.get(FRAMES_PER_SEC_KEY)
        self.total_frames = capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

        self.frame_to_1rm = frame_to_1rm
        self.barbell_width = detected_barbells[0].barbell_width
        self.barbell_height = self.olympic_bar.with_width(self.barbell_width).shape[0]

    def frame_number_to_seconds(self, frame_number):
        return float(frame_number) / self.frames_per_sec

    def output_video(self, filename):
        print "Creating Video..."
        while True:
            success, haystack = self.capture.read()
            if not success:
                break
            self.frame_number += 1
            current_seconds = self.frame_number_to_seconds(self.frame_number)
            if self.start_seconds is not None and current_seconds < self.start_seconds:
                continue
            if self.stop_seconds is not None and current_seconds > self.stop_seconds:
                break
            haystack = resized_frame(haystack)
            haystack = rotate_perpendicular(haystack, self.orientation_id)

            height, width = haystack.shape[0: 2]
            codec = cv2.cv.FOURCC('M', 'J', 'P', 'G')
            self.video_writer = self.video_writer or cv2.VideoWriter(filename, codec, self.frames_per_sec, (width, height))

            # wait_key_time = 5
            if self.frame_number in self.frame_to_detected_barbell:
                barbell_detection = self.frame_to_detected_barbell[self.frame_number]
                self.display_barbell_on_img(haystack, barbell_detection)
            self.draw_metadata_on_canvas(haystack)
            self.draw_percent_error_notches(haystack)
                # wait_key_time = 0
            self.video_writer.write(haystack)
            # cv2.imshow("Final output", haystack)
            # cv2.waitKey(wait_key_time)

        if self.video_writer is not None:
            self.video_writer.release()
        print "Done"

    def draw_percent_error_notches(self, frame):
        if self.frame_number not in self.frame_to_detected_barbell:
            return
        barbell_detection = self.frame_to_detected_barbell[self.frame_number]
        y_offset = barbell_detection.offset_y
        x_offset = barbell_detection.offset_x

        center_x = int(x_offset + (self.barbell_width / 2))
        center_y = int(y_offset + (self.barbell_height / 2))

        line_height = 30

        brush_size = 1

        for percent in (0.9, 1.1):
            if percent > 1.0:
                pixel_color = (0, 0, 255)
            else:
                pixel_color = (255, 0, 0)
            display_percent = "%s%%" % abs(int(float("%.2f" % (percent - 1.0)) * 100))
            if percent >= 1.0:
                display_percent = "-" + display_percent
            else:
                display_percent = "+" + display_percent

            cv2.line(frame, (center_x + int(percent * self.barbell_width / 2.0), center_y - line_height / 2), (center_x + int(percent * self.barbell_width / 2.0), center_y + line_height / 2), pixel_color, brush_size)
            cv2.line(frame, (center_x - int(percent * self.barbell_width / 2.0), center_y - line_height / 2), (center_x - int(percent * self.barbell_width / 2.0), center_y + line_height / 2), pixel_color, brush_size)

            font_scale = 0.4
            font_color = (255, 255, 255, 255)
            font = cv2.FONT_ITALIC
            opaque_black_pixel = (0, 0, 0, 150)
            rectangle_height = 25
            rectangle_width = 40
            overlay = np.zeros((rectangle_height, rectangle_width, 4), dtype=np.uint8)
            overlay[:, :] = opaque_black_pixel
            cv2.putText(overlay, display_percent, (0, 2 * rectangle_height / 3), font, font_scale, font_color)

            if percent < 1.0:
                y_offset = center_y - int(percent * self.barbell_height / 2.0) - rectangle_height - line_height / 2
            else:
                y_offset = center_y + int(percent * self.barbell_height / 2.0) + line_height / 2  # - rectangle_height - line_height / 2
            x_offset1 = center_x + int(percent * self.barbell_width / 2.0) - rectangle_width / 2
            x_offset2 = center_x - int(percent * self.barbell_width / 2.0) - rectangle_width / 2

            for x_offset in (x_offset1, x_offset2):
                if x_offset + rectangle_width < 0:
                    continue
                if x_offset < 0:
                    x_offset = 0
                if x_offset > frame.shape[1]:
                    continue
                if x_offset + rectangle_width > frame.shape[1]:
                    x_offset = frame.shape[1] - rectangle_width

                try:
                    for col in xrange(3):
                        masked_overlay = (overlay[:, :, col] * (overlay[:, :, 3] / 255.0)
                            + frame[y_offset: y_offset
                            + overlay.shape[0], x_offset: x_offset
                            + overlay.shape[1], col]
                            * (1.0 - overlay[:, :, 3] / 255.0))
                        frame[y_offset: y_offset + overlay.shape[0], x_offset: x_offset + overlay.shape[1], col] = masked_overlay
                except ValueError:
                    print "WARNING: Ignoring a frame because of broadcast shapes"
                    continue

    def draw_metadata_on_canvas(self, draw_img):
        height, width, channels = draw_img.shape[0: 3]
        # SBL working here
        opaque_black_pixel = (0, 0, 0, 150)
        percent_img_width = 0.35
        percent_img_height = 0.15

        left_x = int(width * 0.05)
        top_y = int(height * 0.05)
        right_x = int(percent_img_width * width + left_x)
        bottom_y = int(percent_img_height * height + top_y)

        rectangle_width = right_x - left_x
        rectangle_height = bottom_y - top_y

        overlay = np.zeros((rectangle_height, rectangle_width, 4), dtype=np.uint8)
        overlay[:, :] = opaque_black_pixel
        font_scale = 0.4
        font_color = (255, 255, 255, 255)
        font = cv2.FONT_ITALIC
        cv2.putText(overlay, "1RM Multiplier:", (5, 25), font, font_scale, font_color)

        acceleration_color = (0, 255, 0, 255)

        scalar_str = ""
        if self.frame_number in self.frame_to_1rm and self.frame_to_1rm[self.frame_number] > 0:
            scalar_str = "%.3f" % self.frame_to_1rm[self.frame_number]
        cv2.putText(overlay, scalar_str, (110, 25), font, font_scale, acceleration_color)

        y_offset = top_y
        x_offset = left_x
        for col in xrange(3):
            masked_overlay = (overlay[:, :, col] * (overlay[:, :, 3] / 255.0)
                + draw_img[y_offset: y_offset
                + overlay.shape[0], x_offset: x_offset
                + overlay.shape[1], col]
                * (1.0 - overlay[:, :, 3] / 255.0))
            draw_img[y_offset: y_offset + overlay.shape[0], x_offset: x_offset + overlay.shape[1], col] = masked_overlay
        return draw_img

    def display_barbell_on_img(self, draw_img, barbell_detection):
        bar_width = barbell_detection.barbell_width
        x_offset = barbell_detection.offset_x
        y_offset = barbell_detection.offset_y
        angle = barbell_detection.angle
        if self.frame_to_1rm.get(self.frame_number, 1) > 0:
            bar_overlay = self.olympic_bar.with_width(bar_width)
        else:
            bar_overlay = self.negative_olympic_bar.with_width(bar_width)

        angled_overlay = rotate_image_to_scaled_canvas(bar_overlay, angle)
        for col in xrange(3):
            masked_overlay = (angled_overlay[:, :, col] * (angled_overlay[:, :, 3] / 255.0)
                + draw_img[y_offset: y_offset
                + angled_overlay.shape[0], x_offset: x_offset
                + angled_overlay.shape[1], col]
                * (1.0 - angled_overlay[:, :, 3] / 255.0))
            draw_img[y_offset: y_offset + angled_overlay.shape[0], x_offset: x_offset + angled_overlay.shape[1], col] = masked_overlay


def filter_barbells_by_y_values(detected_barbells):
    y_offsets = np.asarray([bar.offset_y for bar in detected_barbells])
    std_dev = np.std(y_offsets)
    mean = np.mean(y_offsets)
    before = len(detected_barbells)
    detected_barbells = [bar for bar in detected_barbells if abs(bar.offset_y - mean) < 3 * std_dev]
    after = len(detected_barbells)
    print "Discard %s barbells based on Y values" % (before - after)
    return detected_barbells


def filter_by_bar_width(detected_barbells):
    ''' barbells will be returned with fixed barbell width; anamolies will be discarded
    based on 3 sigma rule'''
    barbell_widths = np.asarray([bar.barbell_width for bar in detected_barbells])
    std_dev = np.std(barbell_widths)
    mean = np.mean(barbell_widths)

    detected_barbells = [bar for bar in detected_barbells if abs(bar.barbell_width - mean) < 3 * std_dev]
    return detected_barbells


def set_mean_for_barbells(detected_barbells, mean_value=None):
    final_mean = int(np.mean(np.asarray([bar.barbell_width for bar in detected_barbells])))
    final_mean = mean_value or final_mean

    olympic_barbell = OlympicBarbell()
    while True:
        # SBL final_mean is NaN
        valid_or_none = olympic_barbell.with_width(final_mean)
        if valid_or_none is not None:
            break
        final_mean += 1

    for bar in detected_barbells:
        bar.barbell_width = int(final_mean)

    return detected_barbells


def set_bar_offsets_by_average_x(detected_barbells):
    # redo the X offset
    x_offsets = np.asarray([bar.offset_x for bar in detected_barbells])
    avg_x = np.mean(x_offsets)
    for bar in detected_barbells:
        bar.offset_x = avg_x
    return detected_barbells


def fill_detected_barbells_with_interpolations(detected_barbells):
    if len(detected_barbells) == 0:
        return []

    frame_to_detected_barbell = {barbell.frame_number: barbell for barbell in detected_barbells}
    frame_numbers = sorted(frame_to_detected_barbell.keys())

    frames_to_add = []
    for frame_number in range(min(frame_numbers), max(frame_numbers) + 1):
        if frame_number not in frame_numbers:
            frames_to_add.append(frame_number)

    chunked_frames = []
    frame_buffer = []
    previous_frame_number = min(frames_to_add) - 1
    for frame_number in frames_to_add:
        if frame_number - previous_frame_number == 1:
            frame_buffer.append(frame_number)
        else:
            chunked_frames.append(frame_buffer)
            frame_buffer = [frame_number]
        previous_frame_number = frame_number

    if len(frame_buffer) != 0:
        chunked_frames.append(frame_buffer)

    x_offset = frame_to_detected_barbell.values()[0].offset_x
    angle = 0
    barbell_width = frame_to_detected_barbell.values()[0].barbell_width

    for empty_frame_list in chunked_frames:
        frames_to_interpolate = len(empty_frame_list)
        previous_y = frame_to_detected_barbell[min(empty_frame_list) - 1].offset_y
        next_y = frame_to_detected_barbell[max(empty_frame_list) + 1].offset_y
        for index, empty_frame_number in enumerate(empty_frame_list):
            interpolated_y = min(next_y, previous_y) + (index + 1) * abs(next_y - previous_y) / (1 + frames_to_interpolate)
            fake_barbell_detection = BarbellDetection(empty_frame_number, 9999,
                                                      barbell_width,
                                                      x_offset,
                                                      interpolated_y,
                                                      angle)
            detected_barbells.append(fake_barbell_detection)
    return detected_barbells


def filter_smaller_barbells(detected_barbells):
    if len(detected_barbells) == 0:
        return []
    # barbells may only be X% smaller than the next larger.  This should
    # discard super small results...or it may fuck up everything, add a case to
    # prevent that
    percent_threshold = MAX_PERCENT_DIFFERENCE_BETWEEN_BARS
    detected_barbells = sorted(detected_barbells, key=lambda barbell: barbell.barbell_width)
    detected_barbells.reverse()

    previous_width = detected_barbells[0].barbell_width
    barbells_to_keep = []
    for detected_barbell in detected_barbells:
        if detected_barbell.barbell_width < (1.0 - percent_threshold) * previous_width:
            break
        barbells_to_keep.append(detected_barbell)
        previous_width = detected_barbell.barbell_width
    if len(barbells_to_keep) >= len(detected_barbells) / 2:
        # Don't discard if data appears to be really bad
        print "Discarding %s results from bar width filter" % (len(detected_barbells) - len(barbells_to_keep))
        return barbells_to_keep
    return detected_barbells


def dump_detected_barbells_to_json(detected_barbells, video_filename):
    json_str = json.dumps([barbell.to_json() for barbell in detected_barbells], indent=4)
    filename = "json_data/barbell_detections_%s.json" % video_filename
    with open(filename, "w+") as f:
        f.write(json_str)


def get_barbell_pixels_from_frame(bgr_frame, detected_barbell, barbell_overlay):
    template = barbell_overlay.copy()
    # transparency is already taken care of
    y_offset = detected_barbell.offset_y
    x_offset = detected_barbell.offset_x
    bar_height, bar_width = template.shape[0: 2]
    for col in xrange(3):
        bgr_clip = bgr_frame[y_offset: y_offset + bar_height, x_offset: x_offset + bar_width, col]
        bgr_clip_with_opacity = bgr_clip * (template[:, :, 3] / 255.0)
        template[:, :, col] = bgr_clip_with_opacity
    return template


def amend_barbell_pixels_to_barbell_detections(capture, detected_barbells, orientation_id):
    olympic_barbell = OlympicBarbell(force_fill=True)
    overlay = olympic_barbell.with_width(detected_barbells[0].barbell_width)
    frame_to_detected_barbell = {barbell.frame_number: barbell for barbell in detected_barbells}
    frame_number = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        frame = resized_frame(frame)
        frame = rotate_perpendicular(frame, orientation_id)
        frame_number += 1
        if frame_number not in frame_to_detected_barbell:
            continue
        detected_barbell = frame_to_detected_barbell[frame_number]
        barbell_pixels = get_barbell_pixels_from_frame(frame, detected_barbell, overlay)
        detected_barbell.barbell_pixels = barbell_pixels


def amend_likeness_score_to_barbell_detections(detected_barbells):
    print "Amending likeness score...."
    width, height = detected_barbells[0].barbell_pixels.shape[0: 2]
    for i1, barbell_detection1 in enumerate(detected_barbells):
        total_this_barbell = 0
        for i2, barbell_detection2 in enumerate(detected_barbells):
            if i1 == i2:
                continue
            pixels1 = barbell_detection1.barbell_pixels
            pixels2 = barbell_detection2.barbell_pixels
            error = cv2.norm(pixels1, pixels2)
            similarity = error / (width * height)
            total_this_barbell += similarity
        barbell_detection1.likeness_score = total_this_barbell / len(detected_barbells)
    print "Done."


def amend_symmetry_score_to_barbell_detections(detected_barbells):
    print "Amending symmetry score..."
    for barbell_detection in detected_barbells:
        barbell_pixels = barbell_detection.barbell_pixels
        symmetry_score = get_symmetry_of_img(barbell_pixels)
        barbell_detection.symmetry_score = symmetry_score
    print "Done"


def get_similarity_between_images(img1, img2, total_pixels_compared):
    error = cv2.norm(img1, img2)
    similarity = error / total_pixels_compared
    return similarity


def get_symmetry_of_img(img):
    height, width = img.shape[0: 2]
    left_side_img = img[:, 0: width / 2]

    offset = 0
    if width % 2 == 1:
        offset = 1

    right_side_img = img[:, width / 2 + offset: width]
    mirror = np.fliplr(left_side_img)

    return get_similarity_between_images(mirror, right_side_img, (width * height))


def filter_by_y_pos_and_max_rom(detected_barbells):
    barbell_width_pixels = detected_barbells[0].barbell_width
    barbell_width_meters = 2.2
    pixels_per_meter = barbell_width_pixels / barbell_width_meters
    max_rom_in_pixels = MAX_ROM_IN_METERS * pixels_per_meter

    np_y_offsets = np.asarray([barbell.offset_y for barbell in detected_barbells])
    max_delta = np.max(np_y_offsets) - np.min(np_y_offsets)
    while max_delta > max_rom_in_pixels:
        avg = np.mean(np_y_offsets)
        max_y = np.max(np_y_offsets)
        min_y = np.min(np_y_offsets)
        delta_from_min = abs(min_y - avg)
        delta_from_max = abs(max_y - avg)
        if min(delta_from_min, delta_from_max) == delta_from_min:
            # average is closer to the minimum, discard the top var
            np_y_offsets = np.asarray([item for item in np_y_offsets if item != max_y])
        else:
            # average is closer to the maximum, discard the bottom var
            np_y_offsets = np.asarray([item for item in np_y_offsets if item != min_y])
        max_delta = np.max(np_y_offsets) - np.min(np_y_offsets)
    detected_barbells = [barbell for barbell in detected_barbells if barbell.offset_y in set(np_y_offsets)]
    return detected_barbells


def get_best_start_and_end_index_from_sorted_barbells(detected_barbells):
    best_combo_score = 999999999999
    best_combo = (0, 0)
    print "STARTING THIS CRAZINESS"
    previous_best_combo_for_start = 99999999
    for possible_start in xrange(len(detected_barbells)):
        best_combo_score_this_start = 999999999
        for possible_length in xrange(4, len(detected_barbells)):
            if possible_start + possible_length >= len(detected_barbells):
                continue
            detections = detected_barbells[possible_start: possible_start + possible_length]
            score_values = [detection.total_score for detection in detections]
            std_dev = np.std(np.asarray(score_values))
            combo_score = std_dev / (len(score_values) ** 2)
            if combo_score < best_combo_score_this_start:
                best_combo_score_this_start = combo_score
            if combo_score < best_combo_score:
                best_combo_score = combo_score
                best_combo = (possible_start, possible_length)
        if best_combo_score_this_start > previous_best_combo_for_start:
            break
        previous_best_combo_for_start = best_combo_score_this_start

    start_index = best_combo[0]
    end_index = start_index + best_combo[1]
    return start_index, end_index


def get_acceleration_of_gravity_in_pixels_per_frame(barbell_width_pixels, frames_per_second):
    barbell_width_meters = 2.2
    pixels_per_meter = barbell_width_pixels / barbell_width_meters
    one_g_in_pixels_per_ss = 9.80665 * pixels_per_meter
    one_g_in_pixels_per_ff = one_g_in_pixels_per_ss / (frames_per_second ** 2)
    return one_g_in_pixels_per_ff


def filter_by_no_frame_neighbors(detected_barbells):
    frame_numbers = [barbell_detection.frame_number for barbell_detection in detected_barbells]
    avg_num_frames_between_points = int(np.ceil(abs(np.mean(np.gradient(np.asarray(frame_numbers))))))
    frame_set = set(frame_numbers)
    barbells_to_keep = []
    for barbell_detection in detected_barbells:
        for offset in xrange(1, avg_num_frames_between_points + 1):
            if barbell_detection.frame_number + offset in frame_set or barbell_detection.frame_number - offset in frame_set:
                barbells_to_keep.append(barbell_detection)
                break
    return barbells_to_keep


def filter_barbells_with_original_video(capture_path, detected_barbells, orientation_id):
    capture = cv2.VideoCapture(capture_path)
    amend_barbell_pixels_to_barbell_detections(capture, detected_barbells, orientation_id)
    amend_likeness_score_to_barbell_detections(detected_barbells)
    # TODO consider adding a likeness score with the average detections...
    amend_symmetry_score_to_barbell_detections(detected_barbells)
    detected_barbells = sorted(detected_barbells, key=lambda detection: detection.total_score)

    start_index, end_index = get_best_start_and_end_index_from_sorted_barbells(detected_barbells)
    detected_barbells = detected_barbells[start_index: end_index]
    return detected_barbells


def _get_x__barbell_width(file_to_read):
    '''
    this function is heavy on memory so isolating as much as possible
    '''
    motion_detection_frames = ShakyMotionDetector(file_to_read).generate_frames()
    x_offset, barbell_width = BarbellWidthFinder(motion_detection_frames).find_barbell_width()
    return x_offset, barbell_width


def run(file_to_read, orientation_id):
    video_filename = (file_to_read.split("/")[-1]).split(".")[0]
    start_time = datetime.datetime.utcnow()
    capture_path = file_to_read

    x_offset, barbell_width = _get_x__barbell_width(capture_path)
    max_x = x_offset + barbell_width + 1
    capture = cv2.VideoCapture(capture_path)
    barbell_detector = (BarbellDetector(capture).
                        with_bar_size(barbell_width).
                        with_min_and_max_x(x_offset, max_x))

    detected_barbells = barbell_detector.get_barbell_frame_data()

    detected_barbells = filter_barbells_by_y_values(detected_barbells)

    if len(detected_barbells) == 0:
        raise CouldNotDetectException("Could not detect the barbell in the image")

    capture = cv2.VideoCapture(capture_path)
    detected_barbells = filter_barbells_with_original_video(capture_path, detected_barbells, orientation_id)
    detected_barbells = filter_by_y_pos_and_max_rom(detected_barbells)
    detected_barbells = filter_by_no_frame_neighbors(detected_barbells)

    if len(detected_barbells) < MINIMUM_BAR_DETECTIONS:
        raise CouldNotDetectException("Not enough detections in the image")
    print "Fixing fluctuations..."
    detected_barbells.sort(key=lambda barbell: barbell.frame_number)
    while has_fluctuations(detected_barbells):
        fix_one_fluctuation(detected_barbells)

    print "Calculating accelerations..."
    capture = cv2.VideoCapture(capture_path)
    frames_per_second = capture.get(FRAMES_PER_SEC_KEY)
    frame_to_1rm = get_frame_to_1rm(detected_barbells, frames_per_second)

    print "Outputting video..."
    capture = cv2.VideoCapture(capture_path)
    barbell_display = BarbellDisplayer(capture,
                                       detected_barbells,
                                       frame_to_1rm)
    output_file = "output_%s.avi" % video_filename
    barbell_display.output_video(output_file)
    print "Output to %s" % output_file
    print "Finished in %s minutes" % ((datetime.datetime.utcnow() - start_time).total_seconds() / 60.0)

    cv2.destroyAllWindows()
    return output_file


def process(file_to_read, orientation_id, start_seconds, stop_seconds):
    BarbellDetector.orientation_id = orientation_id
    BarbellDisplayer.orientation_id = orientation_id
    ShakyMotionDetector.orientation_id = orientation_id

    BarbellDisplayer.start_seconds = start_seconds
    BarbellDisplayer.stop_seconds = stop_seconds

    BarbellDetector.start_seconds = start_seconds
    BarbellDetector.stop_seconds = stop_seconds

    ShakyMotionDetector.start_seconds = start_seconds
    ShakyMotionDetector.stop_seconds = stop_seconds

    final_file_path = run(file_to_read, orientation_id)
    return final_file_path


if __name__ == "__main__":
    file_to_read = sys.argv[1]
    orientation_id = Orientation.NONE
    BarbellDetector.orientation_id = orientation_id
    BarbellDisplayer.orientation_id = orientation_id

    run(file_to_read, orientation_id)
