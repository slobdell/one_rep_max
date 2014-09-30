import datetime
import json
import cv2
import math
import numpy as np
import random

FRAMES_PER_SEC_KEY = 5
MOTION_THRESHOLD = 75
MIN_BAR_WIDTH_AS_PERCENT_OF_SCREEN = 0.50
MAX_PERCENT_DIFFERENCE_BETWEEN_BARS = 0.10
MAX_ROM_IN_INCHES = 40
METERS_PER_INCH = 0.0254
MAX_ROM_IN_METERS = MAX_ROM_IN_INCHES * METERS_PER_INCH
ACTUAL_FRAME_OFFSET = 2


def determine_acceleration_values(detected_barbells, one_g):
    # this should already be done, but just being cautious
    detected_barbells.sort(key=lambda barbell: barbell.frame_number)
    x_values = np.asarray([detection.frame_number for detection in detected_barbells]).astype(np.float)
    y_values = np.asarray([detection.offset_y for detection in detected_barbells]).astype(np.float)

    min_maxima, max_maxima = find_maxima(x_values, y_values)
    point_pairs = get_accelerations_from_maxima(min_maxima, max_maxima, x_values, y_values)
    frame_to_acceleration_pixels_per_frame = get_acceleration_values_from_acceleration_points(point_pairs)
    frame_to_acceleration_in_gs = {}
    for frame_number, acceleration in frame_to_acceleration_pixels_per_frame.items():
        acceleration_in_gs = -1 * acceleration / one_g
        frame_to_acceleration_in_gs[frame_number] = acceleration_in_gs
    return frame_to_acceleration_in_gs


def _eliminate_0_derivatives(x_values, y_values):
    first_derivative = np.diff(y_values) / np.diff(x_values)
    while len(first_derivative[first_derivative == 0] > 0):
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


def get_accelerations_from_maxima(min_maxima, max_maxima, x_values, y_values):
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
    return point_pairs


def get_acceleration_values_from_acceleration_points(point_pairs):
    ''' returns a dictionary of frame numbers to acceleration in pixels per frame per frame '''
    frame_to_acceleration = {}
    for start_point, end_point in point_pairs:
        delta_time_frames = end_point[0] - start_point[0]
        delta_distance_pixels = end_point[1] - start_point[1]
        acceleration_pixels_per_ff = 2 * delta_distance_pixels / (delta_time_frames ** 2)
        for x in range(int(start_point[0]), int(end_point[0])):
            frame_to_acceleration[x] = acceleration_pixels_per_ff
    return frame_to_acceleration


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
    BRUSH_SIZE = 4

    def __init__(self):
        shape = (70, 2200, 4)
        self.make_notches_transparent = True
        self.canvas = np.zeros(shape, dtype=np.uint8)
        self.canvas[::] = self.TRANSPARENT_PIXEL
        self._draw_ends()
        if not self.make_notches_transparent:
            self._draw_notches()
        self._draw_bar()
        self._fill_with_black()
        if self.make_notches_transparent:
            self._erase_first_plate()

    def with_width(self, width):
        rows, cols = self.canvas.shape[0: 2]
        resize_factor = float(width) / cols
        canvas_copy = cv2.resize(self.canvas, (int(resize_factor * 2200), int(resize_factor * 70)))
        threshold_pixel = 50
        canvas_copy[canvas_copy > threshold_pixel] = 255
        canvas_copy[canvas_copy < threshold_pixel] = 0
        if np.sum(canvas_copy[:, :, 0]) / 255.0 < canvas_copy.shape[1]:
            return None
        return canvas_copy

    def _draw_ends(self):
        cv2.line(self.canvas, (0, 10), (415, 10), self.WHITE_PIXEL, self.BRUSH_SIZE)
        cv2.line(self.canvas, (0, 60), (415, 60), self.WHITE_PIXEL, self.BRUSH_SIZE)
        # cv2.line(self.canvas, (0, 10), (0, 60), self.WHITE_PIXEL, self.BRUSH_SIZE)

        cv2.line(self.canvas, (2200, 10), (2200 - 415, 10), self.WHITE_PIXEL, self.BRUSH_SIZE)
        cv2.line(self.canvas, (2200, 60), (2200 - 415, 60), self.WHITE_PIXEL, self.BRUSH_SIZE)
        # cv2.line(self.canvas, (2200, 10), (2200, 60), self.WHITE_PIXEL, self.BRUSH_SIZE)

    def _draw_notches(self):
        cv2.line(self.canvas, (415, 10), (415, 0), self.WHITE_PIXEL, self.BRUSH_SIZE)
        cv2.line(self.canvas, (415, 0), (415 + 30, 0), self.WHITE_PIXEL, self.BRUSH_SIZE)
        cv2.line(self.canvas, (415 + 30, 0), (415 + 30, 21), self.WHITE_PIXEL, self.BRUSH_SIZE)

        cv2.line(self.canvas, (415, 60), (415, 60 + 10), self.WHITE_PIXEL, self.BRUSH_SIZE)
        cv2.line(self.canvas, (415, 60 + 10), (415 + 30, 60 + 10), self.WHITE_PIXEL, self.BRUSH_SIZE)
        cv2.line(self.canvas, (415 + 30, 60 + 10), (415 + 30, 70 - 21), self.WHITE_PIXEL, self.BRUSH_SIZE)

        cv2.line(self.canvas, (2200 - 415, 10), (2200 - 415, 0), self.WHITE_PIXEL, self.BRUSH_SIZE)
        cv2.line(self.canvas, (2200 - 415, 0), (2200 - 415 - 30, 0), self.WHITE_PIXEL, self.BRUSH_SIZE)
        cv2.line(self.canvas, (2200 - 415 - 30, 0), (2200 - 415 - 30, 21), self.WHITE_PIXEL, self.BRUSH_SIZE)

        cv2.line(self.canvas, (2200 - 415, 60), (2200 - 415, 60 + 10), self.WHITE_PIXEL, self.BRUSH_SIZE)
        cv2.line(self.canvas, (2200 - 415, 60 + 10), (2200 - 415 - 30, 60 + 10), self.WHITE_PIXEL, self.BRUSH_SIZE)
        cv2.line(self.canvas, (2200 - 415 - 30, 60 + 10), (2200 - 415 - 30, 70 - 21), self.WHITE_PIXEL, self.BRUSH_SIZE)

    def _draw_bar(self):
        cv2.line(self.canvas, (415 + 30, 21), (2200 - 415 - 30, 21), self.WHITE_PIXEL, self.BRUSH_SIZE)
        cv2.line(self.canvas, (415 + 30, 70 - 21), (2200 - 415 - 30, 70 - 21), self.WHITE_PIXEL, self.BRUSH_SIZE)

    def _fill_with_black(self):
        pixel_color = self.BLACK_PIXEL
        cv2.rectangle(self.canvas, (0 + self.BRUSH_SIZE, 21 + self.BRUSH_SIZE), (2200 - self.BRUSH_SIZE, 70 - 21 - self.BRUSH_SIZE), self.BLACK_PIXEL, -1)

        cv2.rectangle(self.canvas, (0 + self.BRUSH_SIZE, 10 + self.BRUSH_SIZE), (415, 70 - 10 - self.BRUSH_SIZE), self.BLACK_PIXEL, -1)
        cv2.rectangle(self.canvas, (2200 - 415 - self.BRUSH_SIZE, 10 + self.BRUSH_SIZE), (2200 - self.BRUSH_SIZE, 70 - 10 - self.BRUSH_SIZE), self.BLACK_PIXEL, -1)

        if not self.make_notches_transparent:
            cv2.rectangle(self.canvas, (415 + self.BRUSH_SIZE, self.BRUSH_SIZE), (415 + 30 - self.BRUSH_SIZE, 70 - self.BRUSH_SIZE), self.BLACK_PIXEL, -1)
            cv2.rectangle(self.canvas, (2200 - 415 - 30 + self.BRUSH_SIZE, self.BRUSH_SIZE), (2200 - 415 - self.BRUSH_SIZE, 70 - self.BRUSH_SIZE), self.BLACK_PIXEL, -1)

    def _erase_first_plate(self):
        plate_width = 30
        cv2.rectangle(self.canvas, (415 - plate_width, 0), (415, 70), self.TRANSPARENT_PIXEL, -1)
        cv2.rectangle(self.canvas, (2200 - 415 - plate_width, 0), (2200 - 415, 70), self.TRANSPARENT_PIXEL, -1)


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
    threshold_pixel = 50
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


def rotate_image_to_scaled_canvas(img, angle):
    # TODO this function won't work on negative angles?
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


def get_best_bar_size_position_angle(olympic_bar,
                                     edges,
                                     min_bar_size,
                                     center_motion_detected_bar,
                                     left_x_limit,
                                     right_x_limit,
                                     show_output=False,
                                     max_bar_size=None):
    deleteme_counter = 0
    height, width = edges.shape[0: 2]
    num_pixels_for_motion = 10  # totally arbitrary right now
    best_impact = 999999999
    min_width = int(max(min_bar_size, width * MIN_BAR_WIDTH_AS_PERCENT_OF_SCREEN))
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
            percent_offset = 0.20
            for x_offset in xrange(int(center - center * percent_offset), int(center + center * percent_offset)):
                if x_offset < left_x_limit:
                    continue
                if x_offset + overlay_width > right_x_limit:
                    continue
                for y_offset in xrange(0, height - overlay_height):
                    deleteme_counter += 1
                    center_x = center_motion_detected_bar[0]
                    center_y = center_motion_detected_bar[1]
                    if center_x < x_offset or center_x > x_offset + overlay_width:
                        continue
                    if center_y < y_offset or center_y > y_offset + overlay_height:
                        continue

                    pixel_sum_before_overlay = np.sum(edges[y_offset: y_offset + overlay_height, x_offset: x_offset + overlay_width])
                    if pixel_sum_before_overlay < num_pixels_for_motion:
                        continue
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
                    impact = np.sum(abs(mat_after_overlay - mat_before_overlay))
                    num_non_transparent_pixels = np.sum(angled_overlay[:, :, 3]) / 255.0
                    impact = float(impact) / (num_non_transparent_pixels)

                    if impact < best_impact:
                        best_impact = impact
                        best_x_offset = x_offset
                        best_y_offset = y_offset
                        best_angle = angle
                        best_bar_width = bar_width
                        if show_output:
                            cv2.imshow("Barbell Finder", grayscale_with_overlay)
                            cv2.waitKey(1)
    # print "Time in function: %s" % (datetime.datetime.utcnow() - start_time).total_seconds()
    # print best_impact, best_x_offset, best_y_offset, best_angle, best_bar_width
    return best_impact, best_bar_width, best_x_offset, best_y_offset, best_angle


def resized_frame(frame):
    height, width = frame.shape[0: 2]
    desired_width = 500
    desired_to_actual = float(desired_width) / width
    new_width = int(width * desired_to_actual)
    new_height = int(height * desired_to_actual)
    return cv2.resize(frame, (new_width, new_height))


class BarbellDetection(object):

    def __init__(self, frame_number, impact_score, barbell_width, offset_x, offset_y, angle):
        self.frame_number = frame_number
        self.impact_score = impact_score
        self.barbell_width = barbell_width
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.angle = angle

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
        }

    @classmethod
    def from_json(self, json_dict):
        detection = BarbellDetection(json_dict["frame_number"],
                                     json_dict["impact_score"],
                                     json_dict["barbell_width"],
                                     json_dict["offset_x"],
                                     json_dict["offset_y"],
                                     json_dict["angle"])
        return detection


class BarbellDetector(object):

    olympic_bar = OlympicBarbell()
    show_output = True

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

            haystack = resized_frame(haystack)

            frame = haystack.copy()
            motion_detection_frame = None
            if previous_previous_frame is not None:
                motion_detection_frame = self._get_motion_detection_frame(previous_previous_frame, previous_frame, frame)
                cv2.imshow("original", motion_detection_frame)
                cv2.waitKey(1)
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
        threshold_pixel = MOTION_THRESHOLD
        # SBL EUREKA!  Establish left and right bounds beforehand, change the
        # bar to white
        grayscale_motion_detection[grayscale_motion_detection < threshold_pixel] = 0
        grayscale_motion_detection[grayscale_motion_detection >= threshold_pixel] = 255

        best_row, white_pixel_count = self._get_most_populated_row(grayscale_motion_detection)
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
                min_width = get_width_from_points(*points)
                try:
                    self.find_bar_from_edges_and_line_heuristics(grayscale_motion_detection, center_point, min_width, left_x, right_x)
                except UnboundLocalError:
                    # dirty, dirty hack to just ignore bad frames
                    pass

        if self.show_output:
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
        return best_row, max_pixels

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
        best_impact, barbell_width, best_x_offset, best_y_offset, best_angle = get_best_bar_size_position_angle(self.olympic_bar,
                                                                                                                edges,
                                                                                                                min_width,
                                                                                                                center_point,
                                                                                                                left_x,
                                                                                                                right_x,
                                                                                                                show_output=self.show_output,
                                                                                                                max_bar_size=self.max_bar_size)
        frame_number_for_previous_previous_frame = self.frame_number - ACTUAL_FRAME_OFFSET
        barbell_detection = BarbellDetection(frame_number_for_previous_previous_frame, best_impact, barbell_width, best_x_offset, best_y_offset, best_angle)
        self.barbell_detections.append(barbell_detection)


class BarbellDisplayer(object):

    olympic_bar = OlympicBarbell()

    def __init__(self, capture, detected_barbells):
        self.capture = capture
        self.detected_barbells = detected_barbells
        self.frame_number = 0
        self.frame_to_detected_barbell = {barbell.frame_number: barbell for barbell in detected_barbells}
        self.video_writer = None
        self.frames_per_sec = capture.get(FRAMES_PER_SEC_KEY)

    def output_video(self, filename):
        print "Creating Video..."
        while True:
            success, haystack = self.capture.read()
            if not success:
                break
            self.frame_number += 1
            haystack = resized_frame(haystack)

            height, width = haystack.shape[0: 2]
            codec = cv2.cv.FOURCC('D', 'I', 'V', 'X')
            self.video_writer = self.video_writer or cv2.VideoWriter(filename, codec, self.frames_per_sec, (width, height))

            # wait_key_time = 5
            if self.frame_number in self.frame_to_detected_barbell:
                barbell_detection = self.frame_to_detected_barbell[self.frame_number]
                self.display_barbell_on_img(haystack, barbell_detection)
                # wait_key_time = 0
            self.video_writer.write(haystack)
            # cv2.imshow("Final output", haystack)
            # cv2.waitKey(wait_key_time)

        if self.video_writer is not None:
            self.video_writer.release()
        print "Done"

    def display_barbell_on_img(self, draw_img, barbell_detection):
        bar_width = barbell_detection.barbell_width
        x_offset = barbell_detection.offset_x
        y_offset = barbell_detection.offset_y
        angle = barbell_detection.angle
        bar_overlay = self.olympic_bar.with_width(bar_width)
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


def filter_by_bar_width_and_set_mean_for_barbells(detected_barbells):
    ''' barbells will be returned with fixed barbell width; anamolies will be discarded
    based on 3 sigma rule'''
    barbell_widths = np.asarray([bar.barbell_width for bar in detected_barbells])
    std_dev = np.std(barbell_widths)
    mean = np.mean(barbell_widths)

    detected_barbells = [bar for bar in detected_barbells if abs(bar.barbell_width - mean) < 3 * std_dev]
    final_mean = np.mean(np.asarray([bar.barbell_width for bar in detected_barbells]))

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


def dump_detected_barbells_to_json(detected_barbells):
    json_str = json.dumps([barbell.to_json() for barbell in detected_barbells], indent=4)
    filename = datetime.datetime.now().strftime("json_data/%Y_%m_%d_%H_%M.json")
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


def create_barbell_template_from_detected_barbells(capture, detected_barbells):
    if len(detected_barbells) == 0:
        raise CouldNotDetectException("Could not detect the barbell in the image")
    frame_number = 0
    frame_to_detected_barbell = {barbell.frame_number: barbell for barbell in detected_barbells}
    olympic_barbell = OlympicBarbell()
    overlay = olympic_barbell.with_width(detected_barbells[0].barbell_width)
    average_matrix = np.zeros(shape=overlay.shape, dtype=np.uint32)
    average_matrix[:, :, 3] += overlay[:, :, 3]
    frames_processed = 0
    while True:
        success, bgr_frame = capture.read()
        if not success:
            break
        bgr_frame = resized_frame(bgr_frame)
        frame_number += 1
        if frame_number not in frame_to_detected_barbell:
            continue
        detected_barbell = frame_to_detected_barbell[frame_number]
        average_matrix += get_barbell_pixels_from_frame(bgr_frame, detected_barbell, overlay)
        frames_processed += 1

    if frames_processed == 0:
        frames_processed = 1
    final_template = average_matrix / frames_processed
    final_template = final_template.astype(np.uint8)
    return final_template


def amend_barbell_pixels_to_barbell_detections(capture, detected_barbells):
    olympic_barbell = OlympicBarbell()
    overlay = olympic_barbell.with_width(detected_barbells[0].barbell_width)
    capture = cv2.VideoCapture(capture_path)
    frame_to_detected_barbell = {barbell.frame_number: barbell for barbell in detected_barbells}
    frame_number = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        frame = resized_frame(frame)
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


def WIP_template_matching(capture, detected_barbells):
    capture = cv2.VideoCapture(capture_path)
    barbell_template = create_barbell_template_from_detected_barbells(capture, detected_barbells)
    barbell_template = convert_transparency_to_noise(barbell_template)

    capture = cv2.VideoCapture(capture_path)
    needle = barbell_template
    while True:
        method = cv2.TM_CCOEFF_NORMED
        success, haystack = capture.read()
        if not success:
            break
        haystack = resized_frame(haystack)
        height, width = needle.shape[0: 2]
        res = cv2.matchTemplate(haystack, needle, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + width, top_left[1] + height)
        cv2.rectangle(haystack, top_left, bottom_right, 255, 2)
        cv2.imshow("some_window", haystack)
        cv2.waitKey(0)


def get_symmetry_of_img(img):
    height, width = img.shape[0: 2]
    left_side_img = img[:, 0: width / 2]

    offset = 0
    if width % 2 == 1:
        offset = 1

    right_side_img = img[:, width / 2 + offset: width]
    mirror = np.fliplr(left_side_img)

    error = cv2.norm(mirror, right_side_img)
    similarity = error / (width * height)
    return similarity


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


def filter_by_reasonable_accelerations(detected_barbells, frames_per_second):
    one_g_in_pixels_per_ff = get_acceleration_of_gravity_in_pixels_per_frame(detected_barbells[0].barbell_width)
    max_possible_acceleration = one_g_in_pixels_per_ff
    min_possible_acceleration = -2 * one_g_in_pixels_per_ff
    y_positions = np.asarray([detected_barbell.offset_y for detected_barbell in detected_barbells])
    # RECURSIVE CALL WOULD START HERE
    accelerations = np.gradient(y_positions, 2)
    for index in xrange(len(accelerations)):
        frame_number_at_current_index = detected_barbells[index].frame_number
        try:
            frame_number_at_next_index = detected_barbells[index + 1].frame_number
        except IndexError:
            frame_number_at_next_index = frame_number_at_current_index + 1
        delta_frames = frame_number_at_next_index - frame_number_at_current_index

        if delta_frames > 1:  # slicing is slow, so don't do unless necessary
            accelerations[index] /= delta_frames

    # SBL the shit below is just a test, won't actually work
    accelerations_to_discard = set.union(set(accelerations[accelerations > max_possible_acceleration]), set(accelerations[accelerations < min_possible_acceleration]))
    indexes_to_discard = set([index for index, acceleration in enumerate(accelerations) if acceleration in accelerations_to_discard])
    detected_barbells = [barbell for index, barbell in enumerate(detected_barbells) if index not in indexes_to_discard]

    # for all the positions that we have now...try discarding either left or
    # right point...in one case it will remove an outlier completely: that
    # would cause 2 whacky accelerations to drop

    # in one case there is no effect, meaning the anamolies are on a bigger
    # scale

    return detected_barbells


if __name__ == "__main__":
    start_time = datetime.datetime.utcnow()
    capture_path = "/Users/slobdell/playground/one_rep_max/heavy_incline_bench.mp4"
    if False:
        capture = cv2.VideoCapture(capture_path)
        frames_per_sec = capture.get(FRAMES_PER_SEC_KEY)

        barbell_detector = BarbellDetector(capture)
        detected_barbells = barbell_detector.get_barbell_frame_data()

        frame_num_to_original_val = {
            bar.frame_number: (bar.offset_x, bar.barbell_width) for bar in detected_barbells
        }

        detected_barbells = filter_smaller_barbells(detected_barbells)
        detected_barbells = filter_barbells_by_y_values(detected_barbells)

        detected_barbells = filter_by_bar_width_and_set_mean_for_barbells(detected_barbells)

        capture = cv2.VideoCapture(capture_path)
        if len(detected_barbells) == 0:
            raise CouldNotDetectException("Could not detect the barbell in the image")

        x_offsets = [bar.offset_x for bar in detected_barbells]
        min_x = min(x_offsets)
        max_x = max(x_offsets) + bar.barbell_width

        barbell_detector = BarbellDetector(capture).with_bar_size(detected_barbells[0].barbell_width).with_min_and_max_x(min_x, max_x)
        detected_barbells = barbell_detector.get_barbell_frame_data()
        for bar in detected_barbells:
            if bar.frame_number not in frame_num_to_original_val:
                frame_num_to_original_val[bar.frame_number] = (bar.offset_x, bar.barbell_width)

        detected_barbells = set_bar_offsets_by_average_x(detected_barbells)

        capture = cv2.VideoCapture(capture_path)
        amend_barbell_pixels_to_barbell_detections(capture, detected_barbells)
        amend_likeness_score_to_barbell_detections(detected_barbells)
        amend_symmetry_score_to_barbell_detections(detected_barbells)
        detected_barbells = sorted(detected_barbells, key=lambda detection: detection.total_score)

        start_index, end_index = get_best_start_and_end_index_from_sorted_barbells(detected_barbells)
        detected_barbells = detected_barbells[start_index: end_index]

        print "STARTING THIS AWESOMENESS"
        print "BEFORE: %s" % len(detected_barbells)
        detected_barbells = filter_by_y_pos_and_max_rom(detected_barbells)
        print "AFTER: %s" % len(detected_barbells)

        detected_barbells = filter_by_no_frame_neighbors(detected_barbells)

        print "STARTING MY AWESOME REPEAT!!!!"
        relevant_frames = [barbell.frame_number for barbell in detected_barbells]

        detected_barbells = []
        for frame_number in relevant_frames:
            # got a key error
            original_x_offset = frame_num_to_original_val[frame_number][0]
            original_bar_width = frame_num_to_original_val[frame_number][1]
            fabricated_detection = BarbellDetection(frame_number, None, original_bar_width, original_x_offset, None, None)
            detected_barbells.append(fabricated_detection)

        detected_barbells = filter_smaller_barbells(detected_barbells)
        detected_barbells = filter_by_bar_width_and_set_mean_for_barbells(detected_barbells)

        x_offsets = [bar.offset_x for bar in detected_barbells]
        min_x = min(x_offsets)
        max_x = max(x_offsets) + bar.barbell_width

        frame_set = set([barbell.frame_number + ACTUAL_FRAME_OFFSET for barbell in detected_barbells])
        capture = cv2.VideoCapture(capture_path)
        barbell_detector = (BarbellDetector(capture).
                            with_frame_num_limits(frame_set).
                            with_min_and_max_x(min_x, max_x).
                            with_bar_size(detected_barbells[0].barbell_width))

        detected_barbells = barbell_detector.get_barbell_frame_data()
        print "Length of detected barbells now: %s" % len(detected_barbells)
        detected_barbells = set_bar_offsets_by_average_x(detected_barbells)
        dump_detected_barbells_to_json(detected_barbells)
        #### END TOTAL REPEAT
    else:
        detected_barbells = []
        with open("json_data/flat_bench.json", "rb") as f:
            json_str = f.read()
            json_data = json.loads(json_str)
            for json_dict in json_data:
                detected_barbells.append(BarbellDetection.from_json(json_dict))

    detected_barbells.sort(key=lambda barbell: barbell.frame_number)
    while has_fluctuations(detected_barbells):
        fix_one_fluctuation(detected_barbells)

    '''
    json_objs = [{barbell.frame_number: barbell.offset_y} for barbell in detected_barbells]
    json_str = json.dumps(json_objs)
    with open("json_data/flat_bench_points.json", "w+") as f:
        f.write(json_str)
    print "Hell yes"
    '''

    capture = cv2.VideoCapture(capture_path)
    frames_per_second = capture.get(FRAMES_PER_SEC_KEY)
    print frames_per_second
    one_g = get_acceleration_of_gravity_in_pixels_per_frame(detected_barbells[0].barbell_width, frames_per_second)
    frame_to_acceleration = determine_acceleration_values(detected_barbells, one_g)
    json_str = json.dumps(frame_to_acceleration, indent=4)
    with open("json_data/deleteme.json", "w+") as f:
        f.write(json_str)
    print json_str

    capture = cv2.VideoCapture(capture_path)
    barbell_display = BarbellDisplayer(capture, detected_barbells)
    barbell_display.output_video("output2.avi")
    print "Finished in %s minutes" % ((datetime.datetime.utcnow() - start_time).total_seconds() / 60.0)

cv2.destroyAllWindows()
