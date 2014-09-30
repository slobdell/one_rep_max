import pylab
import json
import numpy as np


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
        for x in range(start_point[0], end_point[0]):
            frame_to_acceleration[x] = acceleration_pixels_per_ff
    return frame_to_acceleration


if __name__ == "__main__":
    with open("final.json", "rb") as f:
    # with open("lobbdawg.json", "rb") as f:
    # with open("flat_bench_points.json", "rb") as f:
        json_str = f.read()
    # TODO go back and change how we're doing this
    frame_number_to_y = json.loads(json_str)
    final_dict = {}
    for frame_to_y in frame_number_to_y:
        final_dict.update(frame_to_y)
    frame_number_to_y = {int(k): v for k, v in final_dict.items()}
    max_frame = max(frame_number_to_y.keys())

    plot_data = [frame_number_to_y.get(frame_number) for frame_number in xrange(1, max_frame)]

    # this straight up does not work
    x_values = np.asarray([index for index, item in enumerate(plot_data) if item is not None])
    y_values = np.asarray([item for item in plot_data if item is not None])
    y_values = y_values.astype(float)

    pylab.figure()

    # START FINDING ACCELERATION
    min_maxima, max_maxima = find_maxima(x_values, y_values)
    min_x = [t[0] for t in min_maxima]
    min_y = [t[1] for t in min_maxima]

    max_x = [t[0] for t in max_maxima]
    max_y = [t[1] for t in max_maxima]
    point_pairs = get_accelerations_from_maxima(min_maxima, max_maxima, x_values, y_values)
    get_acceleration_values_from_acceleration_points(point_pairs)
    # END FINDING ACCELERATION

    pylab.plot(x_values, y_values, marker='o', color='b', label='Position')

    pylab.plot(min_x, min_y, 'o', color='r')
    pylab.plot(max_x, max_y, 'o', color='g')

    for point_pair in point_pairs:
        x_values = [point[0] for point in point_pair]
        y_values = [point[1] for point in point_pair]
        pylab.plot(x_values, y_values, '-', color='r')

    pylab.legend()
    pylab.xlabel('Frame Number')
    pylab.ylabel('Position')
    pylab.show()
