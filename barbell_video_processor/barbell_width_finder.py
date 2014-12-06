import cv2
import numpy as np
DEBUG = False


def grayscale(frame):
    im = cv2.cv.fromarray(frame)
    gray = cv2.cv.CreateImage((im.width, im.height), 8, 1)
    cv2.cv.CvtColor(im, gray, cv2.cv.CV_BGR2GRAY)
    return np.asarray(gray[:, :])


def smooth_list_gaussian(input_list, degree=8):
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
    smoothed = [0 for _ in xrange(degree)] + smoothed
    return smoothed


class BarbellWidthFinder(object):

    MOTION_THRESHOLD = 25
    Y_OFFSET_FOR_INSPECTION = 40
    MIN_BAR_AS_PERCENT_OF_SCREEN = 0.50

    def __init__(self, filtered_motion_detection_frames):
        self.resultant_frame = None
        self.union_frame = self._create_union_frame(filtered_motion_detection_frames)

    def _create_union_frame(self, filtered_motion_detection_frames):
        for frame in filtered_motion_detection_frames:
            grayscale_frame = grayscale(frame)
            # grayscale_frame[grayscale_frame < self.MOTION_THRESHOLD] = 0
            self.resultant_frame = self._get_resultant_frame(grayscale_frame)
        return self.resultant_frame.copy()

    def _get_resultant_frame(self, current_frame):
        if self.resultant_frame is not None:
            matrix_sum = self.resultant_frame + current_frame
            return matrix_sum

        height, width = current_frame.shape[0: 2]
        self.resultant_frame = current_frame.copy().astype(np.uint64)
        return self.resultant_frame

    def _make_frame_displayable(self, matrix_with_large_ints):
        divisor = np.max(matrix_with_large_ints) / 255.0
        display_frame = matrix_with_large_ints.copy()
        display_frame /= divisor
        return display_frame.astype(np.uint8)

    def _collapse_motion_to_one_row(self, frame):
        rows, cols = frame.shape[0: 2]
        summed_values = []
        for col_index in xrange(cols):
            col_values = frame[:, col_index]
            summed_values.append(np.sum(col_values))
        return np.asarray(summed_values)

    def _display_motion_column(self, motion_by_column, shape):
        rows, cols = shape
        stretched_rows = np.zeros((rows, cols), dtype=np.uint64)
        for row in xrange(rows):
            stretched_rows[row, :] = motion_by_column

        divisor = np.max(motion_by_column) / 255.0
        stretched_rows /= divisor
        stretched_rows = stretched_rows.astype(np.uint8)
        cv2.imshow("COLUMN", stretched_rows)

    def find_barbell_width(self):
        probable_barbell_row = self._get_best_object_motion_row(self.union_frame)
        cropped_aggregate_motion = self._get_cropped_matrix(self.union_frame, probable_barbell_row)
        displayable_frame = self._make_frame_displayable(cropped_aggregate_motion)
        motion_by_column = self._collapse_motion_to_one_row(displayable_frame)
        smoothed_motion_by_column = smooth_list_gaussian(motion_by_column)
        x_offset, bar_width = self._find_width(smoothed_motion_by_column)

        if DEBUG:
            cv2.imshow("initial", self._make_frame_displayable(self.union_frame))
            self._display_motion_column(motion_by_column, self.union_frame.shape[0: 2])
            self._plot(motion_by_column, smoothed_motion_by_column, (x_offset, bar_width))
            cv2.waitKey(0)

        return x_offset, bar_width

    def _get_cropped_matrix(self, matrix, best_row):
        return matrix[best_row - self.Y_OFFSET_FOR_INSPECTION: best_row + self.Y_OFFSET_FOR_INSPECTION, :]

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
            if differential_score != 0:
                row_scores[row_index] = motion_score / differential_score
        return row_scores.argmax()

    def _plot(self, motion_by_column, smoothed_motion_by_column, found_bar_tuple):
        import pylab
        x_values = range(len(motion_by_column))
        x_values2 = range(len(smoothed_motion_by_column))

        bar_x = [i + found_bar_tuple[0] for i in xrange(found_bar_tuple[1])]
        bar_y_value = np.min([smoothed_motion_by_column[index] for index in bar_x])
        bar_y = [bar_y_value for _ in xrange(found_bar_tuple[1])]

        pylab.figure()
        pylab.plot(x_values, motion_by_column, '-', color='b', label='Motion Intensity')
        pylab.plot(x_values2, smoothed_motion_by_column, '-', color='r', label='Smoothed Motion Intensity')
        pylab.plot(bar_x, bar_y, '-', color='g', label='Bar')
        pylab.legend()
        pylab.xlabel('Frame Number')
        pylab.ylabel('Position')
        pylab.show()

    def _find_width(self, motion_by_column):
        '''
        Find the best combo of x_offset, width
        where width * max(min(combination)) is greatest
        '''
        min_pixel_width = int(len(motion_by_column) * self.MIN_BAR_AS_PERCENT_OF_SCREEN)

        best_score = 0
        best_width = min_pixel_width
        best_x_offset = 0

        for x_offset in xrange(len(motion_by_column)):
            for bar_width in xrange(min_pixel_width, len(motion_by_column)):
                if x_offset + bar_width >= len(motion_by_column):
                    continue
                y_values = motion_by_column[x_offset: x_offset + bar_width]
                min_val = np.min(y_values)
                score = min_val * bar_width
                if score > best_score:
                    best_score = score
                    best_x_offset = x_offset
                    best_width = bar_width
        return best_x_offset, best_width
