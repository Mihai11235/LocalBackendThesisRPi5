import numpy as np
import cv2

class LaneDepartureDetector:
    def detect(self, mask_bin, overlay):
        row_sums = mask_bin.sum(axis=1)
        nonzero_rows = np.where(row_sums > 0)[0]
        if len(nonzero_rows) == 0:
            cv2.putText(overlay, "No lane detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 1), 2)
            return "no_lane"

        top, bottom = nonzero_rows[0], nonzero_rows[-1]
        height, width = mask_bin.shape

        cv2.line(overlay, (0, top), (width, top), (1, 1), 2)  # White vertical line
        cv2.line(overlay, (0, bottom), (width, bottom), (1, 1), 2)  # White vertical line

        coord_y = (3 * top + 5 * bottom) // 8
        middle_x = width // 2

        def find_edge(img, direction):
            step = -1 if direction == 'left' else 1
            for x in range(middle_x, 0 if step == -1 else width, step):
                if img[coord_y, x] == 1:
                    return x
            return None

        left_x = find_edge(mask_bin, 'left')
        right_x = find_edge(mask_bin, 'right')

        if left_x is None or right_x is None:
            cv2.putText(overlay, "No Lane Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 1), 2)
            return "no_lane"

        lane_middle = (left_x + right_x) // 2
        cv2.line(overlay, (lane_middle, 0), (lane_middle, height), (1, 1), 2)  # White vertical line

        cv2.line(overlay, (middle_x, 0), (middle_x, height), (1, 1), 2)  # White vertical line
        cv2.line(overlay, (0, coord_y), (width, coord_y), (1, 1), 2)

        if abs(middle_x - lane_middle) > 100:
            cv2.putText(overlay, "Lane Departure Warning!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 1), 2)
            return "lane_departure"

        return None
