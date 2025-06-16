import numpy as np
import cv2

class LaneDepartureDetector:
    """
    Detects lane departure or absence of lanes based on a binary mask image.
    """

    def detect(self, mask_bin, overlay):
        """
        Detects lane departure or absence of lanes based on the binary mask image.

        Args:
            mask_bin (numpy.ndarray): A binary mask image where lane markings are represented as 1s.
            overlay (numpy.ndarray): An image on which visual indicators (lines and text) will be drawn.

        Returns:
            str: A string indicating the detection result:
                - "no_lane": No lane detected.
                - "lane_departure": Lane departure warning.
                - None: No lane departure detected.
        """
        # Sum the rows of the binary mask to find rows with lane markings
        row_sums = mask_bin.sum(axis=1)
        nonzero_rows = np.where(row_sums > 0)[0]

        # If no rows contain lane markings, display "No lane detected" and return "no_lane"
        if len(nonzero_rows) == 0:
            cv2.putText(overlay, "No lane detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 1), 2)
            return "no_lane"

        # Determine the top and bottom rows containing lane markings
        top, bottom = nonzero_rows[0], nonzero_rows[-1]
        height, width = mask_bin.shape

        # Draw horizontal lines at the top and bottom of the detected lane markings
        cv2.line(overlay, (0, top), (width, top), (1, 1), 2)  # White vertical line
        cv2.line(overlay, (0, bottom), (width, bottom), (1, 1), 2)  # White vertical line

        # Calculate a coordinate for the middle of the lane markings
        coord_y = (3 * top + 5 * bottom) // 8
        middle_x = width // 2

        def find_edge(img, direction):
            """
            Finds the edge of the lane markings in the specified direction.

            Args:
                img (numpy.ndarray): The binary mask image.
                direction (str): The direction to search for the edge ("left" or "right").

            Returns:
                int or None: The x-coordinate of the edge, or None if no edge is found.
            """
            step = -1 if direction == 'left' else 1
            for x in range(middle_x, 0 if step == -1 else width, step):
                if img[coord_y, x] == 1:
                    return x
            return None

        # Find the left and right edges of the lane markings
        left_x = find_edge(mask_bin, 'left')
        right_x = find_edge(mask_bin, 'right')

        # If either edge is not found, display "No Lane Detected!" and return "no_lane"
        if left_x is None or right_x is None:
            cv2.putText(overlay, "No Lane Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 1), 2)
            return "no_lane"

        # Calculate the middle of the lane and draw a vertical line at this position
        lane_middle = (left_x + right_x) // 2
        cv2.line(overlay, (lane_middle, 0), (lane_middle, height), (1, 1), 2)  # White vertical line

        # Draw a vertical line at the middle of the image and a horizontal line at coord_y
        cv2.line(overlay, (middle_x, 0), (middle_x, height), (1, 1), 2)  # White vertical line
        cv2.line(overlay, (0, coord_y), (width, coord_y), (1, 1), 2)

        # If the lane middle deviates significantly from the image middle, display a warning
        if abs(middle_x - lane_middle) > 100:
            cv2.putText(overlay, "Lane Departure Warning!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 1), 2)
            return "lane_departure"

        return None
