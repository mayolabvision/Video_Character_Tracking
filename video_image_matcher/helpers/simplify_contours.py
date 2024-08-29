import numpy as np
import cv2

def simplify_contours(frame_contour_array, num_points=50, epsilon_factor=0.01):
    simplified_contours = []

    for contour in frame_contour_array:
        if contour.shape[0] == 0:
            # If the contour is empty, append an empty placeholder
            simplified_contours.append(np.full((num_points, 1, 2), np.nan))
        else:
            # Calculate the epsilon value for the contour approximation
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
    
            # Approximate the contour
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
    
            # If the contour has fewer points than desired, interpolate to reach num_points
            if len(approx_contour) < num_points:
                approx_contour = interp_contour(approx_contour, num_points)
    
            simplified_contours.append(approx_contour)

    # Convert the list of simplified contours into a 3D matrix
    contours_3d = np.full((len(simplified_contours), num_points, 2), np.nan)

    for i, contour in enumerate(simplified_contours):
        if contour.shape[0] > 0:
            contours_3d[i] = contour.squeeze()

    return contours_3d

def interp_contour(contour, num_points):
    """ 
    Interpolates a contour to have exactly `num_points` points.
    """
    contour = contour.reshape(-1, 2)  # Reshape to (n_points, 2)
    
    # Create an empty array to store the interpolated points
    interpolated_contour = np.zeros((num_points, 2), dtype=np.float32)
    
    for i in range(num_points):
        index = i * len(contour) / num_points
        prev_index = int(np.floor(index))
        next_index = (prev_index + 1) % len(contour)
    
        weight = index - prev_index
        interpolated_contour[i] = (1 - weight) * contour[prev_index] + weight * contour[next_index]
    
    return interpolated_contour.reshape(-1, 1, 2).astype(np.int32)

# Example usage:
# Assuming frame_contour_array is your input array of contours
# simplified_contours = simplify_contours(frame_contour_array, num_points=50)
