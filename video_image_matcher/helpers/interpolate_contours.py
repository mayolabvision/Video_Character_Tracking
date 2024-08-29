import numpy as np
from scipy.interpolate import interp1d
import cv2

def interpolate_contours(contours_3d, output_path, character_name):
    num_frames, num_points, _ = contours_3d.shape

    # Initialize an array to store the interpolated contours
    interpolated_contours = np.copy(contours_3d)

    # Loop through each contour point (x and y separately)
    for point_index in range(num_points):
        x_coords = contours_3d[:, point_index, 0]
        y_coords = contours_3d[:, point_index, 1]

        # Find the indices where the values are not NaN
        valid_indices = np.where(~np.isnan(x_coords))[0]

        if len(valid_indices) < 2:
            # If less than 2 valid points, we can't interpolate meaningfully
            continue

        # Only interpolate between the first and last valid indices
        first_valid_idx = valid_indices[0]
        last_valid_idx = valid_indices[-1]

        # Interpolate x and y coordinates separately between the valid range
        interp_func_x = interp1d(valid_indices, x_coords[valid_indices], kind='linear', fill_value='extrapolate')
        interp_func_y = interp1d(valid_indices, y_coords[valid_indices], kind='linear', fill_value='extrapolate')

        # Apply the interpolation within the valid range
        interpolated_contours[first_valid_idx:last_valid_idx + 1, point_index, 0] = interp_func_x(np.arange(first_valid_idx, last_valid_idx + 1))
        interpolated_contours[first_valid_idx:last_valid_idx + 1, point_index, 1] = interp_func_y(np.arange(first_valid_idx, last_valid_idx + 1))

    # Now ensure that each contour forms a single blob
    for frame_number in range(num_frames):
        contour = interpolated_contours[frame_number, :, :]
        if np.isnan(contour).all():
            continue  # Skip frames where no contour exists
        
        # Ensure contour is of the correct shape for cv2 functions
        contour = contour.astype(np.int32).reshape((-1, 1, 2))

        # Use convexHull to ensure a single blob
        hull = cv2.convexHull(contour, returnPoints=True)

        # If the hull has fewer points, interpolate it to match the original number of points
        if hull.shape[0] != num_points:
            hull_interpolated = interpolate_contour_to_fixed_size(hull, num_points)
        else:
            hull_interpolated = hull.squeeze()

        # Replace the original contour with the interpolated hull
        interpolated_contours[frame_number, :, :] = hull_interpolated

    return interpolated_contours

def interpolate_contour_to_fixed_size(contour, num_points):
    """ Interpolates the contour to have exactly `num_points` points. """
    contour = contour.squeeze()
    contour_len = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1)))
    contour_len = np.insert(contour_len, 0, 0)  # Add the starting point
    uniform_len = np.linspace(0, contour_len[-1], num_points)

    interp_x = interp1d(contour_len, contour[:, 0], kind='linear')
    interp_y = interp1d(contour_len, contour[:, 1], kind='linear')

    contour_interp = np.vstack((interp_x(uniform_len), interp_y(uniform_len))).T

    return contour_interp

# Example usage:
# interpolated_contours = interpolate_contours(contours_3d, output_path, character_name)
