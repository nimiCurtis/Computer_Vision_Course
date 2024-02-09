"""Projective Homography and Panorama Solution."""
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple


from numpy.linalg import svd
from scipy.interpolate import griddata


PadStruct = namedtuple('PadStruct',
                    ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""
    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
            Based on the linear eq: Ax = b 
        """
        # return homography
        """INSERT YOUR CODE HERE"""
        # Build the linear equation system Ax = b
        
        A = []

        for i in range(len(match_p_src[0])):
            x_src, y_src = match_p_src[:, i]
            x_dst, y_dst = match_p_dst[:, i]
            
            # A_i = [ -X^T   0      u*X'^T
            #          0    -X^T    v*X'^T]
            # b_i = 0
            
            A.append([-x_src, -y_src, -1, 0, 0, 0, x_dst * x_src, x_dst * y_src, x_dst])
            A.append([0, 0, 0, -x_src, -y_src, -1, y_dst * x_src, y_dst * y_src, y_dst])

        A = np.array(A)

        # Solve for x using SVD
        U, S, VT = np.linalg.svd(A)

        # x =  is the last column of V, or last row of V transpose 
        x = VT[-1]

        # Reshape x into the 3x3 homography matrix
        H = x.reshape(3, 3)

        # Normalize the homography matrix by the bottom right value
        H = H / H[2, 2]

        return H

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        """INSERT YOUR CODE HERE"""
        # Initialize the destination (warped) image with zeros
        warped_img = np.zeros(dst_image_shape, dtype=src_image.dtype)
        
        for y_src in range(src_image.shape[0]):
            for x_src in range(src_image.shape[1]):
                # Apply homography transformation for each pixel in the source image
                src_coords_homogeneous = np.array([x_src, y_src, 1])
                dst_coords_homogeneous = homography @ src_coords_homogeneous
                # Normalize
                dst_coords = dst_coords_homogeneous / dst_coords_homogeneous[2]
                
                x_dst, y_dst = dst_coords[0], dst_coords[1]
                x_dst, y_dst = int(round(x_dst)), int(round(y_dst))

                # Check if the computed destination coordinates are within the bounds of the destination image
                if 0 <= x_dst < dst_image_shape[1] and 0 <= y_dst < dst_image_shape[0]:
                    # Assign the pixel value from the source image to the warped image
                    warped_img[y_dst, x_dst] = src_image[y_src, x_src]

        return warped_img

    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # return new_image
        """INSERT YOUR CODE HERE"""
        
        src_H, src_W = src_image.shape[:2]
        
        # (1) Create a meshgrid of columns and rows for the source image
        xv, yv = np.meshgrid(range(src_W), range(src_H))
        
        # (2) Generate a matrix of size 3x(H*W) for pixel locations in homogeneous coordinates
        ones = np.ones_like(xv)
        homogeneous_coordinates = np.stack([xv.flatten(), yv.flatten(), ones.flatten()], axis=0)
        
        # (3) Apply the homography to transform source image coordinates to the destination image's coordinate system
        transformed_coordinates = homography @ homogeneous_coordinates
        transformed_coordinates /= transformed_coordinates[2, :]  # Normalize
        
        # Convert coordinates into integer values and generate masks for valid indices
        x_dst = np.round(transformed_coordinates[0, :]).astype(int)
        y_dst = np.round(transformed_coordinates[1, :]).astype(int)
        
        # Create masks for coordinates within the destination image bounds
        valid_x_mask = (x_dst >= 0) & (x_dst < dst_image_shape[1])
        valid_y_mask = (y_dst >= 0) & (y_dst < dst_image_shape[0])
        valid_mask = valid_x_mask & valid_y_mask
        
        # Initialize the new image with zeros (black pixels)
        new_image = np.zeros(dst_image_shape, dtype=src_image.dtype)
        
        # Use the mask to filter and assign valid pixel values
        valid_indices = np.where(valid_mask)
        new_image[y_dst[valid_indices], x_dst[valid_indices]] = src_image[yv.flatten()[valid_indices], xv.flatten()[valid_indices]]

        return new_image

    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        # return fit_percent, dist_mse
        """INSERT YOUR CODE HERE"""
        # Ground true destination matching points
        x_dst_gt = np.round(match_p_dst[0, :]).astype(int)
        y_dst_gt = np.round(match_p_dst[1, :]).astype(int)
        
        # Mapping of source matching points to the destination image coordinates 
        ones = np.ones_like(match_p_src[0])
        homogeneous_coordinates = np.stack([match_p_src[0].flatten(), match_p_src[1].flatten(), ones.flatten()], axis=0)
        transformed_coordinates = homography @ homogeneous_coordinates
        transformed_coordinates /= transformed_coordinates[2, :]  # Normalize
        x_dst_map = np.round(transformed_coordinates[0, :]).astype(int)
        y_dst_map = np.round(transformed_coordinates[1, :]).astype(int)
        
        # Calculate the distance
        dist = np.sqrt((x_dst_gt-x_dst_map)**2 + (y_dst_gt-y_dst_map)**2)
        
        # calculate prob
        n_inliers = (dist < max_err).sum()
        p_in =  n_inliers / dist.size
        
        # Calculate MSE -> we left with calculating the mean
        mse = np.mean(dist) if n_inliers!=0 else 10 ** 9
        
        return (p_in, mse)

    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        # return mp_src_meets_model, mp_dst_meets_model
        """INSERT YOUR CODE HERE"""
        
        # Ground true destination matching points
        x_dst_gt = np.round(match_p_dst[0, :]).astype(int)
        y_dst_gt = np.round(match_p_dst[1, :]).astype(int)
        
        # Mapping of source matching points to the destination image coordinates 
        ones = np.ones_like(match_p_src[0])
        homogeneous_coordinates = np.stack([match_p_src[0].flatten(), match_p_src[1].flatten(), ones.flatten()], axis=0)
        transformed_coordinates = homography @ homogeneous_coordinates
        transformed_coordinates /= transformed_coordinates[2, :]  # Normalize
        x_dst_map = np.round(transformed_coordinates[0, :]).astype(int)
        y_dst_map = np.round(transformed_coordinates[1, :]).astype(int)
        
        # Calculate the distance
        dist = np.sqrt((x_dst_gt-x_dst_map)**2 + (y_dst_gt-y_dst_map)**2)
        
        inlier_src = match_p_src[:,dist<max_err]
        inlier_dst = match_p_dst[:,dist<max_err]
        
        return (inlier_src, inlier_dst)

    def compute_homography(self,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        inliers_percent: float,
                        max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # # use class notations:
        w = inliers_percent
        t = max_err
        # # p = parameter determining the probability of the algorithm to
        # # succeed
        p = 0.99
        # # the minimal probability of points which meets with the model
        d = 0.5
        # # number of points sufficient to compute the model
        n = 4
        # # number of RANSAC iterations (+1 to avoid the case where w=1)
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1
        # return homography
        """INSERT YOUR CODE HERE"""
        
        h_model_best = np.zeros((4,4))
        prev_error = np.inf
        
        # Iterate k times
        for i in range(k):

            # Select random points from the sets
            random_idx = np.random.choice(range(match_p_src.shape[1]),n,replace=False)
            src_random_pts = match_p_src[:,random_idx]
            dst_random_pts = match_p_dst[:,random_idx]
            
            # Compute model
            h_model = self.compute_homography_naive(match_p_src=src_random_pts,
                                                    match_p_dst=dst_random_pts)

            # Find inliers based on the random points model
            src_inliers, dst_inliers = self.meet_the_model_points(homography=h_model,
                                                                match_p_dst=match_p_dst,
                                                                match_p_src=match_p_src,
                                                                max_err=t)
            
            # Calculate prob
            fit_percent = src_inliers.shape[1] / match_p_src.shape[1]
            
            # Check if number of inliers is bigger then the minimal probability
            if(fit_percent>d):
                # Recompute using all inliers
                h_model = self.compute_homography_naive(match_p_src=src_inliers,
                                                    match_p_dst=dst_inliers)
                
                # Get the error based on all the points
                fit_percent, dist_mse = self.test_homography(homography=h_model,
                                                                match_p_dst=match_p_dst,
                                                                match_p_src=match_p_src,
                                                                max_err=t)

                # If the current error is less then the previous store the current model
                if(dist_mse<prev_error):
                    prev_error = dist_mse
                    h_model_best = h_model
        
        return h_model_best

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """

        # return backward_warp
        """INSERT YOUR CODE HERE"""
        H, W = dst_image_shape[:2]

        # (1) Create a meshgrid of columns and rows for the destination image
        xv, yv = np.meshgrid(range(W), range(H))

        # (2) Generate a matrix of size 3x(H*W) for pixel locations in homogeneous coordinates
        ones = np.ones_like(xv)
        homogeneous_coordinates = np.stack([xv, yv, ones], axis=-1).reshape(-1, 3).T

        # (3) Transform the source homogeneous coordinates to the target homogeneous coordinates
        transformed_coordinates = backward_projective_homography @ homogeneous_coordinates
        transformed_coordinates /= transformed_coordinates[2, :]  # Normalize

        # (4) Convert coordinates into integer values and clip them according to the destination image size
        x_src, y_src = transformed_coordinates[:2, :]
        valid_x_mask = (x_src >= 0) & (x_src < src_image.shape[1]-1)
        valid_y_mask = (y_src >= 0) & (y_src < src_image.shape[0]-1)
        valid_mask = valid_x_mask & valid_y_mask

        # Initialize the new image with zeros (black pixels)
        new_image = np.zeros((H, W, 3), dtype=src_image.dtype)

        # (5) Only plant the pixels for valid coordinates, leave invalid ones as black
        valid_coords = np.where(valid_mask)  # Get indices of valid coordinates
        # Convert valid source coordinates to integers for indexing
        valid_x_src = x_src[valid_mask].round().astype(int)
        valid_y_src = y_src[valid_mask].round().astype(int)

        # Assign pixels from the source image to the target image based on valid mapped coordinates
        new_image[yv.flatten()[valid_coords], xv.flatten()[valid_coords]] = src_image[valid_y_src, valid_x_src]

        return new_image

    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([1, 1, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 1, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        # return final_homography
        """INSERT YOUR CODE HERE"""
        # (1) Build the translation matrix from the pads
        translation_matrix = np.array([[1, 0, pad_left],
                                    [0, 1, pad_up],
                                    [0, 0, 1]])

        # (2) Compose the backward homography and the translation matrix together
        homography_translated = translation_matrix @ backward_homography

        # (3) Assuming scaling is required to normalize the homography matrix
        homography_translated /= homography_translated[2, 2]

        return homography_translated

    def panorama(self,
                src_image: np.ndarray,
                dst_image: np.ndarray,
                match_p_src: np.ndarray,
                match_p_dst: np.ndarray,
                inliers_percent: float,
                max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        # return np.clip(img_panorama, 0, 255).astype(np.uint8)
        """INSERT YOUR CODE HERE"""
        # Placeholder for steps (1) to (7)
        # (1) Compute the forward homography using matched points (potentially with RANSAC)
        # This would typically involve a function call to a homography computation method
        forward_homography = self.compute_homography(match_p_src, match_p_dst, inliers_percent, max_err)         # panorama_shape = determine_panorama_shape(src_image, dst_image, forward_homography)
        backward_homography = np.linalg.inv(forward_homography)
        panorama_shape = self.find_panorama_shape(src_image=src_image,dst_image=dst_image,homography=forward_homography)
        pad_struct = panorama_shape[2]
        translation_adjusted_homography = self.add_translation_to_backward_homography(backward_homography, pad_struct.pad_left, pad_struct.pad_up)
        warped_src_image = self.compute_backward_mapping(translation_adjusted_homography,src_image,(panorama_shape[0],panorama_shape[1]))
        # img_panorama = create_panorama_image(dst_image, warped_src_image)
        img_panorama = np.zeros((panorama_shape[0],panorama_shape[1],3))
        img_panorama[:dst_image.shape[0],:dst_image.shape[1]] = dst_image
        img_panorama[pad_struct.pad_up:pad_struct.pad_up+warped_src_image.shape[0],pad_struct.pad_left:pad_struct.pad_left+warped_src_image.shape[1]] = warped_src_image        
        img_panorama = np.clip(img_panorama, 0, 255).astype(np.uint8)
        import matplotlib.pyplot as plt
        plt.figure()
        panorama = plt.imshow(img_panorama)
        return np.clip(img_panorama, 0, 255).astype(np.uint8)

        
        
        # (2) Compute the backward homography, which is the inverse of the forward homography
        backward_homography = np.linalg.inv(forward_homography)
        
        # (3) Add translation to the backward homography to ensure correct placement in the panorama
        # Assuming functions to compute the required translation based on the image overlap or desired positioning
        pad_left, pad_up = compute_translation_parameters(src_image, dst_image, backward_homography)
        backward_homography_with_translation = add_translation_to_backward_homography(backward_homography, pad_left, pad_up)
        
        # (4) Compute the backward warping of the source image using the modified homography
        warped_src_image = warp_image(src_image, backward_homography_with_translation, output_shape)
        
        # (5) Create an empty panorama image and plant the destination image
        img_panorama = np.zeros_like(panorama_shape)  # Define panorama_shape based on both images
        img_panorama[...] = place_destination_image(dst_image, img_panorama)
        
        # (6) Place the backward warped source image in the indices where the panorama image is zero
        img_panorama = merge_images(warped_src_image, img_panorama)
        
        # (7) Clip the values of the image to [0, 255]
        img_panorama_clipped = np.clip(img_panorama, 0, 255).astype(np.uint8)
        
        return img_panorama_clipped
