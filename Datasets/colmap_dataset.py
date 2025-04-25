"""Colmap dataset for fundametal matrix estimation. Derived from FundamentalMatrixDataset.
"""

import numpy as np
import os
import cv2
from tqdm import tqdm

from dfe.utils import colmap_read, colmap_utils


class ETH3D():
    """Colmap dataset for fundametal matrix estimation. Derived from FundamentalMatrixDataset.
    """

    def __init__(
        self,
        path,
        num_points=-1,
        threshold=1,
        max_F=None,
        random=False,
        min_matches=20,
        compute_virtual_points=True,
        mode="test"  # Added mode parameter, default is "test"
    ):
        """Init.

        Args:
            path (str): path to dataset folder
            num_points (int, optional): number of points per sample. Defaults to -1.
            threshold (int, optional): epipolar threshold. Defaults to 1.
            max_F (int, optional): maximal number of samples (if None: use all). Defaults to None.
            random (bool, optional): random database access. Defaults to False.
            min_matches (int, optional): minimal number of good matches per sample. Defaults to 20.
            compute_virtual_points (bool, optional): whether to compute virtual points. Defaults to True.
            mode (str, optional): "train" or "test" mode. Defaults to "test".
        """        
        self.mode = mode
        self.compute_virtual_points = compute_virtual_points
        
        cameras = colmap_read.read_cameras_text("%s/dslr_calibration_undistorted/cameras.txt" % path)
        images = colmap_read.read_images_text("%s/dslr_calibration_undistorted/images.txt" % path)

        self.img_paths = []
        self.pts = []
        self.R_gt = []  # Ground truth rotation matrix
        self.t_gt = []  # Ground truth translation vector (normalized)
        self.size_1 = []
        self.size_2 = []
        self.intrinsics = []

        # Traverse all image pairs
        img_ids = list(images.keys())
        min_covisible_points = 100  # Minimum number of covisible points
        min_translation_norm = 0.075  # Minimum translation norm threshold
        
        print(f"Processing image pairs from {path}...")
        processed_pairs = 0
        max_pairs = max_F if max_F else float('inf')
        
        for i in tqdm(range(len(img_ids))):
            if processed_pairs >= max_pairs:
                break
                
            img1_id = img_ids[i]
            img1 = images[img1_id]
            
            for j in range(i+1, len(img_ids)):
                if processed_pairs >= max_pairs:
                    break
                    
                img2_id = img_ids[j]
                img2 = images[img2_id]
                
                # Find common 3D points between the two images using the xys dictionary
                common_points = set(img1.xys.keys()).intersection(set(img2.xys.keys()))
                
                # Skip if not enough covisible points
                if len(common_points) < min_covisible_points:
                    continue
                
                # Get cameras
                K1 = colmap_utils.get_cam(img1, cameras)
                
                # Apply rescaling factor (divide width and height by 6)
                scale_factor = 1
                
                # Adjust the intrinsic matrix - divide focal length and principal point by scale_factor
                K1_scaled = K1.copy()
                K1_scaled[0, 0] /= scale_factor  # fx
                K1_scaled[1, 1] /= scale_factor  # fy
                K1_scaled[0, 2] /= scale_factor  # cx
                K1_scaled[1, 2] /= scale_factor  # cy
                
                # Calculate relative pose
                R1 = img1.qvec2rotmat()
                t1 = img1.tvec
                R2 = img2.qvec2rotmat()
                t2 = img2.tvec
                
                # Original calculation
                R_rel = (R2 @ R1.T)
                t_rel = (t2-R2@R1.T@t1)
                
                # Skip pairs with small translation
                if np.linalg.norm(t_rel) < min_translation_norm:
                    continue
                # Normalize t_rel to unit vector
                t_rel_norm = t_rel / np.linalg.norm(t_rel)
                
                # Extract corresponding points
                pts1 = []
                pts2 = []
                
                for point3D_id in common_points:
                    xy1 = img1.xys[point3D_id]
                    xy2 = img2.xys[point3D_id]
                    # Rescale the points by dividing by scale_factor
                    xy1_scaled = [xy1[0] / scale_factor, xy1[1] / scale_factor]
                    xy2_scaled = [xy2[0] / scale_factor, xy2[1] / scale_factor]
                    pts1.append(xy1_scaled)
                    pts2.append(xy2_scaled)
                
                pts1 = np.array(pts1, dtype=np.float64)
                pts2 = np.array(pts2, dtype=np.float64)
                
                # Create pairs with dummy side information (for compatibility)
                pairs = np.hstack((
                    pts1,
                    pts2,
                    np.zeros((len(pts1), 3))  # Placeholder for side info: dist, rel_scale, rel_orient
                ))
                
                if len(pairs) >= min_matches:
                    self.pts.append(pairs)
                    self.R_gt.append(R_rel)
                    self.t_gt.append(t_rel_norm)
                    self.intrinsics.append(K1_scaled)  # Use the scaled intrinsic matrix
                    
                    img1_path = "%s/%s" % (path, img1.name)
                    img2_path = "%s/%s" % (path, img2.name)
                    self.img_paths.append((img1_path, img2_path))
                    
                    processed_pairs += 1

        print(f"Created {len(self.pts)} pairs for training/evaluation")
                
        # Precompute F_gt and virtual points for training mode
        if mode == "train" and compute_virtual_points:
            self.F_gt = []
            self.pts1_virt = []
            self.pts2_virt = []
            
            print("Computing fundamental matrices and virtual points for training...")
            for i in tqdm(range(len(self.pts))):
                # Calculate fundamental matrix from R, t, and K
                R = self.R_gt[i]
                t = self.t_gt[i]
                K_mat = self.intrinsics[i]
                
                # Calculate essential matrix
                t_cross = np.array([
                    [0, -t[2], t[1]],
                    [t[2], 0, -t[0]],
                    [-t[1], t[0], 0]
                ])
                E = t_cross @ R
                
                # Calculate fundamental matrix
                F = np.linalg.inv(K_mat).T @ E @ np.linalg.inv(K_mat)
                F = F / np.linalg.norm(F)  # Normalize F
                self.F_gt.append(F)
                
                # Generate virtual points for training
                if compute_virtual_points:
                    # Try to load sample image to get dimensions
                    try:
                        img1_path, _ = self.img_paths[i]
                        img = cv2.imread(img1_path)
                        if img is not None:
                            h, w = img.shape[:2]
                        else:
                            h, w = 2000, 3000  # Default size for ETH3D
                    except:
                        h, w = 2000, 3000  # Default size if image loading fails
                    
                    pts1_virt = np.zeros((100, 3))
                    pts2_virt = np.zeros((100, 3))
                    
                    # Create grid of points
                    for j in range(10):
                        for k in range(10):
                            idx = j * 10 + k
                            x = w * (k + 0.5) / 10
                            y = h * (j + 0.5) / 10
                            pts1_virt[idx] = [x, y, 1]
                            pts2_virt[idx] = [x, y, 1]
                    
                    self.pts1_virt.append(pts1_virt)
                    self.pts2_virt.append(pts2_virt)

    def __getitem__(self, index):
        """Get dataset sample.

        Args:
            index (int): sample index

        Returns:
            If mode is "train":
                tuple: points, side information, ground truth fundamental matrix, virtual points 1, virtual points 2
            If mode is "test":
                tuple: points, side information, ground truth rotation and translation, camera matrix
        """
        pts = self.pts[index]
        
        # add data if too small for training
        if self.num_points > 0 and pts.shape[0] < self.num_points:
            while pts.shape[0] < self.num_points:
                num_missing = self.num_points - pts.shape[0]
                idx = np.random.permutation(pts.shape[0])[:num_missing]

                pts_pert = pts[idx]
                pts = np.concatenate((pts, pts_pert), 0)

        # normalize side information
        side_info = pts[:, 4:] / np.maximum(np.amax(pts[:, 4:], 0), 1e-10)

        # Get point coordinates
        pts = pts[:, :4]

        # remove data if too big for training
        if self.num_points > 0 and (pts.shape[0] > self.num_points):
            idx = np.random.permutation(pts.shape[0])[: self.num_points]

            pts = pts[idx, :]
            side_info = side_info[idx]

        if self.mode == "train":
            # Return data for training
            F_gt = self.F_gt[index]
            pts1_virt = self.pts1_virt[index]
            pts2_virt = self.pts2_virt[index]
            return (np.float32(pts), side_info, np.float32(F_gt), np.float32(pts1_virt), np.float32(pts2_virt))
        else:
            # Return data for testing
            R_gt = self.R_gt[index]
            t_gt = self.t_gt[index]
            K = self.intrinsics[index]
            return (np.float32(pts), side_info, np.float32(R_gt), np.float32(t_gt), np.float32(K))
    
    def __len__(self):
        """Get length of dataset.

        Returns:
            int: length
        """
        return len(self.R_gt)


