#!/usr/bin/env python3
import os
import shutil
import numpy as np
import glob
import argparse
from tqdm import tqdm
import math

def filter_kitti_sequence(sequence_id, input_path, output_path, min_displacement=0.5):
    """
    Filter KITTI sequence based on minimum displacement threshold.
    
    Args:
        sequence_id: Sequence ID (e.g., '00', '01', etc.)
        input_path: Base input directory containing Datasets folder
        output_path: Base output directory to save filtered data
        min_displacement: Minimum displacement threshold in meters
    """
    # Paths
    pose_file = os.path.join(input_path, 'poses', f'{sequence_id}_rel.txt')
    img_dir = os.path.join(input_path, sequence_id, 'image_0')
    calib_file = os.path.join(input_path, sequence_id, 'calib.txt')
    times_file = os.path.join(input_path, sequence_id, 'times.txt')
    
    # Create output directories
    out_seq_dir = os.path.join(output_path, sequence_id)
    out_img_dir = os.path.join(out_seq_dir, 'image_0')
    os.makedirs(out_img_dir, exist_ok=True)
    
    # Read pose file
    if not os.path.exists(pose_file):
        print(f"Pose file not found: {pose_file}")
        return
        
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = [float(v) for v in line.strip().split()]
            poses.append(values)
    
    # Get all image files
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    
    if (len(img_files)-1) != len(poses):
        print(f"Warning: Number of images ({len(img_files)-1}) doesn't match number of poses ({len(poses)})")
        return
        
    # Read times file if it exists
    times = []
    if os.path.exists(times_file):
        with open(times_file, 'r') as f:
            times = [float(line.strip()) for line in f]
    
    # Filter based on displacement
    filtered_indices = []
    filtered_poses = []
    
    for i, pose in enumerate(poses):
        # Calculate displacement (magnitude of the last 3 values in the pose)
        displacement = math.sqrt(pose[3]**2 + pose[7]**2 + pose[11]**2)
        print(f"Frame {i}: Position ({pose[3]:.3f}, {pose[7]:.3f}, {pose[11]:.3f}) - Displacement: {displacement:.3f}m")
        
        if displacement >= min_displacement:
            filtered_indices.append(i)
            filtered_poses.append(pose)
    
    # Copy filtered images and create new poses file
    print(f"Sequence {sequence_id}: Keeping {len(filtered_indices)} out of {len(poses)} frames (displacement >= {min_displacement}m)")
    
    # Create output poses directory if it doesn't exist
    os.makedirs(os.path.join(output_path, 'poses_rel'), exist_ok=True)
    
    # Save filtered poses with rearranged order
    with open(os.path.join(output_path, 'poses_rel', f'{sequence_id}.txt'), 'w') as f:
        for pose in filtered_poses:
            # Get values to move
            p3, p7, p11 = pose[3], pose[7], pose[11]
            
            # Create a new pose array with all values except those at indices 3, 7, 11
            new_pose = []
            for i, value in enumerate(pose):
                if i not in [3, 7, 11]:
                    new_pose.append(value)
            
            # Add the position values at the end (positions 9, 10, 11)
            new_pose.append(p3)
            new_pose.append(p7)
            new_pose.append(p11)
            
            # Write to file
            f.write(' '.join([str(v) for v in new_pose]) + '\n')
    
    # Copy filtered images
    for i, idx in enumerate(filtered_indices):
        src_img = img_files[idx]
        dst_img = os.path.join(out_img_dir, os.path.basename(src_img))
        shutil.copy(src_img, dst_img)
    
    # Copy calibration file
    if os.path.exists(calib_file):
        shutil.copy(calib_file, os.path.join(out_seq_dir, 'calib.txt'))
    
    # Create filtered times file if original exists
    if times:
        with open(os.path.join(out_seq_dir, 'times.txt'), 'w') as f:
            for idx in filtered_indices:
                f.write(f"{times[idx]}\n")
    
    return len(filtered_indices)

def main():
    parser = argparse.ArgumentParser(description='Filter KITTI sequences based on minimum displacement.')
    parser.add_argument('--input_path', type=str, default='Datasets',
                      help='Path to the KITTI dataset directory')
    parser.add_argument('--output_path', type=str, default='Datasets_filtered',
                      help='Path to save filtered dataset')
    parser.add_argument('--min_displacement', type=float, default=0.5,
                      help='Minimum displacement threshold in meters')
    parser.add_argument('--sequences', type=str, default='00,01,02,03,04,05,06,07,08,09,10',
                      help='Comma-separated list of sequence IDs to process')
    
    args = parser.parse_args()
    
    # Ensure absolute paths
    input_path = os.path.abspath(args.input_path)
    output_path = os.path.abspath(args.output_path)
    
    # Get the list of sequences to process
    sequences = args.sequences.split(',')
    
    # Process each sequence
    total_frames = 0
    filtered_frames = 0
    
    for seq in sequences:
        print(f"Processing sequence {seq}...")
        kept_frames = filter_kitti_sequence(seq, input_path, output_path, args.min_displacement)
        if kept_frames is not None:
            # Get total frames in this sequence
            seq_total = len(glob.glob(os.path.join(input_path, seq, 'image_0', '*.png')))
            total_frames += seq_total
            filtered_frames += kept_frames
    
    # Print summary
    if total_frames > 0:
        print(f"\nFiltering complete: Kept {filtered_frames} out of {total_frames} frames ({filtered_frames/total_frames*100:.2f}%)")
    else:
        print("\nNo frames were processed. Check your input paths and sequence IDs.")

if __name__ == '__main__':
    main()