import h5py
import numpy as np

file_path = '/hpc2hdd/home/qzhang749/yanzhe/test/NACNet/OANet/data_dump/kitti-00-sift-1000-test.hdf5'

with h5py.File(file_path, 'r') as f:
    # 打印所有顶级键
    print("Keys: ", list(f.keys()))
    
    # 遍历所有键
    for key in f.keys():
        item = f[key]
        print(f"\nKey: {key}")
        print(f"  - Type: {type(item)}")
        
        # 区分处理组和数据集
        if isinstance(item, h5py.Group):
            print(f"  - Group containing keys: {list(item.keys())}")
        elif isinstance(item, h5py.Dataset):
            print(f"  - Shape: {item.shape}")
            print(f"  - Dtype: {item.dtype}")
            # 打印数据样本
            try:
                sample = item[0] if item.shape[0] > 0 else "Empty dataset"
                print(f"  - First item shape: {sample.shape if hasattr(sample, 'shape') else 'scalar'}")
                print(f"  - Sample data: {sample[:3] if hasattr(sample, 'shape') and len(sample.shape) > 0 else sample}")
            except Exception as e:
                print(f"  - Could not preview data: {e}")
    
    # 重点查看xs的信息
    if 'xs' in f:
        xs = f['xs']
        print("\n==== DETAILED INFO FOR 'xs' ====")
        print(f"Shape: {xs.shape}")
        print(f"Dtype: {xs.dtype}")
        print(f"Total number of samples: {xs.shape[0]}")
        
        # 查看第一个样本的形状和值
        if xs.shape[0] > 0:
            first_sample = xs[0]
            print(f"First sample shape: {first_sample.shape}")
            print(f"First sample first few points:")
            # 显示前3个点对
            for i in range(min(3, first_sample.shape[0])):
                print(f"  Point pair {i}: {first_sample[i]}")
