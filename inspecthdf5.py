import h5py

file_path = '/hpc2hdd/home/qzhang749/yanzhe/test/NACNet/OANet/data_dump/kitti-sift-1000-test.hdf5'  # 替换为你的 hdf5 文件路径

with h5py.File(file_path, 'r') as f:
    for key in f.keys():
        dataset = f[key]
        print(f"\nKey: {key}")
        print(f"  - Type: {type(dataset)}")
        print(f"  - Shape: {dataset.shape}")
        print(f"  - Dtype: {dataset.dtype}")
        # 打印前几项以查看内容
        try:
            print(f"  - Sample data: {dataset[:5]}")  # 打印前5行数据
        except Exception as e:
            print(f"  - Could not preview data: {e}")
