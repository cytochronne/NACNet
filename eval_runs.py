from models import Models
from utils import path_utils, trainer_utils, arg_parser
from Datasets.dataset_utils import get_test_dataloader
import pandas as pd
import copy
import os


def eval_exp(args, model):
    # Run test
    test_loader = get_test_dataloader(args, known_scenes=args.eval_knwon_scenes)
    print("data loaded")
    trainer = trainer_utils.get_trainer(args)
    exp_errors = trainer.test(model, dataloaders=test_loader)

    # Combine all errors
    samples_err_pd = pd.DataFrame.from_dict(trainer.model.test_step_errors)
    
    return exp_errors, samples_err_pd


def eval_runs():
    args = arg_parser.get_args()

    # Load model
    ckpt_path = '/hpc2hdd/home/qzhang749/yanzhe/test/NACNet/Experiments/test/17_04_2025_14_57_25/chckpt/epoch=9-step=169120.ckpt'
    print(f"Loading model from: {ckpt_path}")
    model = Models.RobustModel.load_from_checkpoint(ckpt_path, args=args)

    print(f"# # # # # # # # Evaluating {args.run_name}:{args.version} # # # # # # # # ")
    
    sequences = args.sequences.split(',') if args.sequences else ['00']
    
    for seq in sequences:
        seq_args = copy.copy(args)  # 正确使用copy模块复制args
        seq_args.data_type = "KITTI"  # 确保使用KITTI数据集
        seq_args.sequence = seq       # 设置当前序列
        print(f"(information from arg)sequence will be dealed with: {seq}")

        exp_errors, samples_errors = eval_exp(seq_args, model)
        
        # Get exp errors
        exp_errors_pd = pd.DataFrame.from_dict(exp_errors)
        exp_errors_pd.index = [args.run_name]

        # 保存每个样本的单独误差
        output_dir = os.path.join("results", args.run_name)
        os.makedirs(output_dir, exist_ok=True)
        samples_csv_path = os.path.join(output_dir, f"{args.run_name}_{seq}_sample_errors.csv")
        samples_errors.to_csv(samples_csv_path)
        print(f"Individual sample errors saved to {samples_csv_path}")
        
        # 同样保存聚合误差
        exp_csv_path = os.path.join(output_dir, f"{args.run_name}_{seq}_aggregate_errors.csv")
        exp_errors_pd.to_csv(exp_csv_path)
        print(f"Aggregate errors saved to {exp_csv_path}")
if __name__ == "__main__":
    eval_runs()