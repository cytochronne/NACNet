from models import Models
from utils import path_utils, trainer_utils, arg_parser
from Datasets.dataset_utils import get_test_dataloader
import pandas as pd
import copy


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
    
    sequences = args.sequences.split(',') if args.sequences else ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    
    for seq in sequences:
        seq_args = copy.copy(args)  # 正确使用copy模块复制args
        seq_args.data_type = "KITTI"  # 确保使用KITTI数据集
        seq_args.sequence = seq       # 设置当前序列
        print(f"(information from arg)sequence will be dealed with: {seq}")

        exp_errors, samples_errors = eval_exp(seq_args, model)
        
        # Get exp errors
        exp_errors_pd = pd.DataFrame.from_dict(exp_errors)
        exp_errors_pd.index = [args.run_name]


if __name__ == "__main__":
    eval_runs()