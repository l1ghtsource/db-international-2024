import argparse
import os
from utils.config import load_config
from train import train_process
from inference import inference

def parse_args():
    parser = argparse.ArgumentParser(description='RKN Model')
    parser.add_argument(
        '--mode', type=str, choices=['train', 'inference'], required=True,
        help='Run mode: train or inference'
    )
    parser.add_argument(
        '--data-path', type=str, required=True,
        help='Path to your dataset in necessary format'
    )
    parser.add_argument(
        '--config', type=str, default='configs/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--save-model-path', type=str, required=True,
        help='Path to model weights of your experiments'
    )
    parser.add_argument(
        '--model-config', type=str, default='configs/model_configs/clip.yaml',
        help='Path to model config file'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to model checkpoint for inference'
    )
    parser.add_argument(
        '--wandb-key', type=str, default=None,
        help='WandB API key'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config, args.model_config)

    if args.data_path:
        config['data']['base_path'] = args.data_path
    if args.wandb_key:
        os.environ['WANDB_API_KEY'] = args.wandb_key
    if args.save_model_path:
        config['training']['save_model_path'] = args.save_model_path

    if args.mode == 'train':
        train_process(config)
    else:
        if args.checkpoint is None:
            raise ValueError('Checkpoint path is required for inference mode')
        inference(config, args.checkpoint)


if __name__ == '__main__':
    main()