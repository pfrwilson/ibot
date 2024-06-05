from omegaconf import OmegaConf
from src.utils import pull_conf_from_wandb
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('wandb_path', type=str)
parser.add_argument('conf_path', type=str)

args = parser.parse_args()

conf = pull_conf_from_wandb(args.wandb_path)
OmegaConf.save(conf, args.conf_path)