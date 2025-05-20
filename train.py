#!/usr/bin/python3

from absl import flags, app
import torch
from alphafold3_pytorch import Alphafold3, Trainer, create_trainer_from_yaml
from alphafold3_pytorch.inputs import PDBDataset

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_bool("weighted_sampling", default = False, help = 'whether weighted sampling')

def main(unused_argv):
  # 从 YAML 配置文件创建训练器
  config_path = './tests/configs/trainer_with_pdb_dataset.yaml' if FLAGS.weighted_sampling == False else \
                './tests/configs/trainer_with_pdb_dataset_and_weighted_sampling.yaml'
  # NOTE: github.com/lucidrains/alphafold3-pytorch/blob/main/tests/test_trainer.py#L166
  trainer = create_trainer_from_yaml(config_path)

  # 训练模型
  trainer()
  print("Training finished.")

if __name__ == "__main__":
  add_options()
  app.run(main)
