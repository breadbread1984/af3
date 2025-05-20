import torch
from alphafold3_pytorch import Alphafold3, Trainer, create_trainer_from_yaml
from alphafold3_pytorch.inputs import PDBDataset

# 从 YAML 配置文件创建训练器
trainer = create_trainer_from_yaml('./tests/configs/trainer_with_pdb_dataset.yaml')

# 训练模型

trainer()

print("Training finished.")
