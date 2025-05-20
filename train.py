import torch
from alphafold3_pytorch import Alphafold3, Trainer, create_trainer_from_yaml
from alphafold3_pytorch.inputs import PDBDataset

# 从 YAML 配置文件创建训练器
trainer = create_trainer_from_yaml('./tests/configs/trainer_with_pdb_dataset.yaml')

# 训练模型
for step in range(trainer.num_train_steps):
    batch = next(trainer.dataloader)
    losses = trainer.model(**batch.model_forward_dict())
    loss = sum(losses.values())

    # 梯度累积
    loss = loss / trainer.grad_accum_every
    loss.backward()

    if (step + 1) % trainer.grad_accum_every == 0:
        # 梯度裁剪
        if trainer.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.clip_grad_norm)

        # 更新参数
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
        trainer.scheduler.step()

    # 打印训练信息
    if step % 10 == 0:
        print(f"Step {step}: Loss = {loss.item()}")

    # 验证模型
    if trainer.needs_valid and (step + 1) % trainer.valid_every == 0:
        trainer.model.eval()
        valid_losses = []
        with torch.no_grad():
            for valid_batch in trainer.valid_dataloader:
                valid_loss = trainer.model(**valid_batch.model_forward_dict())
                valid_loss = sum(valid_loss.values())
                valid_losses.append(valid_loss.item())
        avg_valid_loss = sum(valid_losses) / len(valid_losses)
        print(f"Step {step}: Validation Loss = {avg_valid_loss}")
        trainer.model.train()

    # 保存检查点
    if (step + 1) % trainer.checkpoint_every == 0:
        checkpoint_path = trainer.checkpoint_folder / f"{trainer.checkpoint_prefix}{step}"
        torch.save({
            'step': step,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict()
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

print("Training finished.")
