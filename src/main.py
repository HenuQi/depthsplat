import os
from pathlib import Path
import warnings
import copy

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.wandb import WandbLogger

from pytorch_lightning.plugins.environments import LightningEnvironment


# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.misc.resume_ckpt import find_latest_ckpt
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
# 1. 加载配置文件，设置全局配置对象
    if cfg_dict["mode"] == "train" and cfg_dict["train"]["eval_model_every_n_val"] > 0:
        # 在训练阶段，构造一个专门用于验证（evaluation）的配置 eval_cfg，用于在训练过程中定期评估模型性能。
        # 在 训练模式 且 设置了验证频率 时才会构造验证配置。
        eval_cfg_dict = copy.deepcopy(cfg_dict)
        dataset_dir = str(cfg_dict["dataset"]["roots"]).lower()
        if "re10k" in dataset_dir:
            eval_path = "assets/evaluation_index_re10k.json"
        elif "dl3dv" in dataset_dir:
            if cfg_dict["dataset"]["view_sampler"]["num_context_views"] == 6:
                eval_path = "assets/dl3dv_start_0_distance_50_ctx_6v_tgt_8v.json"
            else:
                raise ValueError("unsupported number of views for dl3dv")
        else:
            raise Exception("Fail to load eval index path")
        eval_cfg_dict["dataset"]["view_sampler"] = {
            "name": "evaluation",
            "index_path": eval_path,
            "num_context_views": cfg_dict["dataset"]["view_sampler"]["num_context_views"],
        }
        eval_cfg = load_typed_root_config(eval_cfg_dict)
    else:
        eval_cfg = None

    cfg = load_typed_root_config(cfg_dict)  # 把 DictConfig类型 转成 RootCfg 类型
    set_cfg(cfg_dict)       # 设置全局配置对象global_cfg.cfg = cfg_dict，供其他模块访问。

# 2. 设置输出目录，
    # Set up the output directory.
    if cfg_dict.output_dir is None: # 如果没有指定输出目录，就用 Hydra 默认的输出目录
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:  # for resuming  如果指定了输出目录，就用指定的目录
        output_dir = Path(cfg_dict.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    print(cyan(f"Saving outputs to {output_dir}."))
# 3. 设置wandb日志记录器和（回调函数）
    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled" and cfg.mode == "train":
        wandb_extra_kwargs = {}
        if cfg_dict.wandb.id is not None:
            wandb_extra_kwargs.update({'id': cfg_dict.wandb.id,
                                       'resume': "must"})
        logger = WandbLogger(
            entity=cfg_dict.wandb.entity,
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=os.path.basename(cfg_dict.output_dir),
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            **wandb_extra_kwargs,
        )
        callbacks.append(LearningRateMonitor("step", True))

        if wandb.run is not None:
            wandb.run.log_code("src")
    else: # 如果不启用W&B，使用本地日志记录器LocalLogger 进行日志记录
        logger = LocalLogger()

# 4. 设置模型检查点（回调函数）
    # Set up checkpointing.
    callbacks.append( # 添加模型检查点回调 ModelCheckpoint，保存训练过程中的模型权重和状态
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            monitor="info/global_step",
            mode="max",
        )
    )
    for cb in callbacks:
        cb.CHECKPOINT_EQUALS_CHAR = '_'

    # Prepare the checkpoint for loading.
    if cfg.checkpointing.resume: # resume 继续训练，就从本地加载
        if not os.path.exists(output_dir / 'checkpoints'):
            checkpoint_path = None
        else:
            checkpoint_path = find_latest_ckpt(output_dir / 'checkpoints')
            print(f'resume from {checkpoint_path}')
    else: # 否则从 wandb 加载预训练模型
        checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

# 5. 设置StepTracker() ：跨进程共享当前 step 的工具
    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()
# 6. 设置 PyTorch Lightning 的 Trainer 对象，指定训练参数、日志记录器、回调函数等。
    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices=torch.cuda.device_count(),
        strategy='ddp' if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=cfg.mode == "test",
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        num_nodes=cfg.trainer.num_nodes,
        plugins=LightningEnvironment() if cfg.use_plugins else None,
        # precision="bf16-mixed",  # （但是Gaussian Rasterizer不支持！！）<--- 我新增了这一行！开启 bfloat16 混合精度，加快训练速度，同时保持数值稳定性。
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
# 7. 设置模型包装器 ModelWrapper 
    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder, cfg.dataset),
        get_losses(cfg.loss),
        step_tracker,
        eval_data_cfg=(
            None if eval_cfg is None else eval_cfg.dataset
        ),
    )
# 8. 设置数据模块 DataModule
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )
# 9. 根据训练模式（train/test）执行相应的训练或测试流程
    if cfg.mode == "train":
        print("train:", len(data_module.train_dataloader()))
        print("val:", len(data_module.val_dataloader()))
        print("test:", len(data_module.test_dataloader()))

    strict_load = not cfg.checkpointing.no_strict_load

    if cfg.mode == "train":
        # only load monodepth
        if cfg.checkpointing.pretrained_monodepth is not None:
            strict_load = False
            pretrained_model = torch.load(cfg.checkpointing.pretrained_monodepth, map_location='cpu')
            if 'state_dict' in pretrained_model:
                pretrained_model = pretrained_model['state_dict']

            model_wrapper.encoder.depth_predictor.load_state_dict(pretrained_model, strict=strict_load)
            print(
                cyan(
                    f"Loaded pretrained monodepth: {cfg.checkpointing.pretrained_monodepth}"
                )
            )

        # load pretrained mvdepth
        if cfg.checkpointing.pretrained_mvdepth is not None:
            pretrained_model = torch.load(cfg.checkpointing.pretrained_mvdepth, map_location='cpu')['model']

            model_wrapper.encoder.depth_predictor.load_state_dict(pretrained_model, strict=False)
            print(
                cyan(
                    f"Loaded pretrained mvdepth: {cfg.checkpointing.pretrained_mvdepth}"
                )
            )
        
        # load full model
        if cfg.checkpointing.pretrained_model is not None:
            pretrained_model = torch.load(cfg.checkpointing.pretrained_model, map_location='cpu')
            if 'state_dict' in pretrained_model:
                pretrained_model = pretrained_model['state_dict']

            model_wrapper.load_state_dict(pretrained_model, strict=strict_load)
            print(
                cyan(
                    f"Loaded pretrained weights: {cfg.checkpointing.pretrained_model}"
                )
            )

        # load pretrained depth
        if cfg.checkpointing.pretrained_depth is not None:
            pretrained_model = torch.load(cfg.checkpointing.pretrained_depth, map_location='cpu')['model']

            strict_load = True
            model_wrapper.encoder.depth_predictor.load_state_dict(pretrained_model, strict=strict_load)
            print(
                cyan(
                    f"Loaded pretrained depth: {cfg.checkpointing.pretrained_depth}"
                )
            )
        # **********trainer.fit() 开始训练**********
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        # load full model
        if cfg.checkpointing.pretrained_model is not None:
            pretrained_model = torch.load(cfg.checkpointing.pretrained_model, map_location='cpu')
            if 'state_dict' in pretrained_model:
                pretrained_model = pretrained_model['state_dict']

            model_wrapper.load_state_dict(pretrained_model, strict=strict_load)
            print(
                cyan(
                    f"Loaded pretrained weights: {cfg.checkpointing.pretrained_model}"
                )
            )

        # load pretrained depth model only
        if cfg.checkpointing.pretrained_depth is not None:
            pretrained_model = torch.load(cfg.checkpointing.pretrained_depth, map_location='cpu')['model']

            strict_load = True
            model_wrapper.encoder.depth_predictor.load_state_dict(pretrained_model, strict=strict_load)
            print(
                cyan(
                    f"Loaded pretrained depth: {cfg.checkpointing.pretrained_depth}"
                )
            )
        # **********trainer.test() 开始测试**********
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')

    train()
