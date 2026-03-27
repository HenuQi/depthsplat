from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

from .dataset.data_module import DataLoaderCfg, DatasetCfg
from .loss import LossCfgWrapper
from .model.decoder import DecoderCfg
from .model.encoder import EncoderCfg
from .model.model_wrapper import OptimizerCfg, TestCfg, TrainCfg

# config.py功能：
# 把 Hydra 的 字典配置DictConfig 转成 带类型的数据结构RootCfg，
# 然后实例化 logger、checkpoint、Trainer、DataModule、encoder、decoder、loss 和 ModelWrapper。

# 1. 先配置 dataclass 定义（类型结构）
@dataclass
class CheckpointingCfg:
    load: Optional[str]  # Not a path, since it could be something like wandb://...
    every_n_train_steps: int
    save_top_k: int
    pretrained_model: Optional[str]
    ############################新增
    pretrained_vggt: Optional[str]
    ############################
    pretrained_monodepth: Optional[str]
    pretrained_mvdepth: Optional[str]
    pretrained_depth: Optional[str]
    no_strict_load: bool
    resume: bool


@dataclass
class ModelCfg:
    decoder: DecoderCfg
    encoder: EncoderCfg


@dataclass
class TrainerCfg:
    max_steps: int
    val_check_interval: int | float | None
    gradient_clip_val: int | float | None
    num_sanity_val_steps: int
    num_nodes: int

# 2. 把所有子配置组合成根配置 RootCfg 
@dataclass
class RootCfg:
    wandb: dict
    mode: Literal["train", "test"]
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    model: ModelCfg
    optimizer: OptimizerCfg
    checkpointing: CheckpointingCfg
    trainer: TrainerCfg
    loss: list[LossCfgWrapper]
    test: TestCfg
    train: TrainCfg
    seed: int
    use_plugins: bool


TYPE_HOOKS = {      # 类型钩子
    Path: Path,     # 目标类型 : 转换函数。 
    # 作用：让 Dacite 在遇到 Path 类型时直接使用 Path 构造函数进行转换
    # 即：如果 dataclass 字段类型是 Path， 就调用 Path(value)
}


T = TypeVar("T") # 定义一个名字为 T 的类型变量(T 代表 任意类型，但在一次使用中保持一致)

# 把 Hydra 的 DictConfig 配置对象转换成一个带类型的 Python dataclass 对象。
# Hydra DictConfig -> 普通 dict -> dataclass -> 返回 typed config
def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],                # 要转换成的 dataclass 类型
    extra_type_hooks: dict = {},        # 此次调用额外增加的 hook。（要在类型转换规则字典里新增的类型转换规则）
) -> T:
    return from_dict(                   # 2.再把普通 dict 转成 dataclass 实例
        data_class,
        OmegaConf.to_container(cfg),    # 1.先把 DictConfig 转成普通 dict 
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
        # {**A, **B}：表示合并两个字典
    )
# dacite 的逻辑：
# if 类型不匹配:
#     查 hook
#     执行 hook

# 由于 loss 配置是一个列表，Dacite 无法直接处理，所以定义一个特殊的函数来处理这个情况。
def separate_loss_cfg_wrappers(joined: dict) -> list[LossCfgWrapper]: # 把 loss 配置列表从 DictConfig 转成 list[LossCfgWrapper]，每个元素都是一个 LossCfgWrapper 实例。
    # The dummy allows the union to be converted.
    @dataclass
    class Dummy:
        dummy: LossCfgWrapper

    return [
        load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy
        for k, v in joined.items()
    ]


def load_typed_root_config(cfg: DictConfig) -> RootCfg: # 把 DictConfig类型的实例 转成 RootCfg 类型，过程中应用类型钩子来处理特殊类型（如 Path）和 wrapper 列表。
    return load_typed_config(
        cfg,
        RootCfg,
        {list[LossCfgWrapper]: separate_loss_cfg_wrappers},
    )
