import os
import sys
import copy
import json
from typing import Any, Dict

import optuna
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

sys.path.append(os.getcwd())

from models.seq2seq import ChartGenerator
from data.dataset import MaiGenDataModule


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_datamodule(data_cfg: Dict[str, Any]) -> pl.LightningDataModule:
    return MaiGenDataModule(**data_cfg)


def build_model(model_cfg: Dict[str, Any]) -> pl.LightningModule:
    return ChartGenerator(**model_cfg)


def get_all_distributions() -> Dict[str, optuna.distributions.BaseDistribution]:
    """
    Define the superset of all possible distributions used in objective variables.
    This allows us to reconstruct the distributions for loading past trials.
    """
    from optuna.distributions import LogUniformDistribution, CategoricalDistribution, IntDistribution, FloatDistribution
    
    return {
        "lr": LogUniformDistribution(1e-5, 1e-3),
        "dropout": FloatDistribution(0.0, 0.3),
        "d_model": CategoricalDistribution([128, 256, 512]),
        "nhead": CategoricalDistribution([4, 8]),
        "num_mamba_layers": IntDistribution(0, 4),
        "num_encoder_attn_layers": IntDistribution(1, 4),
        "num_decoder_layers": IntDistribution(2, 8),
        "dim_feedforward": CategoricalDistribution([512, 1024, 2048]),
    }


def load_trials_from_logs(study_name: str) -> list[optuna.trial.FrozenTrial]:
    """
    Load past trials from optuna_logs/mai_gen/version_*/hparams.yaml & metrics.csv
    Returns a list of FrozenTrial objects to be added to the study.
    """
    log_root = os.path.join('optuna_logs', 'mai_gen')
    if not os.path.exists(log_root):
        print(f"[warn] No existing logs found at {log_root}, cannot resume.")
        return []

    print(f"[info] Scanning for existing logs in {log_root}...")
    trials = []
    
    known_distributions = get_all_distributions()
    
    # Traverse version directories
    existing_dirs = sorted([
        d for d in os.listdir(log_root) 
        if d.startswith("version_") and os.path.isdir(os.path.join(log_root, d))
    ])
    
    for d in existing_dirs:
        version_dir = os.path.join(log_root, d)
        hparams_path = os.path.join(version_dir, "hparams.yaml")
        metrics_path = os.path.join(version_dir, "metrics.csv")
        
        if not os.path.exists(hparams_path) or not os.path.exists(metrics_path):
            continue
            
        try:
            import yaml
            import csv
            
            # Load Params
            with open(hparams_path, "r") as f:
                hparams = yaml.safe_load(f)
            
            # Extract basic info
            monitor_key = hparams.pop('monitor', 'val/loss') # Default if missing
            hparams.pop('trial_number', None) # Remove non-param
            
            # Filter params and build trial distributions
            filtered_params = {}
            trial_distributions = {}
            
            for key, value in hparams.items():
                if key in known_distributions:
                    filtered_params[key] = value
                    trial_distributions[key] = known_distributions[key]
            
            if not filtered_params:
                continue

            # Load Metrics to find best score (min)
            best_score = float('inf')
            found_metric = False
            
            with open(metrics_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if monitor_key in row and row[monitor_key]:
                        try:
                            val = float(row[monitor_key])
                            if val < best_score:
                                best_score = val
                            found_metric = True
                        except ValueError:
                            pass
            
            if not found_metric:
                continue

            # Create FrozenTrial
            trial_obj = optuna.trial.create_trial(
                params=filtered_params,
                distributions=trial_distributions,
                value=best_score,
                state=optuna.trial.TrialState.COMPLETE,
            )
            trials.append(trial_obj)
        except Exception as e:
            print(f"[warn] Failed to load trial from {d}: {e}")
            
    print(f"[info] Recovered {len(trials)} trials from logs.")
    return trials


def objective(
    trial: optuna.Trial,
    base_cfg_path: str,
    max_epochs: int,
    device_count: int,
    limit_train_batches: float = 1.0,
    limit_val_batches: float = 1.0,
    monitor_key: str = "val/loss",
    use_pruning: bool = True,
) -> float:
    cfg = load_yaml(base_cfg_path)

    # --- 采样超参 ---
    model_cfg = copy.deepcopy(cfg.get("model", {}))
    
    model_cfg["lr"] = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    model_cfg["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
    
    model_cfg["d_model"] = trial.suggest_categorical("d_model", [128, 256, 512])
    model_cfg["nhead"] = trial.suggest_categorical("nhead", [4, 8])
    model_cfg["num_mamba_layers"] = trial.suggest_int("num_mamba_layers", 0, 4)
    model_cfg["num_encoder_attn_layers"] = trial.suggest_int("num_encoder_attn_layers", 1, 4)
    model_cfg["num_decoder_layers"] = trial.suggest_int("num_decoder_layers", 2, 8)
    model_cfg["dim_feedforward"] = trial.suggest_categorical("dim_feedforward", [512, 1024, 2048])

    cfg["model"] = model_cfg

    # 构造 DataModule 和 Model
    data_cfg = cfg.get("data", {})
    trainer_cfg = cfg.get("trainer", {})

    datamodule = build_datamodule(data_cfg)
    model = build_model(model_cfg)

    # 回调与日志
    logger = CSVLogger(save_dir="optuna_logs", name="mai_gen")
    try:
        logger.log_hyperparams({
            **trial.params,
            'trial_number': trial.number,
            'monitor': monitor_key,
        })
    except Exception as e:
        print(f"[warn] failed to log hyperparameters to CSVLogger: {e}")
        
    ckpt = ModelCheckpoint(dirpath=os.path.join("/tmp", "checkpoints_mai_gen"), save_top_k=1, monitor=monitor_key, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [ckpt, lr_monitor]
    if use_pruning:
        try:
            from optuna.integration import PyTorchLightningPruningCallback
            callbacks.append(PyTorchLightningPruningCallback(trial, monitor=monitor_key))
        except Exception:
            pass

    # Trainer 基础设置
    trainer_kwargs = {
        "accelerator": trainer_cfg.get("accelerator", "auto"),
        "devices": device_count,
        "max_epochs": max_epochs if trainer_cfg.get("max_epochs") is None else min(max_epochs, trainer_cfg.get("max_epochs")),
        "benchmark": trainer_cfg.get("benchmark", True),
        "enable_progress_bar": False,
        "logger": logger,
        "callbacks": callbacks,
        "limit_train_batches": limit_train_batches,
        "limit_val_batches": limit_val_batches,
        "precision": trainer_cfg.get("precision", "32"),
        "check_val_every_n_epoch": trainer_cfg.get("check_val_every_n_epoch", 1),
        "accumulate_grad_batches": trainer_cfg.get("accumulate_grad_batches", 1),
    }

    trainer = Trainer(**trainer_kwargs)

    # 运行一次拟合
    seed_everything(42)
    try:
        trainer.fit(model, datamodule=datamodule)
    except optuna.TrialPruned:
        try:
            import yaml
            args_global = globals().get('args', None)
            study_name = getattr(args_global, 'study', 'mai_gen_optuna')
            export_root = os.path.join('optuna_trials', study_name)
            os.makedirs(export_root, exist_ok=True)
            log_dir = getattr(logger, 'log_dir', None) or os.path.join('optuna_logs', 'mai_gen', f'version_{trial.number}')
            trial_dir = os.path.join(export_root, f"trial_{trial.number:04d}")
            os.makedirs(trial_dir, exist_ok=True)
            with open(os.path.join(trial_dir, 'config.yaml'), 'w') as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            with open(os.path.join(trial_dir, 'params.json'), 'w') as f:
                json.dump(trial.params, f, indent=2, ensure_ascii=False)
            with open(os.path.join(trial_dir, 'result.json'), 'w') as f:
                json.dump({
                    'value': None,
                    'monitor': monitor_key,
                    'state': 'PRUNED',
                    'log_dir': log_dir
                }, f, indent=2, ensure_ascii=False)
            trial.set_user_attr('log_dir', log_dir)
            trial.set_user_attr('trial_dir', os.path.abspath(trial_dir))
        except Exception as e:
            print(f"[warn] failed to save pruned trial artifacts: {e}")
        raise

    # 取验证度量作为目标
    metrics = trainer.callback_metrics
    score = float("inf")
    key_candidates = [monitor_key, "val/loss"]
    for k in key_candidates:
        if k in metrics and metrics[k] is not None:
            try:
                score = float(metrics[k].item())
            except Exception:
                score = float(metrics[k])
            break

    try:
        import yaml
        args_global = globals().get('args', None)
        study_name = getattr(args_global, 'study', 'mai_gen_optuna')
        export_root = os.path.join('optuna_trials', study_name)
        os.makedirs(export_root, exist_ok=True)
        log_dir = getattr(logger, 'log_dir', None) or os.path.join('optuna_logs', 'mai_gen', f'version_{trial.number}')
        trial_dir = os.path.join(export_root, f"trial_{trial.number:04d}")
        os.makedirs(trial_dir, exist_ok=True)

        with open(os.path.join(trial_dir, 'config.yaml'), 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        with open(os.path.join(trial_dir, 'params.json'), 'w') as f:
            json.dump(trial.params, f, indent=2, ensure_ascii=False)
        with open(os.path.join(trial_dir, 'result.json'), 'w') as f:
            json.dump({
                'value': score,
                'monitor': monitor_key,
                'state': str(trial.state),
                'log_dir': log_dir
            }, f, indent=2, ensure_ascii=False)
        trial.set_user_attr('log_dir', log_dir)
        trial.set_user_attr('trial_dir', os.path.abspath(trial_dir))
    except Exception as e:
        print(f"[warn] failed to save trial artifacts: {e}")

    return score


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="YAML 配置路径 (默认针对Mai-Gen)")
    parser.add_argument("--trials", type=int, default=20, help="Optuna 试验次数")
    parser.add_argument("--max_epochs", type=int, default=20, help="每次 trial 的最大学习 epoch")
    parser.add_argument("--devices", type=int, default=1, help="设备数量，用于单机多卡")
    parser.add_argument("--study", type=str, default="mai_gen_optuna", help="Study 名称")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage，例如 sqlite:///mai_gen.db")
    parser.add_argument("--monitor", type=str, default="val/loss", help="剪枝与目标度量键")
    parser.add_argument("--pruner", type=str, default="median", choices=["median", "hyperband", "none"], help="剪枝器类型")
    parser.add_argument("--limit_train_batches", type=float, default=1.0, help="训练 batch 限制 (0-1 或 int)")
    parser.add_argument("--limit_val_batches", type=float, default=1.0, help="验证 batch 限制 (0-1 或 int)")
    parser.add_argument("--topk", type=int, default=10, help="打印前 K 个最优 trial")
    parser.add_argument("--export_dir", type=str, default="optuna_best_mai_gen", help="导出最佳配置目录")
    parser.add_argument("--resume_from_logs", action="store_true", help="Attempt to resume study from local optuna_logs/ CSV logs")
    args = parser.parse_args()
    globals()['args'] = args

    # 环境检查
    if not torch.cuda.is_available() and not getattr(torch.backends, 'mps', None) and args.devices > 1:
        print("CUDA/MPS 不可用，强制 devices=1")
        args.devices = 1

    # 定义剪枝器
    pruner = None
    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=max(5, args.trials // 10), n_warmup_steps=0)
    elif args.pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()

    # 定义 Study
    study = optuna.create_study(
        study_name=args.study,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
        pruner=pruner,
    )

    if args.resume_from_logs:
        recovered_trials = load_trials_from_logs(args.study)
        if recovered_trials:
            print(f"[info] Adding {len(recovered_trials)} recovered trials to study...")
            for t in recovered_trials:
                study.add_trial(t)

    def _objective(trial: optuna.Trial) -> float:
        # 记录 trial 超参
        use_pruning = args.pruner != "none"
        value = objective(
            trial,
            args.config,
            args.max_epochs,
            args.devices,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
            monitor_key=args.monitor,
            use_pruning=use_pruning,
        )
        return value

    study.optimize(_objective, n_trials=args.trials)

    # 打印 TopK
    trials_sorted = sorted([t for t in study.trials if t.values], key=lambda t: t.values[0])
    topk = trials_sorted[: args.topk]
    print("Top Trials:")
    for rank, t in enumerate(topk, 1):
        print(f"#{rank}: value={t.values[0]:.6f}, params={t.params}")

    # 打印最佳
    print("Best Value:", study.best_value)
    print("Best Params:")
    print(json.dumps(study.best_params, indent=2))

    # 导出完整试验汇总
    try:
        args_global = globals().get('args', None)
        study_name = getattr(args_global, 'study', 'mai_gen_optuna')
        export_root = os.path.join('optuna_trials', study_name)
        os.makedirs(export_root, exist_ok=True)
        # JSON 汇总
        trials_payload = []
        for t in study.trials:
            trials_payload.append({
                'number': t.number,
                'state': str(t.state),
                'value': (t.values[0] if t.values else None),
                'params': t.params,
                'user_attrs': getattr(t, 'user_attrs', {}),
            })
        with open(os.path.join(export_root, 'summary.json'), 'w') as f:
            json.dump({
                'best_value': study.best_value,
                'best_params': study.best_params,
                'monitor': args.monitor,
                'trials': trials_payload,
            }, f, indent=2, ensure_ascii=False)
        # CSV 汇总
        try:
            import csv
            # 收集所有参数列
            param_keys = set()
            for t in study.trials:
                for k in t.params.keys():
                    param_keys.add(k)
            param_keys = sorted(list(param_keys))
            fieldnames = ['number', 'state', 'value'] + param_keys
            with open(os.path.join(export_root, 'summary.csv'), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for t in study.trials:
                    row = {
                        'number': t.number,
                        'state': str(t.state),
                        'value': (t.values[0] if t.values else None),
                    }
                    for k in param_keys:
                        row[k] = t.params.get(k, None)
                    writer.writerow(row)
        except Exception as e:
            print(f"[warn] failed to export CSV summary: {e}")
    except Exception as e:
        print(f"[warn] failed to export trials summary: {e}")

    # 导出最佳配置到 YAML/JSON
    try:
        import yaml
        os.makedirs(args.export_dir, exist_ok=True)
        best_cfg = load_yaml(args.config)
        # 注入最优超参
        model_cfg = best_cfg.get("model", {})
        
        for k, v in study.best_params.items():
            model_cfg[k] = v
                
        best_cfg["model"] = model_cfg

        best_yaml_path = os.path.join(args.export_dir, "best_config_mai_gen.yaml")
        with open(best_yaml_path, "w") as f:
            yaml.safe_dump(best_cfg, f, sort_keys=False, allow_unicode=True)

        best_json_path = os.path.join(args.export_dir, "best_summary_mai_gen.json")
        with open(best_json_path, "w") as f:
            json.dump({
                "best_value": study.best_value,
                "best_params": study.best_params,
                "monitor": args.monitor,
                "config": os.path.abspath(args.config)
            }, f, indent=2, ensure_ascii=False)

        print(f"导出最佳配置到: {best_yaml_path}")
        print(f"导出摘要到: {best_json_path}")
    except Exception as e:
        print(f"导出最佳配置失败: {e}")


if __name__ == "__main__":
    main()
