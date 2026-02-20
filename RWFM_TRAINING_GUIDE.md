# RWFM/Anchor Training Guide (Crab VLA)

Этот файл фиксирует, что было добавлено в проект для reward-weighted flow matching (RWFM), и как запускать обучение.

## Что добавлено

1. Подготовка и контроль датасетов:
- Собран манифест датасетов: `crab/training/docs/dataset_manifest.csv`.
- Добавлены манифесты под цикл обучения:
  - `crab/training/docs/dataset_manifest_cycle1_clean.csv`
  - `crab/training/docs/dataset_manifest_cycle1_with_cracked.csv`

2. Нормализация reward и спецификация RWFM:
- Спецификация: `crab/training/docs/reward_weighting.md`.
- Пересчитанные stats под новый reward:
  - clean: `crab/training/docs/reward_weighting_stats_cycle1_clean.json`
  - with cracked: `crab/training/docs/reward_weighting_stats_cycle1_with_cracked.json`
- Текущий стабильный параметр: `alpha=0.25`.

3. Даталоадер расширен reward-полями:
- `crab/training/crab_dataset.py`
- В батч добавляются: `step_reward`, `chunk_return`, `episode_reward`, `episode_success`, `episode_damage`, `dataset_name`, `task_name`.

4. Внедрён RWFM в loss-редукцию:
- `crab/training/crab_smolvla_wrapper.py`
- Добавлено взвешивание per-sample loss по формуле RWFM.
- Полная обратная совместимость для baseline (`reward_weighting.enabled=false`).

5. Добавлена anchor-регуляризация:
- `crab/training/crab_smolvla_wrapper.py`
- `L_total = L_rwfm + lambda_anchor * L_anchor` с warmup/ramp.

6. Добавлен мониторинг стабильности:
- `crab/training/train.py`
- Логируются: `w_mean`, `w_std`, `w_p95`, `w_max`, `clipped_frac`, `effective_batch_weight`,
  `loss_unweighted`, `loss_weighted`, `loss_anchor`, `alpha`, `lambda_anchor`.
- Per-epoch summary сохраняется в:
  - `epoch_summary.json`
  - `epoch_summary.jsonl`
  - `epoch_summary.csv`

7. Обновлены конфиги цикла:
- `crab/training/configs/train_cycle1_baseline_il.yaml`
- `crab/training/configs/train_cycle1_rwfm.yaml`
- `crab/training/configs/train_cycle1_rwfm_anchor.yaml`
- `crab/training/configs/train_cycle1_rwfm_anchor_cracked_factor.yaml`
- В `task_masks` добавлена строка для `Open the carton...`.

## Что запускать

Запускать из корня проекта:

```bash
cd C:\Users\user\Downloads\vla
python crab/training/train.py --config crab/training/configs/train_cycle1_rwfm_anchor_cracked_factor.yaml
```

Это режим RWFM+Anchor с максимальным покрытием текущего цикла (включая `egg_carton_to_tray_cracked` как controlled factor).

Альтернативные режимы:

```bash
# Baseline IL (без RWFM и без anchor)
python crab/training/train.py --config crab/training/configs/train_cycle1_baseline_il.yaml

# RWFM без anchor (clean)
python crab/training/train.py --config crab/training/configs/train_cycle1_rwfm.yaml

# RWFM + anchor (clean)
python crab/training/train.py --config crab/training/configs/train_cycle1_rwfm_anchor.yaml
```

## Возобновление обучения

```bash
python crab/training/train.py ^
  --config crab/training/configs/train_cycle1_rwfm_anchor_cracked_factor.yaml ^
  --resume outputs/cycle1_rwfm_anchor_with_cracked/step_XXXXX
```

## Минимальные зависимости

Нужны модули Python: `torch`, `pyyaml`, `pyarrow`, `numpy`, `pandas`, `av`, `torchvision`, `lerobot`.

Если обучение не стартует, сначала проверь окружение:

```bash
python -c "import torch, yaml, pyarrow, av; print('env ok')"
```

## Где смотреть результаты

1. Чекпоинты:
- `outputs/cycle1_*/step_*`

2. Онлайн-метрики в логах train-loop:
- RWFM-веса и weighted/unweighted loss.

3. Сводка по эпохам:
- `outputs/cycle1_*/epoch_summary.csv`
- `outputs/cycle1_*/epoch_summary.json`

