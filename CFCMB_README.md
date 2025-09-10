# CFCMB Implementation Summary

## Overview
This implementation extends the current semi-supervised 3D segmentation training with a **Class-Feature Contrastive Memory Bank (CFCMB)** to improve training efficiency and boundary discrimination.

## Key Components

### 1. CFCMB Module (`code/networks/cfcmb.py`)
- **ClassFeatureMemoryBank**: Maintains per-class prototype vectors with EMA updates
- **Per-class queues**: Optional FIFO queues for negative sampling  
- **Prototype contrastive loss**: InfoNCE-based loss against class prototypes
- **Dual momentum**: Different EMA rates for labeled (0.9) vs unlabeled (0.98) samples
- **Feature sampling**: Helper function to sample features per class from spatial maps

### 2. Training Integration (`code/train_cl7.py`)
- **11 new CLI arguments** for CFCMB configuration
- **Two-phase training**:
  - **Warmup phase** (iter < proto_warmup_start): RA-CL only on labeled, build memory bank
  - **Active phase** (iter >= proto_warmup_start): Add prototype contrast for unlabeled
- **Linear warmup**: Proto weight increases from 0 to args.proto_weight over 2000 iterations
- **Enhanced logging**: Proto loss, bank metrics, per-class prototype norms
- **Checkpointing**: Save/load memory bank state

## Usage

### Basic Usage (CFCMB enabled by default)
```bash
python code/train_cl7.py --root_path /path/to/data --exp my_experiment
```

### Disable CFCMB
```bash
python code/train_cl7.py --disable_cfcmb --root_path /path/to/data --exp my_experiment
```

### Custom CFCMB Configuration
```bash
python code/train_cl7.py \
    --proto_weight 0.2 \
    --proto_temp 0.1 \
    --proto_conf_thresh 0.95 \
    --proto_samples_per_class 512 \
    --proto_momentum_labeled 0.85 \
    --proto_momentum_unlabeled 0.99 \
    --root_path /path/to/data \
    --exp my_experiment
```

## New CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--enable_cfcmb` | flag | True | Enable CFCMB (default on) |
| `--disable_cfcmb` | flag | False | Disable CFCMB |
| `--proto_weight` | float | 0.15 | Prototype contrastive loss weight |
| `--proto_temp` | float | 0.15 | Temperature for prototype contrastive loss |
| `--proto_conf_thresh` | float | 0.9 | Confidence threshold for unlabeled updates |
| `--proto_queue_size` | int | 512 | Size of negative queues per class |
| `--proto_feat_dim` | int | 128 | Prototype feature dimension |
| `--proto_warmup_start` | int | contrast_start_iter | Warmup start iteration |
| `--proto_warmup_stop` | int | warmup_start + 2000 | Warmup end iteration |
| `--proto_samples_per_class` | int | 256 | Max samples per class per image |
| `--proto_momentum_labeled` | float | 0.9 | EMA momentum for labeled samples |
| `--proto_momentum_unlabeled` | float | 0.98 | EMA momentum for unlabeled samples |

## Training Flow

### Phase 1: Warmup (iter < proto_warmup_start)
1. Apply RA-CL only on labeled images
2. Extract decoder features from labeled samples  
3. Sample features per class (up to proto_samples_per_class)
4. Update memory bank prototypes with labeled features
5. No prototype loss applied (proto_weight = 0)

### Phase 2: Active (iter >= proto_warmup_start)  
1. Continue labeled RA-CL and bank updates from Phase 1
2. Extract decoder features from unlabeled samples
3. Use teacher pseudo-labels and confidence scores
4. Sample high-confidence unlabeled features per class
5. Compute prototype contrastive loss against memory bank
6. Apply linear warmup to proto_weight (0 → args.proto_weight over 2000 iters)
7. Optionally update bank with high-confidence unlabeled samples

### Final Loss
```
student_loss = supervised_loss + consistency_loss + weighted_contrast_loss + weighted_proto_loss
```

## Logging Metrics

- `loss/proto_loss`: Raw prototype contrastive loss
- `loss/weighted_proto_loss`: Weighted prototype loss (with warmup)
- `bank/num_sampled_unlabeled`: Number of unlabeled features sampled
- `bank/proto_norm_class_X`: L2 norm of prototype for class X
- `bank/proto_norm_mean`: Mean prototype norm across classes

## Implementation Notes

### Memory Efficiency
- Feature sampling limits memory usage (proto_samples_per_class per class per image)
- L2 normalization ensures stable prototype updates
- Optional queues can be disabled by setting proto_queue_size=0

### Robustness
- Safe handling of empty classes (no updates/loss when no samples)
- Confidence thresholding prevents low-quality pseudo-label contamination
- Different momentum rates reduce confirmation bias from unlabeled samples

### Compatibility
- Maintains full backward compatibility when --disable_cfcmb is used
- No changes to VNet architecture (reuses existing decoder_proj)
- Checkpoints include memory bank state for seamless resumption

## Testing

The implementation includes comprehensive tests:
- Unit tests for CFCMB module functionality
- Integration tests with VNet decoder features  
- Syntax verification for training script modifications
- End-to-end training flow demonstration

## Expected Benefits

1. **Better data utilization**: Each labeled image contributes multiple class-aware contrastive samples
2. **Improved robustness**: Memory bank provides stable class representations
3. **Stronger boundaries**: Cross-image prototype contrast improves boundary discrimination  
4. **Efficient training**: Reuses existing VNet features with minimal computational overhead