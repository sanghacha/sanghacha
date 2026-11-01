"""Train the CVAE on action chunks; validate with prior z=0, without future labels."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from action_generator.goal import GOAL_VERSION
from pipeline.types import check
from .data import ChunkDataset, compute_stats, discover_segments, split_segments
from .model import ACTConfig, GoalConditionedACT, act_loss

DEFAULT_CONFIG = Path(__file__).resolve().parents[1]/'configs'/'act.json'
CHECKPOINT_VERSION = 'so101-act-v3'


def save_checkpoint(payload, path):
    """Publish complete files so monitoring/inference cannot read half a ZIP."""
    path = Path(path)
    pending = path.with_suffix('.pending')
    torch.save(payload, pending)
    for attempt in range(5):
        try:
            pending.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.05)


def select_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; install training[cuda] or select --device cpu')
    return device


def train_run(data_dir, output, config, *, allow_diagnostic=False, init_checkpoint=None, transfer_camera=False):
    check(not transfer_camera or init_checkpoint is not None, 'camera transfer requires an initialization checkpoint')
    seed = int(config['seed'])
    check(config['epochs'] > 0 and config['batch_size'] > 0 and config['learning_rate'] > 0 and
          config['kl_weight'] >= 0 and config['weight_decay'] >= 0, 'invalid training hyperparameters')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(config.get('torch_threads', 4)))
    device = select_device(config['device'])
    model_config = ACTConfig(**config['model'])
    segments = discover_segments(data_dir, allow_diagnostic=allow_diagnostic)
    check(tuple(segments[0]['meta']['image_size']) == model_config.image_size,
          'model image_size must match the collected demonstration resolution')
    train_segments, val_segments = split_segments(segments, config['validation_fraction'], seed)
    initial = None
    if init_checkpoint is not None:
        initial = torch.load(init_checkpoint, map_location='cpu', weights_only=True)
        upgrading = initial['checkpoint_version'] == 'so101-act-v2' and initial['goal_version'] == 'so101-goal-v1'
        check((initial['checkpoint_version'] == CHECKPOINT_VERSION and initial['goal_version'] == GOAL_VERSION) or upgrading,
              'incompatible initialization checkpoint')
        old_model_config = ACTConfig(**initial['model_config']).to_dict()
        new_model_config = model_config.to_dict()
        action_mode_changed = old_model_config.pop('residual_actions') != new_model_config.pop('residual_actions')
        old_velocity, new_velocity = old_model_config.pop('include_velocity'), new_model_config.pop('include_velocity')
        old_error, new_error = old_model_config.pop('include_goal_error'), new_model_config.pop('include_goal_error')
        check(old_velocity == new_velocity or not old_velocity and new_velocity and config.get('allow_velocity_transfer', False),
              'velocity input transfer requires allow_velocity_transfer')
        check(old_error == new_error or not old_error and new_error and config.get('allow_goal_error_transfer', False),
              'goal error input transfer requires allow_goal_error_transfer')
        if old_velocity != new_velocity or old_error != new_error:
            weight = initial['model']['state_proj.weight']
            expanded = weight.new_zeros((weight.shape[0], 6+6*new_velocity+3*new_error))
            expanded[:, :6] = weight[:, :6]
            if old_velocity:
                expanded[:, 6:12] = weight[:, 6:12]
            if old_error:
                expanded[:, 6+6*new_velocity:] = weight[:, 6+6*old_velocity:]
            initial['model']['state_proj.weight'] = expanded
        check(old_model_config == new_model_config, 'initial model architecture differs')
        check(not action_mode_changed or config.get('allow_action_mode_transfer', False),
              'changing action mode requires allow_action_mode_transfer')
        old_train = {s['group_id'] for s in initial['split']['train']}
        new_val = {s['meta']['group_id'] for s in val_segments}
        check(not old_train & new_val, 'initial checkpoint trained on the new validation scenes')
        check(allow_diagnostic or not initial.get('diagnostic', False), 'cannot initialize from diagnostic weights')
        for key in ('command_dt', 'lower', 'upper', 'max_delta'):
            check(np.allclose(initial['control'][key], segments[0]['meta'][key]), 'initial control settings differ')
        for key in ('camera_intrinsics', 'camera_to_robot_base'):
            check(transfer_camera or np.allclose(initial['camera'][key], segments[0]['meta'][key]),
                  'initial camera calibration differs; use explicit --transfer-camera for fine-tuning')
        if upgrading:
            old = initial['model']['goal_proj.weight']
            expanded = old.new_zeros((old.shape[0], 25))
            expanded[:, :17] = old
            initial['model']['goal_proj.weight'] = expanded
            initial['stats']['goal_scale'] = initial['stats']['goal_scale'][:17]+[.03]*3+[1.]*5
    stats = initial['stats'] if initial and not transfer_camera else compute_stats(train_segments)
    train_data = ChunkDataset(train_segments, stats, model_config, cache_images=config.get('cache_images', False))
    val_data = ChunkDataset(val_segments, stats, model_config, cache_images=config.get('cache_images', False))
    generator = torch.Generator().manual_seed(seed)
    sampler = (WeightedRandomSampler(train_data.sampling_weights(), len(train_data), generator=generator)
               if config.get('balance_goals', False) else None)
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=sampler is None, sampler=sampler,
                              num_workers=config.get('num_workers', 0), generator=generator)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config.get('num_workers', 0))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    def describe(group):
        return [{'path': str(s['path'].resolve()), 'group_id': s['meta']['group_id'],
                 'primitive': s['meta']['primitive'], 'frames': s['meta']['frames'],
                 'trajectory_sha256': sha256((s['path']/'trajectory.npz').read_bytes()).hexdigest()}
                for s in group]
    split = {'train': describe(train_segments), 'validation': describe(val_segments)}
    (output/'split.json').write_text(json.dumps(split, indent=2)+'\n', encoding='utf-8')
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
    model = GoalConditionedACT(model_config).to(device)
    model.set_normalization(stats)
    if initial:
        model.load_state_dict(initial['model'], strict=True)
        model.set_normalization(stats)
        if action_mode_changed:
            torch.nn.init.normal_(model.action_head.weight, std=.001)
            torch.nn.init.zeros_(model.action_head.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    first = segments[0]['meta']
    checkpoint_meta = {
        'checkpoint_version': CHECKPOINT_VERSION, 'goal_version': GOAL_VERSION,
        'model_config': model_config.to_dict(), 'stats': stats, 'robot_id': 'so101',
        'primitives': sorted({s['meta']['primitive'] for s in train_segments}),
        'diagnostic': any(s['meta'].get('diagnostic', False) for s in segments),
        'control': {k: first[k] for k in ('command_dt', 'lower', 'upper', 'max_delta', 'joint_names', 'frame_id')},
        'camera': {k: first[k] for k in ('camera_intrinsics', 'camera_to_robot_base')},
        'training_config': config, 'torch_version': str(torch.__version__),
        'training_sources': sorted({s['meta']['source'] for s in train_segments}),
        'split': split,
        'initial_checkpoint': str(Path(init_checkpoint).resolve()) if init_checkpoint else None,
        'camera_transfer': bool(transfer_camera),
    }
    print(f'device={device}; train={len(train_data)} frames; validation={len(val_data)} frames; '
          f'primitives={checkpoint_meta["primitives"]}', flush=True)
    best = float('inf')
    with (output/'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_loss', 'train_l1', 'train_kl', 'val_prior_l1'])
        writer.writeheader()
        for epoch in range(1, int(config['epochs'])+1):
            model.train()
            totals = np.zeros(3)
            samples = 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                prediction, mu, logvar = model(batch['image'], batch['state'], batch['goal'],
                                               batch['actions'], batch['padding'])
                loss, l1, kl = act_loss(prediction, batch['actions'], batch['padding'], mu, logvar, config['kl_weight'])
                check(bool(torch.isfinite(loss)), 'nonfinite training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                n = len(batch['state'])
                totals += np.array([loss.item(), l1.item(), kl.item()])*n
                samples += n
            model.eval()
            error, labels = 0., 0
            family_error, family_labels = {}, {}
            with torch.inference_mode():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    prediction, _, _ = model(batch['image'], batch['state'], batch['goal'])
                    valid = (~batch['padding']).unsqueeze(-1)
                    error += ((prediction-batch['actions']).abs().masked_fill(~valid, 0.)).sum().item()
                    labels += int(valid.sum())*6
                    if config.get('balance_goals', False):
                        errors = (prediction-batch['actions']).abs().masked_fill(~valid, 0.).sum((1, 2))
                        sizes = valid.sum((1, 2))*6
                        for family in batch['family'].unique().tolist():
                            mask = batch['family'] == family
                            family_error[family] = family_error.get(family, 0.)+errors[mask].sum().item()
                            family_labels[family] = family_labels.get(family, 0)+int(sizes[mask].sum())
            val_l1 = (float(np.mean([family_error[f]/family_labels[f] for f in family_error]))
                      if config.get('balance_goals', False) else error/labels)
            check(np.isfinite(val_l1), 'nonfinite validation loss')
            row = dict(zip(('train_loss', 'train_l1', 'train_kl'), (totals/samples).tolist()))
            row.update(epoch=epoch, val_prior_l1=val_l1)
            writer.writerow(row)
            stream.flush()
            payload = {**checkpoint_meta, 'model': model.state_dict(), 'epoch': epoch, 'metrics': row}
            save_checkpoint(payload, output/'last.pt')
            if val_l1 < best:
                best = val_l1
                save_checkpoint(payload, output/'best.pt')
            print(f'epoch={epoch} loss={row["train_loss"]:.5f} val_prior_l1={val_l1:.5f}', flush=True)
    return output/'best.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'))
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--init-checkpoint', type=Path, help='warm-start weights/stats without optimizer state; rejects split leakage')
    parser.add_argument('--transfer-camera', action='store_true', help='explicit camera transfer; recompute normalization on new train split')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.device:
        config['device'] = args.device
    if args.epochs is not None:
        config['epochs'] = args.epochs
    try:
        check(not args.transfer_camera or args.init_checkpoint is not None, '--transfer-camera needs --init-checkpoint')
        checkpoint = train_run(args.data, args.output, config, init_checkpoint=args.init_checkpoint,
                               transfer_camera=args.transfer_camera)
    except (ValueError, FileExistsError) as exc:
        parser.exit(1, f'{exc}\n')
    print('checkpoint:', checkpoint)


if __name__ == '__main__':
    main()
