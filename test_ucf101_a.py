import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gcstr_eventmamba_fusion import GCSTREventMambaFusion
from gcstr_secnet_fusion import VIEWS
from eventmamba_encoder import EventMambaEncoder
from ucf101_a_data import UCF101ADataset, UCF_CONFIG, UCF_ROOT, build_manifest


class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        return index


class UCFProtocolCodeTests(unittest.TestCase):
    """CPU-only configuration/data tests: no model, backward or optimizer steps."""
    def test_builder_preserves_every_official_train_and_test_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'splits').mkdir()
            ids = {}
            fields = ['sample_id', 'split', 'npz_relpath', 'event_relpath', 'label_id',
                      'group', 'sensor_width', 'sensor_height']
            for split, clips in (('train', 2), ('test', 1)):
                rows = []
                for label in range(101):
                    for clip in range(clips):
                        identity = f'{split}_{label}_{clip}'
                        relative = f'processed/{split}/{identity}.npz'
                        path = root / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b'fixture: metadata check only')
                        rows.append(dict(sample_id=identity, split=split, npz_relpath=relative,
                            event_relpath=f'{identity}.mat', label_id=str(label),
                            group='1' if split == 'train' else '2', sensor_width='240', sensor_height='180'))
                ids[split] = {r['sample_id'] for r in rows}
                with (root / 'splits' / f'{split}.csv').open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            target = root / 'adapted.csv'
            report = build_manifest(root, target)
            self.assertFalse(report['validation'])
            self.assertEqual(report['samples'], {'train': 202, 'test': 101})
            with target.open(newline='') as handle:
                adapted = list(csv.DictReader(handle))
            self.assertEqual({r['split'] for r in adapted}, {'train', 'test'})
            for split in ids:
                self.assertEqual({r['sample_id'] for r in adapted if r['split'] == split}, ids[split])
            self.assertEqual(build_manifest(root, target), report)

    def test_two_rank_sampling_covers_all_without_duplicates_or_dropped_tail(self):
        from train_ucf101_a import loader
        selections = []
        with patch('train_ucf101_a.seed_all'), patch('torch.cuda.is_available', return_value=False):
            for rank in (0, 1):
                args = SimpleNamespace(seed=0, workers=0, world_size=2, rank=rank)
                batches = list(loader(IndexDataset(70), args, 1, 64, True))
                self.assertEqual([len(b) for b in batches], [32, 3])
                selections.append({int(i) for b in batches for i in b})
        self.assertFalse(selections[0] & selections[1])
        self.assertEqual(selections[0] | selections[1], set(range(70)))

    def test_no_padding_for_odd_rank_partition(self):
        from train_ucf101_a import loader
        args = SimpleNamespace(seed=0, workers=0, world_size=2, rank=0)
        with self.assertRaises(ValueError):
            loader(IndexDataset(69), args, 1, 64, True)

    def test_test_loader_reads_full_dataset_once(self):
        from train_ucf101_a import loader
        args = SimpleNamespace(seed=0, workers=0, world_size=2, rank=0)
        with patch('train_ucf101_a.seed_all'), patch('torch.cuda.is_available', return_value=False):
            batches = list(loader(IndexDataset(69), args, 0, 64, False))
        self.assertEqual([int(i) for b in batches for i in b], list(range(69)))

    def test_final_checkpoint_gate(self):
        from train_ucf101_a import check_final_state
        common = {'validation': False, 'early_stopping': False}
        bank = {'epoch': 50, 'complete': True, 'signature': {'epochs': 50, 'common': common}}
        check_final_state(bank, common)
        for change in ({'epoch': 49}, {'complete': False}):
            with self.assertRaises(ValueError):
                check_final_state({**bank, **change}, common)

    def test_launcher_uses_two_ranks_only_for_training(self):
        from run_ucf101_a import stage_command
        args = SimpleNamespace(devices='0,1', output_root=Path('/tmp/ucf_test_unused'),
                               epochs=50, image_batch=64, fusion_batch=16)
        for stage in ('images', 'fusion'):
            command = stage_command(args, stage, Path('/tmp'), 'python')
            self.assertIn('--nproc_per_node=2', command)
            self.assertEqual(command[command.index('--image-batch') + 1], '64')
        for stage in ('image_test', 'test'):
            self.assertNotIn('torch.distributed.run', stage_command(args, stage, Path('/tmp'), 'python'))

    def test_independent_fusion_epoch_budget_preserves_image_schedule(self):
        from run_ucf101_a import stage_command
        args = SimpleNamespace(devices='0,1', output_root=Path('/tmp/ucf_test_unused'),
                               epochs=50, fusion_epochs=150, image_batch=64, fusion_batch=16)
        for stage, expected in (('images', '50'), ('image_test', '50'),
                                ('fusion', '150'), ('test', '150')):
            command = stage_command(args, stage, Path('/tmp'), 'python')
            self.assertEqual(command[command.index('--epochs') + 1], expected)

    def test_parser_uses_new_protocol_paths_and_no_patience(self):
        from train_ucf101_a import create_parser
        parser = create_parser()
        args = parser.parse_args(['--stage', 'images', '--output-root', '/tmp/ucf_test_unused'])
        self.assertEqual(args.manifest.name, 'ucf101_a_official_split1.csv')
        self.assertEqual(args.epochs, 50)
        self.assertFalse(hasattr(args, 'patience'))

    def test_validation_split_is_rejected(self):
        with self.assertRaises(ValueError):
            UCF101ADataset('/tmp/does_not_exist.csv', 'val')


class UCFAdapterTests(unittest.TestCase):
    def test_actual_manifest_and_sensor(self):
        with (UCF_ROOT / 'splits/train.csv').open(newline='') as handle:
            train = list(csv.DictReader(handle))
        with (UCF_ROOT / 'splits/test.csv').open(newline='') as handle:
            test = list(csv.DictReader(handle))
        self.assertEqual(len(train), 9464)
        self.assertEqual(len(test), 3745)
        for items in (train, test):
            self.assertEqual({int(r['label_id']) for r in items}, set(range(101)))
        self.assertEqual((UCF_CONFIG.sensor_width, UCF_CONFIG.sensor_height), (240, 180))

    def test_real_train_data_shapes_determinism(self):
        # Read only Train recordings, no official Test events.
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'official.csv'
            build_manifest(UCF_ROOT, manifest)
            data = UCF101ADataset(manifest, 'train', augmentation='spatial',
                                 augmentation_count=5, seed=0)
            data.set_epoch(1)
            a, b = data[0], data[0]
        self.assertEqual(tuple(a['events'].shape), (4, 8192))
        self.assertTrue(torch.equal(a['events'], b['events']))
        self.assertTrue(torch.isfinite(a['events']).all())
        self.assertTrue((a['events'][2, 1:] >= a['events'][2, :-1]).all())
        for v in VIEWS:
            self.assertEqual(tuple(a[v].shape), (3, 224, 224))
            self.assertTrue(torch.equal(a[v], b[v]))

    def test_101_heads_and_freeze_with_small_encoder(self):
        encoder = EventMambaEncoder(groups=(16, 8, 4), neighbors=4,
                    mixer_factory=lambda channels, idx: torch.nn.Linear(channels, channels))
        model = GCSTREventMambaFusion(num_classes=101, point_encoder=encoder, point_aux=True).train()
        events = torch.rand(2, 4, 32)
        events[:, 2] = events[:, 2].sort(1).values
        batch = {'events': events, **{v: torch.rand(2, 3, 64, 64) for v in VIEWS}}
        output = model.forward_batch_outputs(batch)
        self.assertEqual(tuple(output['logits'].shape), (2, 101))
        self.assertEqual(tuple(output['point_logits'].shape), (2, 101))
        loss = torch.nn.functional.cross_entropy(output['logits'], torch.tensor([0, 100]))
        loss += 0.5 * torch.nn.functional.cross_entropy(output['point_logits'], torch.tensor([0, 100]))
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
            else:
                self.assertIsNone(p.grad, name)
        self.assertFalse(model.image_encoders.training)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
