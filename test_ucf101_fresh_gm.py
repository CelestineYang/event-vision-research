"""Fresh-initialization policy guards; real branch/CUDA audits are separate."""
import json
from pathlib import Path
import tempfile
import unittest
from ucf101_fresh_gm_runtime import configuration
from ucf101_fresh_gm_fusion import load_model

class FreshGMTests(unittest.TestCase):
    def config(self):
        path = Path(__file__).parent/'configs/examples/ucf101_fresh_gm/GM_batch48.json'
        return json.loads(path.read_text())

    def test_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'config.json';p.write_text(json.dumps(self.config()))
            c=configuration(p)
        self.assertFalse(c['data_version_confirmed'])
        self.assertEqual(c['video_batch_size_per_gpu'],48)
        self.assertEqual(c['lr_existing_fusion'],5e-5)
        self.assertEqual(c['selection_metric'],'video_top1')

    def test_forbid_fusion_checkpoint_in_config_and_loader(self):
        for field in ['source_checkpoint','source_sha256']:
            c=self.config();c[field]='forbidden'
            with tempfile.TemporaryDirectory() as folder:
                p=Path(folder)/'config.json';p.write_text(json.dumps(c))
                with self.assertRaises(ValueError):configuration(p)
            with self.assertRaises(ValueError):load_model(c)

    def test_wrong_initialization_hash_rejected_before_load(self):
        c=self.config();c['point_sha256']='0'*64
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'checkpoint.pth';p.write_bytes(b'checksum fixture, never deserialized')
            c['point_checkpoint']=str(p)
            with self.assertRaisesRegex(ValueError,'Point checkpoint changed'):load_model(c)

if __name__=='__main__':unittest.main()
