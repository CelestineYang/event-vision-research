"""Integration check for official filename V2 and pretrained window frames."""
import argparse
import torch
from ucf101_fusion_window_data import FusionWindowDataset
from ucf101_pretrained_fusion import load_pretrained_fusion
from eventmamba_v2_model import EventMamba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment',default='experiments/ucf101dvs_gcstr_sliding_txtsplit_v2')
    parser.add_argument('--store',default='/remote-home/datasets/event_project_cache/ucf101_txtsplit_v2_events_v1')
    parser.add_argument('--point-checkpoint',default='best_checkpoint.pth')
    args = parser.parse_args()
    torch.manual_seed(0)
    model = load_pretrained_fusion(args.experiment,args.point_checkpoint).cuda().eval()
    dataset = FusionWindowDataset(args.experiment,'test',args.store,
                                 point_count=2048,point_normalization='official')
    batch = dataset[0]
    again = dataset[0]
    assert all(torch.equal(batch[k],again[k]) for k in ('events','cstr3','xt_p','ty_p'))
    events = batch['events'][None].cuda()
    views = {v:batch[v][None].cuda() for v in ('cstr3','xt_p','ty_p')}
    # Compare to unchanged full classifier with identical random FPS starts.
    reference = EventMamba(101).cuda().eval()
    reference.load_state_dict(model.point_encoder.state_dict(),strict=True)
    with torch.no_grad():
        torch.manual_seed(17)
        expected = reference(events)
        torch.manual_seed(17)
        actual = model.point_encoder(events)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        features, coords = model.point_encoder.forward_features(events)
        assert [tuple(f.shape) for f in features] == [(1,1024,64),(1,512,128),(1,256,256)]
        assert [tuple(c.shape) for c in coords] == [(1,1024,3),(1,512,3),(1,256,3)]
    del reference
    model.train()
    assert not model.image_encoders.training
    assert model.point_encoder.training
    # Train-mode classifier BN requires at least two windows per local batch.
    events = events.repeat(2,1,1)
    views = {v:x.repeat(2,1,1,1) for v,x in views.items()}
    running = {k:v.clone() for k,v in model.image_encoders.named_buffers()}
    outputs = model.forward_outputs(events,views)
    assert all(v.shape == (2,101) and torch.isfinite(v).all() for v in outputs.values())
    (outputs['logits'].square().mean()+outputs['point_logits'].square().mean()).backward()
    assert model.point_encoder.mamba_list[0].mixer.in_proj.weight.grad is not None
    assert model.classifier.weight.grad is not None
    assert all(p.grad is None for p in model.image_encoders.parameters())
    assert all(torch.equal(v,running[k]) for k,v in model.image_encoders.named_buffers())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.freeze_points = True
    model.point_encoder.requires_grad_(False)
    model.train()
    assert not model.point_encoder.training
    print('PASS: strict V2/image loading; exact original classifier agreement; '
          'stage shapes; real-window CUDA forward/backward; frozen image BN and point eval mode')
    print(model.pretrained_sources)


if __name__ == '__main__': main()
