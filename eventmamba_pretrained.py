"""Checkpoint-compatible V1 adapter, based on upstream 7f61fd3 (MIT).

Uses the published V1 forward. Extra LSTM tensors in the uploaded artifact
require explicit acknowledgement: their historical forward use is unknown.
"""
import torch
from torch import nn

from eventmamba_v2_upstream.models.modules import LocalGrouper
from eventmamba_v2_layers import MambaBlock
from eventmamba_v2_model import Attention, Linear1Layer, Linear2Layer


class PretrainedEventMamba(nn.Module):
    coordinate_dim = 3

    def __init__(self, channels=(128, 256, 512), groups=(1024, 512, 256),
                 neighbors=24, num_classes=101):
        super().__init__()
        if len(channels) != 3 or channels[1:] != (2*channels[0], 4*channels[0]):
            raise ValueError('V1 requires three doubling channel widths')
        if len(groups) != 3 or min(groups) < 1 or neighbors < 1:
            raise ValueError('Invalid grouping configuration')
        if any(groups[i+1] > groups[i] or neighbors > groups[i] for i in range(2)):
            raise ValueError('Invalid downstream grouping configuration')
        self.stage_channels, self.groups = tuple(channels), tuple(groups)
        self.neighbors = neighbors
        self.group = LocalGrouper(3, groups[0], neighbors, False, 'anchor')
        self.group_1 = LocalGrouper(channels[0], groups[1], neighbors, False, 'anchor')
        self.group_2 = LocalGrouper(channels[1], groups[2], neighbors, False, 'anchor')
        self.embed_dim = Linear1Layer(6, channels[0])
        for i, width in enumerate(channels, 1):
            setattr(self, f'conv{i}', Linear2Layer(width))
            setattr(self, f'conv{i}_1', Linear2Layer(width))
            setattr(self, f'mamba{i}', MambaBlock(width, i-1, 'v2'))
            setattr(self, f'attention_{i}', Attention(width))
        self.attention_4 = Attention(channels[-1])
        self.classifier = nn.Sequential(nn.Linear(channels[-1], 256),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(.5),
            nn.Linear(256, num_classes))

    @classmethod
    def from_checkpoint(cls, path, *, groups, neighbors=24, allow_unused_lstm=False):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state = checkpoint['net']
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        widths = tuple(state[f'attention_{i}.linear.weight'].shape[1] for i in (1,2,3))
        extras = sorted(k for k in state if k.startswith('lstm.'))
        if extras and not allow_unused_lstm:
            raise ValueError('Checkpoint contains LSTM tensors absent from published V1. '
                             'Confirm original forward before allow_unused_lstm=True: '+str(extras))
        model = cls(widths, tuple(groups), neighbors, state['classifier.4.weight'].shape[0])
        model.load_state_dict({k:v for k,v in state.items() if k not in extras}, strict=True)
        model.load_report = {'checkpoint': str(path), 'channels': widths,
                             'groups': tuple(groups), 'neighbors': neighbors,
                             'ignored_lstm_keys': extras, 'epoch': checkpoint.get('epoch')}
        return model

    def forward_features(self, events):
        # Explicit point input is t,x,y, already normalized by the dataset.
        if events.ndim != 3 or events.shape[1] != 3:
            raise ValueError('Expected [B,3,N] t,x,y')
        if events.shape[2] < max(self.groups[0], self.neighbors):
            raise ValueError('Too few input points')
        if not torch.isfinite(events).all() or (events[:,0,1:] < events[:,0,:-1]).any():
            raise ValueError('Points must be finite and chronological')
        xyz = events.transpose(1,2).contiguous()
        x = xyz
        features, coordinates = [], []
        for i, group in enumerate((self.group,self.group_1,self.group_2), 1):
            xyz, x = group(xyz,x)
            b,n,k,d = x.shape
            x = x.permute(0,1,3,2).reshape(b*n,d,k)
            if i == 1: x = self.embed_dim(x)
            x = getattr(self,f'conv{i}')(x).transpose(1,2)
            att = getattr(self,f'attention_{i}')(x)
            x = torch.bmm(att.unsqueeze(1),x).squeeze(1).reshape(b,n,-1)
            x,_ = getattr(self,f'mamba{i}')(x)
            x = getattr(self,f'conv{i}_1')(x.transpose(1,2)).transpose(1,2)
            features.append(x)
            coordinates.append(xyz)
        return features, coordinates

    def classify_features(self, features):
        x = features[-1]
        return self.classifier(torch.bmm(self.attention_4(x).unsqueeze(1),x).squeeze(1))

    def forward(self, events):
        return self.classify_features(self.forward_features(events)[0])
