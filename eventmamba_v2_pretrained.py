"""Feature interface for the official filename V2 checkpoint.

Inherits all parameter names and modules from the existing full V2 classifier;
the original classifier implementation and its training entry remain intact.
"""
import torch

from eventmamba_v2_model import EventMamba


class PretrainedEventMambaV2(EventMamba):
    stage_channels = (64, 128, 256)
    coordinate_dim = 3

    @classmethod
    def from_checkpoint(cls, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state = checkpoint['net']
        state = {key.removeprefix('module.'): value for key, value in state.items()}
        model = cls(num_classes=101)
        # Do not discard unexpected tensors or silently accept a V1 artifact.
        model.load_state_dict(state, strict=True)
        model.load_report = {
            'checkpoint': str(path), 'architecture': 'EventMamba V2 filename',
            'embedding_channels': model.feature_list[0],
            'stage_channels': model.stage_channels,
            'groups': tuple(model.group_number[:model.stages]),
            'neighbors': tuple(model.neighbors[:model.stages]),
            'epoch_zero_based': checkpoint.get('epoch'),
            'checkpoint_acc': checkpoint.get('acc'),
        }
        return model

    def forward_features(self, events):
        if events.ndim != 3 or events.shape[1] != 3:
            raise ValueError('Expected [B,3,N] in normalized t,x,y order')
        if events.shape[2] < max(self.group_number[0], self.neighbors[0]):
            raise ValueError('Too few input points for V2 grouping')
        if not torch.isfinite(events).all() or (events[:,0,1:] < events[:,0,:-1]).any():
            raise ValueError('Points must be finite and chronological')
        xyz = events.permute(0,2,1)
        x = self.embed_dim(events).permute(0,2,1)
        features, coordinates = [], []
        for i in range(self.stages):
            xyz, x = self.local_grouper_list[i](xyz,x)
            x = x.permute(0,1,3,2)
            b,n,d,s = x.shape
            x = self.local_conv_list[i](x.reshape(-1,d,s)).permute(0,2,1)
            att = self.aggregation_list[i](x)
            x = torch.bmm(att.unsqueeze(1),x).squeeze(1).reshape(b,n,-1)
            x,_ = self.mamba_list[i](x)
            x = self.global_conv_list[i](x.permute(0,2,1)).permute(0,2,1)
            features.append(x)
            coordinates.append(xyz)
        return features, coordinates

    def classify_features(self, features):
        x = features[-1]
        return self.classifier(torch.bmm(self.attention(x).unsqueeze(1),x).squeeze(1))

    def forward(self, events):
        return self.classify_features(self.forward_features(events)[0])
