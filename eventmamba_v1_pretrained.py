"""Strict THU 50-class V1 loader and stage features without changing parameters."""
import torch
from eventmamba_v1_model import EventMamba

class PretrainedEventMambaV1(EventMamba):
    stage_channels=(64,128,256)
    coordinate_dim=3

    @classmethod
    def from_checkpoint(cls,path):
        saved=torch.load(path,map_location='cpu',weights_only=True)
        raw=saved['net'];state={k.removeprefix('module.'):v for k,v in raw.items()}
        if len(state)!=len(raw):raise ValueError('Prefix collision')
        model=cls(num_classes=50,num=2048)
        model.load_state_dict(state,strict=True)
        if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite checkpoint')
        model.load_report={'checkpoint':str(path),'architecture':'EventMamba V1 THU',
            'groups':[1024,512,256],'neighbors':24,'knn':'coordinate','epoch_zero_based':saved.get('epoch'),
            'checkpoint_window_accuracy':saved.get('acc'),'strict_load':True}
        return model

    def forward_features(self,events):
        if events.ndim!=3 or events.shape[1]!=3 or events.shape[2]<1024:
            raise ValueError('Expected chronological [B,3,N>=1024] t,x,y')
        if not torch.isfinite(events).all() or (events[:,0,1:]<events[:,0,:-1]).any():
            raise ValueError('Invalid event points')
        xyz=events.transpose(1,2);x=xyz;features=[];coordinates=[]
        stages=((self.group,self.conv1,self.attention_1,self.mamba1,self.conv1_1),
                (self.group_1,self.conv2,self.attention_2,self.mamba2,self.conv2_1),
                (self.group_2,self.conv3,self.attention_3,self.mamba3,self.conv3_1))
        for i,(group,conv,attention,mamba,post) in enumerate(stages):
            xyz,x=group(xyz,x);x=x.permute(0,1,3,2);b,n,d,k=x.shape;x=x.reshape(-1,d,k)
            if i==0:x=self.embed_dim(x)
            x=conv(x).transpose(1,2)
            x=torch.bmm(attention(x).unsqueeze(1),x).squeeze(1).reshape(b,n,-1)
            x,_=mamba(x);x=post(x.transpose(1,2)).transpose(1,2)
            features.append(x);coordinates.append(xyz)
        return features,coordinates

    def classify_features(self,features):
        x=features[-1]
        return self.classifier(torch.bmm(self.attention_4(x).unsqueeze(1),x).squeeze(1))

    def forward(self,events):
        return self.classify_features(self.forward_features(events)[0])
