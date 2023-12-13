import torch
import logging
from torch import nn
# __init__.py를 확인해볼 것
from .__init__ import methods_map

# Namespace Encapsulation
__all__ = ['ModelManager']

class MIA(nn.Module):

    def __init__(self, args):
        # 아무것도 안 적는 것과 기능적으로 동일함.
        super(MIA, self).__init__()

        fusion_method = methods_map[args.method]
        self.model = fusion_method(args)

    # Forward 방식을 보아하니 text, video, audio들은 모두 전처리 되어 Feature의 형태로 입력이 됨.
    # TODO: self.model에서 fusion_method를 우리가 원하는 방식으로 잘 수정해서 성능 개선을 해야함
    def forward(self, text_feats, video_feats, audio_feats):

        video_feats, audio_feats = video_feats.float(), audio_feats.float()
        mm_model = self.model(text_feats, video_feats, audio_feats)

        return mm_model
        
        
# 전체 구조가 ModelManger구조를 통해 Model을 관리하게 되는데
# 관리하는 과정에서 `self._set_model` 메소드 + args 인자를 통해 model의 종류를 결정하게 되고,
# 이렇게 결정된 model을 return하게 됨.
class ModelManager:

    def __init__(self, args):
        
        self.logger = logging.getLogger(args.logger_name)
        self.device = args.device = torch.device('cuda:%d' % int(args.gpu_id) if torch.cuda.is_available() else 'cpu')
        self.model = self._set_model(args)

    def _set_model(self, args):

        model = MIA(args) 
        model.to(self.device)
        return model