import itertools
from transformers import PreTrainedModel, PretrainedConfig
import torch
from torch import nn
import os
from pathlib import Path

class IFESMConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "esm_if"

class IFESMModel(PreTrainedModel):
    config_class = IFESMConfig

    def __init__(self, config):
        super().__init__(config)
        path= os.path.join(Path(__file__).parent, "esm_if1_gvp4_t16_142M_UR50.pt")
        self.state_dict = torch.load(path, map_location="cpu", weights_only=False)
        #print(self.model)
        #self.model.eval() 

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs
    

from transformers import EsmTokenizer

def build_huggingface_model():
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    config = IFESMConfig()
    model = IFESMModel(config)
    model.save_pretrained("esm_if_model")
    tokenizer.save_pretrained("esm_if_model")