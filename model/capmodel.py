import torch.nn as nn
import torch.nn.functional as F
class CapModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(CapModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, cnn_feats, captions, sentence=None,teacher_forcing_ratio=1.0, mode="train"):
        
        if mode=="train":
         
            frame_feats,loss_contrastive,pos_feats = self.encoder(cnn_feats,sentence,mode)
            outputs, module_weights = self.decoder(frame_feats, captions, teacher_forcing_ratio)
            outputs_pos,_ = self.decoder(pos_feats, captions, teacher_forcing_ratio)
            return outputs, module_weights,loss_contrastive,outputs_pos
            
        else:
            frame_feats= self.encoder(cnn_feats,mode="val")
            outputs, module_weights = self.decoder(frame_feats, captions, teacher_forcing_ratio)
            return outputs, module_weights