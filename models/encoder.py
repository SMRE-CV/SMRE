import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
from torch.autograd import Variable
from .attention import *
from pytorch_transformers import BertModel, BertConfig, BertTokenizer
from torch import Tensor

import nltk
class Bert(nn.Module):
    def __init__(self): 
        super(Bert, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased-vocab.txt")
        modelConfig = BertConfig.from_pretrained("bert-base-uncased-config.json")
        self.textExtractor = BertModel.from_pretrained(
            "bert-base-uncased-pytorch_model.bin", config=modelConfig)  
    def pre_process(self, texts):
        tokens, segments, input_masks, text_length = [], [], [], []
        for text in texts:
            text = '[CLS] ' + text + ' [SEP]'
            tokenized_text = self.tokenizer.tokenize(text)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            if len(indexed_tokens) > 30:
                indexed_tokens = indexed_tokens[:30]
                
            tokens.append(indexed_tokens)
            segments.append([0] * len(indexed_tokens))
            input_masks.append([1] * len(indexed_tokens))
        for j in range(len(tokens)):
            padding = [0] * (30 - len(tokens[j]))
            text_length.append(len(tokens[j])-2)
            tokens[j] += padding
            segments[j] += padding
            input_masks[j] += padding
        tokens = torch.tensor(tokens)
        segments = torch.tensor(segments)
        input_masks = torch.tensor(input_masks)
        text_length = torch.tensor(text_length)
        return tokens, segments, input_masks, text_length
    def forward(self, tokens, segments, input_masks):
        output=self.textExtractor(tokens, token_type_ids=segments, attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :] 
        hidden_states = output[0][:, 1:, :]     
        return text_embeddings, hidden_states


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.a_feature_size = opt.a_feature_size
        self.m_feature_size = opt.m_feature_size
        self.hidden_size = opt.hidden_size
        self.concat_size = self.a_feature_size + self.m_feature_size
        self.language_model = Bert()
        # frame feature embedding
        self.frame_feature_embed = nn.Linear(self.concat_size, self.hidden_size)
        self.cal_loss_tri=Tri_Loss()
        self.cal_loss_cos=ContrastiveLoss_cos()
        self.cap_dim=26
        self.word_size = opt.word_size
        self.vocab_size=opt.vocab_size
        self.rnn_size=1024
        self.bert_emb=nn.Linear(768,self.hidden_size)
        self.recover=nn.Linear(self.hidden_size,self.concat_size)
        self.criterion = nn.CrossEntropyLoss()
    def _init_weights(self):
        nn.init.xavier_normal_(self.frame_feature_embed.weight)
        nn.init.constant_(self.frame_feature_embed.bias, 0)

    def _init_lstm_state(self, d):
        batch_size = d.size(0)
        lstm_state_h = d.data.new(2, batch_size, self.hidden_size).zero_()
        lstm_state_c = d.data.new(2, batch_size, self.hidden_size).zero_()
        return lstm_state_h, lstm_state_c

    


    def cosine_sim(self, im, s):
        #   '''cosine similarity between all the image and sentence pairs'''
        im=im.view(im.size(0),-1)
        s=s.view(s.size(0),-1)
        inner_prod = im.mm(s.t())
        im_norm = torch.sqrt((im**2).sum(1).view(-1, 1) + 1e-18)
        s_norm = torch.sqrt((s**2).sum(1).view(1, -1) + 1e-18)
        sim = inner_prod / (im_norm * s_norm)
        return sim

    def forward(self, cnn_feats,sentence=None,mode="train"):
        '''
        :param cnn_feats: (batch_size, max_frames, m_feature_size + a_feature_size)#b_s,26,2560
        :param region_feats: (batch_size, max_frames, num_boxes, region_feature_size)
        :param spatial_feats: (batch_size, max_frames, num_boxes, spatial_feature_size)
        :return: output of Bidirectional LSTM and embedded region features
        :cap:torch.Size([128, 26])
        '''
        # 2d cnn or 3d cnn or 2d+3d cnn
        
        assert self.a_feature_size + self.m_feature_size == cnn_feats.size(2)
        if mode=="train": 
            bsz=cnn_feats.size(0)
            dim=cnn_feats.size(-1)  
            frame_feats_ori = self.frame_feature_embed(cnn_feats)
            tokens, segments, input_masks, caption_length = self.language_model.pre_process(sentence)
            tokens = tokens.cuda()
            segments = segments.cuda()
            input_masks = input_masks.cuda()
            text_features, hidden_states = self.language_model(tokens, segments, input_masks)
            text_features=self.bert_emb(text_features)
            feats=torch.mean(frame_feats_ori,dim=1)
            loss2,score=self.cal_loss_tri(feats,text_features)
            scores=torch.softmax(score*10, dim=1)
            pos_feats = torch.mm(scores, cnn_feats.view(bsz, -1)).view(bsz, -1, dim)
            pos_emb=self.frame_feature_embed(pos_feats)
            loss1=self.cal_loss_cos(pos_emb,frame_feats_ori,1.0)
            loss=loss2+loss1
            return frame_feats_ori,loss,pos_emb
        else:
            frame_feats = self.frame_feature_embed(cnn_feats)
            return frame_feats

    


class ContrastiveLoss_cos(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    if 1 is similar to 2 then label=0
    """
    def __init__(self, margin=0.2):
        super(ContrastiveLoss_cos, self).__init__()
        self.margin = margin

    def cosine_sim(self, im, s):
    #   '''cosine similarity between all the image and sentence pairs'''
        im=im.view(im.size(0),-1)
        s=s.view(s.size(0),-1)
        inner_prod = im.mm(s.t())
        im_norm = torch.sqrt((im**2).sum(1).view(-1, 1) + 1e-18)
        s_norm = torch.sqrt((s**2).sum(1).view(1, -1) + 1e-18)
        sim = inner_prod / (im_norm * s_norm)
        return 1-sim

    def forward(self, output1, output2, label):
        cos_dist = self.cosine_sim(output1, output2)#torch.Size([128, 128])
        loss_contrastive = torch.mean((1-label) * torch.pow(cos_dist, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - cos_dist, min=0.0), 2))
        # loss_contrastive = torch.mean((1-label) * F.log_softmax(cos_dist) +
        #                               (label) * F.log_softmax(torch.clamp(self.margin - cos_dist, min=0.0)))
        return loss_contrastive
class Tri_Loss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0.2, max_violation=True):
        super(Tri_Loss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        
    def cosine_sim(self, im, s):
    #   '''cosine similarity between all the image and sentence pairs'''
        im=im.view(im.size(0),-1)
        s=s.view(s.size(0),-1)
        inner_prod = im.mm(s.t())
        im_norm = torch.sqrt((im**2).sum(1).view(-1, 1) + 1e-18)
        s_norm = torch.sqrt((s**2).sum(1).view(1, -1) + 1e-18)
        sim = inner_prod / (im_norm * s_norm)
        return sim


    def forward(self, im, s):
        
        # compute image-sentence score matrix
        scores = self.cosine_sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        loss2=(cost_im.sum()+cost_s.sum())/im.size(0)
        score_final=loss2
        return score_final,scores




