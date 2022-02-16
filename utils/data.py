import h5py
import torch
import sys
import pickle
import torchtext
import torch.utils.data as data
from .opt import parse_opt
opt = parse_opt()
import numpy
import json

numpy.random.seed(4567)
class V2TDataset(data.Dataset):
    def __init__(self, cap_pkl, frame_feature_h5, filed):

        h5 = h5py.File(frame_feature_h5, 'r')
        self.video_feats = h5[opt.feature_h5_feats]
        if opt.dataset=="msvd":
            self.video_feats=numpy.load(opt.video_feats_msvd,allow_pickle=True)
        elif opt.dataset=="msr-vtt":
            self.video_feats=numpy.load(opt.video_feats_msrvtt,allow_pickle=True)
        self.tokens = []
        self.video_ids = []
        self.filed = filed
        self.sent=[]
        if opt.dataset=="msr-vtt":
            with open(opt.msrvtt_caption_train, 'r') as data:
                lines = data.readlines()
                for line in lines:
                    vid = line.split('\t')[0]
                    sent = line.split('\t')[1].strip()
                    self.video_ids.append(int(vid))
                    self.tokens.append(filed.preprocess(sent))
                    self.sent.append(sent)
        elif opt.dataset=="msvd":
            with open(opt.msvd_caption_train, 'r') as data:
                lines = data.readlines()
                for line in lines:
                    vid=int(line.split('\t')[0].split("vid")[1])-1
                    sent = line.split('\t')[1].strip()
                    self.video_ids.append(int(vid))
                    self.tokens.append(filed.preprocess(sent))
                    self.sent.append(sent)
        
        self.captions, self.lengths = filed.process(self.tokens)


    def __getitem__(self, index):
        caption = self.captions[index]
        video_id = self.video_ids[index]
        video_feat = torch.from_numpy(self.video_feats[video_id])
        lengths = self.lengths[index]
        if opt.dataset=="msvd":
            if opt.data_aug=="True":
                index_sent1=self.video_ids.index(video_id)+numpy.random.randint(25) 
                sent=self.sent[index_sent1]+" and "+self.sent[index]
            else:
                sent=self.sent[index]
        if opt.dataset=="msr-vtt":
            if opt.data_aug=="True":
                index_sent1=self.video_ids.index(video_id)+numpy.random.randint(18) 
                sent=self.sent[index]+" and "+self.sent[index_sent1]
            else:
                sent=self.sent[index]
        if opt.shuffle=="True":
            idx=torch.randperm(video_feat.shape[0])
            video_feat=video_feat[idx,:].view(video_feat.size())
            video_feat=video_feat[:opt.max_frames,:]
        return video_feat, caption, lengths, video_id,sent

    def __len__(self):
        return len(self.captions)


class VideoDataset(data.Dataset):
    def __init__(self, eval_range, frame_feature_h5):
        self.eval_list = tuple(range(*eval_range))
        h5 = h5py.File(frame_feature_h5, 'r')
        self.video_feats = h5[opt.feature_h5_feats]
        if opt.dataset=="msvd":
            self.video_feats=numpy.load(opt.video_feats_msvd,allow_pickle=True)
        elif opt.dataset=="msr-vtt":
            self.video_feats=numpy.load(opt.video_feats_msrvtt,allow_pickle=True)

    def __getitem__(self, index):
        video_id = self.eval_list[index]
        video_feat = torch.from_numpy(self.video_feats[video_id])
        if opt.shuffle=="True":
            idx=torch.randperm(video_feat.shape[0])
            video_feat=video_feat[idx,:].view(video_feat.size())
            video_feat=video_feat[:opt.max_frames,:]
        return video_feat, video_id

    def __len__(self):
        return len(self.eval_list)


def train_collate_fn(data):
    data.sort(key=lambda x: x[2], reverse=True)

    videos, captions, lengths, video_ids,sent = zip(*data)

    videos = torch.stack(videos, 0)
    captions = torch.stack(captions,0)

    return videos, captions, lengths, video_ids,sent


def eval_collate_fn(data):
    data.sort(key=lambda x: x[-1], reverse=False)

    videos, video_ids = zip(*data)

    videos = torch.stack(videos, 0)

    return videos, video_ids


def get_train_loader(cap_pkl, frame_feature_h5, filed, batch_size=100, shuffle=True, num_workers=3, pin_memory=True):
    v2t = V2TDataset(cap_pkl, frame_feature_h5, filed)
    data_loader = torch.utils.data.DataLoader(dataset=v2t,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=train_collate_fn,
                                              pin_memory=pin_memory)
    return data_loader


def get_eval_loader(cap_pkl, frame_feature_h5,  batch_size=100, shuffle=False, num_workers=1, pin_memory=False):
    vd = VideoDataset(cap_pkl, frame_feature_h5)
    data_loader = torch.utils.data.DataLoader(dataset=vd,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=eval_collate_fn,
                                              pin_memory=pin_memory)
    return data_loader


if __name__ == '__main__':
    filed = torchtext.data.Field(sequential=True, tokenize="spacy",
                                eos_token="<eos>",
                                include_lengths=True,
                                batch_first=True,
                                fix_length=26,
                                lower=True,
                                )
    filed.vocab = pickle.load(open(opt.vocab_pkl_path, 'rb'))
    print(len(filed.vocab))
    print(filed.vocab.stoi['<eos>'])

