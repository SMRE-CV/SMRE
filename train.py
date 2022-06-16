import shutil
import pickle
import time

from torch.types import Device
from utils.utils import *
from utils.data import get_train_loader
from utils.opt import parse_opt

import models
from models.encoder import Encoder,Tri_Loss
from models.decoder import Decoder
from models.capmodel import CapModel
import torch
import os
import torch.nn as nn
import numpy as np
from evaluate import evaluate, convert_data_to_coco_scorer_format
import torchtext

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
def main(opt):
    localtime = time.asctime( time.localtime(time.time()) )
    logger = Logger(os.path.join(opt.result_dir, '{}.txt'.format(localtime)))
    logger.write("time:{}".format(localtime))
    logger.write('Learning rate: %.5f' % opt.learning_rate)
    logger.write('Learning rate decay: {}'.format(opt.learning_rate_decay) )
    logger.write('Batch size: %d' % opt.train_batch_size)
    logger.write('results directory: {}'.format(opt.result_dir) )
    # load vocabulary
    filed = torchtext.legacy.data.Field(sequential=True, tokenize="spacy",
                                 eos_token="<eos>",
                                 include_lengths=True,
                                 batch_first=True,
                                 unk_token="<unk>",
                                 fix_length=opt.max_words,
                                 lower=True,
                                 )
    if opt.dataset=="msvd":
        filed.vocab=pickle.load(open(opt.msvd_vocab, 'rb'))
    elif opt.dataset=="msr-vtt":
        filed.vocab = pickle.load(open(opt.msrvtt_vocab,'rb'))
    vocab_size = len(filed.vocab)

    print("vocab_size:{}".format(vocab_size))
    print("data aug:{}".format(opt.data_aug))
    # print parameters
    print('Learning rate: %.5f' % opt.learning_rate)
    print('Learning rate decay: ', opt.learning_rate_decay)


    # build model
    encoder = Encoder(opt)
    decoder = Decoder(opt, filed)
    net = CapModel(encoder, decoder)
    if opt.use_multi_gpu:
        net = torch.nn.DataParallel(net)
    print('Total parameters:', sum(param.numel() for param in net.parameters()))

    if os.path.exists(opt.model_pth_path) and opt.use_checkpoint:
        print('load from checkpoint')
        net.load_state_dict(torch.load(opt.model_pth_path))
    net.to(DEVICE)

    # initialize loss function and optimizer
    contrastive=Tri_Loss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.learning_rate)
#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=opt.learning_rate)
    if os.path.exists(opt.optimizer_pth_path) and opt.use_checkpoint:
        optimizer.load_state_dict(torch.load(opt.optimizer_pth_path))

    # initialize data loader
    train_loader = get_train_loader(opt.train_caption_pkl_path, opt.feature_h5_path, filed,
                                    opt.train_batch_size)
    total_step = len(train_loader)

    # prepare groundtruth
    reference = convert_data_to_coco_scorer_format(opt.test_reference_txt_path)

    # start training
    best_meteor = 0
    best_meteor_epoch = 0
    best_cider = 0
    best_cider_epoch = 0
    loss_count = 0
    count = 0
    saving_schedule = [int(x * total_step / opt.save_per_epoch) for x in list(range(1, opt.save_per_epoch + 1))]
    print('total: ', total_step)
    print('saving_schedule: ', saving_schedule)
    for epoch in range(opt.max_epoch):
        start_time = time.time()
        if opt.learning_rate_decay and epoch % opt.learning_rate_decay_every == 0 and epoch > 0:
            opt.learning_rate /= opt.learning_rate_decay_rate
        epsilon = max(0.6, opt.ss_factor / (opt.ss_factor + np.exp(epoch / opt.ss_factor)))
        logger.write('epoch:%d\tepsilon:%.8f' % (epoch, epsilon))
        print("patten:{}".format(opt.choose))
        for i, (frames, captions, cap_lens, video_ids,sent) in enumerate(train_loader, start=1):
            # convert data to DEVICE mode
            frames = frames.to(DEVICE)#torch.Size([32, 26, 4096])
            targets = captions.to(DEVICE)

            ######################################################################################
            if opt.choose=="sota":
                # compute results of the model
                optimizer.zero_grad()
                outputs, module_weights,loss_contrastive,outputs_pos = net(frames, targets,sent, epsilon,"train")
                tokens = outputs
                tokens_pos=outputs_pos
                bsz = len(captions)
                # remove pad and flatten outputs
                outputs = torch.cat([outputs[j][:cap_lens[j]] for j in range(bsz)], 0)
                outputs = outputs.view(-1, vocab_size)
                # remove pad and flatten outputs
                outputs_pos = torch.cat([outputs_pos[j][:cap_lens[j]] for j in range(bsz)], 0)
                outputs_pos = outputs_pos.view(-1, vocab_size)
                # remove pad and flatten targets
                targets = torch.cat([targets[j][:cap_lens[j]] for j in range(bsz)], 0)
                targets = targets.view(-1)
                # compute captioning loss
                cap_loss = criterion(outputs, targets)
                cap_loss_pos=criterion(outputs_pos,targets)
                total_loss = cap_loss+loss_contrastive*50+cap_loss_pos*0.5
                
            # ################################################
            elif opt.choose=="baseline":
                # compute results of the model
                optimizer.zero_grad()
                outputs, module_weights = net(frames, targets,sent, epsilon,"val")
                tokens = outputs
                bsz = len(captions)
                # remove pad and flatten outputs
                outputs = torch.cat([outputs[j][:cap_lens[j]] for j in range(bsz)], 0)
                outputs = outputs.view(-1, vocab_size)
                # remove pad and flatten targets
                targets = torch.cat([targets[j][:cap_lens[j]] for j in range(bsz)], 0)
                targets = targets.view(-1)
                # compute captioning loss
                cap_loss = criterion(outputs, targets)
                total_loss = cap_loss
            # ###############################################
            

            loss_count += total_loss.item()
            total_loss.backward()
            clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()

            if i % 100 == 0 or bsz < opt.train_batch_size:
                loss_count /= 100.0 if bsz == opt.train_batch_size else i % 100
                logger.write('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' %
                      (epoch, opt.max_epoch, i, total_step, loss_count,
                       np.exp(loss_count)))
                loss_count = 0
                tokens = tokens.max(2)[1]
                tokens = tokens.data[0].squeeze()
                if opt.use_multi_gpu:
                    we = net.module.decoder.decode_tokens(tokens)
                    gt = net.module.decoder.decode_tokens(captions[0].squeeze())
                else:
                    # pos= net.decoder.decode_tokens(tokens_pos)
                    we = net.decoder.decode_tokens(tokens)
                    gt = net.decoder.decode_tokens(captions[0].squeeze())
                if opt.choose=="sota":
                    logger.write("cap_loss:{},loss_contrastive:{},cap_loss_pos:{}".format(cap_loss,loss_contrastive,cap_loss_pos))
                else:
                    logger.write("cap_loss:{}".format(cap_loss))
                print('[vid:%d]' % video_ids[0])
                # print("pos:{}".format(pos))
                print('WE: %s\nGT: %s' % (we, gt))

            if i in saving_schedule:
                torch.save(net.state_dict(), opt.model_pth_path)
                torch.save(optimizer.state_dict(), opt.optimizer_pth_path)

                blockPrint()
                start_time_eval = time.time()
                net.eval()

                # use opt.val_range to find the best hyperparameters
                metrics = evaluate(opt, net, opt.test_range, opt.test_prediction_txt_path, reference,"val")
                end_time_eval = time.time()
                enablePrint()
                logger.write('evaluate time: %.3fs' % (end_time_eval - start_time_eval))

                for k, v in metrics.items():
                    logger.write('%s: %.6f' % (k, v))
                    if k == 'METEOR' and v > best_meteor:
                        shutil.copy2(opt.model_pth_path, opt.best_meteor_pth_path)
                        shutil.copy2(opt.optimizer_pth_path, opt.best_meteor_optimizer_pth_path)
                        best_meteor = v
                        best_meteor_epoch = epoch

                    if k == 'CIDEr' and v > best_cider:
                        shutil.copy2(opt.model_pth_path, opt.best_cider_pth_path)
                        shutil.copy2(opt.optimizer_pth_path, opt.best_cider_optimizer_pth_path)
                        best_cider = v
                        best_cider_epoch = epoch

                logger.write('Step: %d, Learning rate: %.8f' % (epoch * len(saving_schedule) + count, opt.learning_rate))
                optimizer = torch.optim.Adam(net.parameters(), lr=opt.learning_rate)
                count += 1
                count %= 4
                net.train()

        end_time = time.time()
        logger.write("*******One epoch time: %.3fs*******\n" % (end_time - start_time))
        logger.write('best cider: %.3f' % best_cider)

    with open(opt.test_score_txt_path, 'w') as f:
        f.write('MODEL: {}\n'.format(opt.model))
        f.write('best meteor epoch: {}\n'.format(best_meteor_epoch))
        f.write('best cider epoch: {}\n'.format(best_cider_epoch))
        f.write('Learning rate: {:6f}\n'.format(opt.learning_rate))
        f.write('Learning rate decay: {}\n'.format(opt.learning_rate_decay))
        f.write('Batch size: {}\n'.format(opt.train_batch_size))
        f.write('results directory: {}\n'.format(opt.result_dir))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
