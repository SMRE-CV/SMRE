CUDA_VISIBLE_DEVICES=1 python train.py --dataset=msvd --model=RMN \
 --result_dir=results/msvd_best --use_lin_loss \
 --learning_rate_decay --learning_rate_decay_every=5 \
 --learning_rate_decay_rate=3 \
 --learning_rate=1e-4 --attention=soft \
 --choose=sota \
 --data_aug=True \
 --mask=False \
 --hidden_size=1024 --att_size=1024\
 --train_batch_size=32 --test_batch_size=32 --topk=18 --max_epoch=20 --max_words=26 --beam_size=5
