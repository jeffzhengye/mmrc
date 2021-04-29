# mmrc
machine reading comprehension with paddlepaddle

# run training
`CUDA_VISIBLE_DEVICES=0 python main.py -c configs/haihua.yml -o Global.additional_tag='cosine_aug_norandom' Global.only_folds="[0,1]" Global.num_workers=10 Global.train_bs=20 Global.top_k=5 Global.max_len=256 Global.fold_num=10 Global.epochs=10 Global.support_sen_type=0 Global.random_concat=false Global.is_train=true Global.is_eval=false Global.is_test_run=false`

# run eval
`CUDA_VISIBLE_DEVICES=0 python main.py -c configs/haihua.yml -o Global.additional_tag='cosine_aug_norandom' Global.only_folds="[0,1]" Global.num_workers=10 Global.train_bs=20 Global.top_k=5 Global.max_len=256 Global.fold_num=10 Global.epochs=10 Global.support_sen_type=0 Global.random_concat=false Global.is_train=false Global.is_eval=true Global.is_test_run=false`

# notes
uncompress the data.tgz before training.
