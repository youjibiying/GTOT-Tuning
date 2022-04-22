data_path='./'
rg_type=gtot_feature_map
ft_type=gtot
tag='gtot_cosine'
save_file=${data_path}/shell/results.csv
echo $save_file

## for GTOT with best hyper-parameter

dist_metric=norm_cosine
tag=gtot_best_parameter
filename=none
patience=20
gtot_order=1

dataset=bace
python ${data_path}/finetune.py --input_model_file ${data_path}/model_gin/contextpred.pth --split scaffold --gpu 0 --runseed 0 --gnn_type gin --dataset ${dataset} --gtot_order ${gtot_order} --regularization_type ${rg_type} --finetune_type ${ft_type} --attention_epochs 10 --attention_iteration_limit 10 --tag ${tag} --save_file ${save_file} --patience ${patience} --dist_metric ${dist_metric} --lr 0.0005 --dropout_ratio 0.25 --trade_off_backbone 0.01 --trade_off_head 0.0001 --decay 0.0005 --batch_size 16 --filename ${filename}
exit
dataset=bbbp
python ${data_path}/finetune.py --input_model_file ${data_path}/model_gin/contextpred.pth --split scaffold --gpu 0 --runseed 0 --gnn_type gin --dataset ${dataset} --gtot_order ${gtot_order} --regularization_type ${rg_type} --finetune_type ${ft_type} --attention_epochs 10 --attention_iteration_limit 10 --tag ${tag} --save_file ${save_file} --patience ${patience} --dist_metric ${dist_metric} --lr 0.01 --dropout_ratio 0.4 --trade_off_backbone 0.000005 --trade_off_head 0.01 --decay 1e-6 --batch_size 32 --filename ${filename}
dataset=toxcast
python ${data_path}/finetune.py --input_model_file ${data_path}/model_gin/contextpred.pth --split scaffold --gpu 0 --runseed 0 --gnn_type gin --dataset ${dataset} --gtot_order ${gtot_order} --regularization_type ${rg_type} --finetune_type ${ft_type} --attention_epochs 10 --attention_iteration_limit 10 --tag ${tag} --save_file ${save_file} --patience ${patience} --dist_metric ${dist_metric} --lr 0.005 --dropout_ratio 0.1 --trade_off_backbone 5 --trade_off_head 5e-5 --decay 1e-6 --batch_size 64 --filename ${filename}
dataset=sider
python ${data_path}/finetune.py --input_model_file ${data_path}/model_gin/contextpred.pth --split scaffold --gpu 0 --runseed 0 --gnn_type gin --dataset ${dataset} --gtot_order ${gtot_order} --regularization_type ${rg_type} --finetune_type ${ft_type} --attention_epochs 10 --attention_iteration_limit 10 --tag ${tag} --save_file ${save_file} --patience ${patience} --dist_metric ${dist_metric} --lr 0.001 --dropout_ratio 0.25 --trade_off_backbone 10 --trade_off_head 0.001 --decay 0 --batch_size 32 --filename ${filename}
dataset=clintox
python ${data_path}/finetune.py --input_model_file ${data_path}/model_gin/contextpred.pth --split scaffold --gpu 0 --runseed 0 --gnn_type gin --dataset ${dataset} --gtot_order ${gtot_order} --regularization_type ${rg_type} --finetune_type ${ft_type} --attention_epochs 10 --attention_iteration_limit 10 --tag ${tag} --save_file ${save_file} --patience ${patience} --dist_metric ${dist_metric} --lr 0.005 --dropout_ratio 0.05 --trade_off_backbone 0.0005 --trade_off_head 0.1 --decay 1e-7 --batch_size 64 --filename ${filename}
dataset=tox21
python ${data_path}/finetune.py --input_model_file ${data_path}/model_gin/contextpred.pth --split scaffold --gpu 0 --runseed 0 --gnn_type gin --dataset ${dataset} --gtot_order ${gtot_order} --regularization_type ${rg_type} --finetune_type ${ft_type} --attention_epochs 10 --attention_iteration_limit 10 --tag ${tag} --save_file ${save_file} --patience ${patience} --dist_metric ${dist_metric} --lr 0.005 --dropout_ratio 0.3 --trade_off_backbone 0.005 --trade_off_head 0.0 --decay 1e-7 --batch_size 32 --filename ${filename}
dataset=muv
python ${data_path}/finetune.py --input_model_file ${data_path}/model_gin/contextpred.pth --split scaffold --gpu 0 --runseed 0 --gnn_type gin --dataset ${dataset} --gtot_order ${gtot_order} --regularization_type ${rg_type} --finetune_type ${ft_type} --attention_epochs 10 --attention_iteration_limit 10 --tag ${tag} --save_file ${save_file} --patience ${patience} --dist_metric ${dist_metric} --lr 0.005 --dropout_ratio 0.55 --trade_off_backbone 1 --trade_off_head 0.0005 --decay 1e-6 --batch_size 64 --filename ${filename}
dataset=hiv
python ${data_path}/finetune.py --input_model_file ${data_path}/model_gin/contextpred.pth --split scaffold --gpu 0 --runseed 0 --gnn_type gin --dataset ${dataset} --gtot_order ${gtot_order} --regularization_type ${rg_type} --finetune_type ${ft_type} --attention_epochs 10 --attention_iteration_limit 10 --tag ${tag} --save_file ${save_file} --patience ${patience} --dist_metric ${dist_metric} --lr 0.0005 --dropout_ratio 0.15 --trade_off_backbone 5e-6 --trade_off_head 0.05 --decay 1e-6 --batch_size 64 --filename ${filename}