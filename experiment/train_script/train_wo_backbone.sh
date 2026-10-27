export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

accelerate launch --config_file ./accelerate/6_gpus_config.yaml --main_process_port 29630 --mixed_precision="no" train.py \
  --model_version="1_0" \
  --dataset_name="arctic" \
  --dataset_version="1_0" \
  --val_dataset_name="arctic" \
  --val_dataset_version="1_0" \
  --output_dir="../ckpt/v__11_0__none_temporal_wo_body_backbone" \
  --tracker_project_name="training" \
  --train_batch_size=8 \
  --dataloader_num_workers=8 \
  --gradient_accumulation_steps=2 \
  --checkpointing_epochs=5 \
  --num_train_epochs=10 \
  --learning_rate=1e-4 \
  --learning_rate_scale=0.5 \
  --scheduler_pct_start=0.1 \
  --scheduler_anneal_strategy=cos \
  --scheduler_div_factor=10.0 \
  --scheduler_final_div_factor=20.0 \
  --max_grad_norm=1.0 \
  --camera_source="gt_only" \
  --val_camera_source="gt_only" \
  --pretrained_path /data/sunzhihao2/turuirui/egohands/experiment/ckpt/pretrain_v__8_0__arctic_egodex_stage1_single_frame/checkpoint-step-25000 \