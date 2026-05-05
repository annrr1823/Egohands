export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

accelerate launch --config_file ./accelerate/6_gpus_config.yaml --main_process_port 29630 --mixed_precision="no" train_ablation_wo_disill.py \
  --model_version="1_0" \
  --dataset_name="arctic" \
  --dataset_version="2_0" \
  --val_dataset_name="arctic" \
  --val_dataset_version="2_0" \
  --output_dir="../ckpt/v__12_0_wo_distill" \
  --tracker_project_name="training" \
  --train_batch_size=1 \
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
  --sequence_length=8 \
  --camera_source="cache_only" \
  --val_camera_source="cache_only" \
  --pretrained_path /data/sunzhihao2/turuirui/egohands/experiment/ckpt/pretrain_v__8_0__arctic_egodex_stage1_single_frame/checkpoint-step-25000/model.safetensors \