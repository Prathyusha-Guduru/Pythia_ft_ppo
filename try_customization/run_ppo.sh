export HF_HOME=/scratch/cluster/piti/hf_cache
export PIP_CACHE_DIR=/scratch/cluster/piti/pip_cache
export TRITON_CACHE_DIR=/scratch/cluster/piti/triton_cache
export CUDA_HOME=/scratch/cluster/piti/cuda-jul
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH 

accelerate launch --main_process_port 0 --config_file  ../../accelerate_configs/deepspeed_zero2.yaml \
ppo_tldr.py \
--dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
--dataset_test_split validation \
--learning_rate 3e-6 \
--output_dir /scratch/cluster/piti/trl/examples/scripts/ppo/freeze_run_1000 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 4 \
--total_episodes 1000 \
--model_name_or_path EleutherAI/pythia-1b-deduped \
--sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
--reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
--missing_eos_penalty 1.0 \
--stop_token eos \
--response_length 52 \
--eval_strategy steps \
--eval_steps 100 \
