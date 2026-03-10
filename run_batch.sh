#!/bin/bash
#SBATCH --job-name=qwen_aime_eval
#SBATCH --partition=shared           # 你的房间名
#SBATCH --account=uic458             # 你的买单账号 (重点！)
#SBATCH --output=logs/out_%A_%a.log
#SBATCH --error=logs/err_%A_%a.log
#SBATCH --gres=gpu:1                 # 申请 1 块 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-4                  # 同时交 5 个任务
#SBATCH --time=04:00:00              # 4 小时限制

# 自动创建日志目录
mkdir -p logs

# 执行 Python 脚本
# 传参: 数据集, Trial名, 当前分片ID
python eval_vllm.py aime24 trial1 ${SLURM_ARRAY_TASK_ID}
