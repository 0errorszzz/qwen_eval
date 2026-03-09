#!/bin/bash
#SBATCH --job-name=qwen_aime_eval      # 任务在队列里的名字
#SBATCH --output=logs/out_%A_%a.log    # 标准输出日志 (%A是总任务ID, %a是子任务ID)
#SBATCH --error=logs/err_%A_%a.log     # 错误输出日志
#SBATCH --gres=gpu:1                   # 关键：每个子任务申请 1 块显卡
#SBATCH --ntasks=1                     # 每个子任务运行 1 个实例
#SBATCH --cpus-per-task=4              # 每个子任务分配 4 个 CPU 核心
#SBATCH --mem=32G                      # 每个子任务分配 32G 内存
#SBATCH --array=0-4                    # 关键：同时提交 5 个子任务 (编号 0,1,2,3,4)

# 创建日志文件夹，防止因为找不到目录而报错
mkdir -p logs

# 执行你的 Python 脚本
# 这里的 ${SLURM_ARRAY_TASK_ID} 会自动变成 0, 1, 2, 3, 4 传给 Python 的 sys.argv[3]
python eval_vllm.py aime24 trial1 ${SLURM_ARRAY_TASK_ID}
