import subprocess
import time
from pathlib import Path

# ===================== 配置 =====================
PYTHON_EXEC = "python" 
SCRIPT_NAME = "qwen_dataset_persona.py"
TRIALS = ["trial1", "trial2"]
DATASETS = ["aime24", "gpqa-diamond", "gsm8k"]
TRAIT = "high_neuroticism"
TARGET_TRAIT = f"persona_{TRAIT}"

# 对应你截图的结构和题数
GOALS = {"aime24": 30, "gpqa-diamond": 198, "gsm8k": 1319}
BASE_DIR = Path("qwen_eval_logs_3_1.7")

def count_lines(trial, dataset):
    file_path = BASE_DIR / trial / dataset / f"{dataset}_{TARGET_TRAIT}_T0.6.jsonl"
    if not file_path.exists(): return 0
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

def run_double_trial_mission():
    print(f"🌟 启动 Qwen3-1.7B 高神经质自动化补全任务...")
    
    for tr in TRIALS:
        print(f"\n🚀 --- 正在处理 {tr} ---")
        for ds in DATASETS:
            target_goal = GOALS.get(ds, 0)
            while True:
                current = count_lines(tr, ds)
                if current >= target_goal:
                    print(f"   ✅ {tr}/{ds} 已达标 ({current}/{target_goal})")
                    break
                
                print(f"   ⏳ {tr}/{ds} 进度: {current}/{target_goal}，执行推理...")
                try:
                    subprocess.run([PYTHON_EXEC, SCRIPT_NAME, ds, tr], check=True)
                except Exception as e:
                    print(f"   ⚠️ 错误: {e}，5秒后重试...")
                    time.sleep(5)
                time.sleep(1)

if __name__ == "__main__":
    run_double_trial_mission()