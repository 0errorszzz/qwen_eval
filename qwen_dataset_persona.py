import json
import yaml
import os
import re
import sys
from pathlib import Path
from vllm import LLM, SamplingParams

# ===================== 1. 环境与参数配置 =====================
RAW_TRAIT = "high_neuroticism"
DATASET = sys.argv[1] if len(sys.argv) > 1 else "aime24"
CURRENT_TRIAL = sys.argv[2] if len(sys.argv) > 2 else "trial1"

MODEL_PATH = "Qwen/Qwen3-1.7B"
TARGET_TRAIT = f"persona_{RAW_TRAIT}"

# 按照你的要求命名文件夹
BASE_OUTPUT_DIR = Path("qwen_eval_logs_3_1.7") 
# 直接在当前目录下读取
PERSONA_FILE = Path("persona.yaml")
DATA_ROOT = Path("data")

# 官方推荐的非贪婪 Thinking Mode 参数
TEMPERATURE = 0.6 

# ===================== 2. 初始化 =====================
with open(PERSONA_FILE, "r", encoding="utf-8") as f:
    personas = yaml.safe_load(f)
persona_prompt = personas.get(RAW_TRAIT, "")

# 初始化 vLLM
llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    gpu_memory_utilization=0.8,
    enforce_eager=True
)

sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    max_tokens=4096,
    seed=42 if CURRENT_TRIAL == "trial1" else 1337
)

def run_mission():
    # 结果 folder 结构：qwen_eval_logs_3_1.7/{trial}/{dataset}/
    output_dir = BASE_OUTPUT_DIR / CURRENT_TRIAL / DATASET
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 文件名：aime24_persona_high_neuroticism_T0.6.jsonl
    output_file = output_dir / f"{DATASET}_{TARGET_TRAIT}_T{TEMPERATURE}.jsonl"
    
    # 断点续传逻辑
    done_ids = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try: done_ids.add(json.loads(line).get("id"))
                except: continue

    # 读取输入数据 (支持 data/aime24/test.jsonl 这种截图里的结构)
    input_path = DATA_ROOT / DATASET / "test.jsonl"
    with open(input_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    pending_data = [d for d in all_data if d.get("id") not in done_ids]
    if not pending_data:
        print(f"✅ {DATASET} {CURRENT_TRIAL} 已达标")
        return

    # 构造 Prompts
    prompts = []
    for entry in pending_data:
        problem = entry.get("problem") or entry.get("question")
        full_prompt = f"{persona_prompt}\n\n{problem}\n\nFinal answer in \\boxed{{}}."
        # Qwen3 官方 ChatML 格式
        formatted_prompt = f"<|im_start|>user\n{full_prompt}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(formatted_prompt)

    # 推理
    outputs = llm.generate(prompts, sampling_params)

    # 记录结果 (包含 id, condition, gt, preds, thought, outs)
    with open(output_file, "a", encoding="utf-8") as fout:
        for entry, out in zip(pending_data, outputs):
            final_text = out.outputs[0].text
            
            # 提取 boxed 内容
            match = re.search(r"\\boxed\{([^}]*)\}", final_text)
            pred = match.group(1).strip() if match else None
            
            # 将思维过程和全文分离
            thought_part = final_text.split("Final answer")[0].strip()

            record = {
                "id": entry.get("id"), 
                "condition": TARGET_TRAIT,
                "gt": str(entry.get("answer") or entry.get("gt")).replace(',', ''),
                "preds": [pred],
                "thought": thought_part, 
                "outs": [{"content": final_text}]
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    run_mission()
