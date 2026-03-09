import json
import yaml
import re
import sys
import os
from pathlib import Path

# 强制使用特定显卡，避开可能的冲突
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ===================== 1. 配置 =====================
RAW_TRAIT = "high_neuroticism"
DATASET = sys.argv[1] if len(sys.argv) > 1 else "aime24"
CURRENT_TRIAL = sys.argv[2] if len(sys.argv) > 2 else "trial1"

MODEL_PATH = "Qwen/Qwen3-1.7B"
TARGET_TRAIT = f"persona_{RAW_TRAIT}"

BASE_OUTPUT_DIR = Path("qwen_eval_logs_3_1.7")
PERSONA_FILE = Path("persona.yaml")
DATA_ROOT = Path("data")

# 采样参数
TEMPERATURE = 0.6
TRIAL_SEED_MAP = {"trial1": 42, "trial2": 1337}
TRIAL_SEED = TRIAL_SEED_MAP.get(CURRENT_TRIAL, 42)

# ===================== 2. 解析函数 =====================

def extract_boxed_answer(text: str):
    """提取 \boxed{...} 中的内容"""
    match = re.search(r"\\boxed\{([^}]*)\}", text)
    return match.group(1).strip() if match else None

def extract_thought(text: str):
    """提取思考过程"""
    if "<think>" in text:
        parts = text.split("</think>")
        return parts[0].replace("<think>", "").strip()
    return text.split("Final answer")[0].strip()

def build_prompt(tokenizer, persona_prompt: str, problem: str) -> str:
    """使用官方模板构造 Prompt"""
    user_content = (
        f"Roleplay Identity: {persona_prompt}\n\n"
        f"Problem: {problem}\n\n"
        f"Please reason step by step inside <think> tags, "
        f"and provide the final answer in \\boxed{{}}."
    )
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ===================== 3. 初始化 =====================
print(f"🚀 Initializing Trial: {CURRENT_TRIAL} for {DATASET}", flush=True)

with open(PERSONA_FILE, "r", encoding="utf-8") as f:
    personas = yaml.safe_load(f)
persona_prompt = personas.get(RAW_TRAIT, "")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# H100 优化配置
llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    gpu_memory_utilization=0.5,  # 降低比例，防止 Driver Crash
    enforce_eager=True,          # 保持稳定模式
    max_model_len=8192,          # 允许较长上下文
    swap_space=4                 # 增加交换空间作为缓冲
)

sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    top_p=0.95,
    max_tokens=4096,             # 核心：给够模型说话的空间，防止 null
    seed=TRIAL_SEED,
    stop=["<|im_end|>", "<|endoftext|>"]
)

# ===================== 4. 推理逻辑 =====================
def run_mission():
    output_dir = BASE_OUTPUT_DIR / CURRENT_TRIAL / DATASET
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{DATASET}_{TARGET_TRAIT}_T{TEMPERATURE}.jsonl"

    # 读取数据
    input_path = DATA_ROOT / DATASET / "test.jsonl"
    with open(input_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    # 断点续传 (建议重新跑之前先删掉全是 null 的旧文件)
    done_ids = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try: done_ids.add(json.loads(line).get("id"))
                except: continue

    pending_data = [d for d in all_data if d.get("id") not in done_ids]
    if not pending_data:
        print(f"✅ {DATASET} {CURRENT_TRIAL} 已完成。")
        return

    print(f"📌 Total: {len(all_data)} | Pending: {len(pending_data)}", flush=True)

    # 分批推理核心逻辑
    BATCH_SIZE = 5 
    for i in range(0, len(pending_data), BATCH_SIZE):
        batch_chunk = pending_data[i : i + BATCH_SIZE]
        prompts = [build_prompt(tokenizer, persona_prompt, d.get("problem") or d.get("question", "")) for d in batch_chunk]
        
        print(f"🚀 Running Batch {i//BATCH_SIZE + 1}/{(len(pending_data)+BATCH_SIZE-1)//BATCH_SIZE}...", flush=True)
        
        try:
            outputs = llm.generate(prompts, sampling_params)
            
            with open(output_file, "a", encoding="utf-8") as fout:
                for entry, out in zip(batch_chunk, outputs):
                    final_text = out.outputs[0].text
                    pred = extract_boxed_answer(final_text)
                    thought_part = extract_thought(final_text)

                    record = {
                        "id": entry.get("id"),
                        "condition": TARGET_TRAIT,
                        "gt": str(entry.get("answer") or entry.get("gt")).replace(",", ""),
                        "preds": [pred],
                        "thought": thought_part,
                        "outs": [{"content": final_text}]
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush() # 每一批跑完立刻存盘
        except Exception as e:
            print(f"❌ Batch failed with error: {e}")
            continue

    print(f"✅ Mission Accomplished. Saved to: {output_file}", flush=True)

if __name__ == "__main__":
    run_mission()
