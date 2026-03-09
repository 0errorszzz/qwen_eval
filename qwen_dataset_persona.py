import json
import yaml
import re
import sys
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ===================== 1. 环境与参数配置 =====================
RAW_TRAIT = "high_neuroticism"
DATASET = sys.argv[1] if len(sys.argv) > 1 else "aime24"
CURRENT_TRIAL = sys.argv[2] if len(sys.argv) > 2 else "trial1"

MODEL_PATH = "Qwen/Qwen3-1.7B"
TARGET_TRAIT = f"persona_{RAW_TRAIT}"

BASE_OUTPUT_DIR = Path("qwen_eval_logs_3_1.7")
PERSONA_FILE = Path("persona.yaml")
DATA_ROOT = Path("data")

# 官方推荐参数
TEMPERATURE = 0.6
TRIAL_SEED_MAP = {"trial1": 42, "trial2": 1337}
TRIAL_SEED = TRIAL_SEED_MAP.get(CURRENT_TRIAL, 42)

# ===================== 2. 工具函数 =====================

def extract_boxed_answer(text: str):
    """提取 \boxed{...} 中的内容"""
    match = re.search(r"\\boxed\{([^}]*)\}", text)
    return match.group(1).strip() if match else None

def extract_thought(text: str):
    """
    优先提取 <think> 标签内容，如果没有则尝试按分隔符截断，
    最后还没找到就返回全文。
    """
    closed_match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL)
    if closed_match:
        return closed_match.group(1).strip()
    
    if "<think>" in text:
        return text.split("<think>")[-1].split("Final answer")[0].strip()
    
    return text.split("Final answer")[0].strip()

def build_prompt(tokenizer, persona_prompt: str, problem: str) -> str:
    """使用官方 ChatML 模板构造，并引导模型开启 Thinking 模式"""
    user_content = (
        f"You are now roleplaying: {persona_prompt}\n\n"
        f"Problem: {problem}\n\n"
        f"Please reason step by step inside <think> tags, "
        f"and provide the final answer in \\boxed{{}}."
    )
    messages = [{"role": "user", "content": user_content}]
    
    # 显式使用模板，Qwen3 会根据模板自动添加起始符
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

# ===================== 3. 初始化 =====================
print(f"🚀 Initializing Trial: {CURRENT_TRIAL} for Dataset: {DATASET}", flush=True)

with open(PERSONA_FILE, "r", encoding="utf-8") as f:
    personas = yaml.safe_load(f)
persona_prompt = personas.get(RAW_TRAIT, "")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# vLLM 配置：保持 enforce_eager=True 以求最稳
llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    gpu_memory_utilization=0.8,  # 提高利用率，给 KV Cache 留足空间
    enforce_eager=True,          # 你的集群环境下保持 True
    max_model_len=4096           # 硬性限制上下文长度
)

sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    top_p=0.95,
    top_k=20,
    max_tokens=2048,             # 1.7B 建议单次生成不超过 2048
    seed=TRIAL_SEED,
    stop=["<|im_end|>", "<|endoftext|>"] # 强制停止符
)

# ===================== 4. 主逻辑 =====================
def run_mission():
    output_dir = BASE_OUTPUT_DIR / CURRENT_TRIAL / DATASET
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{DATASET}_{TARGET_TRAIT}_T{TEMPERATURE}.jsonl"

    # 断点续传
    done_ids = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try: done_ids.add(json.loads(line).get("id"))
                except: continue

    input_path = DATA_ROOT / DATASET / "test.jsonl"
    with open(input_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    pending_data = [d for d in all_data if d.get("id") not in done_ids]
    if not pending_data:
        print(f"✅ {DATASET} {CURRENT_TRIAL} 已完成。")
        return

    print(f"📌 Total: {len(all_data)} | Pending: {len(pending_data)}", flush=True)

    # 构造 Prompts
    prompts = [build_prompt(tokenizer, persona_prompt, d.get("problem") or d.get("question", "")) for d in pending_data]

    # 推理 (vLLM 自带批处理，不需要手动写 batch 循环除非为了极其严格的显存控制)
    print("🚀 Starting generation...", flush=True)
    outputs = llm.generate(prompts, sampling_params)

    # 记录结果
    with open(output_file, "a", encoding="utf-8") as fout:
        for entry, out in zip(pending_data, outputs):
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
            fout.flush() # 实时写入，防止挂掉时丢失数据

    print(f"✅ Saved to: {output_file}", flush=True)

if __name__ == "__main__":
    run_mission()
