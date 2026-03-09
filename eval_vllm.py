import json
import yaml
import re
import sys
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ===================== 1. 分片逻辑 (新增) =====================
# 用来接收 sbatch 传进来的 ID，把数据切成 5 份
SLICE_ID = int(sys.argv[3]) if len(sys.argv) > 3 else 0
NUM_SLICES = 5 

# ===================== 2. 配置 (原封不动) =====================
RAW_TRAIT = "high_neuroticism"
DATASET = sys.argv[1] if len(sys.argv) > 1 else "aime24"
CURRENT_TRIAL = sys.argv[2] if len(sys.argv) > 2 else "trial1"

MODEL_PATH = "Qwen/Qwen3-4B"
TARGET_TRAIT = f"persona_{RAW_TRAIT}"
BASE_OUTPUT_DIR = Path("qwen_eval_logs_3_4")
PERSONA_FILE = Path("persona.yaml")
DATA_ROOT = Path("data")

TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0
MAX_TOKENS = 32786
BATCH_SIZE = 5

# ===================== 3. 解析函数 (原封不动) =====================
def extract_boxed_answer(text: str):
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1].strip() if matches else None

def extract_thought(text: str):
    closed_match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL)
    if closed_match: return closed_match.group(1).strip()
    if "<think>" in text: return text.split("<think>", 1)[1].strip()
    return text.strip()

def build_prompt(tokenizer, persona_prompt: str, problem: str) -> str:
    user_content = (
        f"{persona_prompt}\n\n{problem}\n\nPlease reason step by step. \n"
        "Crucially, conclude your response with a separate line starting with "
        "'Final Answer: ' followed by your answer in \\boxed{}."
    )
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)

# ===================== 4. 初始化 (原封不动) =====================
with open(PERSONA_FILE, "r", encoding="utf-8") as f:
    personas = yaml.safe_load(f)
persona_prompt = personas.get(RAW_TRAIT, "")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
llm = LLM(model=MODEL_PATH, gpu_memory_utilization=0.8, enforce_eager=True)
sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, min_p=MIN_P, max_tokens=MAX_TOKENS, seed=1)

# ===================== 5. 主逻辑 (仅修改数据切分) =====================
def run_mission():
    output_dir = BASE_OUTPUT_DIR / CURRENT_TRIAL / DATASET
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 文件名加上 SLICE_ID 防止多个进程写同一个文件
    output_file = output_dir / f"{DATASET}_{TARGET_TRAIT}_slice{SLICE_ID}.jsonl"

    input_path = DATA_ROOT / DATASET / "test.jsonl"
    with open(input_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    # --- 核心切分代码 ---
    chunk_size = (len(all_data) + NUM_SLICES - 1) // NUM_SLICES
    start_idx = SLICE_ID * chunk_size
    end_idx = min(start_idx + chunk_size, len(all_data))
    pending_data = all_data[start_idx:end_idx]
    # ------------------

    print(f"📌 Slice {SLICE_ID} | Range: {start_idx}-{end_idx} | Total: {len(pending_data)}")

    for i in range(0, len(pending_data), BATCH_SIZE):
        batch_chunk = pending_data[i:i + BATCH_SIZE]
        prompts = [build_prompt(tokenizer, persona_prompt, d.get("problem") or d.get("question", "")) for d in batch_chunk]
        outputs = llm.generate(prompts, sampling_params)

        with open(output_file, "a", encoding="utf-8") as fout:
            for entry, out in zip(batch_chunk, outputs):
                final_text = out.outputs[0].text
                record = {
                    "id": entry.get("id"),
                    "gt": str(entry.get("answer") or entry.get("gt")).replace(",", ""),
                    "preds": [extract_boxed_answer(final_text)],
                    "thought": extract_thought(final_text),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    run_mission()
