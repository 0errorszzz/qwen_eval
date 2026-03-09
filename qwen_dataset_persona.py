import json
import yaml
import re
import sys
from pathlib import Path

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

# 官方推荐的 thinking mode 参数
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0

# 参考文档，给足
MAX_TOKENS =16384

# 小 batch 降低崩溃概率
BATCH_SIZE = 15

TRIAL_SEED_MAP = {
    "trial1": 42,
    "trial2": 1337,
}
TRIAL_SEED = TRIAL_SEED_MAP.get(CURRENT_TRIAL, 42)


# ===================== 2. 解析函数 =====================
def extract_boxed_answer(text: str):
    # 逻辑 A: 找标准 boxed
    match = re.search(r"\\boxed\{([^}]*)\}", text)
    if match: return match.group(1).strip()
    
    # 逻辑 B: 找最后出现的数字（针对 AIME 填空题非常有效）
    nums = re.findall(r"\d+", text)
    return nums[-1] if nums else None


def extract_thought(text: str):
    closed_match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL)
    if closed_match:
        return closed_match.group(1).strip()

    if "<think>" in text:
        return text.split("<think>", 1)[1].strip()

    return text.strip()


def build_prompt(tokenizer, persona_prompt: str, problem: str) -> str:
    user_content = (
        f"{persona_prompt}\n\n"
        f"{problem}\n\n"
        f"Please reason step by step, and put your final answer within \\boxed{{}}."
    )

    messages = [{"role": "user", "content": user_content}]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
        thinking_budget=3072,
    )


# ===================== 3. 初始化 =====================
print(f"🚀 Initializing {DATASET} | {CURRENT_TRIAL}", flush=True)

with open(PERSONA_FILE, "r", encoding="utf-8") as f:
    personas = yaml.safe_load(f)
persona_prompt = personas.get(RAW_TRAIT, "")

print("📦 Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("✅ Tokenizer loaded.", flush=True)

print("🧠 Initializing vLLM...", flush=True)
llm = LLM(
    model=MODEL_PATH,
    gpu_memory_utilization=0.8,
    enforce_eager=True,
)
print("✅ vLLM initialized.", flush=True)

sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    min_p=MIN_P,
    max_tokens=MAX_TOKENS,
    seed=TRIAL_SEED,
    presence_penalty=1.5,
)


# ===================== 4. 主逻辑 =====================
def run_mission():
    output_dir = BASE_OUTPUT_DIR / CURRENT_TRIAL / DATASET
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{DATASET}_{TARGET_TRAIT}_T{TEMPERATURE}.jsonl"

    input_path = DATA_ROOT / DATASET / "test.jsonl"
    with open(input_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    done_ids = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line).get("id"))
                except Exception:
                    continue

    pending_data = [d for d in all_data if d.get("id") not in done_ids]
    

    if not pending_data:
        print(f"✅ {DATASET} {CURRENT_TRIAL} 已完成。", flush=True)
        return

    print(f"📌 Total: {len(all_data)} | Pending: {len(pending_data)}", flush=True)

    total_batches = (len(pending_data) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(pending_data), BATCH_SIZE):
        batch_chunk = pending_data[i:i + BATCH_SIZE]
        prompts = [
            build_prompt(tokenizer, persona_prompt, d.get("problem") or d.get("question", ""))
            for d in batch_chunk
        ]

        print(
            f"🚀 Running Batch {i // BATCH_SIZE + 1}/{total_batches} ...",
            flush=True
        )

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

                fout.flush()

        except Exception as e:
            print(f"❌ Batch failed with error: {e}", flush=True)
            continue

    print(f"✅ Mission Accomplished. Saved to: {output_file}", flush=True)


if __name__ == "__main__":
    run_mission()
