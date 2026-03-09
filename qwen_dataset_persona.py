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

# 官方推荐的 thinking mode 采样参数
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0

# 先保守一点，避免太长导致不稳定
MAX_TOKENS = 1024

# 小批量跑，避免 vLLM 一次吞太多
BATCH_SIZE = 4

# trial 对应不同 seed，temperature 不变
TRIAL_SEED_MAP = {
    "trial1": 42,
    "trial2": 1337,
}
TRIAL_SEED = TRIAL_SEED_MAP.get(CURRENT_TRIAL, 42)


# ===================== 2. 工具函数 =====================
def extract_boxed_answer(text: str):
    """
    只提取 \\boxed{...}，如果没有就返回 None
    """
    match = re.search(r"\\boxed\{([^}]*)\}", text)
    if match:
        return match.group(1).strip()
    return None


def extract_thought(text: str):
    """
    优先提取 <think>...</think>
    如果没有闭合，就保留从 <think> 开始的内容
    如果完全没有 <think>，就返回整个文本
    """
    closed_match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL)
    if closed_match:
        return "<think>\n" + closed_match.group(1).strip()

    open_match = re.search(r"(<think>.*)", text, flags=re.DOTALL)
    if open_match:
        return open_match.group(1).strip()

    return text.strip()


def build_prompt(tokenizer, persona_prompt: str, problem: str) -> str:
    """
    用官方 chat template 构造 prompt，并显式要求 thinking + boxed final answer
    """
    user_content = (
        f"{persona_prompt}\n\n"
        f"{problem}\n\n"
        f"/think\n"
        f"Please reason step by step, and at the end provide the final answer in the format "
        f"\\boxed{{your_answer}}."
    )

    messages = [{"role": "user", "content": user_content}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return prompt


# ===================== 3. 初始化 =====================
print("1. loading persona", flush=True)
with open(PERSONA_FILE, "r", encoding="utf-8") as f:
    personas = yaml.safe_load(f)
persona_prompt = personas.get(RAW_TRAIT, "")

print("2. loading tokenizer", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("3. tokenizer loaded", flush=True)

print("4. initializing llm", flush=True)
llm = LLM(
    model=MODEL_PATH,
    gpu_memory_utilization=0.2,
    enforce_eager=True,
)
print("5. llm initialized", flush=True)

sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    min_p=MIN_P,
    max_tokens=MAX_TOKENS,
    seed=TRIAL_SEED,
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
                try:
                    done_ids.add(json.loads(line).get("id"))
                except Exception:
                    continue

    print("6. reading dataset", flush=True)
    input_path = DATA_ROOT / DATASET / "test.jsonl"
    with open(input_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    pending_data = [d for d in all_data if d.get("id") not in done_ids]

    if not pending_data:
        print(f"✅ {DATASET} {CURRENT_TRIAL} 已达标", flush=True)
        return

    print(f"📌 Dataset: {DATASET}", flush=True)
    print(f"📌 Trial: {CURRENT_TRIAL}", flush=True)
    print(f"📌 Seed: {TRIAL_SEED}", flush=True)
    print(f"📌 Total: {len(all_data)}", flush=True)
    print(f"📌 Pending: {len(pending_data)}", flush=True)

    print("7. building prompts", flush=True)
    prompts = []
    for entry in pending_data:
        problem = entry.get("problem") or entry.get("question") or ""
        prompt = build_prompt(tokenizer, persona_prompt, problem)
        prompts.append(prompt)

    print("8. starting batched generation", flush=True)

    total_batches = (len(pending_data) + BATCH_SIZE - 1) // BATCH_SIZE

    with open(output_file, "a", encoding="utf-8") as fout:
        for start in range(0, len(pending_data), BATCH_SIZE):
            batch_idx = start // BATCH_SIZE + 1
            batch_data = pending_data[start:start + BATCH_SIZE]
            batch_prompts = prompts[start:start + BATCH_SIZE]

            print(f"🚀 Generating batch {batch_idx}/{total_batches}", flush=True)

            outputs = llm.generate(batch_prompts, sampling_params)

            for entry, out in zip(batch_data, outputs):
                final_text = out.outputs[0].text
                pred = extract_boxed_answer(final_text)
                thought_part = extract_thought(final_text)

                record = {
                    "id": entry.get("id"),
                    "condition": TARGET_TRAIT,
                    "gt": str(entry.get("answer") or entry.get("gt")).replace(",", ""),
                    "preds": [pred],
                    "thought": thought_part,
                    "outs": [{"content": final_text}],
                }

                # 基本结构自检
                assert "id" in record
                assert "condition" in record
                assert "gt" in record
                assert "preds" in record
                assert "thought" in record
                assert "outs" in record
                assert isinstance(record["preds"], list)
                assert isinstance(record["outs"], list)

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

    print(f"✅ Saved to: {output_file}", flush=True)


if __name__ == "__main__":
    run_mission()
