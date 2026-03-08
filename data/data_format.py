import json

file_path = '/expanse/lustre/projects/uic458/mai1/LLM_Persona/eval/data/ifeval/ifeval_input_data.jsonl'
out_path = f'{'/'.join(file_path.split("/")[:-1])}/test.jsonl'

# 使用 'w' 模式打开输出文件
with open(file_path, 'r', encoding='utf-8') as f, \
     open(out_path, 'w', encoding='utf-8') as out_f:
    
    for line in f:
        if not line.strip(): 
            continue 
        
        entry = json.loads(line)
        
        entry['problem'] = entry['prompt']
        entry['id'] = entry['key']
        
        # 提取 '####' 之后的答案部分
        # ans = entry['answer'].split('####')[-1].strip()
        # prob = entry['question']
        
        # 更新 entry 的内容
        # entry['answer'] = ans
        # entry['problem'] = prob
        
        # 将修改后的 entry 转换为 JSON 字符串并写入新文件
        out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"处理完成！文件已保存至: {out_path}")