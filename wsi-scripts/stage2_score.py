import json
import re
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from collections import defaultdict

def is_multiple_choice(response, label):
    """检查是否为选择题（只有A、B、C、D）"""
    choice_pattern = r'^[A-D]$'
    return (re.match(choice_pattern, str(response).strip()) and 
            re.match(choice_pattern, str(label).strip()))

def calculate_accuracy(responses, labels):
    """计算选择题的准确率"""
    correct = 0
    total = len(responses)
    
    for response, label in zip(responses, labels):
        resp_str = str(response).strip()
        label_str = str(label).strip()
        
        if resp_str == label_str:
            correct += 1
    
    return correct / total if total > 0 else 0

def calculate_corpus_bleu(responses, labels):
    """计算整个语料库的BLEU分数"""
    references = []
    candidates = []
    
    for response, label in zip(responses, labels):
        resp_str = str(response).strip()
        label_str = str(label).strip()
        
        if resp_str and label_str:
            references.append([label_str.split()])  # 每个参考文本是一个列表的列表
            candidates.append(resp_str.split())     # 候选文本是一个列表
    
    if not candidates:
        return 0
    
    # 使用平滑函数处理零值
    smoothie = SmoothingFunction().method4
    return corpus_bleu(references, candidates, smoothing_function=smoothie)

def calculate_sentence_bleu(responses, labels):
    """计算句子级别的BLEU分数（平均值）"""
    bleu_scores = []
    
    for response, label in zip(responses, labels):
        resp_str = str(response).strip()
        label_str = str(label).strip()
        
        if resp_str and label_str:
            reference = [label_str.split()]
            candidate = resp_str.split()
            
            if candidate and reference[0]:
                smoothie = SmoothingFunction().method4
                bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
                bleu_scores.append(bleu_score)
    
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

def process_jsonl_file_complete(file_path):
    """处理JSONL文件的完整版本（包含选择题准确率）"""
    
    open_ended_responses = []
    open_ended_labels = []
    multiple_choice_responses = []
    multiple_choice_labels = []
    total_count = 0
    mc_filtered_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_count += 1
            try:
                data = json.loads(line.strip())
                response = data.get('response', '')
                label = data.get('labels', '')
                
                # 分离选择题和开放题
                if is_multiple_choice(response, label):
                    multiple_choice_responses.append(response)
                    multiple_choice_labels.append(label)
                    mc_filtered_count += 1
                else:
                    open_ended_responses.append(response)
                    open_ended_labels.append(label)
                
            except json.JSONDecodeError:
                print(f"警告：第{total_count}行JSON解析失败")
                continue
    
    print(f"总样本数: {total_count}")
    print(f"选择题数量: {len(multiple_choice_responses)}")
    print(f"开放题数量: {len(open_ended_responses)}")
    
    # 计算选择题准确率
    if multiple_choice_responses:
        mc_accuracy = calculate_accuracy(multiple_choice_responses, multiple_choice_labels)
        print(f"\n选择题评估结果:")
        print("-" * 30)
        print(f"准确率: {mc_accuracy:.4f} ({len(multiple_choice_responses)}个样本)")
        
        # 详细统计每个选项的分布
        response_counts = defaultdict(int)
        label_counts = defaultdict(int)
        correct_by_option = defaultdict(int)
        
        for resp, lbl in zip(multiple_choice_responses, multiple_choice_labels):
            resp_str = str(resp).strip()
            lbl_str = str(lbl).strip()
            response_counts[resp_str] += 1
            label_counts[lbl_str] += 1
            if resp_str == lbl_str:
                correct_by_option[lbl_str] += 1
        
        print("\n选项分布:")
        for option in ['A', 'B', 'C', 'D']:
            resp_count = response_counts.get(option, 0)
            label_count = label_counts.get(option, 0)
            correct_count = correct_by_option.get(option, 0)
            accuracy = correct_count / label_count if label_count > 0 else 0
            print(f"  {option}: 预测{resp_count}次, 真实{label_count}次, 准确率{accuracy:.4f}")
    
    # 计算开放题的BLEU和ROUGE
    if open_ended_responses:
        print(f"\n开放题评估结果:")
        print("-" * 30)
        
        # 初始化ROUGE评估器
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # 计算指标
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        valid_samples = 0
        
        for response, label in zip(open_ended_responses, open_ended_labels):
            resp_str = str(response).strip()
            label_str = str(label).strip()
            
            if resp_str and label_str:  # 确保非空
                rouge_scores = scorer.score(label_str, resp_str)
                rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
                rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
                rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
                valid_samples += 1
        
        # 计算BLEU（两种方式）
        corpus_bleu_score = calculate_corpus_bleu(open_ended_responses, open_ended_labels)
        sentence_bleu_score = calculate_sentence_bleu(open_ended_responses, open_ended_labels)
        
        # 打印结果
        print(f"有效样本数: {valid_samples}")
        print(f"BLEU (语料库级别): {corpus_bleu_score:.4f}")
        print(f"BLEU (句子级别平均): {sentence_bleu_score:.4f}")
        
        if rouge1_scores:
            print(f"ROUGE-1: {sum(rouge1_scores)/len(rouge1_scores):.4f}")
            print(f"ROUGE-2: {sum(rouge2_scores)/len(rouge2_scores):.4f}")
            print(f"ROUGE-L: {sum(rougeL_scores)/len(rougeL_scores):.4f}")
        else:
            print("ROUGE: 无有效样本")
    else:
        print("\n没有开放题样本")

def process_jsonl_file_simple(file_path):
    """处理JSONL文件的简化版本（不使用BLEURT）"""
    
    responses = []
    labels = []
    total_count = 0
    mc_filtered_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_count += 1
            try:
                data = json.loads(line.strip())
                response = data.get('response', '')
                label = data.get('labels', '')
                
                # 跳过选择题
                if is_multiple_choice(response, label):
                    mc_filtered_count += 1
                    continue
                
                responses.append(response)
                labels.append(label)
                
            except json.JSONDecodeError:
                print(f"警告：第{total_count}行JSON解析失败")
                continue
    
    print(f"总样本数: {total_count}")
    print(f"过滤的选择题数量: {mc_filtered_count}")
    print(f"用于评估的样本数: {len(responses)}")
    
    if len(responses) == 0:
        print("没有可用于评估的样本")
        return
    
    # 初始化ROUGE评估器
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # 计算指标
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for response, label in zip(responses, labels):
        resp_str = str(response).strip()
        label_str = str(label).strip()
        
        if resp_str and label_str:  # 确保非空
            rouge_scores = scorer.score(label_str, resp_str)
            rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
            rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
            rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
    
    # 计算BLEU（两种方式）
    corpus_bleu_score = calculate_corpus_bleu(responses, labels)
    sentence_bleu_score = calculate_sentence_bleu(responses, labels)
    
    # 打印结果
    print("\n评估结果:")
    print("-" * 50)
    print(f"BLEU (语料库级别): {corpus_bleu_score:.4f}")
    print(f"BLEU (句子级别平均): {sentence_bleu_score:.4f}")
    print(f"ROUGE-1: {sum(rouge1_scores)/len(rouge1_scores):.4f}")
    print(f"ROUGE-2: {sum(rouge2_scores)/len(rouge2_scores):.4f}")
    print(f"ROUGE-L: {sum(rougeL_scores)/len(rougeL_scores):.4f}")

# 使用示例
if __name__ == "__main__":
    file_path = "/data1/liuxiaoyu/outputs/optimus-qwen3-sft-stage2/v53-20251010-101057/checkpoint-7600/infer_result/20251010-145523.jsonl"
    
    process_jsonl_file_complete(file_path)