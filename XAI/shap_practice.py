import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(parent_dir)
import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경에서도 안정적으로 작동
import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer
from AI.AI_utils import create_directory, data_load, set_device, setting_bert_model
from XAI.xai_utils import argparse_xai
import shap
import numpy as np
from collections import defaultdict

def predict_prob_(texts, model, tokenizer, max_len, device):
    """
    LIME이 요구하는 예측 함수 형태에 맞춰, 문자열 리스트 -> 확률 벡터 (numpy array) 반환
    """
    model.eval()
    if max_len > 512 or max_len == -1:
        max_len = 512
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    token_type_ids = encoded["token_type_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids)
        probs = torch.softmax(outputs, dim=-1)
    return probs.cpu().numpy()


def main(args):
    gpu = args.gpu
    data_base_dir = args.data_base_dir
    tabular_data_name = args.tabular_data_name
    bert_file = args.bert_file
    model_dir = args.model_output_dir
    model_name = args.model_name
    device = set_device(gpu)
    random_seed = int(args.random_seed)
    MAXLEN = int(args.MAXLEN)

    # Load data
    tabular_data, nlp_data_label = data_load(base_dir=data_base_dir, tabular_data_name=tabular_data_name,
                                             bert_flie=bert_file, MAXLEN=MAXLEN)
    nlp_data = nlp_data_label[0]
    labels = nlp_data_label[1]

    model_path = os.path.join(model_dir, f"{model_name}_nlp.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir.replace("../", ""), f"{model_name}_nlp.pth")

    output_numbers = len(set(labels))
    bert_model = setting_bert_model(device, model_path, output_numbers=output_numbers)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    # 2. 샘플 문장
    test_sentence = nlp_data[0]
    # 3. 예측 함수 정의
    def f(texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        elif isinstance(texts, str):
            texts = [texts]
        tokenizer_max_len = MAXLEN
        if tokenizer_max_len == -1 or tokenizer_max_len > 512:
            tokenizer_max_len = 512
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAXLEN,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        token_type_ids = encoded["token_type_ids"].to(device)
        with torch.no_grad():
            outputs = bert_model(input_ids, attention_mask, token_type_ids)
            probs = torch.softmax(outputs, dim=-1)
        return probs.cpu().numpy()

    explainer = shap.Explainer(f, shap.maskers.Text(tokenizer))

    shap_values = explainer([test_sentence])
    html = shap.plots.text(shap_values[0], display=False)  # display=False 중요!
    with open("shap_explanation.html", "w", encoding="utf-8") as f:
        f.write(html)


    # 전역 설명을 위해 nlp_data에서 일부 샘플 선택 (예: 처음 100개)
    # 과부화 방지

    global_samples = nlp_data[:100]
    shap_values_global = explainer(global_samples)
    word_contributions = defaultdict(list)
    # shap_values_global = explainer(nlp_data[:100])
    for sv in shap_values_global:
        for token, value in zip(sv.data, sv.values):
            word_contributions[token].append(abs(value))
    # 평균 절대값으로 정렬
    avg_contributions = {
        word: np.mean(vals) for word, vals in word_contributions.items()
    }
    top_words = sorted(avg_contributions.items(), key=lambda x: x[1], reverse=True)[:20]
    tokens, scores = zip(*top_words)
    plt.figure(figsize=(10, 6))
    plt.barh(tokens[::-1], scores[::-1])
    plt.title("Top Global SHAP Tokens (mean abs val)")
    plt.xlabel("Average |SHAP value|")
    plt.tight_layout()
    plt.savefig("shap_text_global_manual.png")
    plt.clf()

    word_contributions = defaultdict(list)
    for sv in shap_values_global:
        for token, value in zip(sv.data, sv.values):
            word_contributions[token].append(value)  # 부호 포함
    summary_stats = []
    for token, vals in word_contributions.items():
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        summary_stats.append((token, mean_val, std_val))

    # 상위 N개 정렬
    top_k = 20
    summary_stats.sort(key=lambda x: abs(x[1]), reverse=True)
    summary_stats = summary_stats[:top_k]

    tokens, means, stds = zip(*summary_stats)
    colors = ["red" if m > 0 else "blue" for m in means]
    plt.figure(figsize=(10, 6))
    for i, (token, mean, std) in enumerate(zip(tokens, means, stds)):
        color = 'red' if mean > 0 else 'blue'
        plt.errorbar(
            x=mean,
            y=i,
            xerr=std,
            fmt='o',
            color='black',
            ecolor=color,
            elinewidth=3,
            capsize=5
        )

    plt.yticks(np.arange(len(tokens)), tokens)
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.title("SHAP Summary Style Plot (Text Features)")
    plt.xlabel("Mean SHAP Value (± std)")
    plt.tight_layout()
    plt.savefig("shap_summary_style_plot.png", dpi=300)
    plt.clf()


if __name__ == "__main__":
    args = argparse_xai()
    main(args)
