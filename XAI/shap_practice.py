import os

import matplotlib.pyplot as plt
import torch
import argparse
import numpy as np
from transformers import BertTokenizer
from lime.lime_text import LimeTextExplainer
from functools import partial
from AI_utils import create_directory, data_load, set_device, setting_bert_model
from AI_utils.TH_BERT import bert_training, bert_MLP_training
from XAI.xai_utils import argparse_xai

def predict_prob_(texts, model, tokenizer, max_len, device):
    """
    LIME이 요구하는 예측 함수 형태에 맞춰, 문자열 리스트 -> 확률 벡터 (numpy array) 반환
    """
    model.eval()
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

    import shap
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification

    # 1. 모델과 토크나이저 로드
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
    model.load_state_dict(torch.load('./your_trained_model.pth'))
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    # 2. 샘플 문장
    test_sentence = nlp_data[0]
    # 3. 예측 함수 정의
    def f(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
        with torch.no_grad():
            output = model(**inputs)
        probs = torch.nn.functional.softmax(output.logits, dim=1)
        return probs.numpy()

    explainer = shap.Explainer(f, shap.maskers.Text(tokenizer))

    shap_values = explainer([test_sentence])
    shap.plots.text(shap_values[0])

    plt.savefig("shap_plot.png")

if __name__ == "__main__":
    args = argparse_xai()
    main(args)
