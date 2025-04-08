import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(parent_dir)
import torch
from AI.AI_utils import data_load, set_device, setting_bert_model
from XAI.xai_utils import argparse_xai
import pickle
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
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
    tabular_data, _ = data_load(base_dir=data_base_dir, tabular_data_name=tabular_data_name, getdatatype="tabular",
                                             bert_flie=bert_file, MAXLEN=MAXLEN)

    model_path = os.path.join(model_dir, f"{model_name}_xgboost.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir.replace("../", ""), f"{model_name}_xgboost.pkl")

    # 모델 로드
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    class_names = tabular_data['label'].unique()
    class_names = np.sort(class_names)
    X = tabular_data.iloc[:, :-1]
    # LIME 설명기 생성
    i = 0  # 분석할 샘플 인덱스
    explainer = LimeTabularExplainer(
        X.values,
        feature_names=X.columns.tolist(),
        class_names=class_names,  # 필요시 수정
        mode="classification"
    )
    instance = X.iloc[i].to_numpy()
    exp = explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=10
    )
    exp.save_to_file("lime_tabular_explanation.html")


if __name__ == "__main__":
    args = argparse_xai()
    main(args)
