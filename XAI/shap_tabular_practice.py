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
import pickle

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

    #모델 로드
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    class_names = tabular_data['label'].unique()
    class_names = np.sort(class_names)
    X = tabular_data.iloc[:, :-1]

    explainer = shap.TreeExplainer(model, X, tree_output="auto", feature_perturbation="interventional", feature_perturbation_types=None)

    shap_values = explainer.shap_values(X)

    i = 0  # 첫 번째 샘플
    shap.plots.waterfall(shap_values[i], show=False)
    plt.title("SHAP Local Explanation (Sample 0)")
    plt.savefig("shap_tabular_local.png", dpi=300)
    plt.clf()

    shap.summary_plot(shap_values, features=X, feature_names=X.columns, show=False)
    plt.title("SHAP Summary Plot (Global)")
    plt.tight_layout()
    plt.savefig("shap_tabular_summary.png", dpi=300)
    plt.clf()


if __name__ == "__main__":
    args = argparse_xai()
    main(args)
