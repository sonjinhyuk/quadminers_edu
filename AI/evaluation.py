import argparse

from AI_utils import evaluate_xgboost, evaluate_bert, data_load, set_device, getXy
from pickle import load
import os

def main(arg):
    gpu = args.gpu
    data_base_dir = args.data_base_dir
    tabular_data_name = args.tabular_data_name
    bert_file = args.bert_file
    eval_model_type = args.eval_model_type
    model_dir = args.model_output_dir
    model_name = args.model_name
    device = set_device(gpu)
    random_seed = int(args.random_seed)
    MAXLEN = int(args.MAXLEN)

    if 'xgboost' in eval_model_type or 'MLP' in eval_model_type:
        tabular_data, _ = data_load(base_dir=data_base_dir, tabular_data_name=tabular_data_name, scale='minmax',
                                    getdatatype="tabular")

    if 'NLP' in eval_model_type or 'bert' in eval_model_type or "MLPNLP" in eval_model_type \
            or "MLPFeatureNLP" in eval_model_type:
        tabular_data, nlp_data_label = data_load(base_dir=data_base_dir, tabular_data_name=tabular_data_name,
                                                 bert_flie=bert_file, MAXLEN=MAXLEN)

    for tmt in eval_model_type:
        if 'xgboost' in tmt:
            tabualr_train_index_file = f"{model_dir}/tabular_train_index_seed_{random_seed}.npy"
            tabular_test_index_file = f"{model_dir}/tabular_test_index_{random_seed}.npy"
            tabular_X_train, tabular_X_test, tabular_y_train, tabular_y_test, train_indexs, test_indexs = \
                getXy(tabular_data, random_seed, tabualr_train_index_file, tabular_test_index_file)

            model_path = f"{model_dir}/{model_name}_{tmt}.pkl"
            try:
                model = load(open(model_path, "rb"))
            except FileNotFoundError:
                model_dir = model_dir.replace("../", "")
                model_path = f"{model_dir}/{model_name}_{tmt}.pkl"
                model = load(open(model_path, "rb"))
            evaluate_xgboost(tabular_X_test, tabular_y_test, model, model_name)
        elif 'bert' in tmt or 'NLP' in tmt:
            # evaluate_bert(test_loader, model, output_path_prefix, device="cuda"):
            nlp_data = nlp_data_label[0]
            labels = nlp_data_label[1]
            _, X_test, _, Y_test = getXy([nlp_data, labels], random_seed, model_type='bert')
            tmt = "nlp"
            model_path = f"{model_dir}/{model_name}_{tmt}.pth"
            if not os.path.exists(model_path):
                model_dir = model_dir.replace("../", "")
                model_path = f"{model_dir}/{model_name}_{tmt}.pth"

            evaluate_bert([X_test, Y_test], MAXLEN, model_path, device=device)
        else:
            raise ValueError(f"Unknown model type: {tmt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--data_base_dir', default='../model_data/', type=str, help='data base directory')
    parser.add_argument('--tabular_data_name', default='tabular', type=str, help='tabular data name')
    parser.add_argument('--bert_file', default='nlp_data.txt', type=str, help='nlp data name')
    parser.add_argument('--hyperprameger_tuning', action="store_true", help='hyperprameger tuning')
    parser.add_argument('--eval_model_type', choices=['xgboost', 'NLP', 'bert', 'MLPNLP', 'MLPFeatureNLP'], default=['bert'], nargs='+', help='train model name')
    parser.add_argument('--model_output_dir', default='../out_model', type=str, help='model output directory')
    parser.add_argument('--model_name', default='quadminers_edu', type=str, help='model name')
    parser.add_argument('--random_seed', default=42, type=int, help='random seed for train test split')
    parser.add_argument('--MAXLEN', default=-1, type=int, help='max length for bert input')

    args = parser.parse_args()
    main(args)