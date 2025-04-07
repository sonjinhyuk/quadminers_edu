from functools import wraps
from time import time
import pandas as pd
import numpy as np
from AI_utils.TH_BERT import BERTClass
import xgboost as xgb
import torch
from tqdm import tqdm
from pickle import dump, load
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
import os
from AI_utils.TH_BERT import CustomDataset, setting_bert_model, validation_bert
from transformers import BertTokenizer
from torch.utils.data import DataLoader
def set_device(gpu):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set the GPU 2 to use
    # device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    device = 'cpu'
    return device

require_columns = ["src_port", "dst_port", "flow_duration", "tot_fwd_pkts", "tot_bwd_pkts", "totlen_fwd_pkts", "totlen_bwd_pkts", "fwd_pkt_len_max", "fwd_pkt_len_min", "fwd_pkt_len_mean", "fwd_pkt_len_std", "bwd_pkt_len_max", "bwd_pkt_len_min", "bwd_pkt_len_mean", "bwd_pkt_len_std", "flow_byts_s", "flow_pkts_s", "flow_iat_mean", "flow_iat_std", "flow_iat_max", "flow_iat_min", "fwd_iat_tot", "fwd_iat_mean", "fwd_iat_std", "fwd_iat_max", "fwd_iat_min", "bwd_iat_tot", "bwd_iat_mean", "bwd_iat_std", "bwd_iat_max", "bwd_iat_min", "fwd_psh_flags", "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags", "fwd_header_len", "bwd_header_len", "fwd_pkts_s", "bwd_pkts_s", "pkt_len_min", "pkt_len_max", "pkt_len_mean", "pkt_len_std", "pkt_len_var", "fin_flag_cnt", "syn_flag_cnt", "rst_flag_cnt", "psh_flag_cnt", "ack_flag_cnt", "urg_flag_cnt", "cwe_flag_count", "ece_flag_cnt", "down_up_ratio", "pkt_size_avg", "fwd_seg_size_avg", "bwd_seg_size_avg", "fwd_byts_b_avg", "fwd_pkts_b_avg", "fwd_blk_rate_avg", "bwd_byts_b_avg", "bwd_pkts_b_avg", "bwd_blk_rate_avg", "subflow_fwd_pkts", "subflow_fwd_byts", "subflow_bwd_pkts", "subflow_bwd_byts", "init_fwd_win_byts", "init_bwd_win_byts", "fwd_act_data_pkts", "fwd_seg_size_min", "active_mean", "active_std", "active_max", "active_min", "idle_mean", "idle_std", "idle_max", "idle_min", "payload_len", 'label']
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % \
          (f.__name__, te-ts))
        return result
    return wrap
@timing
def data_load(base_dir="./model_data/",
              tabular_data_name="merge_tabular", tabular_data_ext="csv",
              bert_flie="merge_nlp_data.txt",
              encoding="utf-8",
              getdatatype="nlp",
              scale=True, scaler_path=None,
              MAXLEN=512):
    in_file = f"{base_dir}/{tabular_data_name}.{tabular_data_ext}"
    XGboost_df = None
    if tabular_data_ext == "csv":
        try:
            XGboost_df = pd.read_csv(in_file, encoding=encoding)
        except FileNotFoundError:
            base_dir = base_dir.replace("../", "")
            in_file = f"{base_dir}/{tabular_data_name}.{tabular_data_ext}"
            XGboost_df = pd.read_csv(in_file, encoding=encoding)

    try:
        XGboost_df = XGboost_df.loc[:, require_columns]
    except KeyError:
        XGboost_df = XGboost_df[require_columns[:-1]]

    XGboost_df = XGboost_df.sort_index()
    XGboost_df = XGboost_df.fillna(0)
    cols_len = np.array([len(i) for i in XGboost_df.columns])
    XGboost_df = XGboost_df.drop(XGboost_df.columns[np.nonzero(cols_len == 1)[0]], axis=1)
    if scale:
        exclude_cols = ["src_port", "dst_port"]  # 제외할 컬럼
        scaler_df = XGboost_df.drop(exclude_cols, axis=1)
        scaler_df = scaler_df.iloc[:, :-1]
        if scaler_path is None:
            scaler = MinMaxScaler()
            scaler_path = f"../out_model/minmax_scaler.pkl"
            scaler.fit(scaler_df)
            dump(scaler, open(scaler_path, 'wb'))
        else:
            assert os.path.exists(scaler_path), "scaler_path is not exist"
            scaler = load(open(scaler_path, 'rb'))

        try:
            return_df = pd.DataFrame(scaler.transform(scaler_df), columns=scaler_df.columns)
        except ValueError:
            return_df = pd.DataFrame(scaler.transform(XGboost_df), columns=XGboost_df.columns)
        return_df = pd.concat([XGboost_df.loc[:, exclude_cols], return_df], axis=1)

    return_df["label"] = XGboost_df["label"]
    if getdatatype == "tabular":
        return return_df, None
    if MAXLEN == -1:
        MAXLEN = None
    with open(f"{base_dir}/{bert_flie}", "r") as f:
        nlp_data = []
        labels = []
        for line in tqdm(f, desc="loading nlp data", total=return_df.shape[0], unit="line", leave=False):
            try:
                l = line.split(",")
                label = int(l[-1].strip())
            except:
                print()
            l = ",".join(l[:-1])
            nlp_data.append(l[:MAXLEN])
            labels.append(label)

    return return_df, (nlp_data, labels)

def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)
        exit()



def getXy(data, random_seed, train_index_file=None, test_index_file=None, model_type="xgboost"):
    if model_type == "bert":
        x = data[0]
        y = data[1]
        X_train, X_test, Y_train, Y_test = \
            train_test_split(x, y, test_size=0.33, random_state=random_seed, shuffle=True, stratify=None)
        return X_train, X_test, Y_train, Y_test
    else:
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        y = y.astype("int")
        if os.path.exists(train_index_file) and os.path.exists(test_index_file):
            train_indexs = np.load(train_index_file)
            test_indexs = np.load(test_index_file)
            X_train = x.loc[train_indexs]
            X_test = x.loc[test_indexs]
            Y_train = y.loc[train_indexs]
            Y_test = y.loc[test_indexs]

        else:
            X_train, X_test, Y_train, Y_test = \
                train_test_split(x, y, test_size=0.33, random_state=random_seed, shuffle=True, stratify=None)
            np.save(f"{train_index_file}", X_train.index)
            np.save(f"{test_index_file}", X_test.index)
            train_indexs = list(X_train.index)
            test_indexs = list(X_test.index)
        return X_train, X_test, Y_train, Y_test, train_indexs, test_indexs



def get_xgboost_model():
    raw_models = xgb.XGBClassifier(n_estimators=10000,
                                   max_depth=15,
                                   learning_rate=0.5,
                                   min_child_weight=0,
                                   # tree_method='gpu_hist',
                                   # device='cuda:0',
                                   tree_method="hist", device="cuda",
                                   sampling_method='gradient_based',
                                   reg_alpha=0.2,
                                   reg_lambda=1.5,
                                   random_state=42)
    return raw_models

def validation_origin_bert(model, testing_loader, device, pred_value_check=False):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    predbs = []
    soft_max = torch.nn.Softmax(dim=1)
    return_df = pd.DataFrame()
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader), leave=False, total=len(testing_loader), unit="validation"):
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.int64)
            outputs = model(ids, mask, token_type_ids)
            if pred_value_check:
                preba = soft_max(outputs)
                predbs.extend(preba.cpu().detach().numpy().tolist())
                temp_df = pd.DataFrame(data['pid'], columns=['pid'])
                temp_df['real_label'] = targets.cpu().detach().numpy()
                temp_df['b_pred'] = outputs.argmax(dim=1).cpu().detach().numpy()
                temp_df[["b_proba0", "b_proba1", "b_proba2", "b_proba3", "b_proba4"]] = preba.cpu().detach().numpy()
                if return_df.empty:
                    return_df = temp_df
                else:
                    return_df = pd.concat([return_df, temp_df])
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            outputs = outputs.argmax(dim=1)
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets, predbs, return_df

def validation_model(model, test_data, device, model_type="bert", pred_value_check=False):
    if model_type == "bert":
        return validation_origin_bert(model, test_data, device, pred_value_check)
    else:
        xgboost_X = test_data.iloc[:, :-1]
        xgboost_y = test_data.iloc[:, -1]
        pred  = model.predict(xgboost_X)
        proba = model.predict_proba(xgboost_X)
        _result, cm = eval(xgboost_y, pred)
        return _result, cm, xgboost_y

def validation(model, x, y, verbose=False):
    y_pred = model.predict(x)
    accuracy = accuracy_score(y, y_pred)
    if verbose:
        print("Train accuracy: %.2f" % (accuracy * 100.0))
        # Print the confusion matrix
        print(confusion_matrix(y, y_pred))
        # Print the precision and recall, among other metrics
        print(classification_report(y, y_pred, digits=3))
    return accuracy, confusion_matrix(y, y_pred)

import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def eval(y, pred):
    acc = accuracy_score(y, pred)
    presision = precision_score(y, pred, average='weighted')
    recall = recall_score(y, pred, average='weighted')
    f1 = f1_score(y, pred, average='weighted')
    cm = confusion_matrix(y, pred)

    return [acc, presision, recall, f1], cm

def save_metrics_and_cm(result, cm, output_path_prefix):
    # result = [accuracy, precision, recall, f1]
    result_df = pd.DataFrame([result], columns=["Accuracy", "Precision", "Recall", "F1-Score"])
    cm_df = pd.DataFrame(cm)

    with pd.ExcelWriter(f"{output_path_prefix}_evaluation.xlsx") as writer:
        result_df.to_excel(writer, sheet_name="Result", index=False)
        cm_df.to_excel(writer, sheet_name="Confusion Matrix")


def evaluate_xgboost(test_X, test_y, model, output_path_prefix="evaluaction"):
    preds = model.predict(test_X)
    output_path_prefix += "_xgboost"
    _result, cm = eval(test_y, preds)
    save_metrics_and_cm(_result, cm, output_path_prefix)


def evaluate_bert(data, max_len, model_path, output_path_prefix="evaluaction", device="cuda"):
    test_dataset = data[0]
    test_labels = data[1]
    output_numbers = len(set(test_labels))
    bert_model = setting_bert_model(device, model_path, output_numbers=output_numbers)
    bert_model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    testing_set = CustomDataset(test_dataset, test_dataset, test_labels, tokenizer, max_len)
    test_params = {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 0
    }
    testing_loader = DataLoader(testing_set, **test_params)
    output_path_prefix += "_bert"
    fin_outputs, fin_targets, predbs, return_df = validation_bert(bert_model, testing_loader, device)
    _result, cm = eval(fin_targets, fin_outputs)
    save_metrics_and_cm(_result, cm, output_path_prefix)
