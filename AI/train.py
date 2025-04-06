import pandas as pd
import os
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

import torch
from AI_utils import create_directory, data_load, getXy, get_xgboost_model, validation_model, set_device
from AI_utils.TH_BERT import bert_training, bert_MLP_training
import argparse
import pickle

def hyperparameter_tuning():
    pipe = Pipeline([
        ('scaler', StandardScaler()),  # 데이터 스케일링 단계
        ('classifier', LogisticRegression())  # 기본 모델 설정
    ])

    # 모델과 하이퍼파라미터 그리드 설정
    param_grid = [
        {
            'classifier': [LogisticRegression()],
            'classifier__C': [0.001, 0.01, 0.1, 1, 10],
            "classifier__max_iter": [100],
        },
        {
            'classifier': [RandomForestClassifier()],
            'classifier__n_estimators': [5, 10],
            'classifier__max_depth': [2, 4, 10],
            'classifier__min_samples_split': [2, 5, 10],
        },
        {
            'classifier': [SVC()],
            'classifier__C': [0.1, 0.5, 1, 10],
            'classifier__gamma': [0.001, 0.01, 0.1],
            'classifier__kernel': ['rbf', 'poly']
        },
        {
            'classifier': [xgb.XGBClassifier()],
            'classifier__n_estimators': [5, 10, 50],
            'classifier__max_depth': [2, 4, 8],
            'classifier__min_child_weight': [2, 4],
            # 'classifier__subsample': [0.5, 0.7, 1.0],
            # 'classifier__colsample_bytree': [0.5, 0.7, 1.0],
            'classifier__learning_rate': [0.001, 0.01, 0.1],
            'classifier__gamma': [0.001, 0.01]
        }
    ]
    grid_search = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=1)
    return grid_search

def main(args):
    # designate gpu
    gpu = args.gpu
    data_base_dir = args.data_base_dir
    tabular_data_name = args.tabular_data_name
    bert_file = args.bert_file
    train_model_type = args.train_model_type
    model_output_dir = args.model_output_dir
    create_directory(model_output_dir)
    model_name = args.model_name
    hyperprameger_tuning = args.hyperprameger_tuning
    random_seed = int(args.random_seed)
    MAXLEN = int(args.MAXLEN)
    # resume = False
    resume = args.resume
    if 'xgboost' in train_model_type or 'MLP' in train_model_type:
        tabular_data, _ = data_load(base_dir=data_base_dir, tabular_data_name=tabular_data_name, scale='minmax', getdatatype="tabular")

    if 'NLP' in train_model_type or 'bert' in train_model_type or "MLPNLP" in train_model_type\
            or "MLPFeatureNLP" in train_model_type:
        tabular_data, nlp_data_label = data_load(base_dir=data_base_dir, tabular_data_name=tabular_data_name,
                                           bert_flie=bert_file, MAXLEN=MAXLEN)

    if tabular_data.empty or len(tabular_data) == 0:
        raise ValueError("tabular data is empty")
        #caution: tabular data에 대한 학습이 먼저 진행되어야 함

    device = set_device(gpu)
    tabualr_train_index_file = f"{model_output_dir}/tabular_train_index_seed_{random_seed}.npy"
    tabular_test_index_file = f"{model_output_dir}/tabular_test_index_{random_seed}.npy"
    for tmt in train_model_type:
        tabular_X_train, tabular_X_test, tabular_y_train, tabular_y_test, train_indexs, test_indexs = \
            getXy(tabular_data, random_seed, tabualr_train_index_file, tabular_test_index_file)
        model_output = f"{model_output_dir}/{model_name}_{tmt}"
        # labeling_df = data_load(xgboost_file='flowmeter_labeling',scale='minmax', scaler_path='output_models/minmax_scaler.pkl')
        print(f"model({tmt}) training start")
        if tmt == 'xgboost':
            if hyperprameger_tuning:
                grid_search = hyperparameter_tuning()
                grid_search.fit(tabular_X_train, tabular_y_train)
                scores_df = pd.DataFrame(grid_search.cv_results_)
                scores_df.to_csv(f"{model_output_dir}/grid_search_scores_df.csv", index=False)
                model = grid_search.best_estimator_
            else:
                model = get_xgboost_model()
                model.fit(tabular_X_train, tabular_y_train)
                # model.save_model(f'{model_output}.model')
                pickle.dump(model, open(f'{model_output}.pkl', 'wb'))

            tr_data = pd.concat([tabular_X_train, tabular_y_train], axis=1)
            te_data = pd.concat([tabular_X_test, tabular_y_test], axis=1)
            tr_result, tr_cm, _ = validation_model(model, tr_data, device=device, model_type=tmt)
            te_result, te_cm, _ = validation_model(model, te_data, device=device, model_type=tmt)
            tr_result_df = pd.DataFrame([tr_result], columns=["Accuracy", "Precision", "Recall", "F1-Score"])
            te_result_df = pd.DataFrame([te_result], columns=["Accuracy", "Precision", "Recall", "F1-Score"])

            tr_cm_df = pd.DataFrame(tr_cm)
            te_cm_df = pd.DataFrame(te_cm)
            # Excel로 저장
            with pd.ExcelWriter("../model_evaluation_results_xgboost.xlsx") as writer:
                tr_result_df.to_excel(writer, sheet_name="Train Results", index=False)
                te_result_df.to_excel(writer, sheet_name="Test Results", index=False)
                tr_cm_df.to_excel(writer, sheet_name="Train Confusion Matrix")
                te_cm_df.to_excel(writer, sheet_name="Test Confusion Matrix")

        elif tmt == 'NLP' or tmt == 'bert' or tmt == 'MLPNLP' or tmt == 'MLPFeatureNLP':
            nlp_data = nlp_data_label[0]
            labels = nlp_data_label[1]
            output_numbers = len(set(labels))
            bert_check_point = args.bert_check_point
            X_train, X_test, Y_train, Y_test = getXy([nlp_data, labels], random_seed, model_type='bert')
            if tmt == 'NLP' or tmt == 'bert':
                bert_training(device, [X_train, X_test, Y_train, Y_test], model_output,
                              output_numbers=output_numbers, bert_check_point=bert_check_point,
                              max_len=MAXLEN, resume=resume)
            elif tmt == 'MLPNLP' or tmt == "MLPFeatureNLP":
                bert_MLP_training(device, nlp_data, tabular_data, train_indexs, test_indexs, train_labels, test_labels, model_output,
                              output_numbers=output_numbers, bert_check_point=bert_check_point, max_len=MAXLEN, resume=resume, tmt=tmt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str)
    ## training data 관련
    parser.add_argument('--data_base_dir', default='../model_data/', type=str, help='data base directory')
    parser.add_argument('--tabular_data_name', default='tabular', type=str, help='tabular data name')
    parser.add_argument('--bert_file', default='nlp_data.txt', type=str, help='nlp data name')
    parser.add_argument('--hyperprameger_tuning', action="store_true", help='hyperprameger tuning')
    parser.add_argument('--train_model_type', choices=['xgboost', 'NLP', 'bert', 'MLPNLP', 'MLPFeatureNLP'], default=['bert'], nargs='+', help='train model name')
    parser.add_argument('--model_output_dir', default='../out_model', type=str, help='model output directory')
    parser.add_argument('--bert_check_point', default=None, type=str, help="bert model check point")
    parser.add_argument('--model_name', default='quadminers_edu', type=str, help='model name')
    parser.add_argument('--random_seed', default=42, type=int, help='random seed for train test split')
    parser.add_argument('--MAXLEN', default=-1, type=int, help='max length for bert input')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=False, help='resume training')

    args = parser.parse_args()
    main(args)