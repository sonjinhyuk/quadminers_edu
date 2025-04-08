import argparse
def argparse_xai():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--data_base_dir', default='../model_data/', type=str, help='data base directory')
    parser.add_argument('--tabular_data_name', default='tabular', type=str, help='tabular data name')
    parser.add_argument('--bert_file', default='nlp_data.txt', type=str, help='nlp data name')
    parser.add_argument('--hyperprameger_tuning', action="store_true", help='hyperprameger tuning')
    parser.add_argument('--eval_model_type', choices=['xgboost', 'NLP', 'bert', 'MLPNLP', 'MLPFeatureNLP'],
                        default=['bert'], nargs='+', help='train model name')
    parser.add_argument('--model_output_dir', default='../out_model', type=str, help='model output directory')
    parser.add_argument('--model_name', default='quadminers_edu', type=str, help='model name')
    parser.add_argument('--random_seed', default=42, type=int, help='random seed for train test split')
    parser.add_argument('--MAXLEN', default=-1, type=int, help='max length for bert input')
    parser.add_argument('--captum_method', default="ig", type=str, help='max length for bert input')

    args = parser.parse_args()
    return args

