from transformers import BertTokenizer
import transformers
import shap
import torch
from captum.attr import LayerConductance, LayerIntegratedGradients
try:
    from utils import create_directory
except ImportError:
    def create_directory(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print("Error: Creating directory. " + directory)
            exit()
from tqdm import tqdm
import os, json
from XAI.xai_utils.xai_utils import post_process_shap_value, get_tokenized_data, bert_shap, post_tokenizer
import pandas as pd

def get_tokenized_data(tokenizer, input_text, MAX_LEN=200):
    input_text = " ".join(input_text.split())
    tokenized_sent = tokenizer.encode_plus(
        input_text,
        None,
        add_special_tokens=True,
        max_length=MAX_LEN,
        pad_to_max_length=True,
        return_token_type_ids=True,
        truncation=True,
    )
    ids = tokenized_sent['input_ids']
    mask = tokenized_sent['attention_mask']
    token_type_ids = tokenized_sent["token_type_ids"]

    return {
        'input_ids': torch.tensor(ids, dtype=torch.long),
        'attention_mask': torch.tensor(mask, dtype=torch.long),
        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
    }

def get_model_input(inputs, device):
    input_shape = inputs['input_ids'].shape[0]
    ids = inputs['input_ids'].to(device, dtype=torch.long).view(-1, input_shape)
    mask = inputs['attention_mask'].to(device, dtype=torch.long).view(-1, input_shape)
    token_type_ids = inputs['token_type_ids'].to(device, dtype=torch.long).view(-1, input_shape)
    return ids, mask, token_type_ids



def post_tokenizer(tokens):
    new_word_index = []
    new_toekn = []
    for i, t in enumerate(tokens):
        if i == 0:
            # new_word_index.append([i])
            continue
        elif t.startswith("##"):
            new_word_index[-1].append(i)
        elif t == "[SEP]":
            # new_word_index.append([i])
            # break
            continue
        else:
            new_word_index.append([i])
    for nwi in new_word_index:
        new_toekn.append("".join([tokens[i].replace("#", "") for i in nwi]))
    return new_word_index, new_toekn


def bert_shap(explainer: shap.Explainer,
              word_index: list, token: list,
              tokenizer: object, input_text: object,
              label: object, label_types: list,
              in_dict: dict, output_name: str, device:object):
    if os.path.exists(f"{output_name}.csv") and os.path.exists(f"{output_name}.html") \
            and os.path.exists(f"{output_name}.json"):
        with open(f"{output_name}.json", 'r', encoding='cp949') as f:
            in_dict = json.load(f)
        return in_dict
    else:
        shap_values_original = explainer([input_text], silent=True)
        shap_values = shap_values_original.__copy__()
        shap_values = post_process_shap_value(shap_values, word_index, token, tokenizer,
                                              output_names=label_types, device=device)
        shap_values = shap_values_original
        # tokenized_sequence = tokenizer.tokenize(input_text)
        # print(tokenized_sequence)


        data = shap.plots.text(shap_values[0, :, :], display=False)
        with open(f"{output_name}.html", "w") as file:
            file.write(data)
        df_columns = label_types.copy()
        df_columns[label] = df_columns[label] + "_o"
        temp_df = pd.DataFrame(shap_values.values[0][1:-1], columns=label_types, index=shap_values.data[0][1:-1]).T
        try:
            temp_df.to_csv(f"{output_name}.csv", encoding='cp949')
        except UnicodeEncodeError:
            pass
        in_dict['key_words'] = list(shap_values.data[0][1:-1])
        in_dict['base_values'] = list(shap_values.base_values[0])
        for c in label_types:
            in_dict[c] = temp_df.loc[c].to_list()
        with open(f"{output_name}.json", 'w', encoding='cp949') as f:
            json.dump(in_dict, f, indent=4, ensure_ascii=False)



def post_process(cls_data=None, tokenizer=None, _type="multi5", model=None,
                 xai_types=['shap'], device=None, out_dir="xai_output"):
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    if cls_data is None:
        print("cls_data is None")
        return None

    if xai_types is None:
        xai_types = ['shap', 'bertviz', 'capton_lig']
    explainer = None
    attention_model = None
    lig = None
    pred_pipline = None
    for xai in xai_types:
        if "shap" == xai:
            pred_pipline = transformers.pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=device,
                # return_all_scores=True,
                top_k=None,
                truncation=True
            )
            explainer = shap.Explainer(pred_pipline)

        # elif 'bertviz' == xai:
        #     attention_model = model.l1
        # elif 'capton_lig' == xai:
        #     def forward_with_softmax(input_ids, attention_mask=None, token_type_ids=None):
        #         # logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        #         output = model(input_ids, attention_mask, token_type_ids)
        #         probs = torch.nn.functional.softmax(output, dim=1)
        #         return probs
        #     embeddings = model.l1.embeddings
        #     lig = LayerIntegratedGradients(forward_with_softmax, embeddings)

    create_directory(f"{out_dir}/{_type}")
    if _type == "multi5":
        label_types = ['정상', '도박', '웹툰', '토렌트', '성인']
    elif _type == "multi9":
        label_types = ['정상', '도박', '웹툰', 'TV', '음란물', '불법링크', '도박홍보', '의약품', 'others']
    elif _type == "multi8":
        label_types = [f"fake_label_{i}" for i in range(9)]
    else:
        label_types = ['정상', '도박']
    for i in range(len(label_types)):
        create_directory(f"{out_dir}/{_type}/{i}")
    # cls_data = cls_data.iloc[-1:, :]
    for i in tqdm(cls_data.index, total=cls_data.shape[0], mininterval=0.001):
        ser = cls_data.loc[i]
        input_text = ser['keyword']
        try:
            label = int(ser['real_label'])
        except TypeError:
            print(ser['real_label'])
            continue
        pid = ser['pid']
        temp_out_dir = f"{out_dir}/{_type}/{label}/{pid}"
        create_directory(temp_out_dir)
        output_name = f"{temp_out_dir}/{pid}"
        inputs = get_tokenized_data(tokenizer, input_text)
        ids, mask, token_type_ids = get_model_input(inputs, device)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'])  # Convert input ids to token strings
        model_pred = pred_pipline(input_text)

        in_dict = {}
        in_dict['label'] = label
        in_dict['pred_label'] = model_pred
        in_dict['index'] = label_types
        in_dict['pred_values'] = [0 for _ in range(len(label_types))]

        for pred in model_pred[0]:
            label_ = pred['label']
            label_index = int(label_.split("LABEL_")[-1])
            in_dict['pred_values'][label_index] = pred['score']

        new_ward_token_index, new_toekn = post_tokenizer(tokens)
        if "shap" in xai_types:
            bert_shap(explainer, new_ward_token_index, new_toekn, tokenizer,
                      input_text, label, label_types, in_dict, output_name, device)
