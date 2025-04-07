import os
import math
import shap
import torch
from XAI.xai_utils.bert_util import MAX_LEN
import numpy as np
from shap.maskers._text import TokenGroup, Token
import pandas as pd
import json
import seaborn as sns

def get_tokenized_data(tokenizer, input_text):
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

def post_process_shap_value(shap_values, new_word_index, token, tokenizer, output_names, device):
    def partition_tree(decoded_tokens, tokenizer, special_tokens=None):

        def merge_score(group1, group2, special_tokens):
            score = 0

            # ensures special tokens are combined last, so 1st subtree is 1st sentence and 2nd subtree is 2nd sentence
            if len(special_tokens) > 0:
                if group1[-1].s in special_tokens and group2[0].s in special_tokens:
                    score -= math.inf  # subtracting infinity to create lowest score and ensure combining these groups last

            # merge broken-up parts of words first
            if group2[0].s.startswith("##"):
                score += 20

            # merge apostrophe endings next
            if group2[0].s == "'" and (len(group2) == 1 or (len(group2) == 2 and group2[1].s in ["t", "s"])):
                score += 15
            if group1[-1].s == "'" and group2[0].s in ["t", "s"]:
                score += 15

            start_ctrl = group1[0].s.startswith("[") and group1[0].s.endswith("]")
            end_ctrl = group2[-1].s.startswith("[") and group2[-1].s.endswith("]")

            if (start_ctrl and not end_ctrl) or (end_ctrl and not start_ctrl):
                score -= 1000
            if group2[0].s in openers and not group2[0].balanced:
                score -= 100
            if group1[-1].s in closers and not group1[-1].balanced:
                score -= 100

            # attach surrounding an openers and closers a bit later
            if group1[0].s in openers and group2[-1] not in closers:
                score -= 2

            # reach across connectors later
            if group1[-1].s in connectors or group2[0].s in connectors:
                score -= 2

            # reach across commas later
            if group1[-1].s == ",":
                score -= 10
            if group2[0].s == ",":
                if len(group2) > 1:  # reach across
                    score -= 10
                else:
                    score -= 1

            # reach across sentence endings later
            if group1[-1].s in [".", "?", "!"]:
                score -= 20
            if group2[0].s in [".", "?", "!"]:
                if len(group2) > 1:  # reach across
                    score -= 20
                else:
                    score -= 1

            score -= len(group1) + len(group2)
            # print(group1, group2, score)
            return score

        openers = {
            "(": ")"
        }
        closers = {
            ")": "("
        }
        enders = [".", ","]
        connectors = ["but", "and", "or"]
        """Build a heriarchial clustering of tokens that align with sentence structure.
        Note that this is fast and heuristic right now.
        TODO: Build this using a real constituency parser.
        """
        if special_tokens is None:
            special_tokens = [tokenizer.sep_token]
        token_groups = [TokenGroup([Token(t)], i) for i, t in enumerate(decoded_tokens)]
        M = len(decoded_tokens)
        new_index = M
        clustm = np.zeros((M - 1, 4))
        for i in range(len(token_groups) - 1):
            scores = [merge_score(token_groups[i], token_groups[i + 1], special_tokens) for i in
                      range(len(token_groups) - 1)]
            ind = np.argmax(scores)

            lind = token_groups[ind].index
            rind = token_groups[ind + 1].index
            clustm[new_index - M, 0] = token_groups[ind].index
            clustm[new_index - M, 1] = token_groups[ind + 1].index
            clustm[new_index - M, 2] = -scores[ind]
            clustm[new_index - M, 3] = (clustm[lind - M, 3] if lind >= M else 1) + (
                clustm[rind - M, 3] if rind >= M else 1)

            token_groups[ind] = token_groups[ind] + token_groups[ind + 1]
            token_groups[ind].index = new_index

            # track balancing of openers/closers
            if token_groups[ind][0].s in openers and token_groups[ind + 1][-1].s == openers[token_groups[ind][0].s]:
                token_groups[ind][0].balanced = True
                token_groups[ind + 1][-1].balanced = True

            token_groups.pop(ind + 1)
            new_index += 1

        # negative means we should never split a group, so we add 10 to ensure these are very tight groups
        # (such as parts of the same word)
        clustm[:, 2] = clustm[:, 2] + 10

        clustm[:, 2] = clustm[:, 3]
        clustm[:, 2] /= clustm[:, 2].max()
        return clustm

    values = shap_values[0].values
    new_values = post_values(values, device, new_word_index)
    new_clustering = partition_tree(token, tokenizer, special_tokens=None)
    new_clustering = new_clustering.reshape(-1, new_clustering.shape[0], new_clustering.shape[1])
    shap_values.data = (token,)
    shap_values.feature_names = token
    shap_values.values = (new_values,)
    shap_values.clustering = new_clustering
    shap_values.output_names = output_names
    shap_values.hierarchical_values = None
    return shap_values

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
        data = shap.plots.text(shap_values[0, :, :], display=False)
        with open(f"{output_name}.html", "w") as file:
            file.write(data)
        df_columns = label_types.copy()
        df_columns[label] = str(df_columns[label]) + "_o"
        temp_df = pd.DataFrame(shap_values.values[0][1:-1], columns=label_types, index=shap_values.data[0][1:-1]).T
        temp_df.to_csv(f"{output_name}.csv", encoding='cp949')
        in_dict['key_words'] = list(shap_values.data[0][1:-1])
        in_dict['base_values'] = list(shap_values.base_values[0])
        for c in label_types:
            in_dict[c] = temp_df.loc[c].to_list()
        with open(f"{output_name}.json", 'w', encoding='cp949') as f:
            json.dump(in_dict, f, indent=4, ensure_ascii=False)

def post_tokenizer(tokens):
    new_word_index = []
    new_toekn = []
    for i, t in enumerate(tokens):
        if i == 0:
            new_word_index.append([i])
        elif t.startswith("##"):
            new_word_index[-1].append(i)
        elif t == "[SEP]":
            new_word_index.append([i])
            break
        else:
            new_word_index.append([i])
    for nwi in new_word_index:
        new_toekn.append("".join([tokens[i].replace("#", "") for i in nwi]))
    return new_word_index, new_toekn

def post_values(values, device, new_word_index):
    new_values = []
    if type(values) == tuple:
        for value in values:

            temp_layer = []
            temp_head = []
            # value = value[0][-1].cpu().detach().numpy()
            heads = value[0]
            for head in heads:
                vs = head
                ## pad 다 버림
                temp_value = []
                for i in range(len(new_word_index)):
                    v = vs[i].cpu().detach().numpy()
                    row_temp = []
                    for nwi in new_word_index:
                        try:
                            row_temp.append(v[nwi].sum())
                        except IndexError:
                            break
                    if len(row_temp) != 0:
                        temp_value.append(row_temp)
                    else:
                        break
                temp_head.append(temp_value)
            temp_layer.append(temp_head)
            temp_layer = torch.tensor(temp_layer).to(device)
            new_values.append(temp_layer)
    elif len(values.shape) == 3:
        value = values[0]
        for nwi in new_word_index:
            new_values.append(value[nwi].sum(axis=0).tolist())
        new_values = torch.tensor(new_values).to(device)
    elif len(values.shape) == 2:
        for nwi in new_word_index:
            new_values.append(values[nwi].sum(axis=0).tolist())
        new_values = np.asarray(new_values)
    return new_values


# def captum_xai(lig,
#                ids, mask, token_type_ids,
#                word_index: list, token: list,
#                label, score,
#                output_name,
#                device, label_types: list):
#
#
#     # load model
#     if os.path.exists(f"{output_name}_visualize_text.html"):
#         return None
#     pred_label = score.argmax().item()
#     position_vis = []
#     for target_label_index, l in enumerate(label_types):
#         attributions, delta = lig.attribute(inputs=ids, baselines=ids * 0,
#                                             target=target_label_index,
#                                             return_convergence_delta=True,
#                                             additional_forward_args=(mask,))
#         # attributions_sum = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
#         # token = all_tokens
#         n_attributions = post_values(attributions, device, word_index)
#         attributions_sum = n_attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
#         attr_score = attributions_sum.sum()
#         position_vi = viz.VisualizationDataRecord(
#             word_attributions=attributions_sum,
#             pred_prob=score[target_label_index].item(),
#             pred_class=pred_label,  # 예시 정답 라벨
#             true_class=label,
#             attr_class=target_label_index,
#             attr_score=attr_score,
#             raw_input_ids=token,
#             convergence_score=delta
#         )
#         position_vis.append(position_vi)
#     html = viz.visualize_text(position_vis, html_action='return')
#     geneate_html(html, f"{output_name}_visualize_text.html")


def geneate_html(html, output_name):
    if ".html" not in output_name:
        output_name += ".html"
    with open(f"{output_name}", "w") as file:
        file.write(html.data)

## 기존 버젼
# def get_shap_with_pipeline(explainer__, tokenizer, input_text, label, output_name="data.html"):
#     shap_values_original = explainer__([input_text])
#     # data = shap.plots.text(shap_values_original[0, :, :], display=False)
#     shap_values = shap_values_original.__copy__()
#     shap_values = post_process_shap_value(shap_values, tokenizer, input_text)
#     # temp_df = pd.DataFrame(shap_values_original.values[0])
#     # temp_df2 = pd.DataFrame(shap_values.values[0])
#     # temp_df.index = shap_values_original.feature_names
#     # temp_df2.index = shap_values.feature_names
#     # temp_df.columns = shap_values.output_names
#     # temp_df2.columns = shap_values.output_names
#     # temp_df.to_csv("shap_origin.csv",encoding='cp949')
#     # temp_df2.to_csv("shap_post.csv",encoding='cp949')
#     # data = shap.plots.text(shap_values_original[0, :, :], display=False)
#     data = shap.plots.text(shap_values[0, :, :], display=False)
#     if ".html" not in output_name:
#         output_name += ".html"
#     with open(f"{output_name}", "w") as file:
#         file.write(data)
#     # shap.plots.bar(shap_values[:, :, "도박"].mean(), show=False)
#     return shap_values
#     #
#     # pred_choi2 = transformers.pipeline(
#     #     "text-classification",
#     #     model=model,
#     #     tokenizer=tokenizer.encode_plus,
#     #     device=0,
#     #     return_all_scores=True,
#     # )
#     # explainer_choi2 = shap.Explainer(pred_choi2)
#     # shap_values_choi2 = explainer_choi2(input_text)
# 
# def get_shap_with_pipeline_parmap(param):
#     explainer__ = param['explainer']
#     tokenizer = param['tokenizer']
#     input_text = param['input_text']
#     output_name = param['output_name']
#     label = param['label']
#     output_names = param['output_names']
#     temp_dict = param['temp_dict']
#     shap_values_original = explainer__([input_text], silent=True)
# 
#     shap_values = shap_values_original.__copy__()
#     shap_values = post_process_shap_value(shap_values, tokenizer, input_text, output_names=output_names)
#     data = shap.plots.text(shap_values[0, :, :], display=False)
#     with open(f"{output_name}.html", "w") as file:
#         file.write(data)
#     df_columns = output_names.copy()
#     df_columns[label] = df_columns[label] + "_o"
#     temp_df = pd.DataFrame(shap_values.values[0][1:-1], columns=output_names, index=shap_values.data[0][1:-1]).T
#     temp_df.to_csv(f"{output_name}.csv", encoding='cp949')
#     temp_dict['key_words'] = list(shap_values.data[0][1:-1])
#     temp_dict['base_values'] = list(shap_values.base_values[0])
#     for c in output_names:
#         temp_dict[c] = temp_df.loc[c].to_list()

#
# def make_masks(cluster_matrix):
#     def _init_masks(cluster_matrix, M, indices_row_pos, indptr):
#         pos = 0
#         for i in range(2 * M - 1):
#             if i < M:
#                 pos += 1
#             else:
#                 pos += int(cluster_matrix[i - M, 3])
#             indptr[i + 1] = pos
#             indices_row_pos[i] = indptr[i]
#     def _rec_fill_masks(cluster_matrix, indices_row_pos, indptr, indices, M, ind):
#         pos = indices_row_pos[ind]
#
#         if ind < M:
#             indices[pos] = ind
#             return
#
#         lind = int(cluster_matrix[ind - M, 0])
#         rind = int(cluster_matrix[ind - M, 1])
#         lind_size = int(cluster_matrix[lind - M, 3]) if lind >= M else 1
#         rind_size = int(cluster_matrix[rind - M, 3]) if rind >= M else 1
#
#         lpos = indices_row_pos[lind]
#         rpos = indices_row_pos[rind]
#
#         _rec_fill_masks(cluster_matrix, indices_row_pos, indptr, indices, M, lind)
#         indices[pos:pos + lind_size] = indices[lpos:lpos + lind_size]
#
#         _rec_fill_masks(cluster_matrix, indices_row_pos, indptr, indices, M, rind)
#         indices[pos + lind_size:pos + lind_size + rind_size] = indices[rpos:rpos + rind_size]
#
#     M = cluster_matrix.shape[0] + 1
#     indices_row_pos = np.zeros(2 * M - 1, dtype=int)
#     indptr = np.zeros(2 * M, dtype=int)
#     indices = np.zeros(int(np.sum(cluster_matrix[:,3])) + M, dtype=int)
#
#     # build an array of index lists in CSR format
#     _init_masks(cluster_matrix, M, indices_row_pos, indptr)
#     _rec_fill_masks(cluster_matrix, indices_row_pos, indptr, indices, M, cluster_matrix.shape[0] - 1 + M)
#     mask_matrix = scipy.sparse.csr_matrix(
#         (np.ones(len(indices), dtype=bool), indices, indptr),
#         shape=(2 * M - 1, M)
#     )
#
#     return mask_matrix



###잘되는거
# token__ = transformers.AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion", use_fat=True)
# model__ = transformers.AutoModelForSequenceClassification.from_pretrained(
#     "nateraw/bert-base-uncased-emotion"
# ).cuda()
# load the emotion dataset
# dataset = pd.read_csv("data/emotion.csv")
# data = pd.DataFrame({"text": dataset["text"], "emotion": dataset["label"]})
# pred__ = transformers.pipeline(
#     "text-classification",
#     model=model__,
#     tokenizer=token__,
#     device=0,
#     return_all_scores=True,
# )
# explainer = shap.Explainer(pred__)
# shap_values = explainer(data["text"][:3])
# data = shap.plots.text(shap_values[0, :, :], display=False)
# with open("testdata.html", "w") as file:
#     file.write(data)
###잘되는거

##choi 모델
