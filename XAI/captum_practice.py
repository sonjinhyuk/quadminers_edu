import os
import sys
import torch
from transformers import BertTokenizer
from captum.attr import IntegratedGradients, DeepLift
from XAI.xai_utils import argparse_xai
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
# 경로 설정
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(parent_dir)
from AI.AI_utils import data_load, set_device, setting_bert_model_captum

# BERT 모델 래핑: Embedding 입력 사용 가능하게
class BertWrapper(torch.nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.embeddings = self.bert.l1.embeddings

    def forward(self, input_embeds, attention_mask=None, token_type_ids=None):
        outputs = self.bert.l1(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        pooled = outputs.pooler_output
        return self.bert.l3(self.bert.l2(pooled))


def explain_with_ig_or_deeplift(text, tokenizer, wrapped_model, bert_model, max_len=128, method='ig', target_class=0, device='cpu'):
    wrapped_model.eval()
    wrapped_model.to(device)

    # 텍스트 토크나이징 및 임베딩 추출
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        return_token_type_ids=True,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    token_type_ids = encoded['token_type_ids'].to(device)
    embeddings = bert_model.l1.embeddings(input_ids)
    embeddings.requires_grad_()

    # Attribution Explainer 선택
    explainer = IntegratedGradients(wrapped_model) if method == 'ig' else DeepLift(wrapped_model)

    attributions, delta = explainer.attribute(
        inputs=embeddings,
        additional_forward_args=(attention_mask, token_type_ids),
        target=target_class,
        return_convergence_delta=True
    )

    # 토큰 + 중요도 매핑
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    token_attr = list(zip(tokens, scores))
    return token_attr, delta


def plot_token_attributions(token_attr, top_k=15, title="Token Attribution Visualization", save_path=None):
    token_attr = sorted(token_attr, key=lambda x: abs(x[1]), reverse=True)[:top_k]
    tokens, scores = zip(*token_attr)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=scores, y=tokens)
    plt.title(title)
    plt.xlabel("Attribution Score")
    plt.ylabel("Token")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
def main(args):
    device = set_device(args.gpu)
    tabular_data, (nlp_data, labels) = data_load(
        base_dir=args.data_base_dir,
        tabular_data_name=args.tabular_data_name,
        bert_flie=args.bert_file,
        MAXLEN=args.MAXLEN
    )
    model_path = os.path.join(args.model_output_dir, f"{args.model_name}_nlp.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(args.model_output_dir.replace("../", ""), f"{args.model_name}_nlp.pth")

    captum_method = args.captum_method
    output_numbers = len(set(labels))
    bert_model = setting_bert_model_captum(device, model_path, output_numbers)
    wrapped_model = BertWrapper(bert_model)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # 테스트 실행
    target_text = nlp_data[0]
    target_class = 0
    token_attr, delta = explain_with_ig_or_deeplift(
        target_text, tokenizer, wrapped_model, bert_model,
        max_len=args.MAXLEN, method=captum_method, target_class=target_class, device=device
    )
    plot_token_attributions(token_attr, top_k=15, save_path=f"{captum_method}_attribution_plot.png")



if __name__ == "__main__":
    args = argparse_xai()
    main(args)
