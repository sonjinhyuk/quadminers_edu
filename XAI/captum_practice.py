from XAI.xai_utils import argparse_xai
import os
from AI_utils import data_load, set_device, setting_bert_model
from transformers import BertTokenizer
from captum.attr import IntegratedGradients, DeepLift

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

    def forward_func(input_ids, attention_mask, token_type_ids):
        return bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

    def explain_with_ig_or_deeplift(text, tokenizer, model, max_len=128, method='ig', target_class=0, device='cpu'):
        model.eval()
        model.to(device)

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

        # Select attribution method
        if method == 'ig':
            explainer = IntegratedGradients(forward_func)
        elif method == 'deeplift':
            explainer = DeepLift(forward_func)
        else:
            raise ValueError("Invalid method. Choose 'ig' or 'deeplift'.")

        input_ids = input_ids.long()
        attention_mask = attention_mask.long()
        token_type_ids = token_type_ids.long()
        attributions, delta = explainer.attribute(
            inputs=input_ids,
            additional_forward_args=(attention_mask, token_type_ids),
            target=target_class,
            return_convergence_delta=True
        )

        # Decode tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        attributions = attributions.sum(dim=-1).squeeze(0)  # Sum over embedding dim

        token_attr = list(zip(tokens, attributions.detach().cpu().numpy()))
        return token_attr, delta

    explain_with_ig_or_deeplift(nlp_data[0], tokenizer, bert_model, max_len=MAXLEN, method='ig', target_class=0, device=device)
if __name__ == "__main__":
    args = argparse_xai()
    main(args)