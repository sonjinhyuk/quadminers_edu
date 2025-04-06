import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForQuestionAnswering, BertModel, BertConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, auc, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

# Defining some key variables that will be used later on in the training
MAX_LEN = 200
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

class CustomDataset(Dataset):
    def __init__(self, indexlist, dataframe, real_label, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.indexlist = indexlist
        self.comment_text = dataframe
        self.targets = real_label
        if max_len == -1:
            self.max_len = None
        else:
            self.max_len = max_len
    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])[:self.max_len]

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=MAX_LEN,
            # padding='longest',
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            # 'pid': self.indexlist[index],
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

    def get_decode_item(self, index):
        comment_text = str(self.comment_text[index])
        return comment_text


import transformers

# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.
class BERTClass(torch.nn.Module):
    def __init__(self, output_numbers=13):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-multilingual-cased',
                                                         problem_type='multi_label_classification', num_labels=output_numbers, output_attentions=True)  # 5
        self.config = self.l1.config
        # self.l1 = AutoModelForSe uenceClassification.from_pretrained('bert-base-multilingual-uncased', problem_type='multi_label_classification', num_labels=5)
        self.l2 = torch.nn.Dropout(0.3)
        # self.l3 = torch.nn.Linear(768, 9)
        self.l3 = torch.nn.Linear(768, output_numbers)

    # def forward(self, x):
        # _, output_1 = self.l1(**x, return_dict=False)
    def forward(self, input_ids, attention_mask, token_type_ids):
        # _, output_1 = self.l1(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        _, output_1, _ = self.l1(input_ids[0], attention_mask=attention_mask[0], token_type_ids=token_type_ids[0], return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


class BERTMLPDataset(Dataset):
    def __init__(self, indexlist, dataframe, real_label, tokenizer, tabulardata, max_len):
        """
           Dataset for BERT and MLP combined model.

           Args:
               indexlist: List of unique identifiers for data points.
               dataframe: List or Series of text data for NLP processing.
               real_label: List or Series of target labels.
               tokenizer: Tokenizer for BERT model.
               tabulardata: Tabular data as a NumPy array or DataFrame (shape: [num_samples, num_features]).
               max_len: Maximum sequence length for BERT tokenizer (-1 for no limit).
       """
        self.tokenizer = tokenizer
        self.indexlist = indexlist
        self.comment_text = dataframe
        self.targets = real_label
        self.tabulardata = tabulardata  # Tabular data should align with text data
        self.max_len = max_len if max_len != -1 else None
    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])[:self.max_len]
        tabular_features = self.tabulardata.iloc[index]

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=MAX_LEN,
            # padding='longest',
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        # Return data dictionary
        return {
            # 'pid': self.indexlist[index],
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'tabular_data': torch.tensor(tabular_features, dtype=torch.float),  # Tabular data as a tensor
            'targets': torch.tensor(self.targets[index], dtype=torch.float)  # Adjust dtype based on task
        }

    def get_decode_item(self, index):
        comment_text = str(self.comment_text[index])
        return comment_text

# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.
class BERTMLPClass(torch.nn.Module):
    def __init__(self, output_numbers=13, tabular_input_dim=77):
        super(BERTMLPClass, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-multilingual-cased',
                                                         problem_type='multi_label_classification',
                                                         num_labels=output_numbers,
                                                         output_attentions=True)  # 5
        self.config = self.bert.config

        # Tabular 데이터를 위한 MLP
        self.tabular_mlp = torch.nn.Sequential(
            torch.nn.Linear(tabular_input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
        )
        # BERT와 Tabular의 출력을 결합
        combined_dim = self.bert.config.hidden_size + 128  # BERT hidden_size + MLP output size
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(combined_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, output_numbers)
        )

    # def forward(self, x):
        # _, output_1 = self.l1(**x, return_dict=False)
    def forward(self, input_ids, attention_mask, token_type_ids, tabular_data):
        _, bert_output, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )

        # Tabular MLP Forward
        tabular_output = self.tabular_mlp(tabular_data)

        # Concatenate BERT and Tabular Outputs
        combined_output = torch.cat((bert_output, tabular_output), dim=1)

        # Final Fully Connected Layer
        output = self.fc(combined_output)

        return output



class MLPFeatureExtractor(torch.nn.Module):
    def __init__(self, input_dim=77, output_dim=768):
        super(MLPFeatureExtractor, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, output_dim)  # BERT 임베딩 차원 (768)에 맞춤
        )

    def forward(self, x):
        return self.layers(x)

class MLPFeatureBert(torch.nn.Module):
    def __init__(self, output_numbers=13):
        super(MLPFeatureBert, self).__init__()
        # MLP for feature extraction from tabular data
        self.bert = transformers.BertModel.from_pretrained('bert-base-multilingual-cased',
                                                         problem_type='multi_label_classification',
                                                         num_labels=output_numbers,
                                                         output_attentions=True)  # 5
        self.config = self.bert.config
        # Final classification layer
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, output_numbers)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, tabular_data):
        batch_size = input_ids.size(0)


        # Step 2: Add MLP features as a new token embedding
        # Create a fake "token" embedding for the MLP output
        mlp_features = mlp_features.unsqueeze(1)  # Shape: (batch_size, 1, 768)

        # Step 3: Extend BERT embeddings with the MLP token
        bert_embeddings = self.bert.embeddings.word_embeddings(input_ids)  # Shape: (batch_size, seq_len, 768)
        extended_embeddings = torch.cat([mlp_features, bert_embeddings], dim=1)  # Add MLP token at the start

        # Update attention mask and token type IDs to account for new token
        extended_attention_mask = torch.cat([torch.ones((batch_size, 1), dtype=torch.long).to(input_ids.device),
                                             attention_mask], dim=1)
        extended_token_type_ids = torch.cat([torch.zeros((batch_size, 1), dtype=torch.long).to(input_ids.device),
                                             token_type_ids], dim=1)

        # Step 4: Forward pass through BERT with extended inputs
        outputs = self.bert(inputs_embeds=extended_embeddings,
                            attention_mask=extended_attention_mask,
                            token_type_ids=extended_token_type_ids,
                            return_dict=True)

        # Use [CLS] token (now the MLP token is at position 0)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, 768)

        # Step 5: Final classification
        output = self.fc(cls_embedding)
        return output


class Custominference(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.keyword
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        try:
            comment_text = str(self.comment_text[index])
        except KeyError:
            comment_text = str(self.comment_text.values[0])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }


def setting_bert_model(device, bert_model_path=None, output_numbers=13):
    if bert_model_path is None:
        model = BERTClass(output_numbers=output_numbers)
        model.to(device)
    else:
        model = BERTClass(output_numbers=output_numbers).to(device)
        model.load_state_dict(torch.load(f'{bert_model_path}', map_location=device))
    return model

def setting_bert_MLP_model(device, bert_model_path=None, output_numbers=13, tmt="MLPNLP", tokenizer=None):
    if tmt == "MLPNLP":
        if bert_model_path is None:
            model = BERTMLPClass(output_numbers=output_numbers)
            model.to(device)
        else:
            model = BERTMLPClass(output_numbers=output_numbers).to(device)
            model.load_state_dict(torch.load(f'{bert_model_path}', map_location=device))
    elif tmt == "MLPFeatureNLP":
        if bert_model_path is None:
            model = MLPFeatureBert(output_numbers=output_numbers, tokenizer=tokenizer)
            model.to(device)
        else:
            model = MLPFeatureBert(output_numbers=output_numbers, tokenizer=tokenizer).to(device)
            model.load_state_dict(torch.load(f'{bert_model_path}', map_location=device))
    return model

def nlp_data_setting(nlp_data, train_indexs, test_indexs, max_len=None):
    train_dataset, test_dataset = [], []
    for index in train_indexs:
        train_dataset.append(nlp_data[index])
    for index in test_indexs:
        test_dataset.append(nlp_data[index])
    return train_dataset, test_dataset

def tabular_data_setting(tabular_data, train_indexs, test_indexs):
    return tabular_data.iloc[train_indexs], tabular_data.iloc[test_indexs]


def train(model, x, epoch=0, optimizer=None, device=None, loss_fn=None):
    model.train()
    curr_loss = 0
    correct = 0
    counts = 0
    preds = []
    labels = []
    for _, data in tqdm(enumerate(x), total=len(x), desc=f'Epoch {epoch}', leave=False):
        ids = data['input_ids'].to(device, dtype=torch.int64)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.int64)
        outputs = model(ids, mask, token_type_ids)
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        outputs = outputs.argmax(dim=1)
        correct += (outputs == targets).sum().item()
        counts += len(targets)
        curr_loss += loss.item()
        preds.extend(outputs.cpu().numpy())
        labels.extend(targets.cpu().numpy())

    loss_result = curr_loss / len(x)
    acc, pre, recall, f1, cm = \
        accuracy_score(labels, preds), precision_score(labels, preds, average='macro'), \
            recall_score(labels, preds, average='macro'), f1_score(labels, preds, average='macro'), \
            confusion_matrix(labels, preds)

    return loss_result, acc, pre, recall, f1, cm


def validation_bert(model, testing_loader, device):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    predbs = []
    return_df = pd.DataFrame()
    with torch.no_grad():
        for _, data in enumerate(testing_loader):
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.int64)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            outputs = outputs.argmax(dim=1)
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets, predbs, return_df


def train_bert_mlp(model, x, epoch=0, optimizer=None, device=None, loss_fn=None):
    model.train()
    curr_loss = 0
    correct = 0
    counts = 0
    preds = []
    labels = []
    for _, data in tqdm(enumerate(x), total=len(x), desc=f'Epoch TR:{epoch}', leave=False):
        ids = data['input_ids'].to(device, dtype=torch.int64)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.int64)
        tabular_data = data['tabular_data'].to(device, dtype=torch.float)
        outputs = model(ids, mask, token_type_ids, tabular_data)
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        outputs = outputs.argmax(dim=1)
        correct += (outputs == targets).sum().item()
        counts += len(targets)
        curr_loss += loss.item()
        preds.extend(outputs.cpu().numpy())
        labels.extend(targets.cpu().numpy())

    loss_result = curr_loss / len(x)
    acc, pre, recall, f1, cm = \
        accuracy_score(labels, preds), precision_score(labels, preds, average='macro'), \
            recall_score(labels, preds, average='macro'), f1_score(labels, preds, average='macro'), \
            confusion_matrix(labels, preds)

    return loss_result, acc, pre, recall, f1, cm


def validation_bert_mlp(model, testing_loader, device, epoch=0):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    predbs = []
    return_df = pd.DataFrame()
    with torch.no_grad():
        # for _, data in enumerate(testing_loader):
        for _, data in tqdm(enumerate(testing_loader), total=len(testing_loader), desc=f'Epoch Val:{epoch}', leave=False):
            ids = data['input_ids'].to(device, dtype=torch.int64)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.int64)
            tabular_data = data['tabular_data'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids, tabular_data)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            outputs = outputs.argmax(dim=1)
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets, predbs, return_df

def bert_validation(device, model_path, nlp_data, train_indexs, test_indexs, train_labels, test_labels, max_len=MAX_LEN):
    bert_model = setting_bert_model(device, model_path)
    _, test_dataset = nlp_data_setting(nlp_data, train_indexs, test_indexs)
    testing_set = CustomDataset(test_dataset, test_dataset, test_labels, tokenizer, max_len)
    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 0
                   }
    testing_loader = DataLoader(testing_set, **test_params)
    outputs, targets, _, _ = validation_bert(bert_model, testing_loader, device)


import wandb
def wandb_init(lr, num_epochs, batch_size, resume=False):
    # wandb.init(project='harmful_bert', reinit=True, resume=True)
    wandb.init(project='harmful_bert', reinit=True, resume=resume)
    wandb_args = {
        "learning_rate": lr,
        "epochs": num_epochs,
        "batch_size": batch_size,
    }
    wandb.config.update(wandb_args, allow_val_change=True)


def bert_training(device, datas, model_output,
                  max_len=MAX_LEN, output_numbers=13,
                  bert_check_point=None, resume=False):
    train_dataset = datas[0]
    test_dataset = datas[1]
    train_labels = datas[2]
    test_labels = datas[3]
    print("-"*15,"bert model training start","-"*15)
    bert_model = setting_bert_model(device, bert_check_point, output_numbers=output_numbers)
    bert_model_output = model_output
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    optimizer = torch.optim.Adam(params=bert_model.parameters(), lr=LEARNING_RATE)
    # __init__(self, indexlist, dataframe, real_label, tokenizer, max_len)
    training_set = CustomDataset(train_dataset, train_dataset, train_labels, tokenizer, max_len)
    testing_set = CustomDataset(test_dataset, test_dataset, test_labels, tokenizer, max_len)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 0
                   }
    wandb_init(LEARNING_RATE, EPOCHS, TRAIN_BATCH_SIZE, resume)
    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    best_loss = 1000000000
    best_f1 = 0
    best_acc = 0

    ### Optimizer setting
    # class_weights = compute_class_weight(class_weight='balanced', classes=[i for i in range(output_numbers)], y=train_labels)
    # weights = torch.tensor(class_weights, dtype=torch.float)
    # loss_fn = nn.CrossEntropyLoss(weights=weights.to(device))

    loss_fn = nn.CrossEntropyLoss()
    for epoch in tqdm(range(EPOCHS)):
        loss_result, acc, pre, recall, f1, cm = train(bert_model, training_loader, epoch=epoch, optimizer=optimizer, device=device, loss_fn=loss_fn)
        outputs, targets, _, _ = validation_bert(bert_model, testing_loader, device=device)
        test_acc = accuracy_score(targets, outputs)
        test_f1 = f1_score(targets, outputs, average='macro')
        if loss_result < best_loss:
            print(f"Best loss:{epoch}, loss:{loss_result}, test_acc:{test_acc}, test_f1:{test_f1}")
            best_loss = loss_result
            torch.save(bert_model.state_dict(), f'{bert_model_output}_best_loss_model.pth')

        if f1 > best_f1:
            print(f"Best F1:{epoch}, loss:{f1}, test_acc:{test_acc}, test_f1:{test_f1}")
            best_f1 = f1
            torch.save(bert_model.state_dict(), f'{bert_model_output}_best_f1_model.pth')

        if acc > best_acc:
            print(f"Best acc:{epoch}, loss:{acc}, test_acc:{test_acc}, test_f1:{test_f1}")
            best_acc = acc
            torch.save(bert_model.state_dict(), f'{bert_model_output}_best_acc_model.pth')
        wandb.log({
            "tr_accuracy": acc, "tr_loss": loss_result, "tr_f1": f1,
            "test_accuracy": test_acc, "test_f1": test_f1
        })
    # Excel로 저장
    # with pd.ExcelWriter("../model_evaluation_results_xgboost.xlsx") as writer:
    #     tr_result_df.to_excel(writer, sheet_name="Train Results", index=False)
    #     te_result_df.to_excel(writer, sheet_name="Test Results", index=False)
    #     tr_cm_df.to_excel(writer, sheet_name="Train Confusion Matrix")
    #     te_cm_df.to_excel(writer, sheet_name="Test Confusion Matrix")
    # print(f"train accuracy: {tr_result[0]}\n train f1_score: {tr_result[-1]}\n train confusion matrix: {tr_cm}")
    # print(f"test accuracy: {te_result[0]}\n test f1_score: {te_result[-1]}\n test confusion matrix: {te_cm}")
    torch.save(bert_model.state_dict(), f'{bert_model_output}.pth')

def bert_MLP_training(device, nlp_data, tabular_data, train_indexs, test_indexs, train_labels, test_labels, model_output,
                      output_numbers=13, max_len=MAX_LEN, bert_check_point=None, resume=False, tmt="MLPNLP"):
    train_bert_dataset, test_bert_dataset = nlp_data_setting(nlp_data, train_indexs, test_indexs)
    train_mlp_dataset, test_mlp_dataset = tabular_data_setting(tabular_data, train_indexs, test_indexs)
    train_mlp_dataset_X, test_mlp_dataset_X = train_mlp_dataset.drop('label', axis=1), test_mlp_dataset.drop('label', axis=1)
    print("-"*15,"bert model training start","-"*15)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    bert_mlp_model = setting_bert_MLP_model(device, bert_check_point, tmt=tmt, output_numbers=output_numbers, tokenizer=tokenizer)
    bert_mlp_model_output = model_output
    optimizer = torch.optim.Adam(params=bert_mlp_model.parameters(), lr=LEARNING_RATE)
    training_set = BERTMLPDataset(train_indexs, train_bert_dataset, train_labels, tokenizer, train_mlp_dataset_X, max_len)
    testing_set = BERTMLPDataset(test_indexs, test_bert_dataset, test_labels, tokenizer, test_mlp_dataset_X, max_len)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 0
                   }
    wandb_init(LEARNING_RATE, EPOCHS, TRAIN_BATCH_SIZE, resume)
    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    best_loss = 1000000000
    best_f1 = 0
    best_acc = 0
    for epoch in tqdm(range(EPOCHS)):
        loss_result, acc, pre, recall, f1, cm = train_bert_mlp(bert_mlp_model, training_loader, epoch=epoch, optimizer=optimizer, device=device, loss_fn=nn.CrossEntropyLoss())
        outputs, targets, _, _ = validation_bert_mlp(bert_mlp_model, testing_loader, device=device, epoch=epoch)
        test_acc = accuracy_score(targets, outputs)
        test_f1 = f1_score(targets, outputs, average='macro')
        if loss_result < best_loss:
            print(f"Best loss:{epoch}, loss:{loss_result}, test_acc:{test_acc}, test_f1:{test_f1}")
            best_loss = loss_result
            torch.save(bert_mlp_model.state_dict(), f'{bert_mlp_model_output}_best_loss_model.pth')

        if f1 > best_f1:
            print(f"Best F1:{epoch}, loss:{f1}, test_acc:{test_acc}, test_f1:{test_f1}")
            best_f1 = f1
            torch.save(bert_mlp_model.state_dict(), f'{bert_mlp_model_output}_best_f1_model.pth')

        if acc > best_acc:
            print(f"Best acc:{epoch}, loss:{acc}, test_acc:{test_acc}, test_f1:{test_f1}")
            best_acc = acc
            torch.save(bert_mlp_model.state_dict(), f'{bert_mlp_model_output}_best_acc_model.pth')
        wandb.log({
            "tr_accuracy": acc, "tr_loss": loss_result, "tr_f1": f1,
            "test_accuracy": test_acc, "test_f1": test_f1
        })
    torch.save(bert_mlp_model.state_dict(), f'{bert_mlp_model_output}.pth')