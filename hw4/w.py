
import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration

from tqdm import tqdm
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import os

torch.manual_seed(3407)


def load_data(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


def preprocess_data(data):
    inputs, outputs = [], []
    for item in data:
        inputs.append(item['nl'])
        outputs.append(item['code'])
    return inputs, outputs


class ConcodeDataset(Dataset):
    def __init__(self, inputs, outputs, tokenizer, max_length=512):
        self.inputs = inputs
        self.outputs = outputs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_txt = self.inputs[idx]
        output_txt = self.outputs[idx]
        inputs = self.tokenizer(input_txt, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        outputs = self.tokenizer(output_txt, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        labels = outputs.input_ids.squeeze()
        labels[labels==self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": labels
        }


def evaluate(model, data_loader, tokenizer, device):
    model.eval()
    predictions = []
    references = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].clone().detach().to(device)
            labels[labels == -100] = tokenizer.pad_token_id
            generated_ids = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=512)
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
            predictions.extend(preds)
            references.extend(refs)
    return predictions, references


def calculate_exact_match(predictions, references):
    total = len(predictions)
    exact_matches = 0
    for pred, ref in zip(predictions, references):
        if pred == ref:
            exact_matches += 1
    return exact_matches / total if total > 0 else 0


def calculate_bleu(predictions, references):
    smoothie = SmoothingFunction().method4
    total_bleu = 0
    count = 0
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = [ref.split()]
        if len(pred_tokens) > 0:
            total_bleu += sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
            count += 1
    return total_bleu / count if count > 0 else 0


train_data = load_data('./concode/train.json')
test_data = load_data('./concode/test.json')
dev_data = load_data('./concode/dev.json')

train_inputs, train_outputs = preprocess_data(train_data)
test_inputs, test_outputs = preprocess_data(test_data)
dev_inputs, dev_outputs = preprocess_data(dev_data)

model_name = "Salesforce/codet5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

train_dataset = ConcodeDataset(train_inputs, train_outputs, tokenizer)
test_dataset = ConcodeDataset(test_inputs, test_outputs, tokenizer)
dev_dataset = ConcodeDataset(dev_inputs, dev_outputs, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10)
dev_loader = DataLoader(dev_dataset, batch_size=10)

device = torch.device("cuda:7" if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

loss_curve = []

num_epoch = 3
for epoch in range(num_epoch):
    model.train()
    cnt = 0
    running_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        inputs = {k: v.to(device) for k, v in batch.items() if k!= "labels"}
        labels = batch["labels"].to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
        cnt += 1
        if cnt % 250 == 0:
            loss_curve.append(running_loss / 250)
            running_loss = 0
            cnt = 0
    dev_pred, dev_ref = evaluate(model, dev_loader, tokenizer, device)
    dev_em = calculate_exact_match(dev_pred, dev_ref)
    dev_bleu = calculate_bleu(dev_pred, dev_ref)
    test_pred, test_ref = evaluate(model, test_loader, tokenizer, device)
    test_em = calculate_exact_match(test_pred, test_ref)
    test_bleu = calculate_bleu(test_pred, test_ref)
    torch.save(model.state_dict(), f"./model/epoch_{epoch}_dev_em_{dev_em:.4%}_dev_bleu_{dev_bleu:.4%}_test_em_{test_em:.4%}_test_bleu_{test_bleu:.4%}.pth")
    print(f"Epoch: {epoch}, dev_em: {dev_em:.4%}, dev_bleu: {dev_bleu:.4%}, test_em: {test_em:.4%}, test_bleu: {test_bleu:.4%}.")

plt.figure()
plt.plot(range(1, len(loss_curve) + 1), loss_curve)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.savefig('./loss_curve.png')
plt.show()

