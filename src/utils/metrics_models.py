import torch
import pandas as pd
from transformers import BertTokenizer, RobertaTokenizer, BertForSequenceClassification, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Cargar los modelos entrenados
model_bert = joblib.load('/content/bert_model.pkl')
model_roberta = joblib.load('/content/roberta_model.pkl')

# Inicializar los tokenizers
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Cargar los datasets de entrenamiento y validación
train_df = pd.read_csv('/content/train_data1.csv')
val_df = pd.read_csv('/content/val_data1.csv')

# Función para tokenizar los datos
def tokenize_data(df, tokenizer, max_length=256):
    return tokenizer(
        df['processed_text'].tolist(),
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

train_encodings_bert = tokenize_data(train_df, bert_tokenizer)
val_encodings_bert = tokenize_data(val_df, bert_tokenizer)
train_encodings_roberta = tokenize_data(train_df, roberta_tokenizer)
val_encodings_roberta = tokenize_data(val_df, roberta_tokenizer)

# Crear un Dataset Personalizado para PyTorch
class HateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_labels = train_df['IsHatespeech'].astype(int).tolist()
val_labels = val_df['IsHatespeech'].astype(int).tolist()

train_dataset_bert = HateSpeechDataset(train_encodings_bert, train_labels)
val_dataset_bert = HateSpeechDataset(val_encodings_bert, val_labels)
train_dataset_roberta = HateSpeechDataset(train_encodings_roberta, train_labels)
val_dataset_roberta = HateSpeechDataset(val_encodings_roberta, val_labels)

# Mover el modelo a la GPU si está disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_bert.to(device)
model_roberta.to(device)

# Función para evaluar el modelo
def evaluate_model(model, dataset):
    model.eval()
    predictions, true_labels = [], []
    for batch in dataset:
        inputs = {key: val.unsqueeze(0).to(device) for key, val in batch.items() if key != 'labels'}
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).tolist())
        true_labels.extend([labels.item()] if labels.dim() == 0 else labels.cpu().tolist())
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    return accuracy, f1

# Evaluar el rendimiento en el conjunto de entrenamiento
train_accuracy_bert, train_f1_bert = evaluate_model(model_bert, train_dataset_bert)
train_accuracy_roberta, train_f1_roberta = evaluate_model(model_roberta, train_dataset_roberta)

# Evaluar el rendimiento en el conjunto de validación
val_accuracy_bert, val_f1_bert = evaluate_model(model_bert, val_dataset_bert)
val_accuracy_roberta, val_f1_roberta = evaluate_model(model_roberta, val_dataset_roberta)

print(f'BERT - Entrenamiento: Accuracy={train_accuracy_bert}, F1={train_f1_bert}')
print(f'BERT - Validación: Accuracy={val_accuracy_bert}, F1={val_f1_bert}')
print(f'RoBERTa - Entrenamiento: Accuracy={train_accuracy_roberta}, F1={train_f1_roberta}')
print(f'RoBERTa - Validación: Accuracy={val_accuracy_roberta}, F1={val_f1_roberta}')

"""BERT - Entrenamiento: Accuracy=0.9388934764657308, F1=0.9454277286135693
BERT - Validación: Accuracy=0.9233576642335767, F1=0.9509727626459143
RoBERTa - Entrenamiento: Accuracy=0.9397192402972749, F1=0.9462840323767476
RoBERTa - Validación: Accuracy=0.927007299270073, F1=0.9532710280373832
BERT:

Overfitting Ratio de Accuracy: 1.65%

Overfitting Ratio de F1-score: -0.59%

RoBERTa:

Overfitting Ratio de Accuracy: 1.35%

Overfitting Ratio de F1-score: -0.74%
"""