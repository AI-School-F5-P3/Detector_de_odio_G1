import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, RobertaTokenizer
import torch
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, Trainer, TrainingArguments

# Función para cargar y procesar un archivo CSV
def load_and_process_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# Cargar los datasets
train_df = load_and_process_csv('/train_data1.csv')
val_df = load_and_process_csv('/val_data1.csv')
test_df = load_and_process_csv('/test_data1.csv')

# Inicializar los tokenizers de BERT y RoBERTa
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Tokenizar los textos
def tokenize_data(df, tokenizer, max_length=256):
    return tokenizer(
        df['processed_text'].tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

train_encodings_bert = tokenize_data(train_df, bert_tokenizer)
val_encodings_bert = tokenize_data(val_df, bert_tokenizer)
train_encodings_roberta = tokenize_data(train_df, roberta_tokenizer)
val_encodings_roberta = tokenize_data(val_df, roberta_tokenizer)

# Crear un Dataset Personalizado para PyTorch
class HateSpeechDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
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

# Inicializar los modelos BERT y RoBERTa para clasificación de secuencias
model_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model_roberta = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# Mover los modelos a GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_bert.to(device)
model_roberta.to(device)

# Configurar los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10
)

# Inicializar los entrenadores
trainer_bert = Trainer(
    model=model_bert,
    args=training_args,
    train_dataset=train_dataset_bert,
    eval_dataset=val_dataset_bert
)

trainer_roberta = Trainer(
    model=model_roberta,
    args=training_args,
    train_dataset=train_dataset_roberta,
    eval_dataset=val_dataset_roberta
)

# Entrenar los modelos
trainer_bert.train()
trainer_roberta.train()

# Evaluar los modelos en el conjunto de validación
results_bert = trainer_bert.evaluate()
results_roberta = trainer_roberta.evaluate()

# Obtener las predicciones de los modelos
predictions_bert = trainer_bert.predict(val_dataset_bert).predictions
predictions_roberta = trainer_roberta.predict(val_dataset_roberta).predictions

# Promediar las predicciones para obtener el ensamble
ensemble_predictions = (predictions_bert + predictions_roberta) / 2

# Convertir las predicciones a etiquetas
ensemble_labels = torch.argmax(torch.tensor(ensemble_predictions), dim=1).numpy()

# Evaluar el ensamble
from sklearn.metrics import accuracy_score, f1_score

accuracy = accuracy_score(val_labels, ensemble_labels)
f1 = f1_score(val_labels, ensemble_labels)

print(f'Accuracy del ensamble: {accuracy}')
print(f'F1 Score del ensamble: {f1}')

# Guardar los modelos entrenados
import joblib

joblib.dump(model_bert, 'bert_model.pkl')
joblib.dump(model_roberta, 'roberta_model.pkl')
