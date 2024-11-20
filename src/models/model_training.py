import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import mlflow
import mlflow.pytorch
import subprocess
import time
import os
import signal
import logging
import sys
from pathlib import Path

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_project_root():
    """Obtiene la ruta raíz del proyecto"""
    current_file = Path(__file__).resolve()
    # Asumiendo que estamos en src/models/model_training.py
    # y queremos llegar a la raíz del proyecto
    return current_file.parent.parent.parent

def get_data_path():
    """Obtiene la ruta correcta para los archivos de datos"""
    project_root = get_project_root()
    return project_root / "data" / "processed"

def start_mlflow_server():
    """Inicia el servidor MLflow como un proceso separado"""
    try:
        # Crear directorio para la base de datos si no existe
        mlruns_dir = get_project_root() / "mlruns"
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        
        # Comando para iniciar el servidor MLflow
        cmd = [
            sys.executable, "-m", "mlflow", "server",
            "--backend-store-uri", f"sqlite:///{mlruns_dir.parent}/mlflow.db",
            "--default-artifact-root", str(mlruns_dir),
            "--host", "0.0.0.0",
            "--port", "5000"
        ]
        
        # Iniciar el servidor como proceso independiente
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        
        # Esperar a que el servidor esté listo
        time.sleep(5)
        
        return process
    except Exception as e:
        logger.error(f"Error al iniciar el servidor MLflow: {e}")
        return None

def stop_mlflow_server(process):
    """Detiene el servidor MLflow"""
    if process is None:
        return
    
    try:
        if os.name == 'nt':  # Windows
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:  # Linux/Mac
            process.terminate()
        process.wait(timeout=5)
    except Exception as e:
        logger.error(f"Error al detener el servidor MLflow: {e}")
        if os.name == 'nt':
            os.system('taskkill /F /T /PID {}'.format(process.pid))
        else:
            os.kill(process.pid, signal.SIGKILL)

class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(model, train_dataset, val_dataset, training_args):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    return trainer

def main():
    # Iniciar el servidor MLflow
    logger.info("Iniciando servidor MLflow...")
    mlflow_server = start_mlflow_server()
    
    if mlflow_server is None:
        logger.error("No se pudo iniciar el servidor MLflow. Abortando...")
        return
    
    try:
        # Configurar MLflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("hate_speech_detection_ensemble")
        
        # Obtener la ruta correcta de los datos
        data_path = get_data_path()
        logger.info(f"Cargando datos desde: {data_path}")
        
        # Cargar los datos balanceados
        train_df = pd.read_csv(data_path / "balanced_train_data.csv")
        val_df = pd.read_csv(data_path / "balanced_val_data.csv")
        test_df = pd.read_csv(data_path / "balanced_test_data.csv")
        
        # Inicializar el tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Crear datasets
        train_dataset = HateSpeechDataset(
            texts=train_df['processed_text'].to_numpy(),
            labels=train_df['IsHatespeech'].astype(int).to_numpy(),
            tokenizer=tokenizer
        )

        val_dataset = HateSpeechDataset(
            texts=val_df['processed_text'].to_numpy(),
            labels=val_df['IsHatespeech'].astype(int).to_numpy(),
            tokenizer=tokenizer
        )

        test_dataset = HateSpeechDataset(
            texts=test_df['processed_text'].to_numpy(),
            labels=test_df['IsHatespeech'].astype(int).to_numpy(),
            tokenizer=tokenizer
        )

        # Definir los modelos para el ensemble
        model_names = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"]
        models = [AutoModelForSequenceClassification.from_pretrained(name, num_labels=2) for name in model_names]

        # Crear directorio para resultados si no existe
        results_dir = get_project_root() / "models" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = get_project_root() / "models" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Definir argumentos de entrenamiento
        training_args = TrainingArguments(
            output_dir=str(results_dir),
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=str(logs_dir),
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
        )

        # Entrenar cada modelo
        trained_models = []
        for idx, model in enumerate(models):
            logger.info(f"Entrenando modelo {model_names[idx]}")
            with mlflow.start_run(run_name=f"model_{model_names[idx]}"):
                trainer = train_model(model, train_dataset, val_dataset, training_args)
                metrics = trainer.evaluate()
                mlflow.log_metrics(metrics)
                mlflow.pytorch.log_model(model, f"model_{model_names[idx]}")
                trained_models.append(model)
            logger.info(f"Modelo {model_names[idx]} entrenado exitosamente")

        logger.info("Entrenamiento de todos los modelos completado")
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}")
        raise
    finally:
        # Detener el servidor MLflow
        logger.info("Deteniendo servidor MLflow...")
        stop_mlflow_server(mlflow_server)

if __name__ == "__main__":
    main()