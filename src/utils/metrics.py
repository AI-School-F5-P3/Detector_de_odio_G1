import logging

def setup_logging(log_filename):
    """Configura el logging para el proyecto"""
    logging.basicConfig(
        filename=log_filename,
        filemode='a',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def calculate_metrics(y_true, y_pred):
    """Calcula métricas de evaluación para el modelo"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    return metrics

def log_metrics(metrics, logger):
    """Registra las métricas en el logger"""
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
