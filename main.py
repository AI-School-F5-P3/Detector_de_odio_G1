import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd

# Añadir el directorio src al PYTHONPATH
project_dir = Path(__file__).resolve().parent
sys.path.append(str(project_dir))

from src.preprocessing.text_processor import TextPreprocessor
from src.utils.metrics import setup_logging, calculate_metrics, log_metrics

def setup_directories():
    """Crear estructura de directorios necesaria"""
    directories = [
        'data/raw',
        'data/processed',
        'logs',
        'models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    # Configurar logging
    log_filename = f"logs/processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_filename)
    logger = logging.getLogger(__name__)
    
    try:
        # Crear directorios necesarios
        setup_directories()
        
        # Inicializar el preprocesador de texto
        logger.info("Iniciando procesamiento de datos...")
        preprocessor = TextPreprocessor()
        
        # Procesar y guardar cada conjunto de datos
        datasets = ['train_data.csv', 'val_data.csv', 'test_data.csv']
        for dataset in datasets:
            data_path = f'data/processed/{dataset}'
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"No se encontró el archivo de datos en {data_path}")
            
            # Cargar y procesar datos
            df = pd.read_csv(data_path)
            processed_df = preprocessor.process_dataframe(df)
            
            # Guardar datos procesados
            output_path = f'data/processed/processed_{dataset}'
            processed_df.to_csv(output_path, index=False)
            logger.info(f"Datos procesados guardados en {output_path}")
        
        logger.info("Preprocesamiento completado.")
        
    except Exception as e:
        logger.error(f"Error durante el procesamiento: {str(e)}")
        raise

if __name__ == "__main__":
    main()
