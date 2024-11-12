from src.utils.data_loader import DataLoader
import logging
from datetime import datetime

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/data_prep_{datetime.now():%Y%m%d_%H%M%S}.log'),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Iniciando preparación de datos")
        
        # Inicializar data loader
        data_loader = DataLoader()
        
        # Preparar datasets
        train, val, test = data_loader.prepare_datasets()
        
        logger.info(f"Preparación completada exitosamente")
        logger.info(f"Train size: {len(train)}")
        logger.info(f"Validation size: {len(val)}")
        logger.info(f"Test size: {len(test)}")
        
    except Exception as e:
        logger.error(f"Error en la preparación de datos: {str(e)}")
        raise

if __name__ == "__main__":
    main()