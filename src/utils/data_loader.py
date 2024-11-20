import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from typing import Tuple, Dict
import logging

class DataLoader:
    def __init__(self, raw_data_path: str = 'data/raw/youtoxic_english_1000.csv'):
        self.raw_data_path = raw_data_path
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
    def load_raw_data(self) -> pd.DataFrame:
        """Carga los datos raw iniciales"""
        try:
            df = pd.read_csv(self.raw_data_path)
            self.logger.info(f"Datos cargados exitosamente: {df.shape[0]} registros")
            return df
        except Exception as e:
            self.logger.error(f"Error cargando datos: {str(e)}")
            raise
            
    def split_data(self, df: pd.DataFrame, 
                   test_size: float = 0.2, 
                   val_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divide los datos en train, test y validation sets
        """
        # Primer split para separar test
        train_val, test = train_test_split(df, test_size=test_size, random_state=42, 
                                         stratify=df['IsHatespeech'])
        
        # Segundo split para separar validation
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(train_val, test_size=val_ratio, random_state=42,
                                    stratify=train_val['IsHatespeech'])
        
        return train, val, test
    
    def save_splits(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
        """Guarda los splits en el directorio processed"""
        processed_dir = 'data/processed'
        os.makedirs(processed_dir, exist_ok=True)
        
        train.to_csv(f'{processed_dir}/train_data.csv', index=False)
        val.to_csv(f'{processed_dir}/val_data.csv', index=False)
        test.to_csv(f'{processed_dir}/test_data.csv', index=False)
        
        self.logger.info(f"Datos guardados en {processed_dir}")
        
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Obtiene estadísticas básicas del dataset"""
        summary = {
            'total_samples': len(df),
            'hate_speech_ratio': df['IsHatespeech'].mean(),
            'unique_users': df['CommentId'].nunique(),
            'avg_text_length': df['Text'].str.len().mean(),
            'label_distribution': {
                col: df[col].mean() 
                for col in df.columns 
                if df[col].dtype == bool
            }
        }
        return summary

    def prepare_datasets(self):
        """Proceso completo de preparación de datasets"""
        # Cargar datos
        df = self.load_raw_data()
        
        # Obtener resumen
        summary = self.get_data_summary(df)
        self.logger.info(f"Resumen del dataset:\n{summary}")
        
        # Dividir datos
        train, val, test = self.split_data(df)
        
        # Guardar splits
        self.save_splits(train, val, test)
        
        return train, val, test

if __name__ == "__main__":
    data_loader = DataLoader()
    data_loader.prepare_datasets()
