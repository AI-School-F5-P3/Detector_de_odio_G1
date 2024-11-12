import os
import shutil
import pandas as pd
from pathlib import Path

def setup_project_data():
    """Configura la estructura inicial de datos del proyecto"""
    # Crear estructura de directorios
    directories = [
        'data/raw',
        'data/processed',
        'data/interim',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Ruta del archivo original
    original_path = r"C:\Users\samir\Downloads\youtoxic_english_1000.csv"
    
    # Ruta destino en el proyecto
    destination_path = 'data/raw/youtoxic_english_1000.csv'
    
    # Copiar archivo a la estructura del proyecto
    shutil.copy2(original_path, destination_path)
    print(f"Archivo copiado a: {destination_path}")
    
    return destination_path

def load_and_verify_data(file_path):
    """Carga y verifica el dataset"""
    df = pd.read_csv(file_path)
    
    print("\nInformación del Dataset:")
    print("-" * 50)
    print(f"Número total de registros: {len(df)}")
    print("\nColumnas disponibles:")
    print("-" * 50)
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")
    
    print("\nPrimeras 5 filas de la columna Text:")
    print("-" * 50)
    print(df['Text'].head())
    
    print("\nDistribución de etiquetas:")
    print("-" * 50)
    for col in df.columns:
        if df[col].dtype == bool:
            positive_ratio = df[col].mean() * 100
            print(f"{col}: {positive_ratio:.2f}% positivos")
    
    return df

def main():
    # Configurar estructura de datos
    data_path = setup_project_data()
    
    # Cargar y verificar datos
    df = load_and_verify_data(data_path)
    
    # Guardar una copia en interim para procesamiento
    df.to_csv('data/interim/initial_data.csv', index=False)
    print("\nDatos guardados en data/interim/initial_data.csv")

if __name__ == "__main__":
    main()