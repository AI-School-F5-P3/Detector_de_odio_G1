""" 
SpaCy tiene algunas ventajas sobre el enfoque tradicional de TF-IDF + Regresión Logística:

Incorpora embeddings pre-entrenados que capturan mejor el significado semántico
Maneja mejor el contexto y las relaciones entre palabras
Tiene funcionalidades integradas para limpieza y preprocesamiento de texto
Puede ser más preciso en tareas de NLP 
"""

import pandas as pd
import spacy
import numpy as np
from sklearn.model_selection import train_test_split
from spacy.tokens import DocBin
import random
