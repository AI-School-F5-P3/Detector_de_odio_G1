import pandas as pd
data = {
    'Text': [
        """             I wonder how big of a crater Thug Brown made in the street when he met the final shot to the melon. You get what you sew...

        BLM - When Truth and Facts Don't then they Shit in their own Kitchens before they get the right directions to the toilet."""
    ]
}
df = pd.DataFrame(data)

# Limpiar la columna 'Text'
df['Text'] = df['Text'].str.replace(r'\s*\n\s*', ' ', regex=True)  # Reemplaza saltos de línea por un espacio
df['Text'] = df['Text'].str.strip()  # Elimina espacios al inicio y al final

# Mostrar el dataframe resultante
print(df['Text'][0])