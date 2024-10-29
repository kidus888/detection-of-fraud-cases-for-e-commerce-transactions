
import numpy as np
import pandas as pd

def preprocess_input(data):
    # Convert JSON data into the required format (modify as needed for your model)
    df = pd.DataFrame([data])
    # Apply any transformations (scaling, encoding) here
    # Example: df['feature'] = some_transformation(df['feature'])
    return df.values
