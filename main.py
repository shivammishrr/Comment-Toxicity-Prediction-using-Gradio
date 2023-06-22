# Importing Data
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import gradio as gr

# Data preparation

df = pd.read_csv(r"train.csv.zip")

# Creating Word Embeddings
from tensorflow.keras.layers import TextVectorization
X = df['comment_text']
y = df[df.columns[2:]].values
MAX_FEATURES = 200000
vectorizer = TextVectorization(max_tokens = MAX_FEATURES, output_sequence_length = 1800, output_mode = 'int')
vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)
print('Vectorization Complete!')

# Loading The Model
model = tf.keras.models.load_model('hate_model.h5')

# To display results
def score_comment(comment):
    vectorize_comment = vectorizer([comment])
    results = model.predict(vectorize_comment)
    
    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx]>0.5)
        
    return text

interface = gr.Interface(fn=score_comment,
                        inputs=gr.inputs.Textbox(lines=2, placeholder='Comment to score'),
                        outputs='text')
interface.launch()