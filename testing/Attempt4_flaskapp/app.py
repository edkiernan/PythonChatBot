import pandas as pd

data = pd.read_csv('Service_Training_Dataset.csv')

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Remove unwanted characters and words
data['clean_text'] = data['text'].apply(lambda x: re.sub(r'[^\w\s]','',x))
data['clean_text'] = data['clean_text'].apply(lambda x: x.lower())
data['clean_text'] = data['clean_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords.words('english'))]))

# Tokenize the text
data['tokenized_text'] = data['clean_text'].apply(lambda x: word_tokenize(x))

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(data.category.unique()))

# Define the data and label arrays
X = data.tokenized_text.values
y = data.category.values

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10
)

# Define the trainer and train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=X,
    eval_dataset=y
)

trainer.train()

from flask import Flask, request, jsonify, render_template
import torch

app = Flask(__name__)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(data.category.unique()))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

@app.route('/')
def home():
    return render_template('index.html')

def predict(text):
    # Preprocess the text
    clean_text = re.sub(r'[^\w\s]','',text)
    clean_text = clean_text.lower()
    clean_text = ' '.join([word for word in clean_text.split() if word not in (stopwords.words('english'))])
    tokenized_text = tokenizer.encode(clean_text, add_special_tokens=True)

    # Convert the tokenized text to tensors
    input_ids = torch.tensor(tokenized_text).unsqueeze(0)

    # Pass the input through the model and get the predicted category
    outputs = model(input_ids)[0]
    _, predicted = torch.max(outputs, 1)

    # Convert the predicted category to text and return it
    category = data.category.unique()[predicted]
    return category

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    category = predict(text)
    return jsonify({'category': category})