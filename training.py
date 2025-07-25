import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))

# Load the dataset
df = pd.read_csv('input/Suicide_Detection.csv')
print("Dataset shape: {}".format(df.shape))
print(df.head())

# Check class distribution
print("\nClass Distribution:")
print(df['class'].value_counts())
print(df['class'].value_counts(normalize=True) * 100)

# Download NLTK resources (if using for the first time)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    else:
        return ""

# Apply preprocessing to the text column
df['processed_text'] = df['text'].apply(preprocess_text)

# Check for and handle missing values
print("\nMissing values in dataset:")
print(df.isnull().sum())

# Remove rows with missing text
df = df.dropna(subset=['processed_text'])
df = df[df['processed_text'] != ""]

# Map class labels to numeric values
df['label'] = df['class'].map({'suicide': 1, 'non-suicide': 0})

# Feature Engineering

# 1. Text length features
df['text_length'] = df['processed_text'].apply(len)
df['word_count'] = df['processed_text'].apply(lambda x: len(x.split()))

# 2. Sentiment analysis using VADER (if NLTK is available)
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['processed_text'].apply(lambda x: sid.polarity_scores(x)['compound'])
    df['sentiment_neg'] = df['processed_text'].apply(lambda x: sid.polarity_scores(x)['neg'])
    df['sentiment_pos'] = df['processed_text'].apply(lambda x: sid.polarity_scores(x)['pos'])
    print("Sentiment analysis features added.")
except:
    print("VADER not available, skipping sentiment analysis.")

# 3. Identify potential suicide-related keywords
suicide_indicators = [
    'kill', 'die', 'suicide', 'end', 'pain', 'life', 'anymore', 'want', 'hope', 
    'help', 'death', 'dead', 'hate', 'tired', 'pills', 'hurt', 'alone', 'sad', 
    'depression', 'anxiety', 'lost', 'cut', 'empty', 'worthless'
]

for word in suicide_indicators:
    df['has_{}'.format(word)] = df['processed_text'].apply(lambda x: 1 if word in x.split() else 0)

# 4. Count first-person pronouns
first_person_pronouns = ['i', 'me', 'my', 'mine', 'myself']
df['first_person_count'] = df['processed_text'].apply(
    lambda x: sum(1 for word in x.split() if word in first_person_pronouns)
)

# Display the dataset with engineered features
print("\nSample of dataset with engineered features:")
print(df.head())

# Visualize some of the engineered features
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(x='class', y='text_length', data=df)
plt.title('Text Length by Class')

plt.subplot(1, 2, 2)
sns.boxplot(x='class', y='word_count', data=df)
plt.title('Word Count by Class')

plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.close()

# Prepare data for BERT
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], 
    df['label'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label']
)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 128  # or another suitable value based on your data

# Create a custom dataset class
class SuicideDetectionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Create data loaders
train_dataset = SuicideDetectionDataset(X_train, y_train, tokenizer, max_length)
test_dataset = SuicideDetectionDataset(X_test, y_test, tokenizer, max_length)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)

model = model.to(device)

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# Calculate total training steps
epochs = 3
total_steps = len(train_loader) * epochs

# Create scheduler with warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Training function
def train_model(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc="Training", leave=True)
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update parameters
        optimizer.step()
        
        # Update learning rate
        scheduler.step()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(train_loader)

# Evaluation function
def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    
    return predictions, actual_labels

# Train the model
training_losses = []

for epoch in range(epochs):
    print("\nEpoch {}/{}".format(epoch+1, epochs))

    
    # Train
    avg_train_loss = train_model(model, train_loader, optimizer, scheduler, device)
    training_losses.append(avg_train_loss)
    
    # Print the average training loss
    print("Average training loss: {:.4f}".format(avg_train_loss))
    
    # Evaluate
    if epoch == epochs - 1:  # Only evaluate on the last epoch to save time
        print("\nEvaluating model...")
        predictions, actual_labels = evaluate_model(model, test_loader, device)
        
        # Calculate metrics
        accuracy = accuracy_score(actual_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual_labels, predictions, average='weighted'
        )
        
        print("Accuracy: {:.4f}".format(accuracy))
        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))
        print("F1 Score: {:.4f}".format(f1))

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(actual_labels, predictions, target_names=['Non-suicidal', 'Suicidal']))
        
        # Create confusion matrix
        cm = confusion_matrix(actual_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-suicidal', 'Suicidal'],
                   yticklabels=['Non-suicidal', 'Suicidal'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1), training_losses, marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.grid(True)
plt.savefig('training_loss.png')
plt.close()

# Function to make predictions on new text
def predict_suicide_risk(text, model, tokenizer, device, max_length=128):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Tokenize
    encoding = tokenizer.encode_plus(
        processed_text,
        add_special_tokens=True,
        max_length=max_length,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
    # Get prediction probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    
    # Get class with highest probability
    _, prediction = torch.max(probs, dim=1)
    
    return {
        'prediction': 'Suicidal' if prediction.item() == 1 else 'Non-suicidal',
        'confidence': probs[0][prediction.item()].item(),
        'suicidal_prob': probs[0][1].item(),
        'non_suicidal_prob': probs[0][0].item()
    }

# Example of using the prediction function
print("\nExample prediction:")
example_text = "I don't know if I can keep going anymore. Everything feels so hopeless."
result = predict_suicide_risk(example_text, model, tokenizer, device)
print("Input text: {}".format(example_text))
print("Prediction: {}".format(result['prediction']))
print("Confidence: {:.4f}".format(result['confidence']))
print("Probability of suicidal: {:.4f}".format(result['suicidal_prob']))
print("Probability of non-suicidal: {:.4f}".format(result['non_suicidal_prob']))

# Save the model
output_dir = './models/suicide_detection_model/'
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("\nModel saved to {}".format(output_dir))

# Define a function to analyze model predictions
def analyze_predictions(text, prediction_result):
    """Analyze why the model made a particular prediction"""
    # Check for suicide indicators
    suicide_indicators_present = []
    for word in suicide_indicators:
        if word in text.lower().split():
            suicide_indicators_present.append(word)
    
    # Check for first-person pronoun usage
    first_person_count = sum(1 for word in text.lower().split() if word in first_person_pronouns)
    
    # Analyze text length
    text_length = len(text)
    word_count = len(text.split())
    
    # Try to get sentiment if available
    sentiment_score = None
    try:
        sentiment_score = sid.polarity_scores(text)['compound']
    except:
        pass
    
    print("\nPrediction Analysis:")
    print("- Prediction: {} with {:.2%} confidence".format(prediction_result['prediction'], prediction_result['confidence']))
    print("- Text length: {} characters, {} words".format(text_length, word_count))
    if sentiment_score:
        sentiment_label = "Positive" if sentiment_score > 0 else ("Negative" if sentiment_score < 0 else "Neutral")
        print("- Sentiment score: {:.2f} ({})".format(sentiment_score, sentiment_label))
    print("- First-person pronouns used: {}".format(first_person_count))

    if suicide_indicators_present:
        print("- Potential concerning words detected: {}".format(', '.join(suicide_indicators_present)))
    else:
        print("- No specific concerning words detected from our indicator list")

    
    return {
        'indicators': suicide_indicators_present,
        'first_person_count': first_person_count,
        'text_length': text_length,
        'word_count': word_count,
        'sentiment': sentiment_score
    }

# Example of using the analysis function
print("\nDetailed prediction analysis:")
example_text = "I'm so tired of everything. I don't think I want to be alive anymore."
result = predict_suicide_risk(example_text, model, tokenizer, device)
analysis = analyze_predictions(example_text, result)

# Create a more comprehensive system
def suicide_risk_assessment_system(text, model, tokenizer, device):
    """Complete system for suicide risk assessment"""
    # Make prediction
    prediction = predict_suicide_risk(text, model, tokenizer, device)
    
    # Analyze prediction
    analysis = analyze_predictions(text, prediction)
    
    # Determine risk level
    risk_level = "Low"
    if prediction['suicidal_prob'] > 0.8:
        risk_level = "High"
    elif prediction['suicidal_prob'] > 0.5:
        risk_level = "Medium"
    
    # Generate response based on risk level
    response = "The system has completed its assessment. "
    
    if risk_level == "High":
        response += "This conversation shows significant indicators of suicide risk. "
        response += "Immediate professional intervention is recommended."
    elif risk_level == "Medium":
        response += "This conversation shows some concerning patterns that may indicate suicide risk. "
        response += "Further assessment by a mental health professional is recommended."
    else:
        response += "This conversation does not show strong indicators of immediate suicide risk. "
        response += "However, regular monitoring is still advised as risk factors can change."
    
    return {
        'text': text,
        'prediction': prediction,
        'analysis': analysis,
        'risk_level': risk_level,
        'response': response
    }

# Example of the complete system
print("\nComplete Suicide Risk Assessment System Example:")
test_conversations = [
    "I had a rough day today but I'll be fine after some rest.",
    "I can't take this pain anymore. I just want it all to end. No one would miss me anyway.",
    "I've been feeling down lately and finding it hard to see the point in things."
]

for i, conversation in enumerate(test_conversations):
    print("\nConversation {}:".format(i+1))
    print("'{}'".format(conversation))
    
    result = suicide_risk_assessment_system(conversation, model, tokenizer, device)
    print("Risk Level: {}".format(result['risk_level']))
    print("System Response: {}".format(result['response']))

print("\nNOTE: This model is intended as a support tool for trained professionals, not as an autonomous decision-maker.")
print("Always consult with mental health professionals for proper assessment and intervention in potential suicide risk cases.")