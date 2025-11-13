# -----------------------old 10/11/2025----------------------------
# BERT-Based Hate Speech Detection Training Script
# This uses transformers for state-of-the-art accuracy

# Expected Accuracy: 90-95%+ (vs 85% with Random Forest)

# Requirements:
# pip install transformers torch scikit-learn pandas tqdm
# """

# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import (
#     BertTokenizer, 
#     BertForSequenceClassification,
#     get_linear_schedule_with_warmup
# )
# from torch.optim import AdamW  # ← Import from torch instead
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
# from tqdm import tqdm
# import os
# import joblib

# print("="*60)
# print("BERT-BASED LGBTQ+ HATE SPEECH DETECTION - TRAINING")
# print("="*60)

# # Check for GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"\n🖥️  Using device: {device}")
# if device.type == 'cuda':
#     print(f"   GPU: {torch.cuda.get_device_name(0)}")
# else:
#     print("   ⚠️  No GPU found - training will be slower")

# # ========================================
# # CONFIGURATION
# # ========================================
# CONFIG = {
#     'model_name': 'bert-base-uncased',  # Pre-trained BERT
#     'max_length': 128,                   # Max tokens per text
#     'batch_size': 16,                    # Adjust based on GPU memory
#     'epochs': 4,                         # Usually 3-5 epochs is enough
#     'learning_rate': 2e-5,               # Standard for BERT fine-tuning
#     'warmup_steps': 0.1,                 # Warmup ratio
#     'weight_decay': 0.01,
#     'dataset_path': 'anti-lgbt-cyberbullying.csv'
# }

# # ========================================
# # STEP 1: LOAD DATA
# # ========================================
# print("\n[STEP 1] Loading dataset...")

# try:
#     df = pd.read_csv(CONFIG['dataset_path'])
#     print(f"✓ Loaded dataset from: {CONFIG['dataset_path']}")
# except FileNotFoundError:
#     print(f"✗ ERROR: File not found at {CONFIG['dataset_path']}")
#     exit()

# df['label'] = df['anti_lgbt']
# df = df.dropna(subset=['text', 'label'])

# print(f"✓ Total samples: {len(df)}")
# print(f"  - Non-hate speech (0): {(df['label'] == 0).sum()}")
# print(f"  - Hate speech (1): {(df['label'] == 1).sum()}")

# # ========================================
# # STEP 2: PREPARE DATA
# # ========================================
# print("\n[STEP 2] Preparing data...")

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     df['text'].values,
#     df['label'].values,
#     test_size=0.2,
#     random_state=42,
#     stratify=df['label']
# )

# # Further split train into train/validation
# X_train, X_val, y_train, y_val = train_test_split(
#     X_train,
#     y_train,
#     test_size=0.1,
#     random_state=42,
#     stratify=y_train
# )

# print(f"✓ Training samples: {len(X_train)}")
# print(f"✓ Validation samples: {len(X_val)}")
# print(f"✓ Testing samples: {len(X_test)}")

# # ========================================
# # STEP 3: TOKENIZATION
# # ========================================
# print("\n[STEP 3] Loading BERT tokenizer...")

# tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])
# print(f"✓ Loaded tokenizer: {CONFIG['model_name']}")

# class HateSpeechDataset(Dataset):
#     """Custom Dataset for BERT"""
    
#     def __init__(self, texts, labels, tokenizer, max_length):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_length = max_length
    
#     def __len__(self):
#         return len(self.texts)
    
#     def __getitem__(self, idx):
#         text = str(self.texts[idx])
#         label = self.labels[idx]
        
#         # Tokenize
#         encoding = self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_length,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )
        
#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'labels': torch.tensor(label, dtype=torch.long)
#         }

# # Create datasets
# print("Creating datasets...")
# train_dataset = HateSpeechDataset(X_train, y_train, tokenizer, CONFIG['max_length'])
# val_dataset = HateSpeechDataset(X_val, y_val, tokenizer, CONFIG['max_length'])
# test_dataset = HateSpeechDataset(X_test, y_test, tokenizer, CONFIG['max_length'])

# # Create dataloaders
# train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
# test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])

# print(f"✓ Created dataloaders (batch_size={CONFIG['batch_size']})")

# # ========================================
# # STEP 4: LOAD MODEL
# # ========================================
# print("\n[STEP 4] Loading BERT model...")

# model = BertForSequenceClassification.from_pretrained(
#     CONFIG['model_name'],
#     num_labels=2,  # Binary classification
#     output_attentions=False,
#     output_hidden_states=False
# )

# model.to(device)
# print(f"✓ Loaded model: {CONFIG['model_name']}")
# print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# # ========================================
# # STEP 5: SETUP TRAINING
# # ========================================
# print("\n[STEP 5] Setting up optimizer and scheduler...")

# # Optimizer
# optimizer = AdamW(
#     model.parameters(),
#     lr=CONFIG['learning_rate'],
#     eps=1e-8,
#     weight_decay=CONFIG['weight_decay']
# )

# # Scheduler
# total_steps = len(train_loader) * CONFIG['epochs']
# warmup_steps = int(total_steps * CONFIG['warmup_steps'])

# scheduler = get_linear_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=warmup_steps,
#     num_training_steps=total_steps
# )

# print(f"✓ Total training steps: {total_steps}")
# print(f"✓ Warmup steps: {warmup_steps}")

# # ========================================
# # STEP 6: TRAINING FUNCTIONS
# # ========================================

# def train_epoch(model, data_loader, optimizer, scheduler, device):
#     """Train for one epoch"""
#     model.train()
#     total_loss = 0
#     predictions = []
#     true_labels = []
    
#     progress_bar = tqdm(data_loader, desc="Training")
    
#     for batch in progress_bar:
#         # Move batch to device
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
        
#         # Forward pass
#         outputs = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=labels
#         )
        
#         loss = outputs.loss
#         logits = outputs.logits
        
#         # Backward pass
#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#         scheduler.step()
        
#         # Track metrics
#         total_loss += loss.item()
#         preds = torch.argmax(logits, dim=1).cpu().numpy()
#         predictions.extend(preds)
#         true_labels.extend(labels.cpu().numpy())
        
#         progress_bar.set_postfix({'loss': loss.item()})
    
#     avg_loss = total_loss / len(data_loader)
#     accuracy = accuracy_score(true_labels, predictions)
    
#     return avg_loss, accuracy

# def evaluate(model, data_loader, device):
#     """Evaluate model"""
#     model.eval()
#     total_loss = 0
#     predictions = []
#     true_labels = []
    
#     with torch.no_grad():
#         for batch in tqdm(data_loader, desc="Evaluating"):
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)
            
#             outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 labels=labels
#             )
            
#             loss = outputs.loss
#             logits = outputs.logits
            
#             total_loss += loss.item()
#             preds = torch.argmax(logits, dim=1).cpu().numpy()
#             predictions.extend(preds)
#             true_labels.extend(labels.cpu().numpy())
    
#     avg_loss = total_loss / len(data_loader)
#     accuracy = accuracy_score(true_labels, predictions)
#     f1 = f1_score(true_labels, predictions)
    
#     return avg_loss, accuracy, f1, predictions, true_labels

# # ========================================
# # STEP 7: TRAIN MODEL
# # ========================================
# print(f"\n[STEP 6] Training BERT model for {CONFIG['epochs']} epochs...")
# print("="*60)

# best_val_f1 = 0
# history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

# for epoch in range(CONFIG['epochs']):
#     print(f"\n📊 Epoch {epoch + 1}/{CONFIG['epochs']}")
#     print("-" * 60)
    
#     # Train
#     train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
    
#     # Validate
#     val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
    
#     # Save history
#     history['train_loss'].append(train_loss)
#     history['train_acc'].append(train_acc)
#     history['val_loss'].append(val_loss)
#     history['val_acc'].append(val_acc)
#     history['val_f1'].append(val_f1)
    
#     print(f"\n✓ Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
#     print(f"✓ Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
    
#     # Save best model
#     if val_f1 > best_val_f1:
#         best_val_f1 = val_f1
#         print(f"💾 New best model! (F1: {val_f1:.4f})")
#         os.makedirs('models', exist_ok=True)
#         torch.save(model.state_dict(), 'models/bert_model_best.pt')
#         # Save tokenizer too
#         tokenizer.save_pretrained('models/bert_tokenizer')

# print("\n" + "="*60)
# print("✓ Training completed!")

# # ========================================
# # STEP 8: EVALUATE ON TEST SET
# # ========================================
# print("\n[STEP 7] Evaluating on test set...")

# # Load best model
# model.load_state_dict(torch.load('models/bert_model_best.pt'))

# test_loss, test_acc, test_f1, y_pred, y_true = evaluate(model, test_loader, device)

# print(f"\n🎯 TEST RESULTS:")
# print("="*60)
# print(f"✓ Test Accuracy: {test_acc*100:.2f}%")
# print(f"✓ Test F1-Score: {test_f1:.4f}")
# print(f"✓ Test Loss: {test_loss:.4f}")

# print("\nClassification Report:")
# print("="*60)
# print(classification_report(y_true, y_pred, target_names=['Non-Hate', 'Hate']))

# print("\nConfusion Matrix:")
# print("="*60)
# cm = confusion_matrix(y_true, y_pred)
# print(f"                 Predicted Non-Hate  |  Predicted Hate")
# print(f"Actual Non-Hate:      {cm[0][0]:5d}          |      {cm[0][1]:5d}")
# print(f"Actual Hate:          {cm[1][0]:5d}          |      {cm[1][1]:5d}")

# # ========================================
# # STEP 9: TEST ON SPECIFIC EXAMPLES
# # ========================================
# print("\n[STEP 8] Testing on challenging examples...")

# test_cases = [
#     ("i am disgusted that i am trans", 1),
#     ("i am disgusted that i am bi", 1),
#     ("i love being trans", 0),
#     ("proud to be gay", 0),
#     ("lgbt people are disgusting", 1),
#     ("i support lgbt rights", 0),
#     ("being gay is wrong", 1),
#     ("trans people deserve respect", 0),
#     ("i hate myself for being gay", 1),
#     ("lgbtq community is amazing", 0),
# ]

# print("\n" + "="*60)
# print("Testing specific examples:")
# print("="*60)

# model.eval()
# correct = 0

# for text, expected in test_cases:
#     # Tokenize
#     encoding = tokenizer.encode_plus(
#         text,
#         add_special_tokens=True,
#         max_length=CONFIG['max_length'],
#         padding='max_length',
#         truncation=True,
#         return_attention_mask=True,
#         return_tensors='pt'
#     )
    
#     input_ids = encoding['input_ids'].to(device)
#     attention_mask = encoding['attention_mask'].to(device)
    
#     # Predict
#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         logits = outputs.logits
#         probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
#         prediction = np.argmax(probs)
    
#     is_correct = prediction == expected
#     correct += is_correct
    
#     result_emoji = "✓" if is_correct else "✗"
#     print(f"\n{result_emoji} Text: '{text}'")
#     print(f"   Expected: {'Hate' if expected == 1 else 'Non-Hate'}")
#     print(f"   Predicted: {'Hate' if prediction == 1 else 'Non-Hate'}")
#     print(f"   Confidence: Non-Hate={probs[0]*100:.1f}%, Hate={probs[1]*100:.1f}%")

# print(f"\n{'='*60}")
# print(f"Test accuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.1f}%)")

# # ========================================
# # STEP 10: SAVE MODEL CONFIG
# # ========================================
# print("\n[STEP 9] Saving model configuration...")

# model_config = {
#     'model_name': CONFIG['model_name'],
#     'max_length': CONFIG['max_length'],
#     'num_labels': 2,
#     'test_accuracy': test_acc,
#     'test_f1': test_f1,
#     'best_val_f1': best_val_f1
# }

# joblib.dump(model_config, 'models/bert_model_config.pkl')

# print("✓ Model saved to: models/bert_model_best.pt")
# print("✓ Tokenizer saved to: models/bert_tokenizer/")
# print("✓ Config saved to: models/bert_model_config.pkl")

# # ========================================
# # FINAL SUMMARY
# # ========================================
# print("\n" + "="*60)
# print("🎉 BERT MODEL TRAINING COMPLETE!")
# print("="*60)
# print(f"\n📊 Final Performance:")
# print(f"  ✓ Test Accuracy: {test_acc*100:.2f}%")
# print(f"  ✓ Test F1-Score: {test_f1:.4f}")
# print(f"  ✓ Challenge Test: {correct}/{len(test_cases)}")
# print(f"\n💾 Model Files:")
# print(f"  • models/bert_model_best.pt (~400 MB)")
# print(f"  • models/bert_tokenizer/ (~200 KB)")
# print(f"  • models/bert_model_config.pkl (~1 KB)")
# print(f"\n🚀 Next Steps:")
# print(f"  1. Run: streamlit run Hate_Speech_BERT.py")
# print(f"  2. Test with your examples")
# print(f"  3. Compare with Random Forest results")
# print("\n" + "="*60)



# ---------------------new 11/11/2025-------------------------

"""
IMPROVED BERT-Based Hate Speech Detection Training Script
Enhancements for better self-hate detection and overall accuracy

Key Improvements:
1. Data augmentation for self-hate cases
2. Class weight balancing
3. Increased epochs with early stopping
4. Better learning rate scheduling
5. Enhanced validation

Requirements:
pip install transformers torch scikit-learn pandas tqdm
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import os
import joblib
import random

print("="*60)
print("IMPROVED BERT HATE SPEECH DETECTION - TRAINING")
print("="*60)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Using device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
else:
    print("   ⚠️  No GPU found - training will be slower")

# ========================================
# CONFIGURATION
# ========================================
CONFIG = {
    'model_name': 'bert-base-uncased',
    'max_length': 128,
    'batch_size': 16,
    'epochs': 6,  # Increased from 4
    'learning_rate': 2e-5,
    'warmup_steps': 0.1,
    'weight_decay': 0.01,
    'dataset_path': 'anti-lgbt-cyberbullying.csv',
    'early_stopping_patience': 3,
    'use_class_weights': True,
    'augment_data': True  # New: Data augmentation
}

# ========================================
# DATA AUGMENTATION FOR SELF-HATE CASES
# ========================================
def augment_self_hate_data(df):
    """
    Augment dataset with variations of self-hate speech
    This helps the model learn self-directed hate better
    """
    print("\n[DATA AUGMENTATION] Creating synthetic self-hate examples...")
    
    # Self-hate patterns
    self_hate_templates = [
        "i am {} that i am {}",
        "i hate that i am {}",
        "i am ashamed that i am {}",
        "i wish i wasn't {}",
        "i am {} for being {}",
        "why am i {}",
        "i hate being {}",
        "i am {} and i hate it",
        "being {} is disgusting",
        "i can't accept that i am {}"
    ]
    
    negative_words = [
        'disgusted', 'ashamed', 'disappointed', 'upset', 
        'angry', 'sad', 'confused', 'scared', 'worried'
    ]
    
    lgbt_identities = [
        'gay', 'lesbian', 'bi', 'bisexual', 'trans', 'transgender',
        'queer', 'lgbt', 'lgbtq', 'non-binary', 'pan', 'pansexual'
    ]
    
    augmented_texts = []
    augmented_labels = []
    
    # Generate synthetic examples
    for template in self_hate_templates[:5]:  # Use first 5 templates
        for identity in lgbt_identities[:6]:  # Use first 6 identities
            if '{}' in template:
                count = template.count('{}')
                if count == 2:
                    neg_word = random.choice(negative_words)
                    text = template.format(neg_word, identity)
                elif count == 1:
                    text = template.format(identity)
                
                augmented_texts.append(text)
                augmented_labels.append(1)  # All are hate speech
    
    # Create augmented dataframe
    aug_df = pd.DataFrame({
        'text': augmented_texts,
        'label': augmented_labels
    })
    
    print(f"✓ Generated {len(aug_df)} synthetic self-hate examples")
    
    return aug_df

# ========================================
# STEP 1: LOAD AND AUGMENT DATA
# ========================================
print("\n[STEP 1] Loading dataset...")

try:
    df = pd.read_csv(CONFIG['dataset_path'])
    print(f"✓ Loaded dataset from: {CONFIG['dataset_path']}")
except FileNotFoundError:
    print(f"✗ ERROR: File not found at {CONFIG['dataset_path']}")
    exit()

df['label'] = df['anti_lgbt']
df = df.dropna(subset=['text', 'label'])

print(f"✓ Original samples: {len(df)}")
print(f"  - Non-hate speech (0): {(df['label'] == 0).sum()}")
print(f"  - Hate speech (1): {(df['label'] == 1).sum()}")

# Augment data if enabled
if CONFIG['augment_data']:
    aug_df = augment_self_hate_data(df)
    df = pd.concat([df, aug_df], ignore_index=True)
    print(f"✓ Total samples after augmentation: {len(df)}")
    print(f"  - Non-hate speech (0): {(df['label'] == 0).sum()}")
    print(f"  - Hate speech (1): {(df['label'] == 1).sum()}")

# ========================================
# STEP 2: PREPARE DATA
# ========================================
print("\n[STEP 2] Preparing data...")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'].values,
    df['label'].values,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# Further split train into train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.1,
    random_state=42,
    stratify=y_train
)

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Validation samples: {len(X_val)}")
print(f"✓ Testing samples: {len(X_test)}")

# Compute class weights for imbalanced data
if CONFIG['use_class_weights']:
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"✓ Class weights: {class_weights.cpu().numpy()}")
else:
    class_weights = None

# ========================================
# STEP 3: TOKENIZATION
# ========================================
print("\n[STEP 3] Loading BERT tokenizer...")

tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])
print(f"✓ Loaded tokenizer: {CONFIG['model_name']}")

class HateSpeechDataset(Dataset):
    """Custom Dataset for BERT"""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create datasets
print("Creating datasets...")
train_dataset = HateSpeechDataset(X_train, y_train, tokenizer, CONFIG['max_length'])
val_dataset = HateSpeechDataset(X_val, y_val, tokenizer, CONFIG['max_length'])
test_dataset = HateSpeechDataset(X_test, y_test, tokenizer, CONFIG['max_length'])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])

print(f"✓ Created dataloaders (batch_size={CONFIG['batch_size']})")

# ========================================
# STEP 4: LOAD MODEL
# ========================================
print("\n[STEP 4] Loading BERT model...")

model = BertForSequenceClassification.from_pretrained(
    CONFIG['model_name'],
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)

model.to(device)
print(f"✓ Loaded model: {CONFIG['model_name']}")
print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ========================================
# STEP 5: SETUP TRAINING
# ========================================
print("\n[STEP 5] Setting up optimizer and scheduler...")

# Optimizer
optimizer = AdamW(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    eps=1e-8,
    weight_decay=CONFIG['weight_decay']
)

# Scheduler
total_steps = len(train_loader) * CONFIG['epochs']
warmup_steps = int(total_steps * CONFIG['warmup_steps'])

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"✓ Total training steps: {total_steps}")
print(f"✓ Warmup steps: {warmup_steps}")

# ========================================
# STEP 6: TRAINING FUNCTIONS (IMPROVED)
# ========================================

def train_epoch(model, data_loader, optimizer, scheduler, device, class_weights=None):
    """Train for one epoch with optional class weights"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(data_loader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        # Apply class weights if provided
        if class_weights is not None:
            # Recompute loss with class weights
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    return avg_loss, accuracy, f1

def evaluate(model, data_loader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    return avg_loss, accuracy, f1, predictions, true_labels

# ========================================
# STEP 7: TRAIN MODEL (WITH EARLY STOPPING)
# ========================================
print(f"\n[STEP 6] Training BERT model for {CONFIG['epochs']} epochs...")
print("="*60)

best_val_f1 = 0
patience_counter = 0
history = {
    'train_loss': [], 'train_acc': [], 'train_f1': [],
    'val_loss': [], 'val_acc': [], 'val_f1': []
}

for epoch in range(CONFIG['epochs']):
    print(f"\n📊 Epoch {epoch + 1}/{CONFIG['epochs']}")
    print("-" * 60)
    
    # Train
    train_loss, train_acc, train_f1 = train_epoch(
        model, train_loader, optimizer, scheduler, device, class_weights
    )
    
    # Validate
    val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['train_f1'].append(train_f1)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)
    
    print(f"\n✓ Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
    print(f"✓ Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
    
    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        print(f"💾 New best model! (F1: {val_f1:.4f})")
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), 'models/bert_model_best.pt')
        tokenizer.save_pretrained('models/bert_tokenizer')
    else:
        patience_counter += 1
        print(f"⏳ Patience: {patience_counter}/{CONFIG['early_stopping_patience']}")
        
        if patience_counter >= CONFIG['early_stopping_patience']:
            print(f"\n⚠️  Early stopping triggered! No improvement for {CONFIG['early_stopping_patience']} epochs.")
            break

print("\n" + "="*60)
print("✓ Training completed!")

# ========================================
# STEP 8: EVALUATE ON TEST SET
# ========================================
print("\n[STEP 7] Evaluating on test set...")

# Load best model
model.load_state_dict(torch.load('models/bert_model_best.pt'))

test_loss, test_acc, test_f1, y_pred, y_true = evaluate(model, test_loader, device)

print(f"\n🎯 TEST RESULTS:")
print("="*60)
print(f"✓ Test Accuracy: {test_acc*100:.2f}%")
print(f"✓ Test F1-Score: {test_f1:.4f}")
print(f"✓ Test Loss: {test_loss:.4f}")

print("\nClassification Report:")
print("="*60)
print(classification_report(y_true, y_pred, target_names=['Non-Hate', 'Hate']))

print("\nConfusion Matrix:")
print("="*60)
cm = confusion_matrix(y_true, y_pred)
print(f"                 Predicted Non-Hate  |  Predicted Hate")
print(f"Actual Non-Hate:      {cm[0][0]:5d}          |      {cm[0][1]:5d}")
print(f"Actual Hate:          {cm[1][0]:5d}          |      {cm[1][1]:5d}")

# ========================================
# STEP 9: TEST ON SPECIFIC EXAMPLES (EXPANDED)
# ========================================
print("\n[STEP 8] Testing on challenging examples...")

test_cases = [
    # Self-hate cases (the problematic ones)
    ("i am disgusted that i am trans", 1),
    ("i am bi and i am disgusted", 1),
    ("i am disgusted that i am bi", 1),
    ("i hate that i am gay", 1),
    ("i am ashamed of being trans", 1),
    ("i wish i wasn't lgbt", 1),
    
    # Direct hate
    ("lgbt people are disgusting", 1),
    ("being gay is wrong", 1),
    ("trans people are mentally ill", 1),
    
    # Support/positive
    ("i love being trans", 0),
    ("proud to be gay", 0),
    ("i support lgbt rights", 0),
    ("trans people deserve respect", 0),
    ("lgbtq community is amazing", 0),
    ("i am proud to be bi", 0),
]

print("\n" + "="*60)
print("Testing specific examples:")
print("="*60)

model.eval()
correct = 0

for text, expected in test_cases:
    # Tokenize
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=CONFIG['max_length'],
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        prediction = np.argmax(probs)
    
    is_correct = prediction == expected
    correct += is_correct
    
    result_emoji = "✓" if is_correct else "✗"
    print(f"\n{result_emoji} Text: '{text}'")
    print(f"   Expected: {'Hate' if expected == 1 else 'Non-Hate'}")
    print(f"   Predicted: {'Hate' if prediction == 1 else 'Non-Hate'}")
    print(f"   Confidence: Non-Hate={probs[0]*100:.1f}%, Hate={probs[1]*100:.1f}%")

print(f"\n{'='*60}")
print(f"Test accuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.1f}%)")

# ========================================
# STEP 10: SAVE MODEL CONFIG
# ========================================
print("\n[STEP 9] Saving model configuration...")

model_config = {
    'model_name': CONFIG['model_name'],
    'max_length': CONFIG['max_length'],
    'num_labels': 2,
    'test_accuracy': test_acc,
    'test_f1': test_f1,
    'best_val_f1': best_val_f1,
    'augmented': CONFIG['augment_data'],
    'class_weighted': CONFIG['use_class_weights']
}

joblib.dump(model_config, 'models/bert_model_config.pkl')

print("✓ Model saved to: models/bert_model_best.pt")
print("✓ Tokenizer saved to: models/bert_tokenizer/")
print("✓ Config saved to: models/bert_model_config.pkl")

# ========================================
# FINAL SUMMARY
# ========================================
print("\n" + "="*60)
print("🎉 IMPROVED BERT MODEL TRAINING COMPLETE!")
print("="*60)
print(f"\n📊 Final Performance:")
print(f"  ✓ Test Accuracy: {test_acc*100:.2f}%")
print(f"  ✓ Test F1-Score: {test_f1:.4f}")
print(f"  ✓ Challenge Test: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.1f}%)")
print(f"\n💾 Model Files:")
print(f"  • models/bert_model_best.pt (~400 MB)")
print(f"  • models/bert_tokenizer/ (~200 KB)")
print(f"  • models/bert_model_config.pkl (~1 KB)")
print(f"\n🚀 Next Steps:")
print(f"  1. Run: streamlit run app_3.py")
print(f"  2. Test with your examples")
print(f"  3. The model should now handle self-hate cases better!")
print("\n" + "="*60)