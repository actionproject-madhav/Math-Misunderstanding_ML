"""
Approach 6: Transformer Models (BERT/DistilBERT)
State-of-the-art approach that understands context deeply
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Note: Requires transformers library
# pip install transformers torch

def transformer_approach():
    print("="*60)
    print("TRANSFORMER (BERT) APPROACH - State of the Art")
    print("Most advanced method for understanding text")
    print("="*60)
    
    try:
        import torch
        from transformers import (
            DistilBertTokenizer, 
            DistilBertForSequenceClassification,
            Trainer, 
            TrainingArguments,
            AutoTokenizer,
            AutoModelForSequenceClassification
        )
        print(f"PyTorch version: {torch.__version__}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
    except ImportError:
        print("\nERROR: Please install required libraries:")
        print("pip install transformers torch")
        return
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print("\n" + "="*60)
    print("HOW BERT/TRANSFORMERS WORK:")
    print("="*60)
    print("1. BERT = Bidirectional Encoder Representations from Transformers")
    print("   - Reads entire text at once (not word by word)")
    print("   - Understands context from both directions")
    print("   - Pre-trained on millions of texts")
    print("\n2. Like having a team of experts:")
    print("   - Each expert (attention head) looks at different aspects")
    print("   - They share insights with each other")
    print("   - Together they deeply understand the text")
    print("\n3. DistilBERT = Smaller, faster version of BERT")
    print("   - 40% smaller, 60% faster")
    print("   - Still 95% as good as full BERT")
    
    # Combine text
    train_df['all_text'] = (
        train_df['QuestionText'].fillna('') + ' [SEP] ' +
        train_df['MC_Answer'].fillna('') + ' [SEP] ' +
        train_df['StudentExplanation'].fillna('')
    )
    
    test_df['all_text'] = (
        test_df['QuestionText'].fillna('') + ' [SEP] ' +
        test_df['MC_Answer'].fillna('') + ' [SEP] ' +
        test_df['StudentExplanation'].fillna('')
    )
    
    # Create labels
    train_df['label'] = train_df['Category'] + ':' + train_df['Misconception']
    all_labels = sorted(train_df['label'].unique())
    label_to_id = {label: i for i, label in enumerate(all_labels)}
    
    # Get unique students
    unique_students = train_df.drop_duplicates(subset=['row_id']).sort_values('row_id')
    
    # Create label matrix
    label_matrix = np.zeros((len(unique_students), len(all_labels)))
    for idx, student in unique_students.iterrows():
        student_labels = train_df[train_df['row_id'] == student['row_id']]['label'].tolist()
        for label in student_labels:
            label_idx = label_to_id[label]
            row_idx = unique_students.index.get_loc(idx)
            label_matrix[row_idx, label_idx] = 1
    
    print("\n" + "="*60)
    print("Preparing data for BERT...")
    
    # Use DistilBERT (smaller, faster)
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    
    # Create custom dataset class
    class MisconceptionDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels=None):
            self.texts = texts
            self.labels = labels
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=256,  # Shorter for speed
                return_tensors='pt'
            )
            
            item = {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }
            
            if self.labels is not None:
                item['labels'] = torch.FloatTensor(self.labels[idx])
            
            return item
    
    # Prepare datasets
    X_train, X_val, y_train, y_val = train_test_split(
        unique_students['all_text'].tolist(),
        label_matrix,
        test_size=0.2,
        random_state=42
    )
    
    train_dataset = MisconceptionDataset(X_train, y_train)
    val_dataset = MisconceptionDataset(X_val, y_val)
    test_dataset = MisconceptionDataset(test_df['all_text'].tolist())
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    # Simple training approach (without Trainer API)
    print("\n" + "="*60)
    print("Training DistilBERT model...")
    print("NOTE: This is a simplified version for demonstration")
    print("For best results, use GPU and train for more epochs")
    print("="*60)
    
    # Load pre-trained model
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(all_labels),
        problem_type="multi_label_classification"
    ).to(device)
    
    # Training settings
    batch_size = 8  # Small batch size for CPU
    learning_rate = 2e-5
    num_epochs = 2  # Very few epochs for demo (use 5-10 for better results)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop (simplified)
    print("\nTraining...")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            # Move batch to device
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
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_count}, Loss: {loss.item():.4f}")
            
            # For demo, train on only first 50 batches
            if batch_count >= 50:
                print("Stopping early for demo...")
                break
        
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    # Make predictions
    print("\n" + "="*60)
    print("Making predictions...")
    model.eval()
    
    all_predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get probabilities
            probs = torch.sigmoid(outputs.logits)
            predictions = (probs > 0.5).cpu().numpy()
            all_predictions.extend(predictions)
    
    # Format results
    results = []
    for i, test_row in test_df.iterrows():
        pred_indices = np.where(all_predictions[i] == 1)[0]
        if len(pred_indices) == 0:
            pred_labels = ["True_Correct:NA"]
        else:
            pred_labels = [all_labels[idx] for idx in pred_indices]
        
        results.append({
            'row_id': test_row['row_id'],
            'Category:Misconception': ' '.join(sorted(pred_labels))
        })
    
    # Save results
    submission = pd.DataFrame(results)
    submission.to_csv('submission_bert.csv', index=False)
    print("\nResults saved to 'submission_bert.csv'")
    
    print("\nExample predictions:")
    print(submission.head())
    
    print("\n" + "="*60)
    print("ADVANTAGES OF TRANSFORMERS:")
    print("="*60)
    print("✓ Best performance on most text tasks")
    print("✓ Deep understanding of context")
    print("✓ Pre-trained on massive datasets")
    print("✓ Handles complex language patterns")
    print("✓ State-of-the-art results")
    
    print("\nDISADVANTAGES:")
    print("✗ Requires GPU for reasonable speed")
    print("✗ Large models (hundreds of MB)")
    print("✗ Expensive to train")
    print("✗ Can be overkill for simple tasks")
    print("✗ Hard to interpret decisions")
    
    print("\nTIPS FOR BETTER RESULTS:")
    print("1. Use GPU (10-100x faster)")
    print("2. Train for more epochs (5-10)")
    print("3. Try different models:")
    print("   - bert-base-uncased (larger, better)")
    print("   - roberta-base (often better than BERT)")
    print("   - albert-base-v2 (smaller, efficient)")
    print("4. Fine-tune hyperparameters")
    print("5. Use larger batch sizes with GPU")
    
    print("\n" + "="*60)
    print("ALTERNATIVE: Using Hugging Face AutoTrain")
    print("="*60)
    print("For production use, consider:")
    print("1. Hugging Face AutoTrain (no-code solution)")
    print("2. Google Colab with free GPU")
    print("3. Cloud services (AWS, GCP, Azure)")

if __name__ == "__main__":
    transformer_approach()