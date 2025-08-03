"""
Approach 5: Deep Learning with LSTM/GRU
Neural networks that read text sequentially and remember important parts
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Note: Requires tensorflow/keras
# pip install tensorflow

def deep_learning_approach():
    print("="*60)
    print("DEEP LEARNING APPROACH - LSTM/GRU Neural Networks")
    print("Reads text word by word and remembers important information")
    print("="*60)
    
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GRU
        from tensorflow.keras.optimizers import Adam
        print(f"TensorFlow version: {tf.__version__}")
    except ImportError:
        print("\nERROR: Please install TensorFlow first:")
        print("pip install tensorflow")
        return
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Combine text
    train_df['all_text'] = (
        train_df['QuestionText'].fillna('') + ' ' +
        train_df['MC_Answer'].fillna('') + ' ' +
        train_df['StudentExplanation'].fillna('')
    )
    
    test_df['all_text'] = (
        test_df['QuestionText'].fillna('') + ' ' +
        test_df['MC_Answer'].fillna('') + ' ' +
        test_df['StudentExplanation'].fillna('')
    )
    
    print("\n" + "="*60)
    print("HOW LSTM/GRU WORKS:")
    print("="*60)
    print("1. LSTM = Long Short-Term Memory")
    print("   - Reads text word by word")
    print("   - Has 'memory' to remember important information")
    print("   - Can forget irrelevant information")
    print("\n2. GRU = Gated Recurrent Unit")
    print("   - Similar to LSTM but simpler and faster")
    print("   - Also has memory mechanisms")
    print("\n3. Think of it like reading a book:")
    print("   - You remember important plot points")
    print("   - You forget minor details")
    print("   - Context from earlier helps understand later parts")
    
    # Create labels
    train_df['label'] = train_df['Category'] + ':' + train_df['Misconception']
    all_labels = sorted(train_df['label'].unique())
    
    # Get unique students
    unique_students = train_df.drop_duplicates(subset=['row_id']).sort_values('row_id')
    
    # Create label matrix
    label_matrix = np.zeros((len(unique_students), len(all_labels)))
    for idx, student in unique_students.iterrows():
        student_labels = train_df[train_df['row_id'] == student['row_id']]['label'].tolist()
        for label in student_labels:
            label_idx = all_labels.index(label)
            row_idx = unique_students.index.get_loc(idx)
            label_matrix[row_idx, label_idx] = 1
    
    # Prepare text for neural network
    print("\n" + "="*60)
    print("Preparing text for neural network...")
    
    # Create tokenizer (converts words to numbers)
    max_words = 10000  # Use top 10000 words
    max_length = 200   # Maximum text length
    
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(unique_students['all_text'])
    
    # Convert text to sequences of numbers
    X_train_sequences = tokenizer.texts_to_sequences(unique_students['all_text'])
    X_test_sequences = tokenizer.texts_to_sequences(test_df['all_text'])
    
    # Pad sequences to same length
    X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding='post')
    X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length, padding='post')
    
    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    print(f"Padded sequence shape: {X_train_padded.shape}")
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_padded, label_matrix, test_size=0.2, random_state=42
    )
    
    # Build LSTM model
    print("\n" + "="*60)
    print("Building LSTM Neural Network...")
    print("="*60)
    
    model = Sequential([
        # Embedding layer: converts word numbers to dense vectors
        Embedding(max_words, 128, input_length=max_length),
        
        # Bidirectional LSTM: reads text forward and backward
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        
        # Another LSTM layer
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        
        # Dense layers for classification
        Dense(64, activation='relu'),
        Dropout(0.5),
        
        # Output layer: one neuron per possible label
        Dense(len(all_labels), activation='sigmoid')  # sigmoid for multi-label
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',  # for multi-label
        metrics=['accuracy']
    )
    
    print("\nModel architecture:")
    model.summary()
    
    # Train model
    print("\n" + "="*60)
    print("Training LSTM model...")
    print("This will take several minutes...")
    
    # Train for fewer epochs for demo
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=5,  # Increase to 10-20 for better results
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Make predictions
    print("\nMaking predictions...")
    predictions_proba = model.predict(X_test_padded)
    predictions = (predictions_proba > 0.5).astype(int)
    
    # Format results
    results = []
    for i, test_row in test_df.iterrows():
        pred_indices = np.where(predictions[i] == 1)[0]
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
    submission.to_csv('submission_lstm.csv', index=False)
    print("\nResults saved to 'submission_lstm.csv'")
    
    # Build GRU model (alternative)
    print("\n" + "="*60)
    print("Building GRU model (faster alternative to LSTM)...")
    
    gru_model = Sequential([
        Embedding(max_words, 128, input_length=max_length),
        Bidirectional(GRU(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(GRU(32)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(all_labels), activation='sigmoid')
    ])
    
    gru_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nTraining GRU model...")
    gru_history = gru_model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=5,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # GRU predictions
    gru_predictions_proba = gru_model.predict(X_test_padded)
    gru_predictions = (gru_predictions_proba > 0.5).astype(int)
    
    # Format GRU results
    gru_results = []
    for i, test_row in test_df.iterrows():
        pred_indices = np.where(gru_predictions[i] == 1)[0]
        if len(pred_indices) == 0:
            pred_labels = ["True_Correct:NA"]
        else:
            pred_labels = [all_labels[idx] for idx in pred_indices]
        
        gru_results.append({
            'row_id': test_row['row_id'],
            'Category:Misconception': ' '.join(sorted(pred_labels))
        })
    
    gru_submission = pd.DataFrame(gru_results)
    gru_submission.to_csv('submission_gru.csv', index=False)
    print("\nGRU results saved to 'submission_gru.csv'")
    
    print("\n" + "="*60)
    print("ADVANTAGES OF DEEP LEARNING:")
    print("="*60)
    print("✓ Understands complex patterns in text")
    print("✓ Remembers important context")
    print("✓ Can learn subtle relationships")
    print("✓ State-of-the-art performance possible")
    
    print("\nDISADVANTAGES:")
    print("✗ Needs lots of data to work well")
    print("✗ Takes long time to train")
    print("✗ Requires GPU for best performance")
    print("✗ Hard to understand why it makes decisions")
    print("✗ Can overfit on small datasets")
    
    print("\nTIPS FOR BETTER RESULTS:")
    print("- Train for more epochs (10-20)")
    print("- Use GPU if available (much faster)")
    print("- Try different architectures")
    print("- Use pre-trained embeddings")
    print("- Add more training data if possible")

if __name__ == "__main__":
    deep_learning_approach()