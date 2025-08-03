"""
Approach 4: Word Embeddings (Understanding word meanings)
This converts words to numbers in a way that similar words have similar numbers
"""

import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Note: For this to work, you need to install:
# pip install gensim

def embedding_approach():
    print("="*60)
    print("WORD EMBEDDINGS APPROACH - Understanding Word Meanings")
    print("Converts words to vectors where similar words are close together")
    print("="*60)
    
    try:
        from gensim.models import Word2Vec
        import gensim.downloader as api
    except ImportError:
        print("\nERROR: Please install gensim first:")
        print("pip install gensim")
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
    print("HOW WORD EMBEDDINGS WORK:")
    print("="*60)
    print("1. Each word becomes a list of numbers (vector)")
    print("   Example: 'fraction' -> [0.2, -0.5, 0.8, ...]")
    print("\n2. Similar words have similar numbers")
    print("   'fraction' ≈ [0.2, -0.5, 0.8, ...]")
    print("   'decimal'  ≈ [0.3, -0.4, 0.7, ...]")
    print("   'cat'      ≈ [0.9, 0.1, -0.2, ...] (very different!)")
    print("\n3. We can do math with words!")
    print("   'king' - 'man' + 'woman' ≈ 'queen'")
    
    # Tokenize text
    print("\n" + "="*60)
    print("Preparing text for embeddings...")
    
    def tokenize(text):
        """Convert text to list of words"""
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    # Tokenize all texts
    train_tokens = [tokenize(text) for text in train_df['all_text']]
    test_tokens = [tokenize(text) for text in test_df['all_text']]
    
    # Create labels
    train_df['label'] = train_df['Category'] + ':' + train_df['Misconception']
    all_labels = sorted(train_df['label'].unique())
    
    # Get unique students
    unique_students = train_df.drop_duplicates(subset=['row_id']).sort_values('row_id')
    unique_tokens = [tokenize(text) for text in unique_students['all_text']]
    
    label_matrix = np.zeros((len(unique_students), len(all_labels)))
    for idx, student in unique_students.iterrows():
        student_labels = train_df[train_df['row_id'] == student['row_id']]['label'].tolist()
        for label in student_labels:
            label_idx = all_labels.index(label)
            row_idx = unique_students.index.get_loc(idx)
            label_matrix[row_idx, label_idx] = 1
    
    # Train Word2Vec model
    print("\n" + "="*60)
    print("Training Word2Vec model on our data...")
    print("This learns number representations for each word")
    
    # Combine all texts for training Word2Vec
    all_tokens = unique_tokens + test_tokens
    
    # Train Word2Vec
    print("Learning word vectors (this may take a minute)...")
    word2vec_model = Word2Vec(
        sentences=all_tokens,
        vector_size=100,  # Each word becomes 100 numbers
        window=5,         # Look at 5 words on each side
        min_count=2,      # Ignore words that appear less than 2 times
        workers=4,
        seed=42
    )
    
    print(f"Learned embeddings for {len(word2vec_model.wv)} words")
    
    # Show example of similar words
    print("\n" + "="*60)
    print("EXAMPLE: Finding similar words")
    print("="*60)
    
    test_words = ['fraction', 'answer', 'correct', 'simplify']
    for word in test_words:
        if word in word2vec_model.wv:
            similar = word2vec_model.wv.most_similar(word, topn=5)
            print(f"\nWords similar to '{word}':")
            for sim_word, score in similar:
                print(f"  - {sim_word}: {score:.3f}")
    
    # Convert documents to vectors
    print("\n" + "="*60)
    print("Converting documents to vectors...")
    print("We average all word vectors in each document")
    
    def document_to_vector(tokens, model):
        """Convert a document (list of words) to a single vector"""
        vectors = []
        for word in tokens:
            if word in model.wv:
                vectors.append(model.wv[word])
        
        if len(vectors) == 0:
            # If no words found, return zeros
            return np.zeros(model.vector_size)
        
        # Average all word vectors
        return np.mean(vectors, axis=0)
    
    # Convert all documents
    X_train = np.array([document_to_vector(tokens, word2vec_model) 
                       for tokens in unique_tokens])
    X_test = np.array([document_to_vector(tokens, word2vec_model) 
                      for tokens in test_tokens])
    
    print(f"Document vectors shape: {X_train.shape}")
    print("Each document is now 100 numbers representing its meaning!")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier
    print("\n" + "="*60)
    print("Training Random Forest on word embeddings...")
    
    classifier = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
    )
    classifier.fit(X_train_scaled, label_matrix)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = classifier.predict(X_test_scaled)
    
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
    submission.to_csv('submission_embeddings.csv', index=False)
    print("\nResults saved to 'submission_embeddings.csv'")
    
    print("\nExample predictions:")
    print(submission.head())
    
    # Alternative: Use pre-trained embeddings
    print("\n" + "="*60)
    print("ALTERNATIVE: Using pre-trained GloVe embeddings")
    print("="*60)
    print("Pre-trained embeddings are trained on massive text datasets")
    print("They often work better than training your own")
    
    # Download pre-trained embeddings (this is large, ~100MB)
    print("\nTo use pre-trained embeddings, uncomment the code below:")
    print("# Load pre-trained GloVe embeddings")
    print("# glove_vectors = api.load('glove-wiki-gigaword-100')")
    
    # Example of how to use pre-trained embeddings
    """
    # Uncomment this section to use pre-trained embeddings
    
    print("Loading pre-trained GloVe embeddings...")
    glove_vectors = api.load('glove-wiki-gigaword-100')
    
    def document_to_glove_vector(tokens, glove_model):
        vectors = []
        for word in tokens:
            if word in glove_model:
                vectors.append(glove_model[word])
        
        if len(vectors) == 0:
            return np.zeros(100)  # GloVe uses 100 dimensions
        
        return np.mean(vectors, axis=0)
    
    # Convert using GloVe
    X_train_glove = np.array([document_to_glove_vector(tokens, glove_vectors) 
                             for tokens in unique_tokens])
    X_test_glove = np.array([document_to_glove_vector(tokens, glove_vectors) 
                            for tokens in test_tokens])
    
    # Scale and train
    X_train_glove_scaled = scaler.fit_transform(X_train_glove)
    X_test_glove_scaled = scaler.transform(X_test_glove)
    
    classifier_glove = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    )
    classifier_glove.fit(X_train_glove_scaled, label_matrix)
    
    # Predict and save
    predictions_glove = classifier_glove.predict(X_test_glove_scaled)
    # ... format and save results ...
    """
    
    print("\n" + "="*60)
    print("ADVANTAGES OF EMBEDDINGS:")
    print("="*60)
    print("✓ Understands that 'fraction' and 'ratio' are related")
    print("✓ Handles synonyms automatically")
    print("✓ Can find patterns even with different words")
    print("✓ Works well with less training data")
    
    print("\nDISADVANTAGES:")
    print("✗ Loses word order information")
    print("✗ Requires more computation")
    print("✗ Harder to interpret results")

if __name__ == "__main__":
    embedding_approach()