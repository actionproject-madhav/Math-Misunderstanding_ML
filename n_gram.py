"""
Approach 3: N-grams (Looking at word combinations)
This looks at phrases and word sequences, not just individual words
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
import re

def ngram_approach():
    print("="*60)
    print("N-GRAM APPROACH - Understanding Word Combinations")
    print("Looks at phrases like 'common denominator' not just 'common' and 'denominator'")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Text preprocessing function
    def preprocess_text(text):
        """Clean and prepare text"""
        if pd.isna(text):
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove special math symbols but keep numbers
        text = re.sub(r'\\[a-z]+', ' ', text)  # Remove LaTeX
        text = re.sub(r'[^\w\s]', ' ', text)   # Remove punctuation
        text = re.sub(r'\s+', ' ', text)       # Multiple spaces to single
        return text.strip()
    
    # Combine and clean text
    print("\nPreparing text with preprocessing...")
    train_df['all_text'] = (
        train_df['QuestionText'].apply(preprocess_text) + ' ' +
        train_df['MC_Answer'].apply(preprocess_text) + ' ' +
        train_df['StudentExplanation'].apply(preprocess_text)
    )
    
    test_df['all_text'] = (
        test_df['QuestionText'].apply(preprocess_text) + ' ' +
        test_df['MC_Answer'].apply(preprocess_text) + ' ' +
        test_df['StudentExplanation'].apply(preprocess_text)
    )
    
    # Create labels
    train_df['label'] = train_df['Category'] + ':' + train_df['Misconception']
    all_labels = sorted(train_df['label'].unique())
    
    # Get unique students
    unique_students = train_df.drop_duplicates(subset=['row_id']).sort_values('row_id')
    label_matrix = np.zeros((len(unique_students), len(all_labels)))
    
    for idx, student in unique_students.iterrows():
        student_labels = train_df[train_df['row_id'] == student['row_id']]['label'].tolist()
        for label in student_labels:
            label_idx = all_labels.index(label)
            row_idx = unique_students.index.get_loc(idx)
            label_matrix[row_idx, label_idx] = 1
    
    print("\n" + "="*60)
    print("UNDERSTANDING N-GRAMS:")
    print("="*60)
    print("1-gram (unigram): Single words")
    print("   Example: 'the', 'fraction', 'simplified'")
    print("\n2-gram (bigram): Two-word phrases")
    print("   Example: 'common denominator', 'simplest form', 'not correct'")
    print("\n3-gram (trigram): Three-word phrases")
    print("   Example: 'lowest common denominator', 'reduce to simplest'")
    print("\nWhy this helps: 'not correct' has opposite meaning from 'correct'!")
    
    # Create different n-gram vectorizers
    print("\n" + "="*60)
    print("Creating multiple n-gram feature sets...")
    
    # 1. Character n-grams (for catching spelling patterns)
    print("\n1. Character n-grams (catches spelling patterns)...")
    char_vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),  # 3 to 5 character sequences
        max_features=3000
    )
    
    # 2. Word n-grams (for catching phrases)
    print("2. Word n-grams (catches meaningful phrases)...")
    word_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),  # 1 to 3 word sequences
        max_features=5000,
        min_df=2,
        max_df=0.95
    )
    
    # 3. Mixed approach - combine both
    print("3. Combining both approaches...")
    
    # Fit and transform
    X_train_char = char_vectorizer.fit_transform(unique_students['all_text'])
    X_test_char = char_vectorizer.transform(test_df['all_text'])
    
    X_train_word = word_vectorizer.fit_transform(unique_students['all_text'])
    X_test_word = word_vectorizer.transform(test_df['all_text'])
    
    # Combine features
    from scipy.sparse import hstack
    X_train_combined = hstack([X_train_char, X_train_word])
    X_test_combined = hstack([X_test_char, X_test_word])
    
    print(f"\nTotal features created: {X_train_combined.shape[1]}")
    print(f"- Character n-grams: {X_train_char.shape[1]}")
    print(f"- Word n-grams: {X_train_word.shape[1]}")
    
    # Show example n-grams
    print("\n" + "="*60)
    print("EXAMPLE N-GRAMS FOUND:")
    print("="*60)
    
    # Get feature names
    word_features = word_vectorizer.get_feature_names_out()
    char_features = char_vectorizer.get_feature_names_out()
    
    print("\nExample word n-grams:")
    # Find n-grams with 2 or 3 words
    bigrams = [f for f in word_features if ' ' in f][:10]
    print("2-word phrases:", bigrams[:5])
    trigrams = [f for f in word_features if f.count(' ') == 2][:5]
    print("3-word phrases:", trigrams)
    
    print("\nExample character n-grams:")
    print("Character patterns:", char_features[:10].tolist())
    
    # Train classifier
    print("\n" + "="*60)
    print("Training Gradient Boosting classifier on n-gram features...")
    
    # Use a smaller model for each label due to many features
    base_classifier = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42
    )
    
    classifier = MultiOutputClassifier(base_classifier, n_jobs=-1)
    
    # Train only on word n-grams (faster)
    print("Training on word n-grams only (for speed)...")
    classifier.fit(X_train_word, label_matrix)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = classifier.predict(X_test_word)
    
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
    submission.to_csv('submission_ngrams.csv', index=False)
    print("\nResults saved to 'submission_ngrams.csv'")
    
    print("\nExample predictions:")
    print(submission.head())
    
    # Analyze which n-grams are important
    print("\n" + "="*60)
    print("IMPORTANT N-GRAMS FOR MISCONCEPTIONS")
    print("="*60)
    
    # For gradient boosting, we can't easily get feature importance per label
    # But we can analyze which n-grams appear most in each misconception type
    
    for label in all_labels[:3]:  # First 3 labels
        print(f"\n{label}:")
        # Get texts for this label
        label_texts = train_df[train_df['label'] == label]['all_text'].tolist()
        if len(label_texts) > 0:
            # Fit a small vectorizer just for this label
            label_vec = TfidfVectorizer(ngram_range=(2, 3), max_features=10)
            try:
                label_vec.fit(label_texts)
                important_ngrams = label_vec.get_feature_names_out()
                print(f"Common phrases: {', '.join(important_ngrams[:5])}")
            except:
                print("Not enough examples for n-gram analysis")

if __name__ == "__main__":
    ngram_approach()