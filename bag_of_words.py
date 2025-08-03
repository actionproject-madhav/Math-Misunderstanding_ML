"""
Approach 1: Bag of Words (Simplest Approach)
This counts how many times each word appears in the text
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

def bag_of_words_approach():
    print("="*60)
    print("BAG OF WORDS APPROACH - Simplest Method")
    print("This just counts how often each word appears")
    print("="*60)
    
    # Step 1: Load the data
    print("\nStep 1: Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Step 2: Combine all text into one column
    print("\nStep 2: Combining text columns...")
    # We combine question + answer + explanation into one text
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
    
    # Step 3: Create labels
    print("\nStep 3: Creating labels...")
    train_df['label'] = train_df['Category'] + ':' + train_df['Misconception']
    all_labels = sorted(train_df['label'].unique())
    print(f"Found {len(all_labels)} different types of misconceptions")
    
    # Step 4: Handle multiple labels per student
    print("\nStep 4: Organizing data (each student can have multiple mistakes)...")
    # Get one row per student
    unique_students = train_df.drop_duplicates(subset=['row_id']).sort_values('row_id')
    
    # Create a matrix where each row is a student, each column is a possible label
    label_matrix = np.zeros((len(unique_students), len(all_labels)))
    
    # Fill in which labels each student has
    for idx, student in unique_students.iterrows():
        student_labels = train_df[train_df['row_id'] == student['row_id']]['label'].tolist()
        for label in student_labels:
            label_idx = all_labels.index(label)
            row_idx = unique_students.index.get_loc(idx)
            label_matrix[row_idx, label_idx] = 1
    
    # Step 5: Convert text to numbers using Bag of Words
    print("\nStep 5: Converting text to word counts...")
    print("Example: 'the cat sat on the mat' -> the:2, cat:1, sat:1, on:1, mat:1")
    
    # Create the word counter
    vectorizer = CountVectorizer(
        max_features=5000,  # Only use top 5000 most common words
        stop_words='english',  # Ignore common words like 'the', 'is', 'at'
        lowercase=True  # Convert everything to lowercase
    )
    
    # Convert training text to word counts
    X_train = vectorizer.fit_transform(unique_students['all_text'])
    X_test = vectorizer.transform(test_df['all_text'])
    
    print(f"Created {X_train.shape[1]} features (unique words)")
    
    # Step 6: Train a simple classifier
    print("\nStep 6: Training Naive Bayes classifier...")
    print("This learns which words are associated with which misconceptions")
    
    # Use Naive Bayes - good for word counts
    base_classifier = MultinomialNB()
    classifier = MultiOutputClassifier(base_classifier)
    classifier.fit(X_train, label_matrix)
    
    # Step 7: Make predictions
    print("\nStep 7: Making predictions...")
    predictions = classifier.predict(X_test)
    
    # Step 8: Format results
    print("\nStep 8: Formatting results...")
    results = []
    for i, test_row in test_df.iterrows():
        # Get all predicted labels for this student
        predicted_indices = np.where(predictions[i] == 1)[0]
        
        if len(predicted_indices) == 0:
            predicted_labels = ["True_Correct:NA"]  # Default if no predictions
        else:
            predicted_labels = [all_labels[idx] for idx in predicted_indices]
        
        results.append({
            'row_id': test_row['row_id'],
            'Category:Misconception': ' '.join(sorted(predicted_labels))
        })
    
    # Save results
    submission = pd.DataFrame(results)
    submission.to_csv('submission_bag_of_words.csv', index=False)
    print("\nResults saved to 'submission_bag_of_words.csv'")
    
    # Show example predictions
    print("\nExample predictions:")
    print(submission.head())
    
    # Show which words are most important for a specific misconception
    print("\n" + "="*60)
    print("INTERPRETATION: Most important words for each misconception")
    print("="*60)
    
    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # For each classifier, show top words
    for i, label in enumerate(all_labels[:3]):  # Show first 3 labels
        if hasattr(classifier.estimators_[i], 'feature_log_prob_'):
            # Get word importance scores
            scores = classifier.estimators_[i].feature_log_prob_[1]
            top_indices = np.argsort(scores)[-10:]  # Top 10 words
            top_words = [feature_names[idx] for idx in top_indices]
            print(f"\n{label}:")
            print(f"Important words: {', '.join(top_words)}")

if __name__ == "__main__":
    bag_of_words_approach()