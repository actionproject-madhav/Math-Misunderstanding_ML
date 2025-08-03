"""
Approach 2: TF-IDF (Term Frequency-Inverse Document Frequency)
This is smarter than bag of words - it gives more weight to rare, important words
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def tfidf_approach():
    print("="*60)
    print("TF-IDF APPROACH - Smart Word Importance")
    print("Gives higher scores to rare but meaningful words")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Combine text
    print("\nCombining text columns...")
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
    
    # Create labels
    train_df['label'] = train_df['Category'] + ':' + train_df['Misconception']
    train_df = train_df.dropna(subset=['label'])

    all_labels = sorted(train_df['label'].astype(str).unique())
    
    # Get unique students and create label matrix
    unique_students = train_df.drop_duplicates(subset=['row_id']).sort_values('row_id')
    label_matrix = np.zeros((len(unique_students), len(all_labels)))
    
    for idx, student in unique_students.iterrows():
        student_labels = train_df[train_df['row_id'] == student['row_id']]['label'].tolist()
        for label in student_labels:
            label_idx = all_labels.index(label)
            row_idx = unique_students.index.get_loc(idx)
            label_matrix[row_idx, label_idx] = 1
    
    print("\n" + "="*60)
    print("HOW TF-IDF WORKS:")
    print("="*60)
    print("1. TF (Term Frequency): How often a word appears in a document")
    print("   - 'fraction' appears 5 times in a math explanation = high TF")
    print("\n2. IDF (Inverse Document Frequency): How rare a word is overall")
    print("   - 'the' appears in every document = low IDF (not important)")
    print("   - 'denominator' appears in few documents = high IDF (important!)")
    print("\n3. TF-IDF = TF × IDF")
    print("   - High score = word is frequent in this document but rare overall")
    print("   - This finds the most meaningful words!")
    
    # Create TF-IDF vectorizer
    print("\n" + "="*60)
    print("Creating TF-IDF features...")
    
    tfidf = TfidfVectorizer(
        max_features=8000,  # Use top 8000 features
        ngram_range=(1, 2),  # Use single words and word pairs
        min_df=2,  # Word must appear in at least 2 documents
        max_df=0.95,  # Ignore words that appear in >95% of documents
        stop_words='english',
        sublinear_tf=True  # Use logarithmic term frequency
    )
    
    # Transform text to TF-IDF features
    X_train = tfidf.fit_transform(unique_students['all_text'])
    X_test = tfidf.transform(test_df['all_text'])
    
    print(f"Created {X_train.shape[1]} TF-IDF features")
    
    # Show example of most important words
    feature_names = tfidf.get_feature_names_out()
    print("\nExample of high TF-IDF words (most meaningful):")
    # Get average TF-IDF scores across all documents
    avg_tfidf = X_train.mean(axis=0).A1
    top_indices = np.argsort(avg_tfidf)[-20:]
    top_words = [feature_names[i] for i in top_indices]
    print(", ".join(top_words))
    
    # Train classifier
    print("\n" + "="*60)
    print("Training Random Forest classifier...")
    print("This learns patterns between TF-IDF features and misconceptions")
    
    classifier = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
    )
    classifier.fit(X_train, label_matrix)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = classifier.predict(X_test)
    
    # Also try with Logistic Regression for comparison
    print("\nAlso training Logistic Regression for comparison...")
    lr_classifier = MultiOutputClassifier(
        LogisticRegression(max_iter=1000, random_state=42)
    )
    lr_classifier.fit(X_train, label_matrix)
    lr_predictions = lr_classifier.predict(X_test)
    
    # Format results for Random Forest
    results_rf = []
    results_lr = []
    
    for i, test_row in test_df.iterrows():
        # Random Forest predictions
        rf_indices = np.where(predictions[i] == 1)[0]
        if len(rf_indices) == 0:
            rf_labels = ["True_Correct:NA"]
        else:
            rf_labels = [all_labels[idx] for idx in rf_indices]
        
        results_rf.append({
            'row_id': test_row['row_id'],
            'Category:Misconception': ' '.join(sorted(rf_labels))
        })
        
        # Logistic Regression predictions
        lr_indices = np.where(lr_predictions[i] == 1)[0]
        if len(lr_indices) == 0:
            lr_labels = ["True_Correct:NA"]
        else:
            lr_labels = [all_labels[idx] for idx in lr_indices]
        
        results_lr.append({
            'row_id': test_row['row_id'],
            'Category:Misconception': ' '.join(sorted(lr_labels))
        })
    
    # Save results
    submission_rf = pd.DataFrame(results_rf)
    submission_rf.to_csv('submission_tfidf_rf.csv', index=False)
    
    submission_lr = pd.DataFrame(results_lr)
    submission_lr.to_csv('submission_tfidf_lr.csv', index=False)
    
    print("\nResults saved:")
    print("- submission_tfidf_rf.csv (Random Forest)")
    print("- submission_tfidf_lr.csv (Logistic Regression)")
    
    print("\nExample predictions (Random Forest):")
    print(submission_rf.head())
    
    # Analyze feature importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # For Random Forest, we can get feature importance
    print("\nMost important features overall (Random Forest):")
    # Average importance across all estimators
    importances = np.mean([est.feature_importances_ for est in classifier.estimators_], axis=0)
    top_feature_indices = np.argsort(importances)[-15:]
    
    print("\nTop 15 most important words/phrases:")
    for idx in reversed(top_feature_indices):
        print(f"- {feature_names[idx]}: {importances[idx]:.4f}")

if __name__ == "__main__":
    tfidf_approach()