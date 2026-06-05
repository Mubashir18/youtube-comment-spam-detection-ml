"""
Preprocessing module for SMS Spam Detection
Handles text cleaning, vectorization, and feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string

class SMSPreprocessor:
    """Text preprocessing and feature extraction for SMS messages"""
    
    def __init__(self, max_features=5000, ngram_range=(1, 2), random_state=42):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state
        self.vectorizer = None
        self.feature_names = None
        
    def clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', 'URL', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', 'EMAIL', text)
        
        # Remove phone numbers (basic pattern)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 'PHONE', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_handcrafted_features(self, texts):
        """Extract hand-crafted security-relevant features"""
        features = []
        
        for text in texts:
            feat_dict = {
                'has_url': 1 if 'URL' in text or 'http' in text.lower() else 0,
                'has_urgent_words': 1 if any(w in text.lower() for w in 
                                             ['urgent', 'urgent', 'immediately', 'now', 'asap', 'click', 'act']) else 0,
                'has_currency': 1 if any(c in text for c in ['$', '£', '€', '¥', 'won', 'prize', 'free']) else 0,
                'has_phone': 1 if 'PHONE' in text else 0,
                'has_email': 1 if 'EMAIL' in text else 0,
                'text_length': len(text),
                'word_count': len(text.split()),
                'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
                'digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1),
            }
            features.append(feat_dict)
        
        return pd.DataFrame(features)
    
    def fit(self, texts):
        """Fit the vectorizer on training data"""
        cleaned_texts = [self.clean_text(t) for t in texts]
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        self.vectorizer.fit(cleaned_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self
    
    def transform(self, texts):
        """Transform texts to TF-IDF vectors"""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        cleaned_texts = [self.clean_text(t) for t in texts]
        return self.vectorizer.transform(cleaned_texts)
    
    def fit_transform(self, texts):
        """Fit and transform in one step"""
        self.fit(texts)
        return self.transform(texts)
    
    def get_top_features(self, n=20):
        """Get top n feature names"""
        if self.feature_names is None:
            return []
        return list(self.feature_names[:min(n, len(self.feature_names))])


def load_and_split_data(filepath, test_size=0.2, random_state=42, stratify=True):
    """
    Load SMS data and split into train/test sets with no leakage
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file with 'label' and 'message' columns
    test_size : float
        Proportion of data for test set
    random_state : int
        Random seed for reproducibility
    stratify : bool
        Whether to use stratified split (respect class balance)
    
    Returns:
    --------
    dict with X_train, X_test, y_train, y_test (text messages and labels)
    """
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(filepath)
    
    # Verify data quality
    print(f"\nData Loading Report:")
    print(f"  Total records: {len(df)}")
    print(f"  Missing labels: {df['label'].isna().sum()}")
    print(f"  Missing messages: {df['message'].isna().sum()}")
    print(f"  Duplicates: {df.duplicated().sum()}")
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates()
    removed = initial_count - len(df)
    if removed > 0:
        print(f"  Removed {removed} duplicates")
    
    # Remove rows with missing values
    df = df.dropna()
    
    # Convert labels to binary (0=ham, 1=spam)
    df['label'] = (df['label'] == 'spam').astype(int)
    
    # Class distribution
    print(f"\nClass Distribution:")
    print(f"  Ham (0): {(df['label'] == 0).sum()} ({100 * (df['label'] == 0).mean():.2f}%)")
    print(f"  Spam (1): {(df['label'] == 1).sum()} ({100 * (df['label'] == 1).mean():.2f}%)")
    
    # Stratified split
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            df['message'],
            df['label'],
            test_size=test_size,
            random_state=random_state,
            stratify=df['label']
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            df['message'],
            df['label'],
            test_size=test_size,
            random_state=random_state
        )
    
    print(f"\nTrain/Test Split:")
    print(f"  Train size: {len(X_train)}")
    print(f"  Test size: {len(X_test)}")
    print(f"  Train spam ratio: {y_train.mean():.2%}")
    print(f"  Test spam ratio: {y_test.mean():.2%}")
    
    return {
        'X_train': X_train.reset_index(drop=True),
        'X_test': X_test.reset_index(drop=True),
        'y_train': y_train.reset_index(drop=True),
        'y_test': y_test.reset_index(drop=True)
    }
