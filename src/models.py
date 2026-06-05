"""
Models for SMS Spam Detection
Includes baseline and improved classifiers
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from typing import Tuple, Dict

class SpamDetectionBaseline:
    """Baseline model: TF-IDF + Logistic Regression"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver='lbfgs',
            n_jobs=-1
        )
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train the baseline model"""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        train_score = self.model.score(X_train, y_train)
        print(f"Baseline Model Trained")
        print(f"  Training accuracy: {train_score:.4f}")
        
        return self
    
    def predict(self, X):
        """Predict class labels"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names, n_top=20):
        """Get top features by coefficient magnitude"""
        coef = self.model.coef_[0]
        top_indices = np.argsort(np.abs(coef))[-n_top:][::-1]
        
        result = pd.DataFrame({
            'feature': [feature_names[i] for i in top_indices],
            'coefficient': coef[top_indices]
        })
        
        return result


class SpamDetectionImproved:
    """Improved model with class balancing and better calibration"""
    
    def __init__(self, random_state=42, use_smote=True):
        self.random_state = random_state
        self.use_smote = use_smote
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver='lbfgs',
            class_weight='balanced',  # Handle class imbalance
            n_jobs=-1
        )
        self.smote = None
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train the improved model with optional SMOTE"""
        
        X_train_processed = X_train
        y_train_processed = y_train
        
        if self.use_smote:
            # Apply SMOTE for class balancing
            self.smote = SMOTE(random_state=self.random_state, k_neighbors=5)
            X_train_processed, y_train_processed = self.smote.fit_resample(X_train, y_train)
            print(f"SMOTE Applied: {X_train.shape[0]} -> {X_train_processed.shape[0]} samples")
        
        self.model.fit(X_train_processed, y_train_processed)
        self.is_trained = True
        
        train_score = self.model.score(X_train_processed, y_train_processed)
        print(f"Improved Model Trained (class_weight='balanced', SMOTE={self.use_smote})")
        print(f"  Training accuracy: {train_score:.4f}")
        
        return self
    
    def predict(self, X, threshold=0.5):
        """Predict class labels with custom threshold"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        proba = self.model.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names, n_top=20):
        """Get top features by coefficient magnitude"""
        coef = self.model.coef_[0]
        top_indices = np.argsort(np.abs(coef))[-n_top:][::-1]
        
        result = pd.DataFrame({
            'feature': [feature_names[i] for i in top_indices],
            'coefficient': coef[top_indices]
        })
        
        return result


class ThresholdOptimizer:
    """Find optimal decision threshold based on PR curve"""
    
    @staticmethod
    def find_optimal_threshold(y_true, y_proba, min_recall=0.90, prefer_f1=True):
        """
        Find threshold that achieves target recall while maximizing precision/F1
        
        Parameters:
        -----------
        y_true : array
            True labels (0=ham, 1=spam)
        y_proba : array
            Predicted probabilities for spam class
        min_recall : float
            Minimum recall threshold (default: 90% to catch most spam)
        prefer_f1 : bool
            If True, maximize F1 score among valid thresholds (balanced metric)
            If False, maximize precision (more conservative)
        
        Returns:
        --------
        dict with optimal threshold and metrics
        """
        from sklearn.metrics import precision_recall_curve, recall_score, precision_score, f1_score
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Find all thresholds where recall >= min_recall
        # Note: precision and recall arrays have len = len(thresholds) + 1
        # So precision[i] corresponds to threshold[i]
        valid_mask = recall[:-1] >= min_recall  # Use :-1 to match thresholds length
        
        if not valid_mask.any():
            # If no threshold achieves target recall, use default 0.5
            optimal_threshold = 0.5
        else:
            # Get indices of valid thresholds
            valid_indices = np.where(valid_mask)[0]
            
            if prefer_f1:
                # Among valid thresholds, maximize F1 score
                f1_scores = []
                for idx in valid_indices:
                    y_pred = (y_proba >= thresholds[idx]).astype(int)
                    f1 = f1_score(y_true, y_pred)
                    f1_scores.append(f1)
                
                # Select threshold with max F1
                best_idx = valid_indices[np.argmax(f1_scores)]
                optimal_threshold = thresholds[best_idx]
            else:
                # Among valid thresholds, maximize precision
                valid_precisions = precision[:-1][valid_mask]
                best_idx_in_valid = np.argmax(valid_precisions)
                best_idx = valid_indices[best_idx_in_valid]
                optimal_threshold = thresholds[best_idx]
        
        # Calculate metrics at optimal threshold
        y_pred = (y_proba >= optimal_threshold).astype(int)
        
        return {
            'threshold': optimal_threshold,
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'accuracy': (y_pred == y_true).mean()
        }
    
    @staticmethod
    def plot_pr_curve(y_true, y_proba, ax=None):
        """Plot precision-recall curve"""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        import matplotlib.pyplot as plt
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(recall, precision, lw=2, label=f'PR Curve (AP={ap:.3f})')
        ax.axhline(y=0.5, color='r', linestyle='--', label='Precision=0.5')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(alpha=0.3)
        
        return ax
