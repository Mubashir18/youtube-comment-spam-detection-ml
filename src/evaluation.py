"""
Evaluation module for spam detection
Metrics, visualizations, and error analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score, balanced_accuracy_score, accuracy_score,
    precision_score, recall_score
)


def comprehensive_evaluation(y_true, y_pred, y_proba=None, model_name="Model"):
    """
    Generate comprehensive evaluation metrics
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    y_proba : array, optional
        Predicted probabilities for positive class
    model_name : str
        Name of the model
    
    Returns:
    --------
    dict with all metrics
    """
    
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name="Model", ax=None, normalize=False):
    """Plot confusion matrix"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'],
                cbar_kws={'label': 'Count'})
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(f'Confusion Matrix - {model_name}')
    
    # Add text summary
    tn, fp, fn, tp = cm.ravel()
    ax.text(0.5, -0.15, f'TN={tn}, FP={fp}, FN={fn}, TP={tp}',
            transform=ax.transAxes, ha='center', fontsize=10)
    
    return ax


def plot_roc_curve(y_true, y_proba, model_name="Model", ax=None):
    """Plot ROC curve"""
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    return ax


def get_false_positives_negatives(X, y_true, y_pred, texts=None):
    """
    Extract false positive and false negative examples
    
    Parameters:
    -----------
    X : array-like
        Feature vectors
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    texts : array, optional
        Original text messages for interpretability
    
    Returns:
    --------
    dict with false positive and negative examples
    """
    
    fn_mask = (y_true == 1) & (y_pred == 0)  # Spam predicted as ham
    fp_mask = (y_true == 0) & (y_pred == 1)  # Ham predicted as spam
    
    result = {
        'false_negatives': {
            'indices': np.where(fn_mask)[0],
            'count': fn_mask.sum(),
            'texts': texts[fn_mask] if texts is not None else None
        },
        'false_positives': {
            'indices': np.where(fp_mask)[0],
            'count': fp_mask.sum(),
            'texts': texts[fp_mask] if texts is not None else None
        }
    }
    
    return result


def error_analysis_summary(y_true, y_pred, y_proba, X_texts, model_name="Model"):
    """
    Generate comprehensive error analysis
    """
    
    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    
    analysis = {
        'model': model_name,
        'false_negatives': {
            'count': fn_mask.sum(),
            'percentage': 100 * fn_mask.sum() / (y_true == 1).sum() if (y_true == 1).sum() > 0 else 0,
            'examples': []
        },
        'false_positives': {
            'count': fp_mask.sum(),
            'percentage': 100 * fp_mask.sum() / (y_true == 0).sum() if (y_true == 0).sum() > 0 else 0,
            'examples': []
        }
    }
    
    # Get top false negatives (spam messages model was confident about being ham)
    fn_indices = np.where(fn_mask)[0]
    if len(fn_indices) > 0:
        fn_scores = y_proba[fn_indices]
        # Sort by confidence in wrong prediction
        sorted_fn = fn_indices[np.argsort(fn_scores)][:3]  # Top 3
        for idx in sorted_fn:
            analysis['false_negatives']['examples'].append({
                'text': X_texts[idx],
                'predicted_proba_spam': y_proba[idx],
                'true_label': 'SPAM'
            })
    
    # Get top false positives (ham messages model was confident about being spam)
    fp_indices = np.where(fp_mask)[0]
    if len(fp_indices) > 0:
        fp_scores = y_proba[fp_indices]
        # Sort by confidence in wrong prediction
        sorted_fp = fp_indices[np.argsort(1 - fp_scores)][:3]  # Top 3
        for idx in sorted_fp:
            analysis['false_positives']['examples'].append({
                'text': X_texts[idx],
                'predicted_proba_spam': y_proba[idx],
                'true_label': 'HAM'
            })
    
    return analysis


def metrics_comparison_table(*results):
    """
    Create comparison table of multiple model results
    
    Parameters:
    -----------
    results : dicts
        Each dict should have metrics including 'model', 'accuracy', 'precision', etc.
    
    Returns:
    --------
    pandas DataFrame
    """
    
    df = pd.DataFrame(results)
    
    # Round to 4 decimals for display
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].round(4)
    
    return df


def plot_metrics_comparison(results, ax=None):
    """Plot bar chart comparing metrics across models"""
    
    df = pd.DataFrame(results)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy']
    metrics = [m for m in metrics if m in df.columns]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, df[metric], width, label=metric.replace('_', ' ').title())
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df['model'])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    return ax
