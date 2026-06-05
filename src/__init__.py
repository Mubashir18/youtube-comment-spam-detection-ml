"""
SMS Spam Detection Package
Track B1: Phishing/Spam Detection for Consumer Protection
"""

__version__ = "1.0.0"
__author__ = "ML Security Course"
__description__ = "Production-ready SMS phishing detection system"

from .preprocessing import SMSPreprocessor, load_and_split_data
from .models import SpamDetectionBaseline, SpamDetectionImproved, ThresholdOptimizer
from .evaluation import comprehensive_evaluation, plot_confusion_matrix, error_analysis_summary

__all__ = [
    'SMSPreprocessor',
    'load_and_split_data',
    'SpamDetectionBaseline',
    'SpamDetectionImproved',
    'ThresholdOptimizer',
    'comprehensive_evaluation',
    'plot_confusion_matrix',
    'error_analysis_summary',
]
