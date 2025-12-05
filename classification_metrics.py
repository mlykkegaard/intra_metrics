#!/usr/bin/env python3
"""
Classification Metrics Calculator for Differential Abundance Analysis
Calculates confusion matrix, precision, recall, F1-score, AUC-ROC, etc.
Also generates a volcano plot (effect_size or log2FC vs -log10 p-value),
colored by true labels with a legend.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
)
import os
import matplotlib.pyplot as plt


def load_data(results_file, labels_file):
    """Load results and ground truth labels"""
    results = pd.read_csv(results_file)
    labels = pd.read_csv(labels_file, index_col=0)
    return results, labels


def calculate_metrics(y_true, y_pred, y_score=None):
    """Calculate classification metrics"""
    metrics = {}
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics['TP'] = int(tp)
    metrics['TN'] = int(tn)
    metrics['FP'] = int(fp)
    metrics['FN'] = int(fn)
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred)
    metrics['Recall'] = recall_score(y_true, y_pred)
    metrics['Sensitivity'] = metrics['Recall']
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['F1_Score'] = f1_score(y_true, y_pred)
    metrics['FPR'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['FDR'] = fp / (fp + tp) if (fp + tp) > 0 else 0
    if y_score is not None:
        try:
            metrics['AUC_ROC'] = roc_auc_score(y_true, y_score)
            metrics['AUC_PR'] = average_precision_score(y_true, y_score)
        except:
            metrics['AUC_ROC'] = np.nan
            metrics['AUC_PR'] = np.nan
    return metrics


def plot_volcano(results, y_true, output_dir, name, fdr_threshold=0.05):
    """
    Create a volcano plot (effect_size or log2FC vs -log10 p-value),
    coloring only the true labels.
    """
    # Detect fold-change/effect size column
    fc_col = None
    for col in ['effect_size', 'logFC', 'Log2FC', 'log2FoldChange', 'FoldChange']:
        if col in results.columns:
            fc_col = col
            break
    if fc_col is None:
        print("Warning: No effect size or fold-change column found. Volcano plot skipped.")
        return

    # Detect p-value column
    p_val_col = None
    for col in ['P.Value', 'p_value', 'pval', 'p-value', 'adj.P.Val']:
        if col in results.columns:
            p_val_col = col
            break
    if p_val_col is None:
        print("Warning: No p-value column found. Volcano plot skipped.")
        return

    logfc = results[fc_col]
    neglogp = -np.log10(results[p_val_col])

    # Color by true labels instead of significance
    colors = pd.Series(y_true, index=results.index).map({1: "red", 0: "grey"})

    plt.figure(figsize=(8,6))
    plt.scatter(logfc, neglogp, c=colors, alpha=0.7)
    plt.axhline(-np.log10(fdr_threshold), color='blue', linestyle='--', linewidth=1)
    plt.xlabel(fc_col)
    plt.ylabel("-Log10 p-value")
    plt.title(f"Volcano Plot: {name} (colored by true labels)")

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='True DE', markerfacecolor='red', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Not DE', markerfacecolor='grey', markersize=8)
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    outpath = os.path.join(output_dir, f"{name}_volcano.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Volcano plot saved to: {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate classification metrics for DAA results'
    )
    parser.add_argument('--results', required=True, dest='results',
                        help='Path to method results CSV file')
    parser.add_argument('--data.true_labels_proteins', required=True, dest='labels',
                        help='Path to true labels CSV file')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory')
    parser.add_argument('--name', required=True,
                        help='Dataset/method name')
    parser.add_argument('--fdr-threshold', type=float, default=0.05,
                        help='FDR threshold for significance (default: 0.05)')
    parser.add_argument('--score-column', default='P.Value',
                        help='Column name for probability scores (default: P.Value)')
    parser.add_argument('--prediction-column', default='Significant',
                        help='Column name for binary predictions (default: Significant)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results, labels = load_data(args.results, args.labels)

    id_col = None
    for col in ['Name', 'ID', 'Protein', 'protein', 'Feature']:
        if col in results.columns:
            id_col = col
            break
    if id_col:
        results = results.set_index(id_col)

    common_features = results.index.intersection(labels.index)
    results = results.loc[common_features]
    labels = labels.loc[common_features]

    if 'label' in labels.columns:
        y_true = labels['label'].values
    elif 'is_differentially_expressed' in labels.columns:
        y_true = labels['is_differentially_expressed'].values
    else:
        y_true = labels.iloc[:, 0].values

    if args.prediction_column in results.columns:
        y_pred = results[args.prediction_column].astype(int).values
    else:
        p_val_col = None
        for col in ['P.Value', 'p_value', 'pval', 'p-value', 'adj.P.Val']:
            if col in results.columns:
                p_val_col = col
                break
        if p_val_col:
            y_pred = (results[p_val_col] < args.fdr_threshold).astype(int).values
        elif args.score_column in results.columns:
            y_pred = (results[args.score_column] < args.fdr_threshold).astype(int).values
        else:
            raise ValueError("Could not find p-value column.")

    y_score = None
    p_val_col = None
    for col in ['P.Value', 'p_value', 'pval', 'p-value', 'adj.P.Val']:
        if col in results.columns:
            p_val_col = col
            break
    if p_val_col:
        y_score = 1 - results[p_val_col].values
    elif args.score_column in results.columns:
        y_score = 1 - results[args.score_column].values

    metrics = calculate_metrics(y_true, y_pred, y_score)

    metrics_file = os.path.join(args.output_dir, f"{args.name}_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_file, index=False)

    detailed_file = os.path.join(args.output_dir, f"{args.name}_detailed.csv")
    detailed_df = results.copy()
    detailed_df['True_Label'] = y_true
    detailed_df['Predicted_Label'] = y_pred
    if y_score is not None:
        detailed_df['Prediction_Score'] = y_score
    detailed_df.to_csv(detailed_file)

    plot_volcano(results, y_true, args.output_dir, args.name, fdr_threshold=args.fdr_threshold)


if __name__ == "__main__":
    main()
