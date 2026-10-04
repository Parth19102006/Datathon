import numpy as np, json
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

def apply_log_offsets(proba, offsets):
    eps = 1e-15
    logp = np.log(proba + eps)
    logp += offsets
    expp = np.exp(logp - np.max(logp, axis=1, keepdims=True))
    return expp / np.sum(expp, axis=1, keepdims=True)

def main():
    val_proba = np.load('lgbm_val_proba.npy')
    y_true = np.load('y_val.npy')
    offsets = np.array([-0.09375, 0.5, -0.25, -0.03125, -0.0625, 0.0, 0.25])
    # Apply offsets
    mod_proba = apply_log_offsets(val_proba, offsets)
    preds = np.argmax(mod_proba, axis=1)
    macro_f1 = f1_score(y_true, preds, average='macro')
    precision, recall, f1, support = precision_recall_fscore_support(y_true, preds, labels=np.arange(7), zero_division=0)
    cm = confusion_matrix(y_true, preds, labels=np.arange(7))
    # most frequent misclass pairs
    mispair = {}
    for i in range(7):
        for j in range(7):
            if i != j:
                cnt = cm[i, j]
                if cnt > 0:
                    mispair[(i, j)] = cnt
    top_mispairs = sorted(mispair.items(), key=lambda x: x[1], reverse=True)[:5]
    # top-1, top-2 probabilities and margin
    top1 = np.max(mod_proba, axis=1)
    top2 = np.partition(mod_proba, -2, axis=1)[:, -2]
    margin = top1 - top2
    report = {
        'macro_f1': macro_f1,
        'per_class': {i+1:{'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support':int(support[i])} for i in range(7)},
        'confusion_matrix': cm.tolist(),
        'most_frequent_misclass_pairs': [((i+1,j+1),cnt) for (i,j),cnt in top_mispairs],
        'avg_top1_prob': float(np.mean(top1)),
        'avg_top2_prob': float(np.mean(top2)),
        'avg_margin': float(np.mean(margin))
    }
    with open('phase1_analysis_report.json','w') as f:
        json.dump(report, f, indent=2)
    # also markdown summary
    md = f"# Phase 1 Validation Error Analysis\n\n**Macro-F1:** {macro_f1:.6f}\n\n## Per-Class Metrics\n| Class | Precision | Recall | F1 | Support |\n|------|-----------|--------|----|---------|\n"
    for i in range(7):
        md += f"| {i+1} | {precision[i]:.4f} | {recall[i]:.4f} | {f1[i]:.4f} | {support[i]} |\n"
    md += "\n## Most Frequent Misclassifications (true → pred)\n"
    for (i,j),cnt in top_mispairs:
        md += f"- Class {i+1} → Class {j+1}: {cnt} instances\n"
    md += f"\n**Avg Top-1 probability:** {np.mean(top1):.4f}\n"
    md += f"**Avg Top-2 probability:** {np.mean(top2):.4f}\n"
    md += f"**Avg prediction margin:** {np.mean(margin):.4f}\n"
    with open('phase1_analysis_report.md','w') as f:
        f.write(md)

if __name__ == '__main__':
    main()
