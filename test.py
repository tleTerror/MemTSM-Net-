def test(test_loader, model, device, threshold=0.5):
    model.eval()
    all_scores = []
    all_errors = []
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for features, targets, _ in tqdm(test_loader):
            features = features.to(device)
            scores, recon, _ = model(features)

            # Calculate reconstruction error
            rec_target = features[:, :, 8:24]
            errors = torch.mean((recon - rec_target)**2, dim=(1,2))

            # Store results
            all_scores.extend(scores.sigmoid().cpu().numpy())
            all_errors.extend(errors.cpu().numpy())            
            all_targets.extend(targets.numpy())

    # Normalize both score and error to [0,1]
    scores = np.array(all_scores)
    errors = np.array(all_errors)
    
    # Min-max normalize both
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
    errors = (errors - errors.min()) / (errors.max() - errors.min() + 1e-6)
    
    # Dynamic fusion: emphasize classification when error is low, reconstruction when error is high
    combined_scores = (0.6 + 0.2 * errors) * scores + (0.4 - 0.2 * errors) * errors
    
    
    # Use Youden's J statistic for threshold selection (better than static median)
    fpr, tpr, thresholds = roc_curve(all_targets, combined_scores)
    youden_idx = np.argmax(tpr - fpr)
    threshold = thresholds[youden_idx]

    preds = (combined_scores >= threshold).astype(int)

    cm = confusion_matrix(all_targets, preds)

    roc_auc = roc_auc_score(all_targets, combined_scores)
    precision, recall, _ = precision_recall_curve(all_targets, combined_scores)
    pr_auc = auc(recall, precision)

    acc = accuracy_score(all_targets, preds)
    prec = precision_score(all_targets, preds, zero_division=0)
    rec = recall_score(all_targets, preds, zero_division=0)
    f1 = f1_score(all_targets, preds, zero_division=0)

    return roc_auc, pr_auc, acc, prec, rec, f1, cm, combined_scores, all_targets
