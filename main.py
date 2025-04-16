def main():
    
    set_seed(156) 
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    train_dataset = VideoFeatureDataset(os.path.join(args.feature_dir, 'xfeat_features_train'), flag="Train")
    test_dataset = VideoFeatureDataset(os.path.join(args.feature_dir, 'xfeat_features_test'), flag="Test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Train class distribution: {np.bincount(train_dataset.labels)}")
    print(f"Test class distribution: {np.bincount(test_dataset.labels)}")

    model = TSMAE(in_channels=args.in_channels).to(device)
    criterion = Loss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)

    # Modified scheduler
    num_steps = len(train_loader)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.max_epoch * len(train_loader),
        lr_min=1e-6,
        warmup_t=5 * len(train_loader),
        warmup_lr_init=1e-5,
    )

    # Training metrics tracking
    train_info = {"epoch": [], "train_loss": [], "train_AUC": [], "train_PR": []}
    test_info = {"epoch": [], "test_AUC": [], "test_PR": [], "accuracy": [], "precision": [], "recall": [], "f1": []}

    best_auc = 0.0

    # Training loop
    for epoch in range(args.max_epoch):
        # Train
        train_loss, train_auc, train_pr_auc = train(train_loader, model, criterion, optimizer, scheduler, device, epoch)

        # Test
        test_auc, test_pr_auc, accuracy, precision, recall, f1, _, _, _ = test(test_loader, model, device)

        # Log info
        train_info["epoch"].append(epoch + 1)
        train_info["train_loss"].append(train_loss)
        train_info["train_AUC"].append(train_auc)
        train_info["train_PR"].append(train_pr_auc)

        test_info["epoch"].append(epoch + 1)
        test_info["test_AUC"].append(test_auc)
        test_info["test_PR"].append(test_pr_auc)
        test_info["accuracy"].append(accuracy)
        test_info["precision"].append(precision)
        test_info["recall"].append(recall)
        test_info["f1"].append(f1)

        print(f"Epoch {epoch+1}/{args.max_epoch}")
        print(f"Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, PR-AUC: {train_pr_auc:.4f}")
        print(f"Test - AUC: {test_auc:.4f}, PR-AUC: {test_pr_auc:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}")
        print("=" * 50)

        # Save best model
        if test_auc > best_auc:
            best_auc = test_auc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Saved new best model with AUC: {best_auc:.4f}")

    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_info["epoch"], train_info["train_AUC"], 'b-', label='Train AUC')
    plt.plot(test_info["epoch"], test_info["test_AUC"], 'r-', label='Test AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title('ROC AUC')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_info["epoch"], train_info["train_PR"], 'b-', label='Train PR-AUC')
    plt.plot(test_info["epoch"], test_info["test_PR"], 'r-', label='Test PR-AUC')
    plt.xlabel('Epochs')
    plt.ylabel('PR-AUC')
    plt.title('PR AUC')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_metrics.png'))
    plt.show()

    print("Final evaluation on test set...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))

    test_auc, test_pr_auc, accuracy, precision, recall, f1, cm, all_scores, all_targets = test(test_loader, model, device)

    print(f"Final Test Results:")
    print(f"AUC: {test_auc:.4f}")
    print(f"PR-AUC: {test_pr_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plot confusion matrix
    if cm is not None and cm.size > 0:
        plot_confusion_matrix(cm, os.path.join(args.save_dir, 'confusion_matrix.png'))
    else:
        print("Warning: Could not plot confusion matrix due to invalid data")

    # Calculate and plot ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(all_targets, all_scores)
    plot_roc_curve(fpr, tpr, test_auc, os.path.join(args.save_dir, 'roc_curve.png'))

    # Calculate and plot PR curve
    precision_arr, recall_arr, _ = precision_recall_curve(all_targets, all_scores)
    plot_pr_curve(precision_arr, recall_arr, test_pr_auc, os.path.join(args.save_dir, 'pr_curve.png'))

main()
