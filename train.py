def train(train_loader, model, criterion, optimizer, scheduler, device, epoch):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    for i, (features, targets_float, targets_long) in enumerate(pbar):
        features = features.to(device)
        targets = targets_float.to(device)

        # === use smoothed targets for loss only ===
        smoothed_targets = 0.9 * targets + 0.05

    
        scores, recon, encoded = model(features)
        loss = criterion(scores, recon, encoded, smoothed_targets, features)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step_update(epoch * len(train_loader) + i)

        
        with torch.no_grad():
            probs = torch.sigmoid(scores)
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())  

        running_loss += loss.item()
        pbar.set_postfix({'loss': running_loss / (i + 1)})

    # Compute proper binary classification AUC
    train_auc = roc_auc_score(all_targets, all_preds)
    return running_loss / len(train_loader), train_auc, 0.0
