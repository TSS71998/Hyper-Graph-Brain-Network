import torch
import numpy as np
import datetime
from sklearn.metrics import f1_score, precision_score, confusion_matrix
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def print_with_timestamp(message):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp}\t{message}")


def train_hyper(args, device):
    from dataset_hyper import Dataset_ADNI_Hyper
    from models_hyper import BrainCHEF_HOC, HyperTokenTransformer

    dataset = Dataset_ADNI_Hyper(
        data_dir    = 'C:/Users/Thesu/OneDrive/Desktop/Research/data/ADNI/AAL90',
        label_file  = 'C:/Users/Thesu/OneDrive/Desktop/Research/data/ADNI/label-2cls_new.csv',
        num_classes = args.num_classes,
        k_neighbors = 12,
        K_bins      = 5,
        roi2net_file=args.roi2net, 
        window      =args.window,
        edge_density=args.edge_density, 
        sca_tau     =args.sca_tau
    )

    skf       = StratifiedKFold(
        n_splits=args.n_folds, random_state=args.seed, shuffle=True)
    generator = torch.Generator().manual_seed(args.seed)

    all_fold_metrics = {
        'accuracy': [], 'precision': [], 'f1': [], 'conf_matrix': []
    }

    print_with_timestamp(
        f"Yeo7 Sequential Transformer | seed={args.seed} | "
        f"hidden={args.hidden_dim} | layers={args.n_layers} | "
        f"dropout={args.dropout}")

    for fold, (train_index, test_index) in enumerate(
            skf.split(dataset.data, dataset.labels)):

        print_with_timestamp(f"Fold {fold+1}/{args.n_folds}")

        train_data   = Subset(dataset, train_index)
        val_data     = Subset(dataset, test_index)
        train_labels = [dataset[i].y.item() for i in train_index]
        val_labels   = [dataset[i].y.item() for i in test_index]

        print_with_timestamp(
            f"Train dist: {np.bincount(train_labels)} | "
            f"Val dist:   {np.bincount(val_labels)}")

        # Class weights
        counts = np.bincount(train_labels)
        class_weights = torch.tensor(
            len(train_labels) / (len(counts) * counts),
            dtype=torch.float32).to(device)

        criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=0.1)

        train_loader = DataLoader(
            train_data, batch_size=args.batch_size,
            shuffle=True, generator=generator)
        val_loader = DataLoader(
            val_data, batch_size=args.batch_size,
            shuffle=False)

        f_dim, n_sys = dataset[0].hyperedge_attr.size(1), dataset[0].sca_attr.size(1)
        if args.model == 'transformer':
            model = HyperTokenTransformer(dataset[0].x.size(1), f_dim, n_sys, d=args.hidden_dim,
                                          n_layers=args.n_layers, dropout=args.dropout,
                                          n_classes=args.num_classes).to(device)
        else:
            model = BrainCHEF_HOC(dataset[0].x.size(1), f_dim, n_sys, args.hidden_dim,
                                  args.n_layers, args.num_classes, args.dropout,
                                  alpha_mode=args.alpha_mode).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate, weight_decay=2e-3)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=30, T_mult=2, eta_min=1e-6)

        best_val_loss      = float('inf')
        best_val_acc       = 0
        best_val_f1        = 0
        best_val_precision = 0
        patience_counter   = 0
        best_conf_matrix   = None

        for epoch in range(args.epochs):
            # ── Train ──
            model.train()
            train_loss, correct, total = 0, 0, 0

            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()

                # Mixup
                lam = np.random.beta(0.4, 0.4) \
                    if data.num_graphs > 1 else 1.0
                logits, _ = model(data)
                loss = criterion(logits, data.y)

                if lam < 1.0 and data.num_graphs > 1:
                    idx       = torch.randperm(
                        data.num_graphs, device=device)
                    targets_b = data.y[idx]
                    loss = (lam * criterion(logits, data.y) +
                            (1-lam) * criterion(logits, targets_b))
                else:
                    loss = criterion(logits, data.y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                correct    += (logits.argmax(1)==data.y).sum().item()
                total      += data.y.size(0)

            scheduler.step(epoch)
            train_acc = correct / total

            # ── Validate ──
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            val_preds, val_labels_list = [], []

            with torch.no_grad():
                for data in val_loader:
                    data      = data.to(device)
                    logits, _ = model(data)
                    loss      = criterion(logits, data.y)
                    val_loss    += loss.item()
                    preds        = logits.argmax(1)
                    val_correct += (preds==data.y).sum().item()
                    val_total   += data.y.size(0)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels_list.extend(data.y.cpu().numpy())

            val_loss /= len(val_loader)
            val_acc   = val_correct / val_total
            val_f1    = f1_score(
                val_labels_list, val_preds,
                average='weighted', zero_division=0)
            val_prec  = precision_score(
                val_labels_list, val_preds,
                average='weighted', zero_division=0)
            conf_mat  = confusion_matrix(val_labels_list, val_preds)

            if (epoch+1) % 10 == 0:
                print_with_timestamp(
                    f"Ep {epoch+1}/{args.epochs} | "
                    f"TrLoss:{train_loss/len(train_loader):.4f} "
                    f"TrAcc:{train_acc:.4f} | "
                    f"VlLoss:{val_loss:.4f} VlAcc:{val_acc:.4f} "
                    f"F1:{val_f1:.4f}")

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print_with_timestamp(
                        f"Early stopping at epoch {epoch+1}")
                    break

            if val_acc > best_val_acc:
                best_val_acc       = val_acc
                best_val_f1        = val_f1
                best_val_precision = val_prec
                best_conf_matrix   = conf_mat

        cm_str = np.array2string(
            best_conf_matrix, separator=' ',
            max_line_width=np.inf).replace('\n', ' ')
        print_with_timestamp(
            f"Fold {fold+1} Best: Acc={best_val_acc:.4f} "
            f"Prec={best_val_precision:.4f} F1={best_val_f1:.4f}")
        print_with_timestamp(
            f"Fold {fold+1} Confusion Matrix: {cm_str}")

        all_fold_metrics['accuracy'].append(best_val_acc)
        all_fold_metrics['precision'].append(best_val_precision)
        all_fold_metrics['f1'].append(best_val_f1)
        all_fold_metrics['conf_matrix'].append(best_conf_matrix)

        avg_acc  = np.mean(all_fold_metrics['accuracy'])
        avg_prec = np.mean(all_fold_metrics['precision'])
        avg_f1   = np.mean(all_fold_metrics['f1'])
        print_with_timestamp(
            f"Average until Fold {fold+1}: "
            f"Acc={avg_acc:.4f} Prec={avg_prec:.4f} F1={avg_f1:.4f}")

    final_acc  = np.mean(all_fold_metrics['accuracy'])
    final_prec = np.mean(all_fold_metrics['precision'])
    final_f1   = np.mean(all_fold_metrics['f1'])

    print_with_timestamp("Training completed.")
    print_with_timestamp(
        f"Final | Acc={final_acc:.4f} "
        f"Prec={final_prec:.4f} F1={final_f1:.4f}")

    avg_conf = np.mean(all_fold_metrics['conf_matrix'], axis=0)
    cm_str   = np.array2string(
        avg_conf, separator=' ',
        max_line_width=np.inf).replace('\n', ' ')
    print_with_timestamp(f"Final Confusion Matrix: {cm_str}")
