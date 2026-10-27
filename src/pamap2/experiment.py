"""Shared model fitting helpers; default entry uses frozen 70/30 and training-only CV."""
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .models import make_model
from .common import seed_all

def metrics(y, pred):
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=range(12), average='macro', zero_division=0)),
            'macro_recall': float(recall_score(y, pred, labels=range(12), average='macro', zero_division=0)),
            'supported_macro_f1': float(f1_score(y,pred,labels=np.unique(y),average='macro',zero_division=0)),
            'supported_classes': int(len(np.unique(y)))}

def feature_indices(name):
    lookup = {'rf_wrist': [0], 'rf_chest': [1], 'rf_ankle': [2],
              'rf_no_wrist': [1,2], 'rf_no_chest': [0,2], 'rf_no_ankle': [0,1]}
    return np.concatenate([np.arange(i*50, (i+1)*50) for i in lookup.get(name, [0,1,2])])

def normalization(x, features, indices):
    subset = np.asarray(x[indices], dtype=np.float32)
    mean = subset.mean(axis=(0,2), dtype=np.float64).astype(np.float32)[None,:,None]
    std = subset.std(axis=(0,2), dtype=np.float64).astype(np.float32)[None,:,None]
    fmean = features[indices].mean(0, dtype=np.float64).astype(np.float32)
    fstd = features[indices].std(0, dtype=np.float64).astype(np.float32)
    return {'mean': mean, 'std': np.maximum(std, 1e-6),
            'fmean': fmean, 'fstd': np.maximum(fstd, 1e-6)}

def make_loader(x, features, y, indices, norm, batch_size, shuffle, seed):
    tx = torch.from_numpy(((np.asarray(x[indices])-norm['mean'])/norm['std']).copy())
    tf = torch.from_numpy(((features[indices]-norm['fmean'])/norm['fstd']).copy())
    ty = torch.from_numpy(y[indices].copy())
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(tx,tf,ty), batch_size=batch_size, shuffle=shuffle,
                      generator=generator, num_workers=0)

def predict_loader(model, loader):
    model.eval()
    results = []
    with torch.inference_mode():
        for xb, fb, _ in loader:
            results.append(model(xb,fb).argmax(1).numpy())
    return np.concatenate(results)

def fit_nn(name, x, features, y, train_idx, val_idx, config, seed, epochs=None):
    seed_all(seed)
    model = make_model(name)
    norm = normalization(x,features,train_idx)
    train_loader = make_loader(x,features,y,train_idx,norm,config['batch_size'],True,seed)
    val_loader = None if val_idx is None else make_loader(x,features,y,val_idx,norm,256,False,seed)
    counts = np.bincount(y[train_idx], minlength=12)
    weights = 1 / np.sqrt(np.maximum(counts,1))
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],weight_decay=config['weight_decay'])
    history = []
    best, selected, stale = -1., 1, 0
    for epoch in range(1,(epochs or config['max_epochs'])+1):
        begin = time.monotonic()
        model.train()
        total, n = 0., 0
        for xb,fb,yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb,fb),yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            total += float(loss.detach())*len(yb)
            n += len(yb)
        row = {'epoch':epoch,'train_loss':total/n,'seconds':time.monotonic()-begin}
        if val_loader is not None:
            score = metrics(y[val_idx],predict_loader(model,val_loader))['macro_f1']
            row['validation_macro_f1'] = score
            if score > best+1e-8:
                best,selected,stale = score,epoch,0
            else:
                stale += 1
        history.append(row)
        if epoch == 1 or epoch%5 == 0:
            print(f'  {name} epoch {epoch}: loss={total/n:.4f}'+
                  (f' val F1={row["validation_macro_f1"]:.4f}' if val_loader is not None else ''),flush=True)
        if val_loader is not None and stale >= config.get('patience',6):
            break
    return model, norm, history, selected

def main():
    from .holdout import main as holdout_main
    return holdout_main()


if __name__ == '__main__':
    main()
