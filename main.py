import os
import time
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from Importdata import LoadData
from model import ASMSTCN_Reg

def compute_metrics(true_vals, pred_vals):
    mae  = mean_absolute_error(true_vals.reshape(-1), pred_vals.reshape(-1))
    rmse = math.sqrt(mean_squared_error(true_vals.reshape(-1), pred_vals.reshape(-1)))
    sig_p, sig_t = pred_vals.std(), true_vals.std()
    if sig_p >= 1e-6 and sig_t >= 1e-6:
        corr = np.mean(
            (pred_vals - pred_vals.mean()) * (true_vals - true_vals.mean())
        ) / (sig_p * sig_t)
    else:
        corr = 0.0
    return mae, rmse, corr


def train_epoch(data, model, criterion, optimizer,
                batch_size, space_loss_weight, ent_loss_weight):
    model.train()
    device = next(model.parameters()).device
    for Xb, Yb in data.get_batches(data.train[0], data.train[1], batch_size, shuffle=True):
        Xb, Yb = Xb.to(device), Yb.to(device)
        optimizer.zero_grad()
        pred, loss_space, ent_reg = model(Xb)  # 期望模型返回 (pred, loss_space, ent_reg)
        pred = pred.squeeze()
        y = Yb.squeeze()
        sup_loss = criterion(pred, y)
        loss = sup_loss + space_loss_weight * loss_space + ent_loss_weight * ent_reg
        loss.backward()
        optimizer.step()

@torch.no_grad()
def evaluate_split(data, model, batch_size, which="valid"):
    model.eval()
    device = next(model.parameters()).device
    if which == "valid":
        X, Y = data.valid
    elif which == "test":
        X, Y = data.test
    elif which == "train":
        X, Y = data.train
    else:
        raise ValueError(f"Unknown split: {which}")

    preds, trues = [], []
    for Xb, Yb in data.get_batches(X, Y, batch_size, shuffle=False):
        Xb, Yb = Xb.to(device), Yb.to(device)
        out = model(Xb)
        pred = out[0] if isinstance(out, tuple) else out
        preds.append(pred.squeeze().cpu())
        trues.append(Yb.squeeze().cpu())
    preds = torch.cat(preds, 0).numpy()
    trues = torch.cat(trues, 0).numpy()
    return preds, trues

def run_training(data,
                 input_seq_len, output_seq_len,
                 learning_rate, batch_size,
                 epochs, hidden_dim,
                 space_loss_weight, ent_loss_weight,
                 nmb_prototype, lap_reg_weight,
                 tau, tau_dyn, alpha_init,
                 early_stop_patience=30):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ASMSTCN_Reg(
        adj_matrix     = data.adj,
        num_nodes      = data.num_nodes,
        input_length   = input_seq_len,
        input_dim      = data.features,
        hidden_dim     = hidden_dim,
        output_dim     = 1,
        output_length  = output_seq_len,
        dropout        = 0.1,
        nmb_prototype  = nmb_prototype,
        lap_reg_weight = lap_reg_weight,
        device         = device,
        coordinates    = data.coordinates,
        tau_dyn        = tau_dyn,
        alpha_init     = alpha_init,
        lambda_ent     = ent_loss_weight
    ).to(device)
    # 超参注入
    model.spatial_reg.tau = torch.tensor(float(tau), device=device)

    criterion = nn.SmoothL1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    # —— 用验证集 MAE 选最佳与早停 ——
    best_val_mae = float('inf')
    best_state = None
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, epochs + 1):
        train_epoch(data, model, criterion, optimizer,
                    batch_size, space_loss_weight, ent_loss_weight)

        v_p, v_t = evaluate_split(data, model, batch_size, which="valid")
        v_mae, v_rmse, v_corr = compute_metrics(v_t, v_p)
        if v_mae < best_val_mae - 1e-8:
            best_val_mae = v_mae
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                break

        print(f"Epoch {epoch:03d} | Val_MAE {v_mae:.4f} | Val_RMSE {v_rmse:.4f} | Val_Corr {v_corr:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    t_p, t_t = evaluate_split(data, model, batch_size, which="test")
    mae, rmse, corr = compute_metrics(t_t, t_p)
    print(f"Best@epoch {best_epoch} | out={output_seq_len}h | Test: MAE={mae:.4f} RMSE={rmse:.4f} Corr={corr:.4f}")

def run_all_experiments():
    out_lens          = [24]
    nmb_prototype     = 50
    lap_reg_weight    = 0.01
    space_loss_weight = 0.5
    tau               = 0.1
    tau_dyn           = 1.2
    alpha_init        = 0.1
    ent_loss_weight   = 0.01

    for out_h in out_lens:
        data = LoadData(0.6, 0.2, 168, out_h)
        run_training(
            data,
            input_seq_len=168,
            output_seq_len=out_h,
            learning_rate=1e-4,
            batch_size=32,
            epochs=100,
            hidden_dim=64,
            space_loss_weight=space_loss_weight,
            ent_loss_weight=ent_loss_weight,
            nmb_prototype=nmb_prototype,
            lap_reg_weight=lap_reg_weight,
            tau=tau,
            tau_dyn=tau_dyn,
            alpha_init=alpha_init,
            early_stop_patience=30
        )
        del data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    run_all_experiments()
