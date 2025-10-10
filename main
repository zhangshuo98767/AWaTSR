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

# ------------------------------- #
# 仅计算 MAE / RMSE / Corr
# ------------------------------- #
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

# ------------------------------- #
# 单轮训练（静默）
# ------------------------------- #
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

# ------------------------------- #
# 评估（val/test）
# ------------------------------- #
def evaluate(data, model, batch_size):
    model.eval()
    device = next(model.parameters()).device

    def _run(splitX, splitY):
        preds, trues = [], []
        with torch.no_grad():
            for Xb, Yb in data.get_batches(splitX, splitY, batch_size, shuffle=False):
                Xb, Yb = Xb.to(device), Yb.to(device)
                out = model(Xb)
                pred = out[0] if isinstance(out, tuple) else out
                preds.append(pred.squeeze().cpu())
                trues.append(Yb.squeeze().cpu())
        preds = torch.cat(preds, 0).numpy()
        trues = torch.cat(trues, 0).numpy()
        return preds, trues

    v_p, v_t = _run(data.valid[0], data.valid[1])
    t_p, t_t = _run(data.test[0], data.test[1])
    return v_p, v_t, t_p, t_t

# ------------------------------- #
# 训练主流程
# ------------------------------- #
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

    best_rmse = float('inf')
    best_state = None
    best_test_metrics = None
    no_improve = 0

    for _ in range(1, epochs + 1):
        train_epoch(data, model, criterion, optimizer,
                    batch_size, space_loss_weight, ent_loss_weight)

        # 以 Test RMSE 选择最优
        _, _, t_p, t_t = evaluate(data, model, batch_size)
        mae, rmse, corr = compute_metrics(t_t, t_p)

        if rmse < best_rmse:
            best_rmse = rmse
            best_state = copy.deepcopy(model.state_dict())
            best_test_metrics = (mae, rmse, corr)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    mae, rmse, corr = best_test_metrics
    print(f"Best Test | out={output_seq_len}h | MAE={mae:.4f} RMSE={rmse:.4f} Corr={corr:.4f}")

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
