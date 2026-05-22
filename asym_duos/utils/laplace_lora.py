# Built on the `bayesian-lora` package (https://github.com/MaximeRobeyns/bayesian_lora).
# Method: Yang et al., "Bayesian Low-Rank Adaptation for Large Language Models",
# ICLR 2024.

import os.path as osp
from copy import deepcopy
from typing import Optional

import numpy as np
import peft
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score
from transformers.tokenization_utils_base import BatchEncoding

from bayesian_lora import (
    calculate_kronecker_factors,
    cholesky_decompose_small_factors,
    model_evidence,
    stable_cholesky,
    variance,
)
from bayesian_lora.main import jacobian_mean

from .optimizer_helpers import (
    _merge_dictionaries,
    _read_or_create_path,
    _sanity_check_on_probs,
    _save_val_test_results,
    _softmax_from_log,
    _sort_by_idx,
)
from .uncertainty_metrics import compute_uncertainty_metrics


def _find_factors(model, trainer, tokenizer, tok_cfg, target_ids, n_kfac, lr_threshold_kfac, device):
    # --- forward hook to find factors -----------------------------------------------
    
    def fwd_call(model, batch_prompts):
        prompts = batch_prompts['prompts']
        inputs = tokenizer(prompts, **tok_cfg).to(device)
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, target_ids]  # Get the last token logits
        logits = logits.softmax(-1)
        return logits

    ## Get train loader
    train_loader = trainer.get_train_dataloader()

    ## Find Kronecker factors
    factors = calculate_kronecker_factors(
        model,            
        fwd_call,         
        train_loader,     
        n_kfac,
        lr_threshold_kfac,
        target_module_keywords = ["lora"]
    )

    ## Cholesky decompose small factors
    factors = cholesky_decompose_small_factors(
            factors, lr_threshold_kfac, device,torch.float32
        )
    return factors

def _find_LL(model, tokenizer, val_loader, tok_cfg, target_ids, device):
    LL = 0
    
    with torch.no_grad(), torch.inference_mode():
        for batch in val_loader:
            prompts = batch['prompts']
            labels = batch['labels'].to(device)
            inputs = tokenizer(prompts, **tok_cfg).to(device)
            logits = model(**inputs).logits[:, -1, target_ids]
            probs = logits.softmax(-1)
            LL += probs.gather(1, labels[:, None].to(device)).log().sum()

    return LL

def _find_optimal_prior_var(prior_var, trainer, model, tokenizer, tok_cfg, val_dataset, target_ids, factors, peft_r, n_kfac, device):
    log_s2 = torch.tensor(prior_var).log().requires_grad_(True)
    opt = torch.optim.Adam([log_s2], lr=1e-2)
    val_loader = trainer.get_eval_dataloader(val_dataset)

    LL = _find_LL(model, tokenizer, val_loader, tok_cfg, target_ids, device)

    for i in range(500):
        opt.zero_grad()
        s2 = log_s2.exp()
        loss = -model_evidence(model, LL, factors, peft_r, n_kfac, s2).log()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([log_s2], 1.0)
        opt.step()
        if i % 50 == 49:
            print(f"  [prior opt] step {i+1}: s2={s2.item():.6f}, loss={loss.item():.4f}")

    return log_s2.exp().detach()

def _get_raw_model_tokenizer(cfg):
    ## Get raw model
    model_cfg = deepcopy(cfg.model.model_cfg)
    model_cls_name = model_cfg.pop('type')
    model_cls = getattr(transformers, model_cls_name)

    model_laplace = model_cls.from_pretrained(cfg.model.model_name_or_path, **model_cfg)
    
    ## Get raw tokenizer
    tokenizer_cfg = deepcopy(cfg.model.tokenizer_cfg)
    tokenizer_cls = getattr(transformers, tokenizer_cfg.pop('type'))
    tokenizer_laplace = tokenizer_cls.from_pretrained(cfg.model.model_name_or_path, **tokenizer_cfg)
    tokenizer_special_tokens = {
        k: getattr(tokenizer_laplace, v.split('.')[-1]) if isinstance(v, str) and v.startswith('tokenizer') else v
        for k,
        v in cfg.model.special_tokens.items()
    }
    if len(tokenizer_special_tokens) > 0:
        tokenizer_laplace.add_special_tokens(tokenizer_special_tokens)
    if tokenizer_laplace.pad_token is None:
        tokenizer_laplace.pad_token = tokenizer_laplace.eos_token

    return model_laplace, tokenizer_laplace

def _get_laplace_predictions(model, tokenizer, tok_cfg, trainer, dataset, target_ids, factors, s2, peft_r, n_kfac, device):
    # --- output callback for Laplace predictions -----------------------------------------------
    def output_callback(outputs):
        logits = outputs.logits
        target_logits = logits[:, -1, target_ids]
        return target_logits

    loader = trainer.get_eval_dataloader(dataset)
    #loader = DataLoader(
    #    base_loader.dataset,
    #    batch_size=laplace_batch_size,
    #    shuffle=False,
    #    collate_fn=base_loader.collate_fn,
    #)

    # Release fragmented allocations left over from training before the
    # Jacobian loop, which is the most memory-intensive part of Laplace.
    torch.cuda.empty_cache()
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)

    pred_mu_list     = []
    pred_var_list    = []
    true_labels_list = []
    pred_probs_list  = []

    def _process_single_batch(batch_inputs, labels):
        jacobian, f_mu = jacobian_mean(model, batch_inputs, output_callback=output_callback)

        actual_bs = batch_inputs.input_ids.size(0)
        if actual_bs == 1:
            # Work-around for bayesian_lora bug: variance() squeezes without dim,
            # dropping the batch dim when bs==1. Expand to bs=2 then slice back.
            var_inputs   = BatchEncoding({k: v.expand(2, *v.shape[1:]) for k, v in batch_inputs.items()})
            var_jacobian = {k: v.expand(2, *v.shape[1:]) for k, v in jacobian.items()}
            f_var = variance(var_inputs, var_jacobian, factors, s2, len(target_ids), peft_r, n_kfac, device)[:1]
        else:
            f_var = variance(batch_inputs, jacobian, factors, s2, len(target_ids), peft_r, n_kfac, device)

        diag_mean = f_var.diagonal(dim1=-2, dim2=-1).mean().item()
        
        print(f"  [variance]: f_var diagonal mean={diag_mean:.2f}")
        L = stable_cholesky(f_var)
        samples = 100_000
        f_mu_exp = f_mu.float().expand(samples, *f_mu.shape)
        L_exp    = L.float().expand(samples, *L.shape)
        eps      = torch.randn_like(f_mu_exp).unsqueeze(-1)
        logits_sampled = f_mu_exp[..., None] + L_exp @ eps
        probs_sampled  = logits_sampled.squeeze(-1).softmax(-1).mean(0)

        return f_mu, f_var, probs_sampled, labels

    with torch.no_grad():
        for batch in loader:
            prompts = batch['prompts']
            labels = batch['labels'].to(device)
            batch_inputs = tokenizer(prompts, **tok_cfg).to(device)
            torch.cuda.empty_cache()
            try:
                f_mu, f_var, probs_sampled, labels = _process_single_batch(batch_inputs, labels)
            except torch.cuda.OutOfMemoryError:
                # Fall back to processing one sample at a time
                f_mu_parts, f_var_parts, probs_parts, lbl_parts = [], [], [], []
                for i in range(batch_inputs.input_ids.size(0)):
                    single_inputs = BatchEncoding({k: v[i:i+1] for k, v in batch_inputs.items()})
                    single_labels = labels[i:i+1]

                    fm, fv, ps, lb = _process_single_batch(single_inputs, single_labels)
                    f_mu_parts.append(fm.cpu()); f_var_parts.append(fv.cpu())
                    probs_parts.append(ps.cpu()); lbl_parts.append(lb.cpu())
                    del fm, fv, ps, lb
                    torch.cuda.empty_cache()

                f_mu          = torch.cat(f_mu_parts).to(device)
                f_var         = torch.cat(f_var_parts).to(device)
                probs_sampled = torch.cat(probs_parts).to(device)
                labels        = torch.cat(lbl_parts).to(device)

            pred_mu_list.append(f_mu.clone().cpu())
            pred_var_list.append(f_var.clone().cpu())
            pred_probs_list.append(probs_sampled.cpu())
            true_labels_list.append(labels.cpu())

    all_probs  = torch.cat(pred_probs_list,  dim=0)   # (N, C)
    all_labels = torch.cat(true_labels_list, dim=0).numpy()

    kfac_avg_probs = all_probs.float().numpy().astype(np.float32)
    kfac_avg_logits  = torch.log(all_probs.clamp(min=1e-10)).float().numpy().astype(np.float32)
    kfac_true_labels = all_labels.astype(int)

    kfac_pred_mu  = torch.cat(pred_mu_list,  dim=0).float().numpy().astype(np.float32)  # (N, C)
    kfac_pred_var = torch.cat(pred_var_list, dim=0).float().numpy().astype(np.float32)

    return kfac_avg_logits, kfac_avg_probs, kfac_true_labels, kfac_pred_mu, kfac_pred_var


def _build_metrics(model_type_logits, laplace_logits, labels, model_type):

    probs_model_type = _softmax_from_log(model_type_logits)
    _sanity_check_on_probs(probs_model_type, model_type)
    pred_model_type = probs_model_type.argmax(axis=1)

    probs_laplace = _softmax_from_log(laplace_logits)
    _sanity_check_on_probs(probs_laplace, f'{model_type}_laplace')
    pred_laplace = probs_laplace.argmax(axis=1)

    prefix = f"{model_type}_laplace"
    metrics = {
        f"{model_type}_accuracy": float(accuracy_score(labels, pred_model_type)),
        f"{model_type}_f1_macro": float(f1_score(labels, pred_model_type, average="macro")),
        f"{prefix}_accuracy": float(accuracy_score(labels, pred_laplace)),
        f"{prefix}_f1_macro": float(f1_score(labels, pred_laplace, average="macro")),
    }

    metrics.update({
        f"{model_type}_{k}": v
        for k, v in compute_uncertainty_metrics(probs_model_type, labels).items()
    })
    metrics.update({
        f"{prefix}_{k}": v
        for k, v in compute_uncertainty_metrics(probs_laplace, labels).items()
    })
    return metrics, pred_laplace


def optimize_laplace(
    result_dir: str,
    saved_file_name: str,
    model_type: str,
    trainer,
    val_dataset,
    test_dataset,
    checkpoint_path: str,
    seed: int = 42,
    prior_var: Optional[float] = 0.1,
    n_kfac: int = 10,
    lr_threshold_kfac: int = 100
) -> None:
    """Fit diagonal Laplace over LoRA params, evaluate on val/test, persist.

    Args
    ----
    result_dir      Experiment output directory (same root as train.py uses).
    saved_file_name NPZ filename, e.g. 'arc_e.npz'.
    model_type      'base' or 'sidekick' — controls the output subdirectory.
    trainer         The FTTrainer (or EvidentialTrainer) after training.
                    Provides: model, processing_class, cfg, args.device,
                    get_train_dataloader(), get_eval_dataloader().
    val_dataset     Used for prior precision tuning + val evaluation.
    test_dataset    Used for test evaluation.
    seed            Written as the key into the NPZ (same as other optimisers).
    prior_precision If None, tuned via grid search on val NLL (recommended).
    """
    model      = trainer.model
    device     = trainer.args.device
    target_ids = trainer.target_ids
    cfg        = trainer.cfg
    tokenizer  = trainer.processing_class
    tok_cfg    = cfg.tokenizer_run_cfg
    peft_r     = cfg.model.peft_cfg.r


    # --- load MAP predictions -----------------------------------------------
    def _load(split):
        path  = osp.join(result_dir, model_type, f"{split}_preds", saved_file_name)
        data  = dict(np.load(path, allow_pickle=True))[str(seed)].item()
        order = np.argsort(data["idx"])
        return data, order

    map_val, mv_ord = _load("val")
    map_val_logits = _sort_by_idx(map_val['logits'], mv_ord)
    map_tst, mt_ord = _load("test")
    map_tst_logits = _sort_by_idx(map_tst['logits'], mt_ord)

    print(f"\n[laplace] Snapshotting MAP weights for {model_type} ...")
    map_params = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    n_lora = sum(p.numel() for p in map_params.values())
    print(f"  LoRA parameter count: {n_lora:,}")

    ## Find Kronecker factors
    factors = _find_factors(model, trainer, tokenizer, tok_cfg, target_ids, n_kfac, lr_threshold_kfac, device)

    ## Find optimal prior
    s2 = _find_optimal_prior_var(
        prior_var, 
        trainer, 
        model, 
        tokenizer, 
        tok_cfg, 
        val_dataset, 
        target_ids, 
        factors, 
        peft_r, 
        n_kfac, 
        device
    )
    print(f"Optimized prior variance is: {s2.item()}")

    # Move trainer's model off GPU before loading model_laplace.
    # del model only drops the local reference; trainer.model still pins it on GPU.
    trainer.model.cpu()
    del model
    trainer.model = None
    torch.cuda.empty_cache()

    ## Get raw model and tokenizer
    model_laplace, tokenizer_laplace = _get_raw_model_tokenizer(cfg)

    ## Get peft model
    model_laplace = peft.PeftModel.from_pretrained(model_laplace, checkpoint_path, is_trainable=True)
    model_laplace = model_laplace.to(device)

    ## Freeze non-lora params
    non_lora = [p for n, p in model_laplace.named_parameters()
                if p.requires_grad and "lora" not in n.lower()]
    for p in non_lora:
        p.requires_grad_(False)

    

    print("[laplace] Generating val predictions...")
    val_logits, val_probs, val_labels, val_pred_mu, val_pred_var = _get_laplace_predictions(
        model_laplace, tokenizer_laplace, tok_cfg, trainer, val_dataset, target_ids, factors,
        s2, peft_r, n_kfac, device
    )

    print("[laplace] Generating test predictions...")
    tst_logits, tst_probs, tst_labels, tst_pred_mu, tst_pred_var = _get_laplace_predictions(
        model_laplace, tokenizer_laplace, tok_cfg, trainer, test_dataset, target_ids, factors,
        s2, peft_r, n_kfac, device
    )

    val_metrics, val_preds = _build_metrics(map_val_logits, 
                                            _sort_by_idx(val_logits, mv_ord), 
                                            _sort_by_idx(val_labels, mv_ord), 
                                            model_type)
    tst_metrics, tst_preds = _build_metrics(map_tst_logits, 
                                            _sort_by_idx(tst_logits, mt_ord), 
                                            _sort_by_idx(tst_labels, mt_ord), 
                                            model_type)
    print(f"  [laplace] val  acc={val_metrics[f'{model_type}_laplace_accuracy']:.4f}  "
          f"ece={val_metrics[f'{model_type}_laplace_ece']:.4f}  nll={val_metrics[f'{model_type}_laplace_nll']:.4f}")
    print(f"  [laplace] test acc={tst_metrics[f'{model_type}_laplace_accuracy']:.4f}  "
          f"ece={tst_metrics[f'{model_type}_laplace_ece']:.4f}  nll={tst_metrics[f'{model_type}_laplace_nll']:.4f}")


    metrics_path = osp.join(result_dir, "metrics", saved_file_name)
    lap_val_path = osp.join(result_dir, "laplace_lora", model_type, "val_preds",  saved_file_name)
    lap_tst_path = osp.join(result_dir, "laplace_lora", model_type, "test_preds", saved_file_name)

    ex_val = _read_or_create_path(lap_val_path)
    ex_tst = _read_or_create_path(lap_tst_path)
    ex_met = _read_or_create_path(metrics_path)

    _save_val_test_results(
        seed, saved_file_name,
        _sort_by_idx(map_val["idx"], mv_ord),
        _sort_by_idx(map_val["input"], mv_ord),
        _sort_by_idx(val_logits, mv_ord),
        _sort_by_idx(val_probs, mv_ord),
        _sort_by_idx(val_preds, mv_ord),
        _sort_by_idx(map_val["true_labels"], mv_ord),
        ex_val, lap_val_path,
    )
    _save_val_test_results(
        seed, saved_file_name,
        _sort_by_idx(map_tst["idx"], mt_ord),
        _sort_by_idx(map_tst["input"], mt_ord),
        _sort_by_idx(tst_logits, mt_ord),
        _sort_by_idx(tst_probs, mt_ord),
        _sort_by_idx(tst_preds, mt_ord),
        _sort_by_idx(map_tst["true_labels"], mt_ord),
        ex_tst, lap_tst_path,
    )

    new_metrics = {
        "data": saved_file_name,
        "laplace_optimized_prior_variance": s2.item(),
        "laplace_n_kfac": n_kfac,
        "laplace_peft_r": peft_r,
        "val": val_metrics,
        "test": tst_metrics,
    }
    seed_key = str(seed)
    existing = ex_met.get(seed_key, {})
    if hasattr(existing, "item"):
        existing = existing.item()
    ex_met[seed_key] = _merge_dictionaries(existing, new_metrics)
    np.savez_compressed(metrics_path, **ex_met)
    print(f"[laplace] Results saved → {lap_val_path}")
    print(f"[laplace] Results saved → {lap_tst_path}")
    print(f"[laplace] Metrics saved → {metrics_path}")