"""
Generative training/eval loops for weaver.

These mirror `train_classification` / `evaluate_classification` from
`weaver/utils/nn/tools.py` but accept a loss callable with signature
    loss_func(model, data, *cond, mask=mask) -> (loss, info_dict)
so the loss owns the noise sampling / JVP / target construction (e.g. a
MeanFlow loss), and the wrapping model `forward(z, t, r, *cond)` returns the
predicted velocity.

Hook into weaver by exposing `get_train_fn` / `get_evaluate_fn` from the
user's network-config module so weaver picks these up instead of the default
classification trainer (see weaver/train.py:851-853).
"""

import time
import torch
import tqdm

from weaver.utils.nn.tools import (
    AllGather,
    _save_live_checkpoint,
    get_autocast_config,
)
from weaver.utils.logger import _logger


def _select_inputs(X, data_config, input_key=None, cond_keys=None, mask_key=None):
    """Split a weaver batch into (data, cond, mask) tensors.

    By default `data` is the first input group and `cond` is whatever the
    network-config declared. `mask_key` is auto-detected as the first input
    group whose name ends in `_mask` (or you can pass it explicitly).
    Override `input_key` / `cond_keys` (list[str]) when constructing a custom
    train function from a network-config.
    """
    if input_key is None:
        input_key = data_config.input_names[0]
        cond_keys = list(data_config.input_names[1:])
    elif cond_keys is None:
        cond_keys = [k for k in data_config.input_names if k != input_key]
    if mask_key is None:
        mask_key = next((k for k in data_config.input_names if k.endswith("_mask")), None)
    data = X[input_key]
    cond = [X[k] for k in cond_keys]
    mask = X[mask_key] if mask_key is not None and mask_key in X else None
    return data, cond, mask


def make_train_generative(input_key=None, cond_keys=None):
    """Factory returning a `train_generative` function pinned to specific
    data/cond keys, suitable for use as `get_train_fn` in a network config."""

    def train_generative(model, loss_func, opt, scheduler, train_loader, dev, epoch,
                         steps_per_epoch=None, grad_scaler=None, tb_helper=None, extra_args=None):
        model.train()
        data_config = train_loader.dataset.config
        clip_grad_norm = getattr(opt, "_clip_grad_norm", float("inf"))

        enable_autocast, autocast_dtype = get_autocast_config(extra_args["args"])
        args_obj = extra_args["args"]
        save_steps = getattr(args_obj, "save_steps", 0)
        live_local_rank = extra_args.get("local_rank", 0) if extra_args else 0

        total_loss = 0.0
        num_batches = 0
        grad_norm_max = 0.0
        start = time.time()
        entry_count = 0
        # running mean of every key the loss reports in its `info` dict (mf,
        # seal, seal_lambda, mse, ...). Lets us print per-component averages
        # in the epoch summary instead of needing a tensorboard reader.
        info_sums = {}

        # Keys we want surfaced in the tqdm postfix (these are the typical
        # MeanFlow / MeanFlowSEAL outputs). Anything else still ends up in TB
        # and the per-epoch summary.
        _postfix_keys = ("mf", "seal", "seal_lambda")

        with tqdm.tqdm(train_loader) as tq:
            for X, _y, _Z in tq:
                data, cond, mask = _select_inputs(X, data_config, input_key, cond_keys)
                data = data.to(dev)
                cond = [c.to(dev) for c in cond]
                mask = mask.to(dev) if mask is not None else None
                entry_count += data.shape[0]
                if tb_helper:
                    tb_helper.global_step += 1
                opt.zero_grad()
                with torch.autocast("cuda", enabled=enable_autocast, dtype=autocast_dtype):
                    loss, info = loss_func(model, data, *cond, mask=mask)
                if grad_scaler is None:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=clip_grad_norm).item()
                    opt.step()
                else:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(opt)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=clip_grad_norm).item()
                    grad_scaler.step(opt)
                    grad_scaler.update()

                if scheduler and getattr(scheduler, "_update_per_step", False):
                    scheduler.step()

                loss_val = loss.item()
                total_loss += loss_val
                num_batches += 1
                grad_norm_max = max(grad_norm_max, grad_norm)
                for k, v in info.items():
                    info_sums[k] = info_sums.get(k, 0.0) + float(v)

                postfix = {
                    "lr": "%.2e" % scheduler.get_last_lr()[0] if scheduler else opt.defaults["lr"],
                    "Loss": "%.4f" % loss_val,
                    "Avg": "%.4f" % (total_loss / num_batches),
                }
                for k in _postfix_keys:
                    if k in info:
                        postfix[k] = "%.2e" % float(info[k])
                tq.set_postfix(postfix)

                if tb_helper:
                    rows = [
                        ("Loss/train", loss_val, tb_helper.global_step),
                        ("GradNorm/train", grad_norm, tb_helper.global_step),
                    ]
                    for k, v in info.items():
                        rows.append((f"{k}/train", v, tb_helper.global_step))
                    tb_helper.write_scalars(rows)

                if save_steps and num_batches > 0 and num_batches % save_steps == 0:
                    _save_live_checkpoint(model, opt, args_obj, live_local_rank,
                                          epoch=epoch, step=num_batches)
                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

        dt = time.time() - start
        _logger.info("Processed %d entries (avg %.1f e/s)", entry_count, entry_count / max(dt, 1e-6))
        _logger.info("Train AvgLoss: %.5f (max grad-norm %.3f)", total_loss / max(num_batches, 1), grad_norm_max)
        if info_sums:
            comps = "  ".join(f"{k}={v / max(num_batches, 1):.4e}" for k, v in info_sums.items())
            _logger.info("Train component avgs: %s", comps)
        if torch.cuda.is_available() and getattr(dev, "type", str(dev)) == "cuda":
            _logger.info("Max CUDA memory: %.1f MB", torch.cuda.max_memory_allocated(dev) / 1024.0 ** 2)
        if scheduler and not getattr(scheduler, "_update_per_step", False):
            scheduler.step()

    return train_generative


def make_evaluate_generative(input_key=None, cond_keys=None):
    """Factory returning an `evaluate_generative` matching the same I/O contract."""

    def evaluate_generative(model, test_loader, dev, epoch, for_training=True, loss_func=None,
                            steps_per_epoch=None, eval_metrics=None,
                            tb_helper=None, extra_args=None):
        model.eval()
        data_config = test_loader.dataset.config
        enable_autocast, autocast_dtype = get_autocast_config(extra_args["args"])

        total_loss = 0.0
        num_batches = 0
        count = 0
        info_sums = {}

        _postfix_keys = ("mf", "seal", "seal_lambda")

        # NB: MeanFlow loss needs grads enabled for JVP-via-autograd; with
        # torch.func.jvp we can skip torch.no_grad and use inference_mode.
        with tqdm.tqdm(test_loader) as tq:
            for X, _y, _Z in tq:
                data, cond, mask = _select_inputs(X, data_config, input_key, cond_keys)
                data = data.to(dev)
                cond = [c.to(dev) for c in cond]
                mask = mask.to(dev) if mask is not None else None
                num = data.shape[0]
                with torch.autocast("cuda", enabled=enable_autocast, dtype=autocast_dtype):
                    loss, info = loss_func(model, data, *cond, mask=mask)
                lv = loss.item()
                num_batches += 1
                count += num
                total_loss += lv * num
                for k, v in info.items():
                    info_sums[k] = info_sums.get(k, 0.0) + float(v) * num

                postfix = {
                    "Loss": "%.4f" % lv,
                    "Avg": "%.4f" % (total_loss / max(count, 1)),
                }
                for k in _postfix_keys:
                    if k in info:
                        postfix[k] = "%.2e" % float(info[k])
                tq.set_postfix(postfix)
                if tb_helper and tb_helper.custom_fn:
                    tb_helper.custom_fn(model_output=None, model=model, epoch=epoch,
                                        i_batch=num_batches, mode="eval" if for_training else "test")
                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

        avg = total_loss / max(count, 1)
        _logger.info("Eval AvgLoss: %.5f over %d entries", avg, count)
        if info_sums:
            comps = "  ".join(f"{k}={v / max(count, 1):.4e}" for k, v in info_sums.items())
            _logger.info("Eval component avgs: %s", comps)
        if tb_helper:
            tb_mode = "eval" if for_training else "test"
            rows = [(f"Loss/{tb_mode} (epoch)", avg, epoch)]
            for k, v in info_sums.items():
                rows.append((f"{k}/{tb_mode} (epoch)", v / max(count, 1), epoch))
            tb_helper.write_scalars(rows)
        if for_training:
            return avg
        # For prediction mode we don't have scores/labels in the
        # classification sense; return placeholders so the caller doesn't
        # crash. Users wanting actual generated samples should call the
        # sampler (e.g. weaver.nn.loss.meanflow.euler_sample) directly.
        return avg, None, {}, {}

    return evaluate_generative
