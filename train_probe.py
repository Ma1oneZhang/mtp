"""One-GPU DFlash training probe with live target hidden-state generation.

This is an observability experiment, not a reproduction trainer.  It keeps the
frozen target and trainable draft on one GPU, materializes target features for
one clean sequence, consumes them immediately, and emits one JSON record per
optimizer step.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from probe_utils import (
    build_dflash_batch,
    capture_parameter_snapshot,
    grouped_gradient_norms,
    parameter_update_metrics,
    parse_args,
    rows,
    tensor_l2_norm,
    tokenize_last_answer,
    validate_args,
)


@torch.no_grad()
def target_features(
    target: torch.nn.Module, input_ids: torch.Tensor, layer_ids: list[int]
) -> torch.Tensor:
    output = target.model(
        input_ids=input_ids,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    # Transformers includes the embedding output at hidden_states[0].
    features = torch.cat(
        [output.hidden_states[layer_id + 1] for layer_id in layer_ids], dim=-1
    )
    return features.detach()


@dataclass(frozen=True)
class DFlashLossOutput:
    objective: torch.Tensor
    pal_loss: torch.Tensor
    plain_ce: torch.Tensor
    position_ce: torch.Tensor
    position_accuracy: torch.Tensor
    greedy_prefix_matches: torch.Tensor
    soft_expected_acceptance: torch.Tensor
    pal_weight_by_position: torch.Tensor


def pal_objective(
    log_target_probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean PAL loss, mean soft acceptance, and mean slot weights."""
    if log_target_probabilities.ndim != 2:
        raise ValueError("PAL expects [anchors, draft_positions] log probabilities")
    if log_target_probabilities.shape[1] == 0:
        raise ValueError("PAL requires at least one draft position")

    log_prefix_probabilities = log_target_probabilities.cumsum(dim=-1)
    zero_length_prefix = torch.zeros_like(log_prefix_probabilities[:, :1])
    log_expected_acceptance = torch.logsumexp(
        torch.cat((zero_length_prefix, log_prefix_probabilities), dim=-1),
        dim=-1,
    )

    suffix_log_sums = torch.flip(
        torch.logcumsumexp(torch.flip(log_prefix_probabilities, dims=(-1,)), dim=-1),
        dims=(-1,),
    )
    weights = torch.exp(suffix_log_sums - log_expected_acceptance[:, None])
    return (
        -log_expected_acceptance.mean(),
        log_expected_acceptance.exp().mean().detach(),
        weights.mean(dim=0).detach(),
    )


def dflash_loss(
    *,
    target: torch.nn.Module,
    draft: torch.nn.Module,
    input_ids: torch.Tensor,
    features: torch.Tensor,
    anchors: list[int],
    loss_name: str,
    pal_ce_weight: float,
) -> DFlashLossOutput:
    block_size = draft.block_size
    noise_ids, position_ids, attention_mask, labels = build_dflash_batch(
        input_ids,
        anchors,
        block_size,
        draft.mask_token_id,
    )
    noise_embedding = target.model.embed_tokens(noise_ids)
    draft_hidden = draft(
        target_hidden=features,
        noise_embedding=noise_embedding,
        position_ids=position_ids,
        attention_mask=attention_mask,
        use_cache=False,
        is_causal=False,
    )
    predicted_hidden = draft_hidden.view(
        1, len(anchors), block_size, draft_hidden.shape[-1]
    )[:, :, 1:, :]
    logits = target.lm_head(predicted_hidden.flatten(1, 2))
    per_position_ce = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    ).view(len(anchors), block_size - 1)
    pal_loss, soft_expected_acceptance, pal_weights = pal_objective(-per_position_ce)
    plain_ce = per_position_ce.mean()
    if loss_name == "ce":
        objective = plain_ce
    elif loss_name == "pal":
        objective = pal_loss + pal_ce_weight * plain_ce
    else:
        raise ValueError(f"unsupported loss: {loss_name}")
    accuracy = logits.view(1, len(anchors), block_size - 1, -1).argmax(dim=-1)
    accuracy = accuracy.eq(labels).float().squeeze(0)
    greedy_prefix_matches = accuracy.cumprod(dim=-1).sum(dim=-1).mean().detach()
    return DFlashLossOutput(
        objective=objective,
        pal_loss=pal_loss.detach(),
        plain_ce=plain_ce.detach(),
        position_ce=per_position_ce.detach().mean(dim=0),
        position_accuracy=accuracy.detach().mean(dim=0),
        greedy_prefix_matches=greedy_prefix_matches,
        soft_expected_acceptance=soft_expected_acceptance,
        pal_weight_by_position=pal_weights,
    )


def load_models(args: Any) -> tuple[Any, Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, local_files_only=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    target.requires_grad_(False)
    if args.draft_init == "checkpoint":
        draft = AutoModel.from_pretrained(
            args.draft_model,
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        )
    else:
        draft_config = AutoConfig.from_pretrained(
            args.draft_model,
            local_files_only=True,
            trust_remote_code=True,
        )
        draft = AutoModel.from_config(
            draft_config,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")
    draft.train()
    if draft.mask_token_id is None:
        raise ValueError("draft config has no mask_token_id")
    if draft.mask_token_id >= target.config.vocab_size:
        raise ValueError("draft mask_token_id exceeds target vocabulary")
    return tokenizer, target, draft


def build_optimizer(
    optimizer_name: str,
    draft: torch.nn.Module,
    learning_rate: float,
) -> torch.optim.Optimizer | OptimizerBundle:
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            draft.parameters(),
            lr=learning_rate,
            momentum=0.0,
            weight_decay=0.0,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            draft.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
    if optimizer_name == "muon":
        muon_params = []
        auxiliary_params = []
        for name, parameter in draft.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim == 2 and not name.startswith(
                ("embed_tokens.", "lm_head.")
            ):
                muon_params.append(parameter)
            else:
                auxiliary_params.append(parameter)
        if not muon_params:
            raise ValueError("Muon requires at least one trainable 2D parameter")
        if not auxiliary_params:
            raise ValueError("Muon draft has no parameters for auxiliary AdamW")
        muon = torch.optim.Muon(
            muon_params,
            lr=learning_rate,
            weight_decay=0.01,
            momentum=0.95,
            nesterov=True,
            ns_coefficients=(3.4445, -4.775, 2.0315),
            eps=1e-7,
            ns_steps=5,
            adjust_lr_fn="match_rms_adamw",
        )
        auxiliary_adamw = torch.optim.AdamW(
            auxiliary_params,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
        return OptimizerBundle(
            muon,
            auxiliary_adamw,
            metric_config={
                "muon_momentum": 0.95,
                "muon_nesterov": True,
                "muon_ns_steps": 5,
                "muon_adjust_lr_fn": "match_rms_adamw",
                "auxiliary_optimizer": "adamw",
            },
        )
    raise ValueError(f"unsupported optimizer: {optimizer_name}")


class OptimizerBundle:
    """One optimizer-step boundary over disjoint parameter groups."""

    def __init__(
        self,
        *optimizers: torch.optim.Optimizer,
        metric_config: dict[str, Any] | None = None,
    ) -> None:
        if not optimizers:
            raise ValueError("optimizer bundle cannot be empty")
        self.optimizers = optimizers
        self.param_groups = [
            group for optimizer in optimizers for group in optimizer.param_groups
        ]
        self.metric_config = metric_config or {}

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()


def main(
    *,
    optimizer_name: str = "sgd",
    loss_name: str = "ce",
    pal_ce_weight: float = 0.0,
    architecture_name: str = "dflash",
    draft_transform: Callable[[torch.nn.Module], torch.nn.Module] | None = None,
) -> None:
    if loss_name not in {"ce", "pal"}:
        raise ValueError(f"unsupported loss: {loss_name}")
    if pal_ce_weight < 0:
        raise ValueError("PAL CE weight cannot be negative")
    if loss_name == "ce" and pal_ce_weight != 0:
        raise ValueError("CE loss cannot have a PAL CE weight")
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer, target, draft = load_models(args)
    if draft_transform is not None:
        draft = draft_transform(draft)
        draft.train()
    optimizer = build_optimizer(optimizer_name, draft, args.learning_rate)
    stream = rows(args.data)
    torch.cuda.reset_peak_memory_stats()

    for step in range(1, args.steps + 1):
        prepared = None
        row = None
        while prepared is None:
            row = next(stream)
            prepared = tokenize_last_answer(
                row, tokenizer, args.max_length, draft.block_size
            )
        input_ids, valid_anchors = prepared
        input_ids = input_ids.to("cuda")
        anchors = random.sample(
            valid_anchors, k=min(args.anchors_per_sample, len(valid_anchors))
        )

        torch.cuda.synchronize()
        started = time.perf_counter()
        features = target_features(target, input_ids, draft.target_layer_ids)
        optimizer.zero_grad(set_to_none=True)
        loss_output = dflash_loss(
            target=target,
            draft=draft,
            input_ids=input_ids,
            features=features,
            anchors=anchors,
            loss_name=loss_name,
            pal_ce_weight=pal_ce_weight,
        )
        loss_output.objective.backward()

        if any(parameter.grad is not None for parameter in target.parameters()):
            raise RuntimeError("frozen target unexpectedly received gradients")
        group_norms = grouped_gradient_norms(draft)
        gradient_norm = tensor_l2_norm(
            parameter.grad
            for parameter in draft.parameters()
            if parameter.grad is not None
        )
        observe_update = args.update_log_interval > 0 and (
            step == 1 or step % args.update_log_interval == 0
        )
        parameter_snapshot = (
            capture_parameter_snapshot(draft) if observe_update else None
        )
        optimizer.step()
        update_metrics = (
            parameter_update_metrics(draft, parameter_snapshot)
            if parameter_snapshot is not None
            else None
        )
        del parameter_snapshot
        torch.cuda.synchronize()

        metrics = {
            "step": step,
            "draft_init": args.draft_init,
            "architecture": architecture_name,
            "optimizer": optimizer_name,
            "loss_name": (
                loss_name if pal_ce_weight == 0 else f"pal+{pal_ce_weight:g}ce"
            ),
            "pal_ce_weight": pal_ce_weight,
            "sample_id": row["id"],
            "tokens": input_ids.shape[1],
            "anchors": anchors,
            "loss": loss_output.objective.detach().item(),
            "pal_loss": loss_output.pal_loss.item(),
            "plain_ce": loss_output.plain_ce.item(),
            "soft_expected_acceptance": loss_output.soft_expected_acceptance.item(),
            "pal_weight_by_position": loss_output.pal_weight_by_position.tolist(),
            "position_ce": loss_output.position_ce.tolist(),
            "position_accuracy": loss_output.position_accuracy.tolist(),
            "greedy_prefix_matches": loss_output.greedy_prefix_matches.item(),
            "greedy_acceptance_length": 1 + loss_output.greedy_prefix_matches.item(),
            "gradient_norm_by_group": group_norms,
            "gradient_norm": gradient_norm,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0]["weight_decay"],
            "update_log_interval": args.update_log_interval,
            "elapsed_seconds": time.perf_counter() - started,
            "cuda_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        }
        if update_metrics is not None:
            metrics.update(update_metrics)
            metrics["parameter_update_gradient_basis"] = "raw"
        metrics.update(getattr(optimizer, "metric_config", {}))
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        del features, input_ids, loss_output


if __name__ == "__main__":
    main()
