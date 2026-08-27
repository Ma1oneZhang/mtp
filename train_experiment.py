from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from architectures import configure_architecture, selector_objective
from training_utils import (
    build_dflash_batch,
    capture_parameter_snapshot,
    gradient_norms,
    parameter_update_metrics,
    parse_args,
    rows,
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
    return torch.cat(
        [output.hidden_states[layer_id + 1] for layer_id in layer_ids], dim=-1
    ).detach()


@dataclass(frozen=True)
class LossOutput:
    objective: torch.Tensor
    plain_ce: torch.Tensor
    pal_loss: torch.Tensor
    proxy_acceptance_length: torch.Tensor
    position_ce: torch.Tensor
    position_accuracy: torch.Tensor
    greedy_acceptance_length: torch.Tensor
    selector_loss: torch.Tensor
    selector_candidate_recall: torch.Tensor
    selector_teacher_accuracy: torch.Tensor
    selector_path_acceptance_length: torch.Tensor


def proxy_acceptance_loss(
    logits: torch.Tensor, target_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    correct_log_probabilities = (
        logits.float()
        .log_softmax(dim=-1)
        .gather(dim=-1, index=target_ids.unsqueeze(-1))
        .squeeze(-1)
    )
    prefix_log_probabilities = correct_log_probabilities.cumsum(dim=-1)
    log_proxy_acceptance = torch.logsumexp(
        torch.cat(
            (
                torch.zeros_like(prefix_log_probabilities[:, :1]),
                prefix_log_probabilities,
            ),
            dim=-1,
        ),
        dim=-1,
    )
    return -log_proxy_acceptance.mean(), log_proxy_acceptance.exp().mean()


def compute_loss(
    *,
    target: torch.nn.Module,
    draft: torch.nn.Module,
    input_ids: torch.Tensor,
    features: torch.Tensor,
    anchors: list[int],
    ngram_beta: float,
    loss_objective: str = "ce",
    draft_forward: torch.nn.Module | None = None,
) -> LossOutput:
    noise_ids, position_ids, attention_mask, block_tokens = build_dflash_batch(
        input_ids, anchors, draft.block_size, draft.mask_token_id
    )
    noise_embedding = target.model.embed_tokens(noise_ids)
    forward_model = draft if draft_forward is None else draft_forward
    hidden = forward_model(
        target_hidden=features,
        noise_embedding=noise_embedding,
        position_ids=position_ids,
        attention_mask=attention_mask,
        use_cache=False,
        is_causal=False,
    )
    hidden = hidden.view(1, len(anchors), draft.block_size, -1)[:, :, 1:, :]
    base_logits = target.lm_head(hidden.flatten(1, 2)).view(
        len(anchors), draft.block_size - 1, -1
    )
    target_ids = block_tokens[0, :, 1:]
    anchor_ids = block_tokens[0, :, 0]
    if hasattr(draft, "markov_head"):
        teacher_predecessors = torch.cat(
            (anchor_ids[:, None], target_ids[:, :-1]), dim=-1
        )
        logits = draft.markov_head(base_logits, teacher_predecessors)
        predicted = draft.markov_head.greedy_path(base_logits, anchor_ids)
    else:
        logits = base_logits
        predicted = logits.argmax(dim=-1)
    per_position_ce = F.cross_entropy(
        logits.float().flatten(0, 1), target_ids.flatten(), reduction="none"
    ).view_as(target_ids)
    plain_ce = per_position_ce.mean()
    pal_loss, proxy_acceptance_length = proxy_acceptance_loss(logits, target_ids)
    if loss_objective == "ce":
        token_objective = plain_ce
    elif loss_objective == "pal10":
        token_objective = pal_loss + 0.1 * plain_ce
    else:
        raise ValueError(f"unknown loss objective: {loss_objective}")
    accuracy = predicted.eq(target_ids).float()
    greedy_acceptance_length = 1 + accuracy.cumprod(dim=-1).sum(dim=-1).mean()

    if hasattr(draft, "candidate_selector"):
        selector = selector_objective(
            draft.candidate_selector,
            hidden[0],
            logits,
            target_ids,
            anchor_ids,
            input_ids[0],
            anchors,
            ngram_beta,
        )
        selector_loss = selector.loss
        objective = token_objective + selector_loss
        candidate_recall = selector.candidate_recall
        teacher_accuracy = selector.teacher_accuracy
        selector_acceptance = 1 + selector.path_acceptance
    else:
        zero = plain_ce.detach().new_zeros(())
        selector_loss = plain_ce * 0
        objective = token_objective
        candidate_recall = zero
        teacher_accuracy = zero
        selector_acceptance = zero

    return LossOutput(
        objective=objective,
        plain_ce=plain_ce.detach(),
        pal_loss=pal_loss.detach(),
        proxy_acceptance_length=proxy_acceptance_length.detach(),
        position_ce=per_position_ce.detach().mean(dim=0),
        position_accuracy=accuracy.detach().mean(dim=0),
        greedy_acceptance_length=greedy_acceptance_length.detach(),
        selector_loss=selector_loss.detach(),
        selector_candidate_recall=candidate_recall,
        selector_teacher_accuracy=teacher_accuracy,
        selector_path_acceptance_length=selector_acceptance,
    )


class OptimizerBundle:
    def __init__(self, *optimizers: torch.optim.Optimizer) -> None:
        self.optimizers = optimizers
        self.param_groups = [
            group for optimizer in optimizers for group in optimizer.param_groups
        ]

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> list[dict[str, Any]]:
        return [optimizer.state_dict() for optimizer in self.optimizers]

    def load_state_dict(self, states: list[dict[str, Any]]) -> None:
        if len(states) != len(self.optimizers):
            raise ValueError("optimizer checkpoint has an unexpected group count")
        for optimizer, state in zip(self.optimizers, states, strict=True):
            optimizer.load_state_dict(state)

    def set_learning_rate(self, learning_rate: float) -> None:
        for group in self.param_groups:
            group["lr"] = learning_rate


def build_optimizer(draft: torch.nn.Module, learning_rate: float) -> OptimizerBundle:
    matrix_parameters = []
    auxiliary_parameters = []
    for parameter in draft.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 2:
            matrix_parameters.append(parameter)
        else:
            auxiliary_parameters.append(parameter)
    if not matrix_parameters or not auxiliary_parameters:
        raise ValueError("Muon requires matrix and auxiliary parameter groups")
    return OptimizerBundle(
        torch.optim.Muon(
            matrix_parameters,
            lr=learning_rate,
            weight_decay=0.01,
            momentum=0.95,
            nesterov=True,
            ns_coefficients=(3.4445, -4.775, 2.0315),
            eps=1e-7,
            ns_steps=5,
            adjust_lr_fn="match_rms_adamw",
        ),
        torch.optim.AdamW(
            auxiliary_parameters,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        ),
    )


def load_models(args: Any) -> tuple[Any, Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, local_files_only=True)
    target = (
        AutoModelForCausalLM.from_pretrained(
            args.target_model,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    target.requires_grad_(False)
    config = AutoConfig.from_pretrained(
        args.draft_model, local_files_only=True, trust_remote_code=True
    )
    draft = AutoModel.from_config(
        config,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    draft.train()
    if draft.mask_token_id is None or draft.mask_token_id >= target.config.vocab_size:
        raise ValueError("draft mask token is incompatible with target vocabulary")
    return tokenizer, target, draft


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def save_final_weights(
    draft: torch.nn.Module, output_dir: Path, metadata: dict[str, Any]
) -> None:
    temporary = output_dir / "final-model.pt.tmp"
    destination = output_dir / "final-model.pt"
    state = {name: tensor.detach().cpu() for name, tensor in draft.state_dict().items()}
    torch.save(state, temporary)
    os.replace(temporary, destination)
    atomic_json(output_dir / "completed.json", metadata)


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer, target, draft = load_models(args)
    architecture = configure_architecture(draft, args.experiment)
    optimizer = build_optimizer(draft, args.learning_rate)
    stream = rows(args.data)
    torch.cuda.reset_peak_memory_stats()
    started_run = time.time()

    atomic_json(
        args.output_dir / "run.json",
        {
            **architecture,
            "steps": args.steps,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "anchors_per_sample": args.anchors_per_sample,
            "update_log_interval": args.update_log_interval,
            "loss": "equal_weight_token_ce",
            "selector_loss": "candidate_ce"
            if hasattr(draft, "candidate_selector")
            else None,
            "git_revision": os.environ.get("MTP_RUN_REVISION"),
            "target_model": args.target_model.name,
            "draft_model": args.draft_model.name,
            "data": str(args.data),
        },
    )

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
            valid_anchors, min(args.anchors_per_sample, len(valid_anchors))
        )
        torch.cuda.synchronize()
        started_step = time.perf_counter()
        features = target_features(target, input_ids, draft.target_layer_ids)
        optimizer.zero_grad(set_to_none=True)
        output = compute_loss(
            target=target,
            draft=draft,
            input_ids=input_ids,
            features=features,
            anchors=anchors,
            ngram_beta=architecture["ngram_beta"],
        )
        output.objective.backward()
        if any(parameter.grad is not None for parameter in target.parameters()):
            raise RuntimeError("frozen target unexpectedly received gradients")
        grad_metrics = gradient_norms(draft)
        if not all(
            torch.isfinite(parameter.grad).all()
            for parameter in draft.parameters()
            if parameter.grad is not None
        ):
            raise FloatingPointError("non-finite draft gradient")
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
        elapsed = time.perf_counter() - started_step
        metrics = {
            "step": step,
            "experiment": args.experiment,
            "seed": args.seed,
            "sample_id": row["id"],
            "tokens": input_ids.shape[1],
            "anchors": anchors,
            "loss": output.objective.detach().item(),
            "plain_ce": output.plain_ce.item(),
            "position_ce": output.position_ce.tolist(),
            "position_accuracy": output.position_accuracy.tolist(),
            "greedy_acceptance_length": output.greedy_acceptance_length.item(),
            "selector_loss": output.selector_loss.item(),
            "selector_candidate_recall": output.selector_candidate_recall.item(),
            "selector_teacher_accuracy": output.selector_teacher_accuracy.item(),
            "selector_path_acceptance_length": output.selector_path_acceptance_length.item(),
            "gradient_norms": grad_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": elapsed,
            "cuda_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        }
        if update_metrics is not None:
            metrics.update(update_metrics)
            metrics["parameter_update_gradient_basis"] = "raw"
        if not all(
            torch.isfinite(torch.tensor(metrics[key]))
            for key in ("loss", "plain_ce", "greedy_acceptance_length")
        ):
            raise FloatingPointError("non-finite training metric")
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        del features, input_ids, output

    completion = {
        **architecture,
        "steps": args.steps,
        "seed": args.seed,
        "elapsed_seconds": time.time() - started_run,
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "final_weights": "final-model.pt" if args.save_final else None,
    }
    if args.save_final:
        save_final_weights(draft, args.output_dir, completion)
    else:
        atomic_json(args.output_dir / "completed.json", completion)


if __name__ == "__main__":
    main()
