import argparse
import json
import os
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# Monkeypatch load_state_dict to ignore stripped EMA training buffers
_orig_load = torch.nn.Module.load_state_dict
def _safe_load_state_dict(self, state_dict, strict=True, *args, **kwargs):
    res = _orig_load(self, state_dict, strict=False, *args, **kwargs)
    real_missing = [
        k for k in res.missing_keys
        if not any(buf in k for buf in ("ema_cluster_size", "ema_w", "codebook_hit"))
    ]
    if strict and (real_missing or res.unexpected_keys):
        raise RuntimeError(f"Missing: {real_missing}\nUnexpected: {res.unexpected_keys}")
    return res
torch.nn.Module.load_state_dict = _safe_load_state_dict

from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer


def setup_models(config_path, ck_dir=None, vqvae_ckpt=None):
    CFG = OmegaConf.load(config_path)

    if vqvae_ckpt:
        CFG.tokenizer.vqvae.checkpoint_path = vqvae_ckpt
    elif ck_dir and "checkpoints" in CFG and CFG.checkpoints.get("tokenizer"):
        CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck_dir, CFG.checkpoints.tokenizer)

    if ck_dir and "checkpoints" in CFG:
        for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
            if CFG.checkpoints.get(name):
                CFG.tokenizer.pre_tokenizer[key] = os.path.join(ck_dir, CFG.checkpoints[name])

    tk = VQVAETokenizer.from_config(CFG.tokenizer)
    tk.vqvae = tk.vqvae.cuda().eval()
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).cuda()
    return CFG, tk, net


def load_generator_checkpoint(net, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_sd = sd.get("model", sd)
    clean_sd = {
        k.replace("_orig_mod.", "").replace("module.", ""): v
        for k, v in model_sd.items()
    }
    net.load_state_dict(clean_sd, strict=True)
    net.eval()
    return net


@torch.no_grad()
def extract_token_logits(tk, net, x, mode="generator"):
    """
    Returns [B, 128, 8192] token logits over the BPE vocabulary.
    - mode == 'vqvae': encodes x through VQ-VAE and decodes back to token logits.
    - mode == 'generator': seeds level-0 from x (if needed), generates all 12 concept levels
      via NextConceptTransformer.generate(), and decodes predicted concept codes via tk.vqvae.decode().
    """
    if mode == "vqvae":
        logits, _, _, _, _ = tk.vqvae(x)
        return logits

    # Prepare prefix_tokens of length >= net.prefix_len
    P = net.prefix_len
    if x.shape[1] < P:
        pad_id = getattr(tk.pre_tokenizer, "pad_token_id", 1)
        prefix_tokens = F.pad(x, (0, P - x.shape[1]), value=pad_id)
    else:
        prefix_tokens = x[:, :P]

    # Seed level 0 from ground-truth VQ-VAE encoding of x if model doesn't predict level 0 directly
    gt_idx = tk.vqvae.encode(x)
    l0 = tk.vqvae.vq.levels[0]
    idx_seed = None if (net.query_level0 or net.predict_level0) else gt_idx[:, :l0]

    # Greedy concept rollout (temperature=0.0) across the 12 hierarchy levels
    pred_idx = net.generate(
        idx=idx_seed,
        prefix_tokens=prefix_tokens,
        temperature=0.0,
        use_kv_cache=False,
    )
    # Decode predicted multiscale concept indices -> [B, 128, 8192] token logits
    logits = tk.vqvae.decode(pred_idx)
    return logits


def evaluate_spread_and_contradiction(tk, net, dataset, mode="generator"):
    group_stats = {}

    for item in dataset:
        interval_str = f"{item['interval'][0]}-{item['interval'][1]}"
        slot_i, slot_j = item["slot_i"], item["slot_j"]
        cand_ids = torch.tensor(item["candidate_ids"], device="cuda")

        x = torch.tensor([item["input_ids"]], dtype=torch.long, device="cuda")
        logits = extract_token_logits(tk, net, x, mode=mode)[0]  # [128, 8192]

        probs_i = F.softmax(logits[slot_i, cand_ids], dim=-1)
        probs_j = F.softmax(logits[slot_j, cand_ids], dim=-1)

        entropy_i = -(probs_i * torch.log(probs_i + 1e-9)).sum().item()
        entropy_j = -(probs_j * torch.log(probs_j + 1e-9)).sum().item()

        same_word_prob = torch.sum(probs_i * probs_j).item()
        contradiction_prob = 1.0 - same_word_prob

        if interval_str not in group_stats:
            group_stats[interval_str] = {
                "contradiction": [],
                "same_word": [],
                "entropy": [],
            }
        group_stats[interval_str]["contradiction"].append(contradiction_prob)
        group_stats[interval_str]["same_word"].append(same_word_prob)
        group_stats[interval_str]["entropy"].append(0.5 * (entropy_i + entropy_j))

    return {
        interval: {
            "contradiction_prob": sum(v["contradiction"]) / len(v["contradiction"]),
            "same_word_prob": sum(v["same_word"]) / len(v["same_word"]),
            "entropy_spread": sum(v["entropy"]) / len(v["entropy"]),
        }
        for interval, v in group_stats.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/eval/tinystories.yaml", help="Path to tinystories.yaml")
    parser.add_argument("--ck-dir", default=None, help="Directory `ck` containing checkpoints")
    parser.add_argument("--vqvae-ckpt", default=None, help="Explicit path to VQ-VAE .pt checkpoint")
    parser.add_argument("--gen-ckpts", nargs="*", default=[], help="Paths to generator checkpoints (10k, 25k, 50k, 75k)")
    parser.add_argument("--data", default="ncm_coherence_110.jsonl")
    parser.add_argument("--out", default="contradiction_results.json")
    args = parser.parse_args()

    with open(args.data, "r") as f:
        dataset = [json.loads(line) for line in f]

    CFG, tk, net = setup_models(args.config, args.ck_dir, args.vqvae_ckpt)
    all_results = {}

    print("\n=== Evaluating VQ-VAE Bottleneck ===")
    vq_res = evaluate_spread_and_contradiction(tk, net, dataset, mode="vqvae")
    all_results["vqvae"] = vq_res
    print(f"{'Interval':<10} | {'Contradiction %':<17} | {'Same-Word %':<13} | {'Entropy Spread'}")
    print("-" * 62)
    for interval_str, m in vq_res.items():
        print(
            f"{interval_str:<10} | {m['contradiction_prob']*100:>15.2f}% | "
            f"{m['same_word_prob']*100:>11.2f}% | {m['entropy_spread']:.4f}"
        )

    for ckpt_path in args.gen_ckpts:
        print(f"\n=== Evaluating Generator Checkpoint: {ckpt_path} ===")
        load_generator_checkpoint(net, ckpt_path)
        res = evaluate_spread_and_contradiction(tk, net, dataset, mode="generator")
        all_results[ckpt_path] = res

        print(f"{'Interval':<10} | {'Contradiction %':<17} | {'Same-Word %':<13} | {'Entropy Spread'}")
        print("-" * 62)
        for interval_str, m in res.items():
            print(
                f"{interval_str:<10} | {m['contradiction_prob']*100:>15.2f}% | "
                f"{m['same_word_prob']*100:>11.2f}% | {m['entropy_spread']:.4f}"
            )

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()