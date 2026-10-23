import argparse
import json
import os
import torch
from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer

# The 10 adjacent intervals from the VQ-VAE multiscale ladder (8 -> 128)
LADDER = [8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]
INTERVALS = [(LADDER[i], LADDER[i + 1]) for i in range(len(LADDER) - 1)]

# 11 constant subjects (10 intervals x 11 subjects = 110 sequences)
SUBJECTS = [
    ("dog", "Sam", ["little", "big", "small", "huge", "tiny", "tall"]),
    ("cat", "Tom", ["little", "big", "small", "huge", "tiny", "black"]),
    ("dress", "Mia", ["red", "blue", "green", "yellow", "white", "pink"]),
    ("ball", "Leo", ["red", "blue", "green", "yellow", "big", "small"]),
    ("car", "Dad", ["red", "blue", "green", "black", "white", "fast"]),
    ("box", "Lily", ["little", "big", "small", "huge", "heavy", "old"]),
    ("house", "Max", ["little", "big", "small", "huge", "white", "old"]),
    ("shoes", "Tim", ["clean", "dirty", "new", "old", "red", "black"]),
    ("water", "Sara", ["clean", "dirty", "cold", "warm", "clear", "dark"]),
    ("bird", "Ben", ["little", "big", "small", "tiny", "blue", "yellow"]),
    ("rock", "Anna", ["little", "big", "small", "huge", "heavy", "smooth"]),
]

FILLER_CHUNKS = [
    "went outside to the park.",
    "walked around the yard slowly.",
    "looked around the quiet room.",
    "waited near the wooden door.",
    "stayed by the green tree.",
    "ran across the soft grass.",
    "sat down on the warm floor.",
    "played happily in the morning sun.",
    "watched the clouds in the sky.",
    "listened to the wind blowing softly.",
    "stepped onto the little path.",
    "smiled and waved at a friend.",
]


def load_tokenizer_from_yaml(yaml_path, ck_dir=None, vqvae_ckpt=None):
    CFG = OmegaConf.load(yaml_path)

    # Handle both full generator config (has CFG.tokenizer) and standalone vqvae.yaml
    if "tokenizer" not in CFG:
        tok_cfg = OmegaConf.create({
            "vqvae": CFG.model,
            "pre_tokenizer": CFG.pre_tokenizer,
        })
    else:
        tok_cfg = CFG.tokenizer

    if vqvae_ckpt:
        tok_cfg.vqvae.checkpoint_path = vqvae_ckpt
    elif ck_dir and "checkpoints" in CFG and CFG.checkpoints.get("tokenizer"):
        tok_cfg.vqvae.checkpoint_path = os.path.join(ck_dir, CFG.checkpoints.tokenizer)

    if ck_dir and "checkpoints" in CFG:
        for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
            if CFG.checkpoints.get(name):
                tok_cfg.pre_tokenizer[key] = os.path.join(ck_dir, CFG.checkpoints[name])

    tk = VQVAETokenizer.from_config(tok_cfg)
    return tk.pre_tokenizer


def encode_text(pre_tok, text):
    ids = pre_tok.encode(text)
    if hasattr(ids, "ids"):
        return list(ids.ids)
    if isinstance(ids, torch.Tensor):
        return ids.tolist()
    return list(ids)


def single_token_id(pre_tok, word):
    for prefix in [" ", ""]:
        ids = encode_text(pre_tok, prefix + word)
        if len(ids) == 1:
            return ids[0]
    return None


def build_sequence_with_distance(pre_tok, pad_id, subject, name, adj, min_d, max_d, ctx_len=128):
    target_d = (min_d + max_d) // 2

    prefix_ids = encode_text(pre_tok, f"Once {name} saw a")
    adj_id = single_token_id(pre_tok, adj)
    subj_ids = encode_text(pre_tok, f" {subject}.")

    slot_i = len(prefix_ids)
    current_ids = prefix_ids + [adj_id] + subj_ids
    pre_slot2_ids = encode_text(pre_tok, f" Then {name} looked at the")

    filler_pool = []
    for chunk in FILLER_CHUNKS * 3:
        filler_pool.extend(encode_text(pre_tok, " " + chunk))

    needed = max(0, (slot_i + target_d) - len(current_ids) - len(pre_slot2_ids))
    seq_up_to_slot2 = current_ids + filler_pool[:needed] + pre_slot2_ids
    slot_j = len(seq_up_to_slot2)

    dist = slot_j - slot_i
    if dist <= min_d:
        extra = (min_d + 1) - dist
        seq_up_to_slot2 = current_ids + filler_pool[: needed + extra] + pre_slot2_ids
        slot_j = len(seq_up_to_slot2)
    elif dist >= max_d:
        trim = dist - (max_d - 1)
        seq_up_to_slot2 = current_ids + filler_pool[: max(0, needed - trim)] + pre_slot2_ids
        slot_j = len(seq_up_to_slot2)

    full_ids = seq_up_to_slot2 + [adj_id] + subj_ids
    assert slot_j < ctx_len, f"slot_j ({slot_j}) >= {ctx_len}"
    if len(full_ids) < ctx_len:
        full_ids = full_ids + [pad_id] * (ctx_len - len(full_ids))
    else:
        full_ids = full_ids[:ctx_len]

    return full_ids, slot_i, slot_j


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/eval/tinystories.yaml", help="Path to YAML config")
    parser.add_argument("--ck-dir", default=None, help="Checkpoint directory if using relative paths")
    parser.add_argument("--vqvae-ckpt", default=None, help="Direct path to VQ-VAE checkpoint")
    parser.add_argument("--out", default="ncm_coherence_110.jsonl")
    args = parser.parse_args()

    pre_tok = load_tokenizer_from_yaml(args.config, args.ck_dir, args.vqvae_ckpt)
    pad_id = getattr(pre_tok, "pad_token_id", 1)

    samples = []
    sample_id = 0

    for group_idx, (x1, x2) in enumerate(INTERVALS):
        for subject, name, cand_words in SUBJECTS:
            valid_cands = {
                w: single_token_id(pre_tok, w)
                for w in cand_words
                if single_token_id(pre_tok, w) is not None
            }
            gt_word = list(valid_cands.keys())[0]
            input_ids, slot_i, slot_j = build_sequence_with_distance(
                pre_tok, pad_id, subject, name, gt_word, x1, x2
            )
            samples.append({
                "id": sample_id,
                "group_idx": group_idx,
                "interval": [x1, x2],
                "distance": slot_j - slot_i,
                "subject": subject,
                "ground_truth_adj": gt_word,
                "slot_i": slot_i,
                "slot_j": slot_j,
                "candidate_words": list(valid_cands.keys()),
                "candidate_ids": list(valid_cands.values()),
                "input_ids": input_ids,
            })
            sample_id += 1

    with open(args.out, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"Saved {len(samples)} sequences across {len(INTERVALS)} intervals to {args.out}")


if __name__ == "__main__":
    main()