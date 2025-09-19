import functools
import sys
from pathlib import Path
from typing import Callable

import circuitsvis as cv
import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from eindex import eindex
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part2_intro_to_mech_interp"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part2_intro_to_mech_interp.tests as tests
from plotly_utils import (
    hist,
    imshow,
    plot_comp_scores,
    plot_logit_attribution,
    plot_loss_difference,
)

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

MAIN = __name__ == "__main__"

print("Getting GPT2...")
gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")

cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal",
    attn_only=True,  # defaults to False
    tokenizer_name="EleutherAI/gpt-neox-20b",
    seed=398,
    use_attn_result=True,
    normalization_type=None,  # defaults to "LN", i.e. layernorm with weights & biases
    positional_embedding_type="shortformer",
)

from huggingface_hub import hf_hub_download

REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"

print("Getting Callum model...")

weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

print("Making model with weights...")

model = HookedTransformer(cfg)
pretrained_weights = t.load(weights_path, map_location=device, weights_only=True)
model.load_state_dict(pretrained_weights)

text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

print("Running model...")

logits, cache = model.run_with_cache(text, remove_batch_dim=True)
attention_patterns = [cache["pattern", 0], cache["pattern", 1]]
tokens = model.to_str_tokens(text)


def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch_size: int = 1
) -> Int[Tensor, "batch_size full_seq_len"]:
    """
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
    """
    t.manual_seed(0)  # for reproducibility
    prefix = (t.ones(batch_size, 1) * model.tokenizer.bos_token_id).long()
    random_tokens = t.randint(0, cfg.d_vocab, [batch_size, seq_len])
    return t.cat((prefix, random_tokens, random_tokens), dim=1)

# %%

print(f"""
Number of layers: {gpt2_small.cfg.n_layers}
Number of heads per layer: {gpt2_small.cfg.n_heads}
Maximum context window: {gpt2_small.cfg.window_size}
""")

# %%

model_description_text = """## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly.

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!"""

logits: Tensor = gpt2_small(model_description_text, return_type="logits")
prediction = logits.argmax(dim=-1).squeeze()[:-1]
tokens = gpt2_small.to_tokens(model_description_text)[:, 1:]
num_correct = (prediction == tokens).sum().item()
print(f'Got {num_correct} right of {tokens.shape[0]} total')

# %%

gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on task-specific datasets."
gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)

print(type(gpt2_logits), type(gpt2_cache))

layer0_pattern_from_cache = gpt2_cache["pattern", 0]

q, k = gpt2_cache["q", 0], gpt2_cache["k", 0]
seq, nhead, headsize = q.shape
layer0_attn_scores = t.einsum('qnh,knh->nqk', q, k)
mask = t.triu(t.ones((seq, seq), dtype=t.bool), diagonal=1).to(device)
layer0_attn_scores.masked_fill_(mask, -1e9)
layer0_pattern_from_q_and_k = (layer0_attn_scores / np.sqrt(headsize)).softmax(-1)
t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
print("Tests passed!")

# %%

print(attention_patterns[0].shape)
for i, attention_pattern in enumerate(attention_patterns):
    html = cv.attention.attention_patterns(
        tokens=tokens,
        attention=attention_pattern,
        attention_head_names=[f"L{i}H{j}" for j in range(cfg.n_heads)]
    )
    with open(f"/home/jwedgwoo/ARENA_3.0/chapter1_transformer_interp/exercises/part2_intro_to_mech_interp/assets/cv_layer{i}.html", "w") as f:
        f.write(str(html))

# %%

def current_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    """

    attn_patterns = [
        cache["pattern", i] for i in range(cfg.n_layers)
    ]

    # define a current-token head as one where at least 50% of total attention pattern is on current token
    current_token_heads = []
    for i, attn_pattern in enumerate(attn_patterns):
        for j in range(cfg.n_heads):
            total_pattern_value = attn_pattern[j].sum().item()
            current_token_value = attn_pattern[j].diagonal().sum().item()
            if current_token_value / total_pattern_value >= 0.5:
                current_token_heads.append(f"{i}.{j}")
    return current_token_heads


def prev_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    """
    
    attn_patterns = [
        cache["pattern", i] for i in range(cfg.n_layers)
    ]

    # define a previous-token head as one where at least 50% of total attention pattern is on current token
    previous_token_heads = []
    for i, attn_pattern in enumerate(attn_patterns):
        for j in range(cfg.n_heads):
            total_pattern_value = attn_pattern[j].sum().item()
            previous_token_value = attn_pattern[j].diagonal(-1).sum().item()
            if previous_token_value / total_pattern_value >= 0.5:
                previous_token_heads.append(f"{i}.{j}")
    return previous_token_heads


def first_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    """

    attn_patterns = [
        cache["pattern", i] for i in range(cfg.n_layers)
    ]

    # define a first-token head as one where at least 50% of total attention pattern is on first token
    first_token_heads = []
    for i, attn_pattern in enumerate(attn_patterns):
        for j in range(cfg.n_heads):
            total_pattern_value = attn_pattern[j].sum().item()
            first_token_value = attn_pattern[j][:, 0].sum().item()
            if first_token_value / total_pattern_value >= 0.5:
                first_token_heads.append(f"{i}.{j}")
    return first_token_heads


print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))

# %%

def run_and_cache_model_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch_size: int = 1
) -> tuple[Tensor, Tensor, ActivationCache]:
    """
    Generates a sequence of repeated random tokens, and runs the model on it, returning (tokens,
    logits, cache). This function should use the `generate_repeated_tokens` function above.

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
        rep_logits: [batch_size, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    """
    
    tokens = generate_repeated_tokens(model, seq_len, batch_size)
    logits, cache = model.run_with_cache(tokens)
    return tokens, logits, cache


def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    logprobs = logits.log_softmax(dim=-1)
    # We want to get logprobs[b, s, tokens[b, s+1]], in eindex syntax this looks like:
    correct_logprobs = eindex(logprobs, tokens, "b s [b s+1]")
    return correct_logprobs

print("Testing repeated tokens...")

seq_len = 50
batch_size = 1
(rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(
    model, seq_len, batch_size
)
rep_cache.remove_batch_dim()
rep_str = model.to_str_tokens(rep_tokens)
model.reset_hooks()
log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()

print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")

plot_loss_difference(log_probs, rep_str, seq_len,
    filename="/home/jwedgwoo/ARENA_3.0/chapter1_transformer_interp/exercises/part2_intro_to_mech_interp/assets/log_diffs.html")

def induction_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    """
    
    attn_patterns = [
        cache["pattern", i] for i in range(cfg.n_layers)
    ]

    # define a induction head as one where at least 50% of total attention pattern is on current token
    seq_len = attn_patterns[0].shape[-1]
    induction_heads = []
    for i, attn_pattern in enumerate(attn_patterns):
        for j in range(cfg.n_heads):
            total_pattern_value = attn_pattern[j].sum().item()
            induction_token_value = attn_pattern[j].diagonal(-(seq_len - 1)).sum().item()
            if induction_token_value / total_pattern_value >= 0.5:
                induction_heads.append(f"{i}.{j}")
    return induction_heads


print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))

# %%

seq_len = 50
batch_size = 10
rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch_size)

# We make a tensor to store the induction score for each head.
# We put it on the model's device to avoid needing to move things between the GPU and CPU,
# which can be slow.
induction_score_store = t.zeros(
    (model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device
)


def induction_score_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"], hook: HookPoint
):
    """
    Calculates the induction score, and stores it in the [layer, head] position of the
    `induction_score_store` tensor.
    """

    induction_stripe = t.diagonal(pattern, dim1=-2, dim2=-1, offset=1-seq_len)
    print(induction_stripe.shape)
    induction_scores = induction_stripe.mean(dim=(0, -1))
    print(induction_scores.shape)
    induction_score_store[hook.layer()] = induction_scores


# We make a boolean filter on activation names, that's true only on attention pattern names
pattern_hook_names_filter = lambda name: name.endswith("pattern")

# Run with hooks (this is where we write to the `induction_score_store` tensor`)
model.run_with_hooks(
    rep_tokens_10,
    return_type=None,  # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(pattern_hook_names_filter, induction_score_hook)],
)

# Plot the induction scores for each head in each layer
fig = imshow(
    induction_score_store,
    labels={"x": "Head", "y": "Layer"},
    title="Induction Score by Head",
    text_auto=".2f",
    width=900,
    height=350,
    return_fig=True
)
fig.write_html("/home/jwedgwoo/ARENA_3.0/chapter1_transformer_interp/exercises/part2_intro_to_mech_interp/assets/induction_scores.html")