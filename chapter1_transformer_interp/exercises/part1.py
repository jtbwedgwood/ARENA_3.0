# %%

import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import datasets
import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import wandb
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part1_transformer_from_scratch"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_transformer_from_scratch.solutions as solutions
import part1_transformer_from_scratch.tests as tests
from plotly_utils import imshow

MAIN = __name__ == "__main__"


@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


def rand_float_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randn(shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple):
        output = output[0]
    print("Output shape:", output.shape, "\n")


def rand_int_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randint(100, 1000, shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple):
        output = output[0]
    print("Output shape:", output.shape, "\n")


def load_gpt2_test(cls, gpt2_layer, input):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
    print("Input shape:", input.shape)
    output = layer(input)
    if isinstance(output, tuple):
        output = output[0]
    print("Output shape:", output.shape)
    try:
        reference_output = gpt2_layer(input)
    except:
        reference_output = gpt2_layer(input, input, input)
    print("Reference output shape:", reference_output.shape, "\n")
    comparison = t.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    print(f"{comparison.sum() / comparison.numel():.2%} of the values are correct\n")
    assert 1 - (comparison.sum() / comparison.numel()) < 1e-5, (
        "More than 0.01% of the values are incorrect"
    )

reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,  # you'll learn about these arguments later!
)
reference_text = "bla bla bla"
tokens = reference_gpt2.to_tokens(reference_text).to(device)
logits, cache = reference_gpt2.run_with_cache(tokens)


class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        mean = residual.mean(dim=-1, keepdim=True)
        var = residual.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (residual - mean) / t.sqrt(var + self.cfg.layer_norm_eps)
        return normalized * self.w + self.b


class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:

        # tokens: [B, P]
        assert tokens.dtype == t.long, tokens.dtype
        assert tokens.device == self.W_E.device, (tokens.device, self.W_E.device)

        V = self.W_E.size(0)  # = cfg.d_vocab
        mn = int(tokens.min().item()); mx = int(tokens.max().item())
        assert mn >= 0 and mx < V, f"Bad token id(s): min={mn}, max={mx}, d_vocab={V}"
        return self.W_E[tokens]


class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        batch, position = tokens.shape
        return einops.repeat(self.W_pos[:position], "position d_model -> batch position d_model", batch=batch)


class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=device))

    def forward(
        self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        
        # set q, k, v
        q = t.einsum('bse,neh->bsnh', normalized_resid_pre, self.W_Q) + self.b_Q
        k = t.einsum('bse,neh->bsnh', normalized_resid_pre, self.W_K) + self.b_K
        v = t.einsum('bse,neh->bsnh', normalized_resid_pre, self.W_V) + self.b_V

        # calculate attention scores
        attn_scores = t.einsum('bqnh,bknh->bnqk', q, k)
        attn_scores_scaled = attn_scores / np.sqrt(self.cfg.d_head)
        attn_scores_masked = self.apply_causal_mask(attn_scores_scaled)
        attn_pattern = F.softmax(attn_scores_masked, dim=-1)

        # take weighted sum of value vectors
        z = t.einsum('bnqk,bknh->bqnh', attn_pattern, v)

        # calculate output
        result = t.einsum('bqnh,nhe->bqne', z, self.W_O)
        attn_out = result.sum(dim=2) + self.b_O
        return attn_out

    def apply_causal_mask(
        self,
        attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"],
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        all_ones = t.ones(attn_scores.shape[-2], attn_scores.shape[-1], device=attn_scores.device)
        upper_triangle = t.triu(all_ones, diagonal=1).bool()
        return attn_scores.masked_fill(
            upper_triangle,
            self.IGNORE
        )


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        linear1_out = t.einsum('bpe,em->bpm', normalized_resid_mid, self.W_in) + self.b_in
        gelu_out = gelu_new(linear1_out)
        linear2_out = t.einsum('bpm,me->bpe', gelu_out, self.W_out) + self.b_out
        return linear2_out


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(
        self, resid_pre: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        resid = resid_pre
        attn_out = self.attn(self.ln1(resid_pre))
        resid = resid + attn_out
        mlp_out = self.mlp(self.ln2(resid))
        resid = resid + mlp_out
        return resid

class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))

    def forward(
        self, normalized_resid_final: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        return t.einsum('bpe,ev->bpv', normalized_resid_final, self.W_U) + self.b_U

class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        resid = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            resid = block(resid)
        resid = self.ln_final(resid)
        return self.unembed(resid)

# %%

print(logits.shape)

probs = logits.softmax(dim=-1)
print(probs.shape)

next_token = logits[0, -1].argmax(dim=-1)
next_char = reference_gpt2.to_string(next_token)
print(repr(next_char))

print(f"Sequence so far: {reference_gpt2.to_string(tokens)[0]!r}")

for i in range(10):
    print(f"{tokens.shape[-1] + 1}th char = {next_char!r}")
    # Define new input sequence, by appending the previously generated token
    tokens = t.cat([tokens, next_token[None, None]], dim=-1)
    # Pass our new sequence through the model, to get new output
    logits = reference_gpt2(tokens)
    # Get the predicted token at the end of our sequence
    next_token = logits[0, -1].argmax(dim=-1)
    # Decode and print the result
    next_char = reference_gpt2.to_string(next_token)

# %%

rand_float_test(LayerNorm, [2, 4, 768])
load_gpt2_test(LayerNorm, reference_gpt2.ln_final, cache["resid_post", 11])

rand_int_test(Embed, [2, 4])
load_gpt2_test(Embed, reference_gpt2.embed, tokens)

rand_int_test(PosEmbed, [2, 4])
load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)

# %%

tests.test_causal_mask(Attention.apply_causal_mask)
rand_float_test(Attention, [2, 4, 768])
load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["normalized", 0, "ln1"])

# %%

rand_float_test(MLP, [2, 4, 768])
load_gpt2_test(MLP, reference_gpt2.blocks[0].mlp, cache["normalized", 0, "ln2"])

# %%

rand_float_test(TransformerBlock, [2, 4, 768])
load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])

# %%

rand_float_test(Unembed, [2, 4, 768])
load_gpt2_test(Unembed, reference_gpt2.unembed, cache["ln_final.hook_normalized"])

# %%

rand_int_test(DemoTransformer, [2, 4])
load_gpt2_test(DemoTransformer, reference_gpt2, tokens)

# %%

demo_gpt2 = DemoTransformer(Config(debug=False)).to(device)
demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)

demo_logits = demo_gpt2(tokens)

test_string = """Flibbityfloops are most often used by"""
for i in range(100):
    test_tokens = reference_gpt2.to_tokens(test_string).to(device)
    demo_logits = demo_gpt2(test_tokens)
    test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())

print(test_string)

# %%

def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = (
        log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    )

    return log_probs_for_tokens

model_cfg = Config(
    debug=False,
    d_model=32,
    n_heads=16,
    d_head=2,
    d_mlp=32 * 4,
    n_layers=4,
    n_ctx=128,
    d_vocab=reference_gpt2.cfg.d_vocab,
)
model = DemoTransformer(model_cfg)

@dataclass
class TransformerTrainingArgs:
    batch_size: int = 32
    epochs: int = 10
    max_steps_per_epoch: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-2
    wandb_project: str | None = "day1-demotransformer"
    wandb_name: str | None = None


args = TransformerTrainingArgs()

dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")

tokenized_dataset = tokenize_and_concatenate(
    dataset,
    reference_gpt2.tokenizer,
    streaming=False,
    max_length=model.cfg.n_ctx,
    column_name="text",
    add_bos_token=True,
    num_proc=4,
)

dataset_dict = tokenized_dataset.train_test_split(test_size=1000)


class TransformerTrainer:
    def __init__(self, args: TransformerTrainingArgs, model: DemoTransformer):
        super().__init__()
        self.model = model
        self.args = args
        self.sampler = solutions.TransformerSampler(self.model, reference_gpt2.tokenizer)
        self.optimizer = t.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.step = 0

        self.train_loader = DataLoader(
            dataset_dict["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            dataset_dict["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def training_step(self, batch: dict[str, Int[Tensor, "batch seq"]]) -> Float[Tensor, ""]:
        """
        Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

        Remember that `batch` is a dictionary with the single key 'tokens'.
        """
        tokens = batch['tokens'].to(device)
        logits = self.model(tokens)
        pred_log_probs = get_log_probs(logits, tokens)
        loss = -pred_log_probs.mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1
        wandb.log({"train_loss": loss}, step=self.step)
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        """
        Evaluate the model on the test set and return the accuracy.
        """
        self.model.eval()
        total_correct, total_samples = 0, 0

        for batch in tqdm(self.test_loader):
            tokens = batch['tokens'].to(device)
            logits = self.model(tokens)[:, :-1]
            pred = logits.argmax(dim=-1)
            total_correct += (pred == tokens[:, 1:]).sum().item()
            total_samples += tokens.size(0) * (tokens.size(1) - 1)

        accuracy = total_correct / total_samples
        wandb.log({"accuracy": accuracy}, step=self.step)

        self.model.train()
        return accuracy

    def train(self):
        """
        Trains the model, for `self.args.epochs` epochs. Also handles wandb initialisation, and early stopping
        for each epoch at `self.args.max_steps_per_epoch` steps.
        """
        wandb.login(key=open('/home/jwedgwoo/ARENA_3.0/api_key_wandb').read())
        wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)
        accuracy = np.nan

        progress_bar = tqdm(total=self.args.max_steps_per_epoch * self.args.epochs)

        for epoch in range(self.args.epochs):
            for i, batch in enumerate(self.train_loader):
                loss = self.training_step(batch)
                progress_bar.update()
                progress_bar.set_description(
                    f"Epoch {epoch + 1}, loss: {loss:.3f}, accuracy: {accuracy:.3f}"
                )
                if i >= self.args.max_steps_per_epoch:
                    break

            accuracy = self.evaluate()
            sample_text = self.sampler.sample("Once upon a time", max_tokens_generated=50)
            print(sample_text)

        wandb.finish()


# See the full run here: https://api.wandb.ai/links/dquarel/nrxuwnv7
model = DemoTransformer(model_cfg).to(device)
args = TransformerTrainingArgs()
trainer = TransformerTrainer(args, model)
trainer.train()

# %%

class TransformerSampler:
    def __init__(self, model: DemoTransformer, tokenizer: GPT2TokenizerFast):
        self.model = model
        self.cfg = model.cfg
        self.tokenizer = tokenizer

    @t.inference_mode()
    def sample(self, prompt: str, max_tokens_generated=100, verbose=False, **kwargs) -> str:
        """
        Returns a string of autoregressively generated text, starting from the prompt.

        Sampling terminates at max_tokens_generated, or when the model generates an end-of-sequence token. kwargs are
        passed to sample_next_token, to give detailed instructions on how new tokens are chosen.
        """

        self.model.eval()
        tokens = self.tokenizer.encode(prompt)
        tokens_tensor = t.tensor(tokens, dtype=t.long).unsqueeze(0).to(device)
        for _ in range(max_tokens_generated):
            logits = self.model(tokens_tensor)[0, -1]
            next_token = self.sample_next_token(tokens_tensor[0], logits, **kwargs)
            tokens_tensor = t.cat([tokens_tensor, t.tensor([[next_token]]).to(device)], dim=1)
            if next_token == self.tokenizer.eos_token_id:
                break
        return self.tokenizer.decode(tokens_tensor[0])

    @staticmethod
    def sample_next_token(
        input_ids: Int[Tensor, "seq_len"],
        logits: Float[Tensor, "d_vocab"],
        temperature=1.0,
        top_k=0,
        top_p=0.0,
        frequency_penalty=0.0,
        seed=None,
    ) -> int:
        assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
        assert temperature >= 0, "Temperature should be non-negative"
        assert 0 <= top_p <= 1.0, "Top-p must be a probability"
        assert 0 <= top_k, "Top-k must be non-negative"
        assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

        # Set random seeds for reproducibility
        if seed is not None:
            t.manual_seed(seed)
            np.random.seed(seed)

        # Apply all the specialized sampling methods
        if temperature == 0:
            return TransformerSampler.greedy_search(logits)
        elif temperature != 1.0:
            logits = TransformerSampler.apply_temperature(logits, temperature)
        if frequency_penalty != 0.0:
            logits = TransformerSampler.apply_frequency_penalty(
                input_ids, logits, frequency_penalty
            )
        if top_k > 0:
            return TransformerSampler.sample_top_k(logits, top_k)
        if top_p > 0.0:
            return TransformerSampler.sample_top_p(logits, top_p)
        return TransformerSampler.sample_basic(logits)

    @staticmethod
    def greedy_search(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Returns the most likely token (as an int).
        """
        return logits.argmax().item()

    @staticmethod
    def apply_temperature(
        logits: Float[Tensor, "d_vocab"], temperature: float
    ) -> Float[Tensor, "d_vocab"]:
        """
        Applies temperature scaling to the logits.
        """
        return logits / temperature

    @staticmethod
    def apply_frequency_penalty(
        input_ids: Int[Tensor, "seq_len"], logits: Float[Tensor, "d_vocab"], freq_penalty: float
    ) -> Float[Tensor, "d_vocab"]:
        """
        Applies a frequency penalty to the logits.
        """
        bin_counts = t.bincount(input_ids, minlength=logits.shape[-1])
        return logits - freq_penalty * bin_counts

    @staticmethod
    def sample_basic(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Samples from the distribution defined by the logits.
        """
        return t.distributions.categorical.Categorical(logits=logits).sample().item()

    @staticmethod
    def sample_top_k(logits: Float[Tensor, "d_vocab"], k: int) -> int:
        """
        Samples from the top k most likely tokens.
        """
        values, indices = t.topk(logits, k)
        mask = t.full_like(logits, float("-inf"))
        mask.scatter_(0, indices, values)
        return t.distributions.categorical.Categorical(logits=mask).sample().item()

    @staticmethod
    def sample_top_p(
        logits: Float[Tensor, "d_vocab"], top_p: float, min_tokens_to_keep: int = 1
    ) -> int:
        """
        Samples from the most likely tokens which make up at least p cumulative probability.
        """
        sorted_logits, _ = logits.sort(descending=True)
        sorted_softmax = sorted_logits.softmax(dim=-1)
        cumsum = 0
        for k, prob in enumerate(sorted_softmax):
            cumsum += prob
            if cumsum >= top_p and k >= min_tokens_to_keep:
                break
        values, indices = t.topk(logits, k)
        mask = t.full_like(logits, float("-inf"))
        mask.scatter_(0, indices, values)
        return t.distributions.categorical.Categorical(logits=mask).sample().item()

    @t.inference_mode()
    def beam_search(
        self,
        prompt: str,
        num_return_sequences: int,
        num_beams: int,
        max_new_tokens: int,
        no_repeat_ngram_size: int | None = None,
    ) -> list[tuple[float, str]]:
        """
        Implements a beam search, by repeatedly performing the `generate` and `filter` steps (starting from the initial
        prompt) until either of the two stopping criteria are met: (1) we've generated `max_new_tokens` tokens, or (2)
        we've generated `num_returns_sequences` terminating sequences.
        """
        raise NotImplementedError()


t.set_grad_enabled(False)  # gradients are not necessary for sampling

model = DemoTransformer(Config()).to(device)
model.load_state_dict(reference_gpt2.state_dict(), strict=False)
tokenizer = reference_gpt2.tokenizer
sampler = TransformerSampler(model, tokenizer)

prompt = "John and Mary went to the"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
logits = model(input_ids)[0, -1]

expected_top_10pct = {
    " church": 0.0648,
    " house": 0.0367,  # These are the two most likely tokens, and add up to >10%
}
top_10pct_sum = sum(expected_top_10pct.values())

observed_freqs = defaultdict(int)

N = 10_000
for _ in tqdm(range(N)):
    token = TransformerSampler.sample_next_token(input_ids.squeeze(), logits, top_p=0.1)
    observed_freqs[tokenizer.decode(token)] += 1

for word in expected_top_10pct:
    expected_freq = expected_top_10pct[word] / top_10pct_sum
    observed_freq = observed_freqs[word] / N
    print(
        f"Word: {word!r:<9}. Expected freq {expected_freq:.4f}, observed freq {observed_freq:.4f}"
    )
    assert abs(observed_freq - expected_freq) < 0.01, (
        "Try increasing N if this fails by a small amount."
    )