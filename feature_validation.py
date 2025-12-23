"""
Difference in Means: Validate Feature 107788 is a "Truth" feature
"""

import torch
import numpy as np
from scipy import stats
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens.saes.sae import SAE

FEATURE_IDX_L9 = 80216  # L9 Uncertainty
FEATURE_IDX_L20 = 107788  # L20 Truth

# FEATURE_IDX_L9 = 10441  # L9 Uncertainty
# FEATURE_IDX_L20 = 59115  # L20 Truth

FEATURE_IDX_L9 = 122482
FEATURE_IDX_L20 = 80603

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
print(f"Model loaded: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Load SAE on CPU to save GPU memory
print("Loading L20 SAE on CPU...")
sae_l20, cfg, sparsity = SAE.from_pretrained(
    release="gemma-scope-9b-it-res-canonical",  # IT = instruction-tuned
    sae_id="layer_20/width_131k/canonical",  # Also different SAE variant
    device="cpu"
)
print("SAE loaded on CPU")

# True/False pairs (matched structure)
pairs = [
    ("The capital of France is Paris", "The capital of France is Berlin"),
    ("The capital of Japan is Tokyo", "The capital of Japan is Beijing"),
    ("Water freezes at zero degrees Celsius", "Water freezes at fifty degrees Celsius"),
    ("The Earth orbits the Sun", "The Sun orbits the Earth"),
    ("Humans have two lungs", "Humans have four lungs"),
    ("Light travels faster than sound", "Sound travels faster than light"),
    ("Two plus two equals four", "Two plus two equals five"),
    ("A triangle has three sides", "A triangle has four sides"),
    ("Cats are mammals", "Cats are reptiles"),
    ("Birds have feathers", "Birds have scales"),
    ("Fish breathe through gills", "Fish breathe through lungs"),
    ("Spiders have eight legs", "Spiders have six legs"),
    ("Whales are mammals", "Whales are fish"),
    ("The moon orbits Earth", "Earth orbits the moon"),
    ("Diamonds are made of carbon", "Diamonds are made of silicon"),
    ("Oxygen is needed for fire", "Nitrogen is needed for fire"),
    ("Ice is frozen water", "Ice is frozen air"),
    ("The sun is a star", "The sun is a planet"),
    ("Penguins are birds", "Penguins are mammals"),
    ("Sound needs a medium to travel", "Sound travels through vacuum"),
]

def get_max_feature_activation(text, layer, feature_idx):
    """Get MAX activation of a feature across all token positions."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    activation = {}
    def hook(module, input, output):
        activation['resid'] = output[0].detach()
    
    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    
    # Get all positions: [seq_len, hidden_dim]
    resid_all = activation['resid'][0, :, :].cpu().float()
    
    # Encode all positions through SAE
    encoded = sae_l20.encode(resid_all)  # [seq_len, n_features]
    
    # Max pool across sequence for this feature
    max_act = encoded[:, feature_idx].max().item()
    return max_act


def get_l20_residual(text):
    """Get residual stream at layer 20, last token."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    activation = {}
    def hook(module, input, output):
        activation['resid'] = output[0].detach()
    
    handle = model.model.layers[20].register_forward_hook(hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    
    return activation['resid'][0, -1, :].cpu().float()  # Move to CPU for SAE


def get_max_feature_activation_l9(text, feature_idx):
    """Same as before but for layer 9"""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    activation = {}
    def hook(module, input, output):
        activation['resid'] = output[0].detach()
    handle = model.model.layers[9].register_forward_hook(hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    resid_all = activation['resid'][0, :, :].cpu().float()
    encoded = sae_l9.encode(resid_all)
    return encoded[:, feature_idx].max().item()


# Collect activations - BOTH last-token and max-pooled
print("\nRunning difference-in-means experiment...")
true_acts_last = []
false_acts_last = []
true_acts_max = []
false_acts_max = []

for true_stmt, false_stmt in tqdm(pairs):
    # Last-token method (original)
    resid_true = get_l20_residual(true_stmt)
    resid_false = get_l20_residual(false_stmt)
    with torch.no_grad():
        feat_true_last = sae_l20.encode(resid_true.unsqueeze(0))[0, FEATURE_IDX_L20].item()
        feat_false_last = sae_l20.encode(resid_false.unsqueeze(0))[0, FEATURE_IDX_L20].item()
    true_acts_last.append(feat_true_last)
    false_acts_last.append(feat_false_last)
    
    # Max-pooled method (new)
    feat_true_max = get_max_feature_activation(true_stmt, layer=20, feature_idx=FEATURE_IDX_L20)
    feat_false_max = get_max_feature_activation(false_stmt, layer=20, feature_idx=FEATURE_IDX_L20)
    true_acts_max.append(feat_true_max)
    false_acts_max.append(feat_false_max)

# Print both
print("\n" + "="*50)
print("Per-pair breakdown (Last-token | Max-pooled):")
print("="*50)
for i, (t, f) in enumerate(pairs):
    print(f"True: {true_acts_last[i]:.2f} | {true_acts_max[i]:.2f}  "
          f"False: {false_acts_last[i]:.2f} | {false_acts_max[i]:.2f}  | {t[:30]}...")

print("\n" + "="*50)
print("SUMMARY:")
print("="*50)
print("Last-token method:")
print(f"  Mean TRUE:  {np.mean(true_acts_last):.2f}, Mean FALSE: {np.mean(false_acts_last):.2f}")
print("Max-pooled method:")
print(f"  Mean TRUE:  {np.mean(true_acts_max):.2f}, Mean FALSE: {np.mean(false_acts_max):.2f}")

t_stat, p_val = stats.ttest_rel(true_acts_max, false_acts_max)
print(f"Max-pooled t-test: t={t_stat:.3f}, p={p_val:.4f}")



print("\n" + "="*50)
print("Lexical check for 'truth' keyword activation:")
test_prompts = [
    # Should NOT fire (true statements, no word "truth")
    "The capital of France is Paris",
    "Water freezes at zero degrees",
    "The Earth orbits the Sun",
    
    # SHOULD fire (contains "truth" or related words)
    "The truth is that Paris is the capital",
    "To tell the truth, I don't know",
    "This statement is true",
    "The truth matters",
    "In truth, water freezes at zero",
]

for prompt in test_prompts:
    resid = get_l20_residual(prompt)
    encoded = sae_l20.encode(resid.unsqueeze(0))
    act = encoded[0, FEATURE_IDX_L20].item()
    print(f"{act:.2f} | {prompt}")






# Test
print("\n" + "="*50)
print(f"Max-pooled activation for feature {FEATURE_IDX_L20}:")
print("\nShould NOT fire:")
for prompt in ["The capital of France is Paris", "Water freezes at zero degrees", "The Earth orbits the Sun"]:
    act = get_max_feature_activation(prompt, layer=20, feature_idx=FEATURE_IDX_L20)
    print(f"  {act:.2f} | {prompt}")

print("\nSHOULD fire:")
for prompt in ["The truth is that Paris is the capital", "To tell the truth, I don't know", "The truth matters"]:
    act = get_max_feature_activation(prompt, layer=20, feature_idx=FEATURE_IDX_L20)
    print(f"  {act:.2f} | {prompt}")




# Load L9 SAE
print("Loading L9 SAE...")
sae_l9, _, _ = SAE.from_pretrained(
    release="gemma-scope-9b-it-res-canonical",
    sae_id="layer_9/width_131k/canonical",  # Check Neuronpedia for exact ID
    device="cpu"
)

print("\n" + "="*50)
print(f"Max-pooled activation for feature {FEATURE_IDX_L9} (L9 'Uncertainty'):")
print(f"\nL9 Feature {FEATURE_IDX_L9} (Uncertainty):")
print("\nShould NOT fire:")
for p in ["The capital is Paris", "Water freezes at zero", "The answer is 42"]:
    print(f"  {get_max_feature_activation_l9(p, FEATURE_IDX_L9):.2f} | {p}")

print("\nSHOULD fire:")
for p in ["Maybe the capital is Paris", "Perhaps water freezes", "The answer might be 42"]:
    print(f"  {get_max_feature_activation_l9(p, FEATURE_IDX_L9):.2f} | {p}")