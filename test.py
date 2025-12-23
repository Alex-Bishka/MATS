"""
Minimal test: Can we load Gemma-2-9b and GemmaScope SAEs?
"""

from sae_lens import SAE
import torch

print("=== GPU CHECK ===")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    used_mem = torch.cuda.memory_allocated() / 1e9
    print(f"Memory: {total_mem:.1f} GB total, {used_mem:.2f} GB used")

# Step 1: Load model
print("\n=== LOADING GEMMA-2-9B ===")
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "google/gemma-2-9b-it"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model (bf16)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

print(f"✅ Model loaded! Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Quick test
print("\n=== QUICK INFERENCE TEST ===")
inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=5)
print(f"Test output: {tokenizer.decode(outputs[0])}")

# Step 2: Try loading SAEs
print("\n=== LOADING SAEs ===")
try:
    from sae_lens import SAE
    
    print("Loading Layer 20 SAE (for truth feature 107788)...")
    sae_l20, cfg, sparsity = SAE.from_pretrained(
        release="gemma-scope-9b-pt-res-canonical",
        sae_id="layer_20/width_131k/canonical",
        device="cpu"
    )
    print(f"✅ L20 SAE loaded! Shape: {sae_l20.W_enc.shape}")
    
    # print("Loading Layer 9 SAE (for uncertainty feature 80216)...")
    # sae_l9, cfg, sparsity = SAE.from_pretrained(
    #     release="gemma-scope-9b-pt-res-canonical",
    #     sae_id="layer_9/width_131k/canonical", 
    #     device="cuda"
    # )
    # print(f"✅ L9 SAE loaded! Shape: {sae_l9.W_enc.shape}")
    
    print(f"\nTotal memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print("\n✅ ALL SYSTEMS GO - Ready for experiments!")
    
except ImportError as e:
    print("❌ sae_lens not installed. Run: pip install sae-lens")
    print(f"ImportError: {e}")
except Exception as e:
    print(f"❌ SAE loading failed: {e}")
    import traceback
    traceback.print_exc()