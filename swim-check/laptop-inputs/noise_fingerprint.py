# Fingerprint of ACE-Step's initial noise for a seed: same call as prepare_noise() (turbo), shape (1, 6000, 64) for 240 s.
import sys, hashlib, torch
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 5555
g = torch.Generator(device="cuda").manual_seed(seed)
n = torch.randn((1, 6000, 64), generator=g, device="cuda", dtype=torch.float32).cpu()
print(torch.cuda.get_device_name(0), "SMs", torch.cuda.get_device_properties(0).multi_processor_count, "torch", torch.__version__)
print("seed", seed, "first5", [round(v, 6) for v in n.flatten()[:5].tolist()], "mean", round(n.mean().item(), 6), "sha256", hashlib.sha256(n.numpy().tobytes()).hexdigest()[:16])
