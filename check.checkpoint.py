import sys
from pathlib import Path
import torch

project_dir = Path(__file__).parent
simswap_dir = project_dir / "SimSwap"

sys.path.insert(0, str(simswap_dir))

from models.fs_networks import Generator_Adain_Upsample


checkpoint_path = simswap_dir / "checkpoints/people/latest_net_G.pth"

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

model = Generator_Adain_Upsample(
    input_nc=3,
    output_nc=3,
    latent_size=512,
    n_blocks=9,
    deep=False,
)

model_state = model.state_dict()

missing_keys = set(model_state) - set(checkpoint)
unexpected_keys = set(checkpoint) - set(model_state)

print("Checkpoint keys:", len(checkpoint))
print("Model keys:", len(model_state))

print("\nMissing keys:")
print(missing_keys)

print("\nUnexpected keys:")
print(unexpected_keys)


print("\nIdentity-conditioning weights:")

for key, value in checkpoint.items():
    if "style" in key and "weight" in key:
        print(
            key,
            "mean abs:",
            value.abs().mean().item(),
            "max abs:",
            value.abs().max().item(),
        )