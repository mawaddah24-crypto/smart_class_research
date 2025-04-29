import torch
import time
from fvcore.nn import FlopCountAnalysis

from modules.HybridFERNet import HybridFERNet
from modules.HybridFERNet_Light import HybridFERNetLight
# Dummy configurations
input_sizes = [(64, 64), (224, 224)]  # Low resolution vs High resolution
batch_size = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Testing each configuration
for input_size in input_sizes:
    print(f"Testing HybridFERNet with input size: {input_size}")
    dummy_input = torch.randn(batch_size, 3, input_size[0], input_size[1]).to(device)

    # Initialize model
    model = HybridFERNet(
        num_classes=7,
        in_channels=3,
        use_residual=True,
        use_dynamic=True,
        use_se=True
    ).to(device)
    model.eval()

    # Forward pass timing
    with torch.no_grad():
        start_time = time.time()
        output = model(dummy_input)
        end_time = time.time()
        # --- FLOPs Count
        print("=== FLOPs Analysis ===")
        dummy_input_single = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
        flops = FlopCountAnalysis(model, dummy_input_single)
        print(f"Total FLOPs: {flops.total():,} FLOPs")
        print(f"Total MACs (Multiply-Accumulate): {flops.total() / 2:,} MACs")
    print(f"Output shape: {output.shape}")
    print(f"Inference time for batch of {batch_size}: {end_time - start_time:.4f} seconds")
    print(f"Inference time per image: {(end_time - start_time) / batch_size:.6f} seconds\n")
