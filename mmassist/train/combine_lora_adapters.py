"""
Utility script for combining multiple LoRA adapters trained on different datasets.

This script provides different methods to combine LoRA adapters:
1. Sequential loading with manual weight averaging
2. Multi-adapter loading with dynamic switching
3. Merged adapter creation

Usage:
    # Method 1: Load single adapter
    python combine_lora_adapters.py --adapters ./adapter1 --output ./merged_adapter
    
    # Method 2: Combine multiple adapters with equal weights
    python combine_lora_adapters.py --adapters ./adapter1 ./adapter2 ./adapter3 --output ./merged_adapter
    
    # Method 3: Combine with custom weights
    python combine_lora_adapters.py --adapters ./adapter1 ./adapter2 --weights 0.7 0.3 --output ./merged_adapter
    
    # Method 4: Create multi-adapter model (no merging, switchable)
    python combine_lora_adapters.py --adapters ./adapter1 ./adapter2 --mode multi --output ./multi_adapter
"""

import argparse
import json
import os
import torch
from typing import List, Optional
from collections import OrderedDict
from peft import PeftModel
from transformers import AutoProcessor, AutoModelForImageTextToText


def load_adapter_metadata(adapter_path: str) -> dict:
    """Load metadata from an adapter directory."""
    metadata_path = os.path.join(adapter_path, "adapter_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}


def weighted_average_adapters(
    base_model_path: str,
    adapter_paths: List[str],
    weights: List[float],
    output_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Create a new adapter by weighted averaging of multiple adapters.
    
    This method manually combines the adapter weights using weighted averaging.
    """
    print(f"Loading base model from {base_model_path}...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load all adapters
    adapter_state_dicts = []
    for i, adapter_path in enumerate(adapter_paths):
        print(f"Loading adapter {i+1}/{len(adapter_paths)} from {adapter_path}")
        metadata = load_adapter_metadata(adapter_path)
        print(f"  Dataset: {metadata.get('train_datasets', 'unknown')}")
        print(f"  Weight: {weights[i]:.3f}")
        
        model = PeftModel.from_pretrained(base_model, adapter_path)
        adapter_state_dicts.append(model.state_dict())
    
    # Weighted average of adapter parameters
    print("\nCombining adapter weights...")
    combined_state_dict = OrderedDict()
    
    # Get all parameter names from first adapter
    param_names = [k for k in adapter_state_dicts[0].keys() if 'lora' in k.lower()]
    
    for param_name in param_names:
        # Weighted sum
        combined_param = sum(
            weights[i] * adapter_state_dicts[i][param_name] 
            for i in range(len(adapter_state_dicts))
        )
        combined_state_dict[param_name] = combined_param
    
    # Save combined adapter
    print(f"\nSaving combined adapter to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    # Load first adapter as template and update with combined weights
    combined_model = PeftModel.from_pretrained(base_model, adapter_paths[0])
    combined_model.load_state_dict(combined_state_dict, strict=False)
    combined_model.save_pretrained(output_path)
    
    # Save processor
    processor = AutoProcessor.from_pretrained(base_model_path)
    processor.save_pretrained(output_path)
    
    # Save metadata about the combination
    combined_metadata = {
        "base_model": base_model_path,
        "source_adapters": [
            {
                "path": path,
                "weight": weight,
                "metadata": load_adapter_metadata(path)
            }
            for path, weight in zip(adapter_paths, weights)
        ],
        "combination_method": "weighted_average"
    }
    
    metadata_path = os.path.join(output_path, "combined_adapter_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(combined_metadata, f, indent=2)
    
    print(f"✓ Combined adapter saved successfully!")
    print(f"✓ Metadata saved to {metadata_path}")


def create_multi_adapter_model(
    base_model_path: str,
    adapter_paths: List[str],
    output_path: str,
):
    """
    Create a multi-adapter model that can switch between adapters.
    
    This doesn't merge the adapters but allows runtime switching.
    """
    print(f"Loading base model from {base_model_path}...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load first adapter
    print(f"Loading adapter 0 from {adapter_paths[0]}")
    model = PeftModel.from_pretrained(base_model, adapter_paths[0], adapter_name="adapter_0")
    
    # Load additional adapters
    for i, adapter_path in enumerate(adapter_paths[1:], 1):
        print(f"Loading adapter {i} from {adapter_path}")
        model.load_adapter(adapter_path, adapter_name=f"adapter_{i}")
    
    # Save the multi-adapter model
    print(f"\nSaving multi-adapter model to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    
    # Save processor
    processor = AutoProcessor.from_pretrained(base_model_path)
    processor.save_pretrained(output_path)
    
    # Save metadata
    multi_adapter_metadata = {
        "base_model": base_model_path,
        "adapters": [
            {
                "name": f"adapter_{i}",
                "path": path,
                "metadata": load_adapter_metadata(path)
            }
            for i, path in enumerate(adapter_paths)
        ],
        "combination_method": "multi_adapter",
        "usage_example": {
            "load": "model = PeftModel.from_pretrained(base_model, output_path)",
            "switch_adapter": "model.set_adapter('adapter_0')  # or 'adapter_1', etc."
        }
    }
    
    metadata_path = os.path.join(output_path, "multi_adapter_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(multi_adapter_metadata, f, indent=2)
    
    print(f"✓ Multi-adapter model saved successfully!")
    print(f"✓ Available adapters: {', '.join([f'adapter_{i}' for i in range(len(adapter_paths))])}")
    print(f"✓ Use model.set_adapter('adapter_X') to switch between them")


def main():
    parser = argparse.ArgumentParser(description="Combine LoRA adapters trained on different datasets")
    parser.add_argument(
        "--base_model",
        type=str,
        default="HuggingFaceTB/SmolVLM-Instruct",
        help="Base model path"
    )
    parser.add_argument(
        "--adapters",
        type=str,
        nargs="+",
        required=True,
        help="Paths to adapter directories"
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Weights for each adapter (must sum to 1.0). If not provided, equal weights are used."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for combined/multi adapter"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["weighted", "multi"],
        default="weighted",
        help="Combination mode: 'weighted' for weight averaging, 'multi' for switchable adapters"
    )
    
    args = parser.parse_args()
    
    # Validate weights
    if args.weights is None:
        # Equal weights
        args.weights = [1.0 / len(args.adapters)] * len(args.adapters)
    
    if len(args.weights) != len(args.adapters):
        raise ValueError(f"Number of weights ({len(args.weights)}) must match number of adapters ({len(args.adapters)})")
    
    if abs(sum(args.weights) - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {sum(args.weights)}")
    
    # Print configuration
    print("=" * 60)
    print("LoRA Adapter Combination")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Number of adapters: {len(args.adapters)}")
    print(f"Mode: {args.mode}")
    print("\nAdapters:")
    for i, (adapter_path, weight) in enumerate(zip(args.adapters, args.weights)):
        metadata = load_adapter_metadata(adapter_path)
        dataset = metadata.get('train_datasets', 'unknown')
        print(f"  {i+1}. {adapter_path} (weight={weight:.3f}, dataset={dataset})")
    print("=" * 60)
    print()
    
    # Combine adapters
    if args.mode == "weighted":
        weighted_average_adapters(
            args.base_model,
            args.adapters,
            args.weights,
            args.output
        )
    elif args.mode == "multi":
        create_multi_adapter_model(
            args.base_model,
            args.adapters,
            args.output
        )


if __name__ == "__main__":
    main()
