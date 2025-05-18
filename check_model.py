import torch
import os

def check_model(model_path):
    print(f"Checking model file: {model_path}")
    
    try:
        # Try loading as a regular PyTorch model
        state_dict = torch.load(model_path, map_location='cpu')
        print("\nSuccessfully loaded as PyTorch model!")
        print("Type of loaded data:", type(state_dict))
        
        if isinstance(state_dict, dict):
            print("\nModel contains following keys:")
            for key in state_dict.keys():
                print(f"- {key}")
                if hasattr(state_dict[key], 'shape'):
                    print(f"  Shape: {state_dict[key].shape}")
    except Exception as e:
        print(f"\nError loading as PyTorch model: {str(e)}")

if __name__ == "__main__":
    model_path = "exps/pretrain.model"
    check_model(model_path) 