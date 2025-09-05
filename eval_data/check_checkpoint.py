#!/usr/bin/env python3
import torch
import os
import sys

# Add FunASR to path
sys.path.append('./FunASR')

checkpoint_path = "../outputs/model.pt.ep1"

if os.path.exists(checkpoint_path):
    try:
        print(f"Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location='cpu')
        print(f"Checkpoint type: {type(state)}")
        
        if isinstance(state, dict):
            print(f"Keys in checkpoint: {list(state.keys())}")
            
            # Check each key
            for key, value in state.items():
                print(f"  {key}: {type(value)}")
                if hasattr(value, '__len__') and not isinstance(value, (str, int, float)):
                    try:
                        print(f"    - Length: {len(value)}")
                    except:
                        pass
                        
        else:
            print("Checkpoint is not a dictionary")
            print(f"Has eval method: {hasattr(state, 'eval')}")
            print(f"Has to method: {hasattr(state, 'to')}")
            print(f"Has state_dict method: {hasattr(state, 'state_dict')}")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"Checkpoint file does not exist: {checkpoint_path}")
