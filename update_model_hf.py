#!/usr/bin/env python3

import argparse
import os
from huggingface_hub import HfApi

def update_model(model_path, commit_message="Update model", create_tag=None, config_path=None):
    """Update model and config on Hugging Face Hub"""
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    # Auto-detect config.json in the same directory if not specified
    if config_path is None:
        model_dir = os.path.dirname(model_path)
        potential_config = os.path.join(model_dir, "config.json")
        if os.path.exists(potential_config):
            config_path = potential_config
            print(f"📄 Auto-detected config: {config_path}")
    
    api = HfApi()
    repo_id = "JustDIt/SenseVoiceSmall"
    
    try:
        print(f"🚀 Uploading model from {model_path}...")
        
        # Upload the model file
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="sensevoice_finetuned_final.pt",
            repo_id=repo_id,
            commit_message=commit_message
        )
        
        print("✅ Model uploaded successfully!")
        
        # Upload config file if exists
        if config_path and os.path.exists(config_path):
            print(f"📄 Uploading config from {config_path}...")
            api.upload_file(
                path_or_fileobj=config_path,
                path_in_repo="config.json",
                repo_id=repo_id,
                commit_message=f"{commit_message} - config update"
            )
            print("✅ Config uploaded successfully!")
        else:
            print("⚠️  No config file found - skipping config upload")
        
        # Create tag if requested
        if create_tag:
            api.create_tag(
                repo_id=repo_id,
                tag=create_tag,
                tag_message=commit_message,
                revision='main'
            )
            print(f"✅ Tag {create_tag} created!")
        
        # Get file info
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"📊 Updated model size: {model_size:.1f} MB")
        
        if config_path and os.path.exists(config_path):
            config_size = os.path.getsize(config_path) / 1024  # KB
            print(f"📊 Updated config size: {config_size:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating model: {e}")
        return False

#python update_model.py [model.pt] -m "description" -c [config file path] -t v1.x.0
def main():
    parser = argparse.ArgumentParser(description="Update SenseVoice model on Hugging Face Hub")
    parser.add_argument("model_path", help="Path to the model file")
    parser.add_argument("-m", "--message", default="Update model", 
                       help="Commit message")
    parser.add_argument("-t", "--tag", help="Create version tag (e.g., v1.1.0)")
    parser.add_argument("-c", "--config", help="Path to config.json (auto-detected if not specified)")
    
    args = parser.parse_args()
    print(f"parser args: {args}")
    success = update_model(args.model_path, args.message, args.tag, args.config)
    exit(0 if success else 1)

if __name__ == "__main__":
    main()