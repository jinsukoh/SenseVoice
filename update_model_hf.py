#!/usr/bin/env python3

import argparse
import os
from huggingface_hub import HfApi

def update_model(model_path, commit_message="Update model", create_tag=None, config_path=None, extra_files=None):
    """Update model, config, and extra files on Hugging Face Hub"""
    import glob
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    model_dir = os.path.dirname(model_path)
    # Auto-detect config.json, configuration.json, config.yaml
    config_candidates = [
        os.path.join(model_dir, "config.json"),
        os.path.join(model_dir, "configuration.json"),
        os.path.join(model_dir, "config.yaml")
    ]
    found_configs = [f for f in config_candidates if os.path.exists(f)]
    if config_path is not None and os.path.exists(config_path):
        found_configs.append(config_path)
    api = HfApi()
    repo_id = "JustDIt/SenseVoiceSmall"
    try:
        print(f"🚀 Uploading model from {model_path}...")
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=os.path.basename(model_path),
            repo_id=repo_id,
            commit_message=commit_message
        )
        print("✅ Model uploaded successfully!")
        # Upload all found config files
        for config_file in found_configs:
            print(f"📄 Uploading config: {config_file}")
            api.upload_file(
                path_or_fileobj=config_file,
                path_in_repo=os.path.basename(config_file),
                repo_id=repo_id,
                commit_message=f"{commit_message} - config update"
            )
            print(f"✅ {os.path.basename(config_file)} uploaded!")
        if not found_configs:
            print("⚠️  No config file found - skipping config upload")
        # Upload extra files (model.py, README.md, tokenizer, etc.)
        default_patterns = ["model.py", "README.md", "tokenizer.json", "vocab.txt"]
        files_to_upload = extra_files if extra_files else []
        for pattern in default_patterns:
            files_to_upload.extend(glob.glob(os.path.join(model_dir, pattern)))
        uploaded = set()
        for file_path in files_to_upload:
            if os.path.exists(file_path) and file_path not in uploaded:
                print(f"📦 Uploading extra file: {file_path}")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=os.path.basename(file_path),
                    repo_id=repo_id,
                    commit_message=f"{commit_message} - add {os.path.basename(file_path)}"
                )
                print(f"✅ {os.path.basename(file_path)} uploaded!")
                uploaded.add(file_path)
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
        for config_file in found_configs:
            config_size = os.path.getsize(config_file) / 1024  # KB
            print(f"📊 {os.path.basename(config_file)} size: {config_size:.1f} KB")
        return True
    except Exception as e:
        print(f"❌ Error updating model: {e}")
        return False

#python update_model.py [model.pt] -m "description" -t v1.x.0
def main():
    parser = argparse.ArgumentParser(description="Update SenseVoice model on Hugging Face Hub")
    parser.add_argument("model_path", help="Path to the model file")
    parser.add_argument("-m", "--message", default="Update model", help="Commit message")
    parser.add_argument("-t", "--tag", help="Create version tag (e.g., v1.1.0)")
    parser.add_argument("-c", "--config", help="Path to config.json (auto-detected if not specified)")
    parser.add_argument("-e", "--extra", nargs="*", help="Extra files to upload (e.g. model.py README.md tokenizer.json)")
    args = parser.parse_args()
    print(f"parser args: {args}")
    success = update_model(args.model_path, args.message, args.tag, args.config, args.extra)
    exit(0 if success else 1)

if __name__ == "__main__":
    main()