"""
Script to download IDM-VTON model from HuggingFace and push to your own account.

Usage:
    1. Install required package: pip install huggingface_hub
    2. Login to HuggingFace: huggingface-cli login
    3. Run this script with your repository name: python load_and_save_model.py
"""

from huggingface_hub import snapshot_download, create_repo, upload_folder
import os
import argparse

def download_and_push_model(
    source_repo="yisol/IDM-VTON",
    target_repo="mkddatascience/IDM-VTON",  # e.g., "your-company/IDM-VTON"
    local_dir="./downloaded_model",
    repo_type="model"
):
    """
    Download a model from HuggingFace and push it to your own account.
    
    Args:
        source_repo: Source repository ID (e.g., "yisol/IDM-VTON")
        target_repo: Your target repository ID (e.g., "your-company/IDM-VTON")
        local_dir: Local directory to save the model temporarily
        repo_type: Type of repository ("model", "dataset", or "space")
    """
    
    if target_repo is None:
        raise ValueError("Please provide target_repo (e.g., 'your-company/IDM-VTON')")
    
    print(f"📥 Downloading model from {source_repo}...")
    try:
        # Download the entire repository
        snapshot_download(
            repo_id=source_repo,
            local_dir=local_dir,
            repo_type=repo_type,
            resume_download=True,  # Resume if interrupted
        )
        print(f"✅ Model downloaded successfully to {local_dir}")
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False
    
    print(f"\n📤 Creating/updating repository {target_repo}...")
    try:
        # Create the repository if it doesn't exist
        create_repo(
            repo_id=target_repo,
            repo_type=repo_type,
            exist_ok=True,  # Don't error if repo already exists
            private=False,  # Set to True if you want a private repo
        )
        print(f"✅ Repository {target_repo} is ready")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        return False
    
    print(f"\n📤 Uploading model to {target_repo}...")
    try:
        # Upload the entire folder to your repository
        upload_folder(
            folder_path=local_dir,
            repo_id=target_repo,
            repo_type=repo_type,
            commit_message=f"Upload IDM-VTON model from {source_repo}",
        )
        print(f"✅ Model successfully uploaded to {target_repo}")
        print(f"\n🎉 Done! Your model is now available at: https://huggingface.co/{target_repo}")
        return True
    except Exception as e:
        print(f"❌ Error uploading model: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download IDM-VTON model and push to your HuggingFace account"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="yisol/IDM-VTON",
        help="Source repository ID (default: yisol/IDM-VTON)"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target repository ID (e.g., your-company/IDM-VTON)"
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default="./downloaded_model",
        help="Local directory to save the model (default: ./downloaded_model)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    
    args = parser.parse_args()
    
    # Make sure user is logged in
    print("⚠️  Make sure you're logged in to HuggingFace!")
    print("   Run: huggingface-cli login")
    print()
    
    # Modify the function to support private repos
    from huggingface_hub import create_repo, upload_folder
    
    # Download and push
    download_and_push_model(
        source_repo=args.source,
        target_repo=args.target,
        local_dir=args.local_dir
    )
