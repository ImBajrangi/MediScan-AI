import os
import time
from huggingface_hub import HfApi

# Configuration
REPO_ID = "mdark4025/MediScan-AI"

def upload_with_retry():
    api = HfApi()
    print(f"🚀 Starting Professional SOTA Asset Upload to {REPO_ID}...")
    
    while True:
        try:
            # 1. Upload Models Folder
            print("📥 Syncing Models folder...")
            api.upload_folder(
                folder_path="./models",
                path_in_repo="models",
                repo_id=REPO_ID,
                repo_type="space",
                commit_message="Sync SOTA models [Automated]"
            )
            
            # 2. Upload Datasets
            print("📊 Syncing Clinical Datasets...")
            api.upload_folder(
                folder_path="./datasets",
                path_in_repo="datasets",
                repo_id=REPO_ID,
                repo_type="space",
                commit_message="Sync clinical datasets [Automated]",
                allow_patterns=["*.csv", "*.py"],
                ignore_patterns=["CKD/*", "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/*"]
            )
            
            print(f"\n✨ SUCCESS! Your HF Space structure is now COMPLETE.")
            break # Exit loop on success
            
        except Exception as e:
            if "429" in str(e):
                print("\n⚠️ Rate Limit Hit (429). Retrying in 60 seconds...")
                time.sleep(60)
            else:
                print(f"❌ Critical Error: {e}")
                break

if __name__ == "__main__":
    upload_with_retry()
