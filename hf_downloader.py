import os
from huggingface_hub import hf_hub_download

# Configuration
REPO_ID = "mdark4025/MediScan-AI"
REPO_TYPE = "space"
MODELS_DIR = "./models"

# List of all required model assets as seen in the Space
MODEL_ASSETS = [
    "models/ckd_clinical_model.joblib",
    "models/ckd_vision_model_enhanced.pth",
    "models/ckd_hybrid_sota.pth",
    "models/enhanced_disease_model.joblib",
    "models/label_encoder.joblib",
    "models/symptoms_list.joblib",
    "models/vision_disease_model.pth",
    "models/vision_label_map.joblib"
]

def download_models():
    print(f"🚀 Starting model sync from Hugging Face: {REPO_ID}")
    
    # Ensure models directory exists
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"📁 Created directory: {MODELS_DIR}")

    for asset_path in MODEL_ASSETS:
        filename = os.path.basename(asset_path)
        print(f"📥 Syncing {filename}...")
        
        try:
            # Download file from Space
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=asset_path,
                local_dir=".",
                local_dir_use_symlinks=False,
                force_download=False  # Only download if missing or updated
            )
            print(f"✅ Successfully synced: {local_path}")
        except Exception as e:
            # Fallback to curl if HF library fails (useful for xet/LFS issues)
            target_path = os.path.join(MODELS_DIR, filename)
            print(f"⚠️ HF library failed for {filename}. Attempting fallback curl...")
            
            # Construct the direct download URL for a HF Space asset
            # Format: https://huggingface.co/spaces/[REPO_ID]/resolve/main/[ASSET_PATH]?download=true
            download_url = f"https://huggingface.co/spaces/{REPO_ID}/resolve/main/{asset_path}?download=true"
            
            curl_cmd = f"curl -L -o {target_path} \"{download_url}\""
            success = os.system(curl_cmd)
            
            if success == 0 and os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
                print(f"✅ Successfully recovered with curl: {target_path}")
            else:
                print(f"❌ Critical failure: Could not download {asset_path}. Error: {e}")

if __name__ == "__main__":
    download_models()
    print("✨ Model synchronization complete.")
