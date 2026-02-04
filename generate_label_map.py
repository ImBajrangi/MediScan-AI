import joblib
import os

models_dir = "./models"
os.makedirs(models_dir, exist_ok=True)

# Map indices to codes expected by app.py
label_map = {
    0: "akiec",
    1: "bcc",
    2: "bkl",
    3: "df",
    4: "mel",
    5: "nv",
    6: "vasc"
}

try:
    joblib.dump(label_map, os.path.join(models_dir, "vision_label_map.joblib"))
    print("Successfully generated vision_label_map.joblib with codes")
    print("Label Map:", label_map)
except Exception as e:
    print(f"Error generating label map: {e}")
