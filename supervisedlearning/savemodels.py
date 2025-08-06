import os
import joblib

def save_model(model, model_name):
    # Absolute path to savedmodels directory
    save_dir = os.path.join(os.path.dirname(__file__), "..", "savedmodels")
    os.makedirs(save_dir, exist_ok=True)  # Make sure it exists

    save_path = os.path.join(save_dir, f"{model_name.replace(' ', '_').lower()}.pkl")
    joblib.dump(model, save_path)
    print(f"[INFO] Model saved at: {save_path}")