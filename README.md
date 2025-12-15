# In your Python script
from optimize_patch import load_image_from_path, add_noisy_block, optimize_patch_detection_ema

# 1. Load
clean_tensor = load_image_from_path("path/to/image.jpg")

# 2. Attack (Simulated)
corrupted_tensor, gt_mask = add_noisy_block(clean_tensor, size=(60, 60))

# 3. Defend
restored_img, detected_mask = optimize_patch_detection_ema(
    corrupted_tensor, 
    unet, 
    scheduler, 
    num_steps=1500
)
