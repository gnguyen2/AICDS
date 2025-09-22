import cv2, os, random, numpy as np
from glob import glob

def add_gaussian_noise(img):
    noise = np.random.normal(0, 25, img.shape).astype(np.uint16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy
def block_dropout(img, size=16):
    h, w, _ = img.shape
    x= random.randint( 0, w-size)
    y= random.randint( 0, h-size)
    img[y:y+size, x:x+size] = 0
    return img

def apply_random_corruption(img):
    funcs =[add_gaussian_noise, block_dropout]
    func = random.choice(funcs)
    return func(img)

src_images = glob("../dataset/train/clean/*.*")  # relative to scripts/
if not src_images:  
    src_images = glob("../raw_images/*.*")  # fallback

print(f"Found {len(src_images)} images to process")

os.makedirs("dataset/train/clean", exist_ok=True)
os.makedirs("dataset/train/corrupted", exist_ok=True)

# --- process images ---
for i, path in enumerate(src_images):
    img = cv2.imread(path)
    if img is None:
        print(f"⚠️ Skipped unreadable file: {path}")
        continue

    clean_path = f"../dataset/train/clean/img_{i:04d}.png"
    corrupted_path = f"../dataset/train/corrupted/img_{i:04d}.png"

    cv2.imwrite(clean_path, img)
    corrupted = apply_random_corruption(img)
    cv2.imwrite(corrupted_path, corrupted)

print("✅ Dataset generation finished")