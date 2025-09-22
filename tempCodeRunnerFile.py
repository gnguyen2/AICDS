import cv2, os, random, numpy as np
from glob import glob

def add_gaussian_noise(img):
    noise = np.random.normal(0, 25, img.shape).astype(np.uint16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy
def block_dropout(img, size=16):
    h, w, _ = img.shape
    x= random.randint( o, w-size)
    y= random.randint( o, h-size)
    img[y:y+size, x:x+size] = 0
    return img

def apply_random_corruption(img):
    funcs =[add_gaussian_noise, block_dropout]
    func = random.choice(funcs)
    return func(img)

src_images = glob("raw_images/*.jpg")
os.makedirs("dataset/train/clean", exist_ok=True)
os.makedirs("dataset/train/corrupted", exist_ok=True)

for i, path in enumerate(src_images):
    img = cv2.imread(path)
    cv2.imwrite(f"dataset/train/clean/img_{i:04d}.png", img)
    corrupted = apply_random_corruption(img)
    cv2.imwrite(f"dataset/train/corrupted/img_{i:04d}.png", corrupted)
    