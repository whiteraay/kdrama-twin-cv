import cv2
import numpy as np
import os

TARGET_SIZE  = (128, 128)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


class ImagePreprocessor:
    def __init__(self, target_size=TARGET_SIZE):
        self.target_size  = target_size
        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    def to_grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def resize(self, img, size=None):
        return cv2.resize(img, size or self.target_size, interpolation=cv2.INTER_AREA)

    def equalize_histogram(self, gray):
        return cv2.equalizeHist(gray)

    def normalize(self, img):
        return img.astype(np.float32) / 255.0

    def detect_faces(self, img):
        gray = self.to_grayscale(img)
        for (sf, mn, ms) in [(1.1, 5, (60,60)), (1.1, 3, (40,40)), (1.05, 2, (30,30))]:
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=sf, minNeighbors=mn, minSize=ms)
            if len(faces) > 0:
                return faces
        return []

    def crop_face(self, img, rect, padding=0.25):
        x, y, w, h   = rect
        H, W         = img.shape[:2]
        px, py       = int(w * padding), int(h * padding)
        return img[max(0,y-py):min(H,y+h+py), max(0,x-px):min(W,x+w+px)]

    def process(self, img, for_display=False):
        faces = self.detect_faces(img)
        if len(faces) > 0:
            rect      = max(faces, key=lambda r: r[2]*r[3])
            crop      = self.crop_face(img, rect)
        else:
            H, W      = img.shape[:2]
            s         = min(H, W)
            crop      = img[(H-s)//2:(H-s)//2+s, (W-s)//2:(W-s)//2+s]
            rect      = ((W-s)//2, (H-s)//2, s, s)

        gray      = self.to_grayscale(crop)
        equalized = self.equalize_histogram(gray)
        resized   = self.resize(equalized)
        if for_display:
            return resized, rect
        return self.normalize(resized), rect


def preprocess_dataset(dataset_dir="dataset/actors", output_dir="dataset/processed"):
    os.makedirs(output_dir, exist_ok=True)
    pre = ImagePreprocessor()
    ok = fail = 0
    results = {}
    for actor in sorted(os.listdir(dataset_dir)):
        apath = os.path.join(dataset_dir, actor)
        if not os.path.isdir(apath):
            continue
        opath = os.path.join(output_dir, actor)
        os.makedirs(opath, exist_ok=True)
        results[actor] = []
        for fname in sorted(os.listdir(apath)):
            if not fname.lower().endswith((".jpg",".jpeg",".png")):
                continue
            img = cv2.imread(os.path.join(apath, fname))
            if img is None:
                continue
            face, _ = pre.process(img, for_display=True)
            out = os.path.join(opath, fname)
            cv2.imwrite(out, face)
            results[actor].append(out)
            ok += 1
            print(f"  [OK] {actor}/{fname}")
    print(f"\n✅ Preprocessed {ok} images, {fail} failed.")
    return results


if __name__ == "__main__":
    preprocess_dataset()
