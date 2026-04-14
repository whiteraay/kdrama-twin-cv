import cv2
import numpy as np
import os
import pickle


class FeatureExtractor:
    def __init__(self, method="HOG"):
        self.method = method.upper()
        self.orb    = cv2.ORB_create(nfeatures=500)
        self.sift   = cv2.SIFT_create(nfeatures=500)
        self.hog    = cv2.HOGDescriptor(
            (128,128), (16,16), (8,8), (8,8), 9)

    def _to_u8(self, img):
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        return img

    def extract_hog(self, img):
        img = cv2.resize(self._to_u8(img), (128,128))
        return self.hog.compute(img).flatten()

    def extract_orb(self, img, n=64):
        img = self._to_u8(img)
        _, desc = self.orb.detectAndCompute(img, None)
        if desc is None:
            return np.zeros(n*32, np.float32)
        desc = desc.astype(np.float32)
        if len(desc) < n:
            desc = np.vstack([desc, np.zeros((n-len(desc),32),np.float32)])
        return desc[:n].flatten()

    def extract_sift(self, img, n=64):
        img = self._to_u8(img)
        _, desc = self.sift.detectAndCompute(img, None)
        if desc is None:
            return np.zeros(n*128, np.float32)
        if len(desc) < n:
            desc = np.vstack([desc, np.zeros((n-len(desc),128),np.float32)])
        return desc[:n].flatten()

    def extract(self, img):
        if self.method == "HOG":      return self.extract_hog(img)
        if self.method == "ORB":      return self.extract_orb(img)
        if self.method == "SIFT":     return self.extract_sift(img)
        if self.method == "COMBINED":
            return np.concatenate([self.extract_hog(img),
                                   cv2.resize(self._to_u8(img),(128,128)).flatten().astype(np.float32)])
        raise ValueError(f"Unknown method: {self.method}")


def build_feature_database(processed_dir="dataset/processed",
                            output_path="models/feature_db.pkl",
                            method="HOG"):
    os.makedirs("models", exist_ok=True)
    ext     = FeatureExtractor(method)
    db      = {}
    labels  = []
    vectors = []

    if not os.path.exists(processed_dir):
        print(f"⚠️  {processed_dir} not found. Run preprocessing first.")
        return None

    for actor in sorted(os.listdir(processed_dir)):
        apath = os.path.join(processed_dir, actor)
        if not os.path.isdir(apath):
            continue
        db[actor] = []
        for fname in sorted(os.listdir(apath)):
            if not fname.lower().endswith((".jpg",".png")):
                continue
            img = cv2.imread(os.path.join(apath, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            feat = ext.extract(img)
            db[actor].append(feat)
            labels.append(actor)
            vectors.append(feat)
            print(f"  [FEAT] {actor}/{fname} dim={feat.shape[0]}")

    data = {"db": db, "labels": labels,
            "vectors": np.array(vectors, np.float32), "method": method}
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print(f"\n✅ Saved {len(vectors)} vectors → {output_path}")
    return data


if __name__ == "__main__":
    build_feature_database()
