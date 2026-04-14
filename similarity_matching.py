import numpy as np
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize


def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))


def euclidean_distance(a, b):
    return float(np.linalg.norm(a - b))


def top_k_linear_search(query, vectors, labels, k=5, metric="cosine"):
    scores = []
    for i, v in enumerate(vectors):
        s = cosine_similarity(query, v) if metric == "cosine" \
            else 1.0 / (1.0 + euclidean_distance(query, v) / 100.0)
        scores.append((labels[i], s))
    best = {}
    for actor, s in scores:
        if actor not in best or s > best[actor]:
            best[actor] = s
    return sorted(best.items(), key=lambda x: x[1], reverse=True)[:k]


class KNNMatcher:
    def __init__(self, k=5):
        self.k   = k
        self.knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute")
        self.is_trained = False
        self.labels = []

    def train(self, vectors, labels):
        self.labels = labels
        self.knn.fit(normalize(vectors, norm="l2"), labels)
        self.is_trained = True
        print(f"✅ KNN trained — {len(labels)} samples, {vectors.shape[1]} features, k={self.k}")

    def predict(self, query, top_k=5):
        if not self.is_trained:
            return []
        q = normalize(query.reshape(1,-1), norm="l2")
        dists, idxs = self.knn.kneighbors(q, n_neighbors=min(self.k, len(self.labels)))
        seen, out = set(), []
        for d, i in zip(dists[0], idxs[0]):
            actor = self.labels[i]
            if actor not in seen:
                seen.add(actor)
                out.append((actor, round(1.0 - d/2.0, 4)))
        return out[:top_k]


class FaceSimilarityMatcher:
    def __init__(self, db_path="models/feature_db.pkl"):
        self.db  = None
        self.knn = KNNMatcher(k=5)
        if os.path.exists(db_path):
            with open(db_path, "rb") as f:
                self.db = pickle.load(f)
            if len(self.db["vectors"]) > 0:
                self.knn.train(self.db["vectors"], self.db["labels"])
        else:
            print(f"⚠️  DB not found: {db_path}")

    def match(self, query, top_k=5, method="BOTH"):
        if self.db is None:
            return {}
        out = {}
        if method in ("COSINE","BOTH","cosine"):
            out["cosine"] = top_k_linear_search(
                query, self.db["vectors"], self.db["labels"], k=top_k, metric="cosine")
        if method in ("EUCLIDEAN","BOTH","euclidean"):
            out["euclidean"] = top_k_linear_search(
                query, self.db["vectors"], self.db["labels"], k=top_k, metric="euclidean")
        if method in ("KNN","BOTH","knn") and self.knn.is_trained:
            out["knn"] = self.knn.predict(query, top_k=top_k)
        return out

    def format_results(self, results):
        lines = []
        for method, hits in results.items():
            lines.append(f"\n── {method.upper()} ──")
            for i, (actor, score) in enumerate(hits, 1):
                lines.append(f"  #{i} {actor.replace('_',' '):<22} {score:.1%}  {'█'*int(score*20)}")
        return "\n".join(lines)
