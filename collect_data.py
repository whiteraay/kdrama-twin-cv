"""
collect_data.py — download K-drama actor images via DuckDuckGo
Run:  python collect_data.py
"""
import os, time, re, json
import urllib.request, urllib.parse
import cv2
import numpy as np

SAVE_DIR   = "dataset/actors"
MAX_IMAGES = 10
DELAY      = 1.5
HEADERS    = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

ACTORS = [
    "Lee Min Ho actor",
    "Park Seo Jun actor",
    "Hyun Bin actor",
    "Kim Soo Hyun actor",
    "Song Joong Ki actor",
    "Jun Ji Hyun actress",
    "Song Hye Kyo actress",
    "IU singer actress",
    "Park Bo Gum actor",
    "Gong Yoo actor",
    "Bae Suzy actress",
    "Han Hyo Joo actress",
]

CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def get_urls(query, n=15):
    url = "https://duckduckgo.com/?q=" + urllib.parse.quote(query) + "&iax=images&ia=images"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        html = urllib.request.urlopen(req, timeout=10).read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"    [ERR] {e}"); return []
    m = re.search(r'vqd=([\d-]+)', html) or re.search(r'"vqd":"([\d-]+)"', html)
    if not m: return []
    params = urllib.parse.urlencode({"l":"us-en","o":"json","q":query,"vqd":m.group(1),"f":",,,,,","p":"1"})
    try:
        req = urllib.request.Request("https://duckduckgo.com/i.js?"+params,
                                      headers={**HEADERS,"Referer":url})
        data = json.loads(urllib.request.urlopen(req, timeout=10).read())
        return [r.get("image") or r.get("thumbnail") for r in data.get("results",[])[:n] if r.get("image") or r.get("thumbnail")]
    except Exception as e:
        print(f"    [ERR] {e}"); return []


def download(url, path):
    try:
        data = urllib.request.urlopen(urllib.request.Request(url,headers=HEADERS), timeout=8).read()
        arr  = np.frombuffer(data, np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or img.shape[0] < 80 or img.shape[1] < 80:
            return False
        cv2.imwrite(path, img)
        return True
    except:
        return False


def has_face(path):
    img = cv2.imread(path)
    if img is None: return False
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = CASCADE.detectMultiScale(gray, 1.1, 4, minSize=(50,50))
    return len(faces) > 0


def collect_all():
    os.makedirs(SAVE_DIR, exist_ok=True)
    total = 0
    for query in ACTORS:
        name   = query.replace(" actor","").replace(" actress","").replace(" singer","").strip().replace(" ","_")
        folder = os.path.join(SAVE_DIR, name)
        os.makedirs(folder, exist_ok=True)
        print(f"\n🎬 {name}")
        urls  = get_urls(query, MAX_IMAGES + 8)
        saved = 0
        for url in urls:
            if saved >= MAX_IMAGES: break
            path = os.path.join(folder, f"{name}_{saved:02d}.jpg")
            if os.path.exists(path): saved += 1; continue
            if download(url, path) and has_face(path):
                saved += 1; print(f"   ✅ {saved}/{MAX_IMAGES}")
            elif os.path.exists(path):
                os.remove(path)
            time.sleep(0.3)
        print(f"   📁 {saved} images saved")
        total += saved
        time.sleep(DELAY)
    print(f"\n✅ Total: {total} images collected")
    if total == 0:
        print("⚠️  0 downloaded — DuckDuckGo may have blocked the request.")
        print("   Manually put actor photos into dataset/actors/Actor_Name/")


if __name__ == "__main__":
    collect_all()
