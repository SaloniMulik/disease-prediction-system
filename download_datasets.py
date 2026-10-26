"""
download_datasets.py - downloads the 5 datasets that are not bundled with scikit-learn
into the 'data' folder using exact file names expected by train_models.py.
(Breast cancer needs no download - it ships with scikit-learn.)

If a download fails (UCI sometimes moves files), the script tells you which page to
open and which file name to save the data as.
"""
import io
import urllib.request
import zipfile

import config

UA = {"User-Agent": "Mozilla/5.0 (educational ML project)"}

# file name -> (list of candidate URLs, manual-download page)
SIMPLE = {
    "diabetes.csv": (
        ["https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
         "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"],
        "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database"),
    "heart_cleveland.data": (
        ["https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
         "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"],
        "https://archive.ics.uci.edu/dataset/45/heart+disease"),
    "ilpd_liver.csv": (
        ["https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv"],
        "https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset"),
    "parkinsons.data": (
        ["https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"],
        "https://archive.ics.uci.edu/dataset/174/parkinsons"),
}
CKD_ZIP_URLS = ["https://archive.ics.uci.edu/static/public/336/chronic+kidney+disease.zip"]
CKD_PAGE = "https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease"


def fetch(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def looks_like_data(raw):
    text = raw[:2000].decode("utf-8", errors="ignore").lower()
    return len(raw) > 5000 and "<html" not in text and "<!doctype" not in text


def get_simple(fname, urls, page):
    dest = config.DATA_DIR / fname
    if dest.exists():
        print(f"[skip] {fname} already exists")
        return True
    for url in urls:
        try:
            raw = fetch(url)
            if url.endswith(".zip"):  # heart-disease.zip contains processed.cleveland.data
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    raw = z.read("processed.cleveland.data")
            if not looks_like_data(raw):
                raise ValueError("response is not a data file")
            dest.write_bytes(raw)
            print(f"[ ok ] {fname}  <- {url}")
            return True
        except Exception as e:
            print(f"[fail] {url}\n       {type(e).__name__}: {e}")
    print(f"[MANUAL] Open {page} , download the data and save it as data/{fname}")
    return False


def get_ckd():
    dest = config.DATA_DIR / "chronic_kidney_disease.arff"
    if dest.exists():
        print("[skip] chronic_kidney_disease.arff already exists")
        return True
    for url in CKD_ZIP_URLS:
        try:
            with zipfile.ZipFile(io.BytesIO(fetch(url))) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".arff")
                         and "full" not in n.lower()]
                if not names:
                    raise ValueError("no .arff file inside zip")
                dest.write_bytes(z.read(names[0]))
            print(f"[ ok ] chronic_kidney_disease.arff  <- {url}")
            return True
        except Exception as e:
            print(f"[fail] {url}\n       {type(e).__name__}: {e}")
    print(f"[MANUAL] Open {CKD_PAGE} , click Download, unzip, and copy "
          f"'chronic_kidney_disease.arff' into the data folder.")
    return False


if __name__ == "__main__":
    config.DATA_DIR.mkdir(exist_ok=True)
    ok = [get_simple(f, u, p) for f, (u, p) in SIMPLE.items()]
    ok.append(get_ckd())
    print("\nBreast cancer: no download needed (bundled with scikit-learn).")
    print(f"\n{sum(ok)}/5 datasets ready in '{config.DATA_DIR}'.")
    if not all(ok):
        print("Fix the [MANUAL] items above, then run: python train_models.py")
