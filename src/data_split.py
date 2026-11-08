import os
import csv
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter

def split_dataset(raw_dir, train_dir, val_dir, test_dir, test_size=0.15, val_size=0.15,
                  random_state=42, manifest_path="data/split_manifest.csv"):
    """
    Split dataset dari raw_dir ke train, val, dan test directory
    dengan proporsi stratified per kelas.
    """
    print("Memulai proses split dataset...")

    # Refuse to overwrite an existing split or manifest. The caller must
    # explicitly move/remove prior outputs before creating a new split.
    existing_paths = [path for path in (train_dir, val_dir, test_dir, manifest_path)
                      if os.path.exists(path)]
    if existing_paths:
        raise FileExistsError(
            "Pembagian dihentikan karena output sudah ada: "
            + ", ".join(existing_paths)
            + ". Pindahkan atau hapus output tersebut secara manual jika memang ingin membuat split baru."
        )
    
    # Kumpulkan semua file gambar per kelas
    if not os.path.exists(raw_dir):
        print(f"Error: Folder {raw_dir} tidak ditemukan.")
        return
        
    classes = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
    
    filepaths = []
    labels = []
    
    for cls in classes:
        cls_dir = os.path.join(raw_dir, cls)
        # Ambil file gambar saja
        files = sorted(
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        )
        for f in files:
            filepaths.append(os.path.join(cls_dir, f))
            labels.append(cls)
            
    if not filepaths:
        print(f"Tidak ada gambar ditemukan di {raw_dir}. Pastikan dataset sudah di-download.")
        return

    # Hitung proporsi untuk split pertama: train vs (val + test)
    val_test_ratio = val_size + test_size
    
    # Split stratified: X_train (70%) dan X_val_test (30%)
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        filepaths, labels, 
        test_size=val_test_ratio, 
        stratify=labels, 
        random_state=random_state
    )
    
    # Split stratified: X_val (15%) dan X_test (15%)
    # Karena val_size dan test_size masing-masing 15%, rasionya 50% dari sisa 30%
    test_ratio = test_size / val_test_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, 
        test_size=test_ratio, 
        stratify=y_val_test, 
        random_state=random_state
    )
    
    # Buat dictionary untuk mempermudah loop
    splits = {
        'Train': (X_train, y_train, train_dir),
        'Val': (X_val, y_val, val_dir),
        'Test': (X_test, y_test, test_dir)
    }
    
    print("\n--- Ringkasan Split Dataset ---")
    
    manifest_rows = []
    for split_name, (X, y, target_dir) in splits.items():
        print(f"\n{split_name} Set: {len(X)} gambar total")

        os.makedirs(target_dir, exist_ok=True)
            
        # Hitung jumlah per kelas
        counts = Counter(y)
        for cls in classes:
            print(f"  - {cls}: {counts.get(cls, 0)} gambar")
            
            # Buat sub-direktori kelas di target folder
            os.makedirs(os.path.join(target_dir, cls), exist_ok=True)
            
        # Proses copy file ke folder masing-masing
        for src, cls in zip(X, y):
            filename = os.path.basename(src)
            dst = os.path.join(target_dir, cls, filename)
            shutil.copy2(src, dst)
            manifest_rows.append({
                "relative_path": os.path.relpath(src, raw_dir).replace(os.sep, "/"),
                "class": cls,
                "split": split_name.lower(),
                "random_state": random_state,
                "train_ratio": 1.0 - val_size - test_size,
                "val_ratio": val_size,
                "test_ratio": test_size,
            })

    # Write the manifest from entries whose file copies completed successfully.
    # Exclusive creation prevents replacing a manifest created concurrently.
    manifest_parent = os.path.dirname(manifest_path)
    if manifest_parent:
        os.makedirs(manifest_parent, exist_ok=True)
    with open(manifest_path, "x", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=[
            "relative_path", "class", "split", "random_state",
            "train_ratio", "val_ratio", "test_ratio"
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)
            
    print(f"\nSplit dataset selesai! Manifest disimpan ke: {manifest_path}")

if __name__ == "__main__":
    # Setup path relatif terhadap lokasi script dijalankan
    # Asumsinya script dijalankan dari root project: `python src/data_split.py`
    split_dataset(
        raw_dir="data/raw",
        train_dir="data/train",
        val_dir="data/val",
        test_dir="data/test",
        test_size=0.15,
        val_size=0.15,
        random_state=42,
        manifest_path="data/split_manifest.csv"
    )
