# create_subset.py
import zipfile, random, os
from collections import defaultdict

ZIP_PATH  = "Stanford_Online_Products.zip.1"
OUT_DIR   = "mini_dataset"
N_CLASSES    = 200
MIN_IMGS     = 4
MAX_IMGS     = 8

with zipfile.ZipFile(ZIP_PATH) as z:
    imgs = [f for f in z.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"{len(imgs)} images disponibles")

    # Grouper par produit : category_final/PRODUCTID_N.jpg → clé = "category_PRODUCTID"
    by_class = defaultdict(list)
    for f in imgs:
        parts = f.split('/')
        if len(parts) >= 2:
            category   = parts[-2]                          # ex: bicycle_final
            filename   = parts[-1]                          # ex: 111085122871_0.jpg
            product_id = '_'.join(filename.split('_')[:-1]) # ex: 111085122871
            class_key  = f"{category}_{product_id}"
            by_class[class_key].append(f)

    eligible = [cls for cls, files in by_class.items() if len(files) >= MIN_IMGS]
    selected = random.sample(eligible, min(N_CLASSES, len(eligible)))

    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0
    for cls in selected:
        class_dir = os.path.join(OUT_DIR, cls)
        os.makedirs(class_dir, exist_ok=True)
        for f in by_class[cls][:MAX_IMGS]:
            data = z.read(f)
            dest = os.path.join(class_dir, f.split('/')[-1])
            with open(dest, 'wb') as out:
                out.write(data)
            total += 1

print(f"Subset créé : {OUT_DIR}/ — {total} images, {len(selected)} classes")