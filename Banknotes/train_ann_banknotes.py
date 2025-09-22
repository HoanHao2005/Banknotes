import os, json, numpy as np, cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import joblib

# --------------------------
# Paths & config
# --------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(SCRIPT_DIR, "data")
MODEL_PATH   = os.path.join(SCRIPT_DIR, "vn_banknotes_color_pixel_ann.h5")
LABELS_PATH  = os.path.join(SCRIPT_DIR, "vn_banknotes_color_pixel_labels.json")
SCALER_PATH  = os.path.join(SCRIPT_DIR, "vn_banknotes_color_pixel_scaler.pkl")

PIXEL_SIZE   = (16, 16)   # flatten pixel
RESIZE_BIG   = (64, 64)   # mean/std
TEST_SPLIT   = 0.2
BATCH_SIZE   = 32
EPOCHS       = 100
LR           = 1e-3

valid_exts   = (".jpg", ".jpeg", ".png")

# --------------------------
# Build class_names from TRAIN only (sorted, cố định)
# --------------------------
train_dir = os.path.join(DATA_DIR, "train")
if not os.path.isdir(train_dir):
    raise SystemExit(f"Thiếu thư mục: {train_dir}")

class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
label_map = {c: i for i, c in enumerate(class_names)}
print("[Info] class_names:", class_names)

# --------------------------
# Extract features from all splits
# --------------------------
def extract_feature_from_path(path):
    img = cv2.imread(path)
    if img is None:
        # thử imdecode (unicode path)
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            return None
    if img is None:
        return None

    big = cv2.resize(img, RESIZE_BIG).astype("float32") / 255.0
    mean_col = big.mean(axis=(0, 1))  # B,G,R
    std_col  = big.std(axis=(0, 1))

    small = cv2.resize(img, PIXEL_SIZE).astype("float32") / 255.0
    flat_pixels = small.flatten()

    feat = np.concatenate([mean_col, std_col, flat_pixels]).astype("float32")
    return feat

X, y = [], []

def add_split(split):
    split_dir = os.path.join(DATA_DIR, split)
    if not os.path.isdir(split_dir): 
        return
    for cls in class_names:  # theo thứ tự cố định
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir): 
            continue
        for fn in os.listdir(cls_dir):
            if not fn.lower().endswith(valid_exts): 
                continue
            p = os.path.join(cls_dir, fn)
            feat = extract_feature_from_path(p)
            if feat is None:
                print("⚠️ Bỏ qua ảnh lỗi:", p)
                continue
            X.append(feat)
            y.append(label_map[cls])

add_split("train")
add_split("val")
add_split("test")

X = np.asarray(X, dtype="float32")
y = np.asarray(y, dtype="int64")
print("[Info] data shape:", X.shape, y.shape)

if X.size == 0:
    raise SystemExit("Không load được ảnh nào. Kiểm tra đường dẫn/định dạng.")

# --------------------------
# Scale features (SAVE SCALER!)
# --------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_PATH)
print("[Info] Saved scaler ->", SCALER_PATH)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SPLIT, stratify=y, random_state=42
)

# --------------------------
# ANN model
# --------------------------
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(512, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(class_names), activation="softmax")
])
model.compile(optimizer=keras.optimizers.Adam(LR),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# --------------------------
# Save model & labels
# --------------------------
model.save(MODEL_PATH)
with open(LABELS_PATH, "w", encoding="utf-8") as f:
    json.dump({"class_names": class_names}, f, ensure_ascii=False, indent=2)
print("[Info] Saved model ->", MODEL_PATH)
print("[Info] Saved labels ->", LABELS_PATH)

