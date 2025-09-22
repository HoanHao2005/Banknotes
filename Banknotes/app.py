import os, sys, json, time
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np
import cv2
from tensorflow import keras
import joblib

# =========================
# Paths & Feature Config
# =========================
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, "vn_banknotes_color_pixel_ann.h5")
LABELS_PATH = os.path.join(SCRIPT_DIR, "vn_banknotes_color_pixel_labels.json")
SCALER_PATH = os.path.join(SCRIPT_DIR, "vn_banknotes_color_pixel_scaler.pkl")

# PHẢI khớp lúc train
PIXEL_SIZE  = (16, 16)   # flatten pixel
RESIZE_BIG  = (64, 64)   # mean/std

VALID_EXTS  = (".jpg", ".jpeg", ".png")

# =========================
# Load assets
# =========================
def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "class_names" in data:
        return data["class_names"]
    if isinstance(data, list):
        return data
    raise ValueError("Labels JSON không hợp lệ")

def robust_imread_bgr(path: str):
    # 1) cv2.imread
    img = cv2.imread(path)
    if img is not None:
        return img
    # 2) PIL -> RGB -> BGR
    try:
        from PIL import Image
        pil = Image.open(path).convert("RGB")
        arr = np.array(pil)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    # 3) imdecode (unicode/space)
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def extract_feature_from_path(path):
    img = robust_imread_bgr(path)
    if img is None:
        return None
    big = cv2.resize(img, RESIZE_BIG).astype("float32") / 255.0
    mean_col = big.mean(axis=(0, 1))  # B,G,R
    std_col  = big.std(axis=(0, 1))
    small = cv2.resize(img, PIXEL_SIZE).astype("float32") / 255.0
    flat_pixels = small.flatten()
    feat = np.concatenate([mean_col, std_col, flat_pixels]).astype("float32")
    return feat.reshape(1, -1)

# Model / Scaler / Labels
try:
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    class_names = load_labels(LABELS_PATH)
except Exception as e:
    messagebox.showerror("Lỗi", f"Không thể load tài nguyên:\n{e}")
    sys.exit(1)

# =========================
# UI: Dark-Glass Theme
# =========================
BG_MAIN   = "#0f172A"  # slate-900
BG_CARD   = "#111827"  # gray-900
BG_SOFT   = "#1f2937"  # gray-800
FG_TEXT   = "#e5e7eb"  # gray-200
ACCENT    = "#22d3ee"  # cyan-400
ACCENT_2  = "#34d399"  # emerald-400
DANGER    = "#f87171"  # red-400
MUTED     = "#9ca3af"  # gray-400

BTN_STYLE = dict(font=("Segoe UI", 11, "bold"), fg=FG_TEXT, bd=0, padx=10, pady=8, activeforeground=FG_TEXT)

def fmt_money(x):  # "100000" -> "100.000"
    try:
        n = int(x)
        return f"{n:,}".replace(",", ".")
    except:
        return x

# =========================
# App
# =========================
class AltApp:
    def __init__(self, root):
        self.root = root
        self.root.title("💵 Vietnamese Banknote Detector — Alt UI")
        self.root.geometry("1100x680")
        self.root.configure(bg=BG_MAIN)

        # --- Keyboard shortcuts
        self.root.bind("<Control-o>", lambda e: self.on_open())
        self.root.bind("<Control-d>", lambda e: self.on_detect())
        self.root.bind("<Delete>",    lambda e: self.on_clear())

        # --- Sidebar (left)
        sidebar = tk.Frame(root, bg=BG_CARD, highlightthickness=1, highlightbackground=BG_SOFT)
        sidebar.pack(side="left", fill="y", padx=(10, 6), pady=10)

        tk.Label(sidebar, text="🧭 Điều hướng", font=("Segoe UI", 12, "bold"),
                 bg=BG_CARD, fg=ACCENT).pack(anchor="w", padx=14, pady=(12, 6))

        self.btn_open = tk.Button(sidebar, text="📂  Chọn ảnh (Ctrl+O)", command=self.on_open,
                                  bg=BG_SOFT, **BTN_STYLE)
        self.btn_open.pack(fill="x", padx=12, pady=(0, 8))

        self.btn_detect = tk.Button(sidebar, text="🔍  Detect (Ctrl+D)", command=self.on_detect,
                                    bg=ACCENT_2, **BTN_STYLE)
        self.btn_detect.pack(fill="x", padx=12, pady=(0, 8))

        self.btn_clear = tk.Button(sidebar, text="🗑  Xóa (Del)", command=self.on_clear,
                                   bg=DANGER, **BTN_STYLE)
        self.btn_clear.pack(fill="x", padx=12, pady=(0, 8))

        # History
        tk.Label(sidebar, text="🕘 Lịch sử", font=("Segoe UI", 12, "bold"),
                 bg=BG_CARD, fg=ACCENT).pack(anchor="w", padx=14, pady=(16, 6))

        self.lst = tk.Listbox(sidebar, bg=BG_SOFT, fg=FG_TEXT, selectbackground=ACCENT, height=16,
                              highlightthickness=1, highlightcolor=BG_SOFT, relief="flat")
        self.lst.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.lst.bind("<<ListboxSelect>>", self.on_history_select)

        # --- Main area
        main = tk.Frame(root, bg=BG_MAIN)
        main.pack(side="right", fill="both", expand=True, padx=(6, 10), pady=10)

        header = tk.Frame(main, bg=BG_MAIN)
        header.pack(fill="x")
        tk.Label(header, text="💵 Banknote Recognition — Alt Layout",
                 font=("Segoe UI Semibold", 18), bg=BG_MAIN, fg=FG_TEXT).pack(side="left")
        self.status = tk.StringVar(value="Sẵn sàng.")
        tk.Label(header, textvariable=self.status, font=("Segoe UI", 10),
                 bg=BG_MAIN, fg=MUTED).pack(side="right")

        content = tk.Frame(main, bg=BG_MAIN)
        content.pack(fill="both", expand=True, pady=(8, 0))

        # Preview Card
        self.preview_card = tk.Frame(content, bg=BG_CARD, highlightthickness=1, highlightbackground=BG_SOFT)
        self.preview_card.pack(side="top", fill="both", expand=True)
        tk.Label(self.preview_card, text="Ảnh xem trước", font=("Segoe UI", 12, "bold"),
                 bg=BG_CARD, fg=ACCENT).pack(anchor="w", padx=14, pady=(12, 6))

        self.preview = tk.Label(self.preview_card, bg=BG_SOFT)
        self.preview.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self.preview.configure(text="Chưa có ảnh.\nBấm “📂 Chọn ảnh”.", fg=MUTED, font=("Segoe UI", 12))

        # Result Card
        self.result_card = tk.Frame(content, bg=BG_CARD, highlightthickness=1, highlightbackground=BG_SOFT)
        self.result_card.pack(side="bottom", fill="x", pady=(10, 0))
        tk.Label(self.result_card, text="Kết quả", font=("Segoe UI", 12, "bold"),
                 bg=BG_CARD, fg=ACCENT).pack(anchor="w", padx=14, pady=(10, 6))

        self.var_cls   = tk.StringVar(value="—")
        self.var_prob  = tk.StringVar(value="—")
        self.var_took  = tk.StringVar(value="—")

        row = tk.Frame(self.result_card, bg=BG_CARD); row.pack(fill="x", padx=14, pady=(0, 12))
        tk.Label(row, text="Mệnh giá:", bg=BG_CARD, fg=MUTED, font=("Segoe UI", 11)).grid(row=0, column=0, sticky="w")
        tk.Label(row, textvariable=self.var_cls, bg=BG_CARD, fg=FG_TEXT, font=("Segoe UI", 13, "bold")).grid(row=0, column=1, sticky="w", padx=(8,0))

        tk.Label(row, text="Độ tin cậy:", bg=BG_CARD, fg=MUTED, font=("Segoe UI", 11)).grid(row=1, column=0, sticky="w")
        tk.Label(row, textvariable=self.var_prob, bg=BG_CARD, fg=FG_TEXT, font=("Segoe UI", 12)).grid(row=1, column=1, sticky="w", padx=(8,0))

        tk.Label(row, text="Thời gian:", bg=BG_CARD, fg=MUTED, font=("Segoe UI", 11)).grid(row=2, column=0, sticky="w")
        tk.Label(row, textvariable=self.var_took, bg=BG_CARD, fg=FG_TEXT, font=("Segoe UI", 12)).grid(row=2, column=1, sticky="w", padx=(8,0))

        # State
        self.image_path = None
        self.tk_img_ref = None
        self.history = []  # [(path, cls, prob%), ...]

    # --------------------- Actions ---------------------
    def on_open(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh tiền",
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png")]
        )
        if not path:
            return
        if not path.lower().endswith(VALID_EXTS):
            messagebox.showwarning("File không hỗ trợ", "Vui lòng chọn JPG/JPEG/PNG.")
            return
        self.image_path = path
        self.show_preview(path)
        self.set_result(None, None, None)
        self.status.set(f"Đã chọn: {os.path.basename(path)}")

    def on_detect(self):
        if not self.image_path:
            messagebox.showwarning("Thiếu ảnh", "Bạn chưa chọn ảnh!")
            return
        feat = extract_feature_from_path(self.image_path)
        if feat is None:
            messagebox.showerror("Lỗi", "Không đọc được ảnh (đường dẫn/định dạng).")
            return

        t0 = time.time()
        feat_scaled = scaler.transform(feat)  # 🔑 scaler như lúc train
        preds = model.predict(feat_scaled, verbose=0)
        took = (time.time() - t0) * 1000

        idx = int(np.argmax(preds))
        prob = float(np.max(preds)) * 100.0
        cls  = class_names[idx] if idx < len(class_names) else f"Class {idx}"

        self.set_result(cls, prob, took)
        self.push_history(self.image_path, cls, prob)
        self.status.set("Detect xong.")

    def on_clear(self):
        self.image_path = None
        self.tk_img_ref = None
        self.preview.configure(image="", text="Chưa có ảnh.\nBấm “📂 Chọn ảnh”.", fg=MUTED)
        self.set_result(None, None, None)
        self.status.set("Đã xóa.")

    def on_history_select(self, _):
        if not self.lst.curselection():
            return
        i = self.lst.curselection()[0]
        path, cls, prob = self.history[i]
        if os.path.isfile(path):
            self.image_path = path
            self.show_preview(path)
            self.set_result(cls, prob, None)
            self.status.set(f"Mở lại: {os.path.basename(path)}")
        else:
            messagebox.showwarning("Thiếu file", "File lịch sử không còn tồn tại.")

    # ------------------ Helpers ------------------
    def show_preview(self, path):
        try:
            img = Image.open(path).convert("RGB")
            w = self.preview.winfo_width() or 800
            h = self.preview.winfo_height() or 420
            img = ImageOps.contain(img, (w-20, h-20))
            self.tk_img_ref = ImageTk.PhotoImage(img)
            self.preview.configure(image=self.tk_img_ref, text="")
        except Exception as e:
            messagebox.showerror("Lỗi ảnh", f"Không mở được ảnh:\n{e}")

    def set_result(self, cls, prob, took_ms):
        if cls is None:
            self.var_cls.set("—")
            self.var_prob.set("—")
            self.var_took.set("—")
            return
        self.var_cls.set(f"{fmt_money(cls)} VND")
        self.var_prob.set(f"{prob:.2f}%")
        self.var_took.set(f"{took_ms:.0f} ms" if took_ms is not None else "—")

    def push_history(self, path, cls, prob):
        entry = f"{os.path.basename(path)}  →  {fmt_money(cls)} VND  ({prob:.1f}%)"
        self.history.append((path, cls, prob))
        self.lst.insert(tk.END, entry)

# =========================
# Main
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    AltApp(root)
    root.mainloop()
