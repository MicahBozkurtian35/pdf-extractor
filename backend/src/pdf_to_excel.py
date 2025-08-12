import uuid
import os
import io
import re
import csv
import json
import math
import base64
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import fitz  # PyMuPDF
import cv2
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import requests
from dotenv import load_dotenv

load_dotenv()

# =============================
# Env helpers
# =============================
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default

def _env_choice(name: str, choices: List[str], default: str) -> str:
    v = (os.getenv(name) or "").strip().lower()
    return v if v in choices else default

# =============================
# Config
# =============================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("OPENROUTER_MODEL", "openrouter/auto")
LLM_MODEL_FALLBACK = os.getenv("LLM_MODEL_FALLBACK", "").strip() or None
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://localhost:5173")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "pdf-extractor")

POPPLER_PATH = os.getenv("POPPLER_PATH") or None
PDF_DPI = _env_int("PDF_DPI", 350)

# Axis-aware padding (extra emphasis for typical charts so ticks/labels stay inside)
REGION_PAD_LEFT_PCT   = _env_float("REGION_PAD_LEFT_PCT", 0.08)
REGION_PAD_RIGHT_PCT  = _env_float("REGION_PAD_RIGHT_PCT", 0.06)
REGION_PAD_TOP_PCT    = _env_float("REGION_PAD_TOP_PCT", 0.06)
REGION_PAD_BOTTOM_PCT = _env_float("REGION_PAD_BOTTOM_PCT", 0.10)
REGION_MIN_PAD_ABS_PX = _env_int("REGION_MIN_PAD_ABS_PX", 12)
# axis bias for line/bar: add % to left/bottom pads
AXIS_PAD_BOOST = _env_float("AXIS_PAD_BOOST", 0.04)

UPSCALE_FACTOR = _env_float("UPSCALE_FACTOR", 1.0)  # keep 1.0 by default (no resize)

# Upstage
USE_UPSTAGE_DETECTOR = _env_bool("USE_UPSTAGE_DETECTOR", True)
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY", "")
UPSTAGE_URL = os.getenv("UPSTAGE_URL", "https://api.upstage.ai/v1/document-digitization")
UPSTAGE_MODEL = os.getenv("UPSTAGE_MODEL", "document-parse")
UPSTAGE_FORCE_OCR = os.getenv("UPSTAGE_FORCE_OCR", "auto")  # "auto" or "force"

# Filtering knobs
PREFERRED_SERIES_RE = re.compile(os.getenv("PREFERRED_SERIES_REGEX", r"^(current|close|price|index|value|actual|series|y)$"), re.I)
EXCLUDED_SERIES_RE  = re.compile(os.getenv("EXCLUDED_SERIES_REGEX", r"(^avg$|average|\+?\d+sd|\-?\d+sd|sd$|band|upper|lower|guide|benchmark|target)"), re.I)
ONLY_ONE_SERIES     = os.getenv("ONLY_ONE_SERIES", "false").strip().lower() in ("1","true","yes","y","on")

# Accuracy controls
DEDUP_POLICY = _env_choice("DEDUP_POLICY", ["first", "last", "mean"], "first")
VALIDATION_RETRY = _env_bool("VALIDATION_RETRY", True)
DATE_PARSE_THRESHOLD = _env_float("DATE_PARSE_THRESHOLD", 0.6)  # % of X values that must parse as date to consider date-like
NUMERIC_THRESHOLD = _env_float("NUMERIC_THRESHOLD", 0.85)       # % of values that must be numeric to consider a column numeric

# =============================
# Paths
# =============================
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
THUMB_DIR = os.path.join(BASE_DIR, "thumbnails")
TEMP_DIR = os.path.join(BASE_DIR, "temp_images")
ENHANCED_DIR = os.path.join(BASE_DIR, "enhanced_images")
UPSTAGE_CACHE_DIR = os.path.join(BASE_DIR, "upstage_cache")
DEBUG_DIR = os.path.join(BASE_DIR, "debug")

for d in (UPLOAD_DIR, THUMB_DIR, TEMP_DIR, ENHANCED_DIR, UPSTAGE_CACHE_DIR, DEBUG_DIR):
    os.makedirs(d, exist_ok=True)

# =============================
# Utils
# =============================
def deep_sanitize(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: deep_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_sanitize(v) for v in obj]
    return obj

def _to_number(v):
    if v is None: return None
    s = str(v)
    s = re.sub(r"[,\s]", "", s)
    s = re.sub(r"(usd|mn|bn|%|€|£|\$)", "", s, flags=re.I)
    if s in ("", "-", ".", "--"): return None
    try: return float(s)
    except: return None

def _is_numeric_series(s: pd.Series) -> bool:
    return s.apply(lambda x: _to_number(x) is not None).mean() >= NUMERIC_THRESHOLD

def _tries_parse_date(x: Any) -> bool:
    try:
        pd.to_datetime(str(x), errors="raise", infer_datetime_format=True)
        return True
    except Exception:
        return False

def _date_like_fraction(col: pd.Series) -> float:
    vals = col.dropna().astype(str).tolist()
    if not vals: return 0.0
    ok = sum(_tries_parse_date(v) for v in vals)
    return ok / len(vals)

# =============================
# Page count + thumbnails
# =============================
def get_pdf_page_count(pdf_path: str) -> int:
    with fitz.open(pdf_path) as doc:
        return doc.page_count

def generate_page_thumbnail(pdf_path: str, page_number: int, max_w: int = 300) -> str:
    images = convert_from_path(
        pdf_path,
        poppler_path=POPPLER_PATH,
        first_page=page_number,
        last_page=page_number,
        dpi=PDF_DPI,
    )
    if not images:
        raise RuntimeError("No image for thumbnail.")
    img = images[0]
    w, h = img.size
    if w > max_w:
        new_h = int(h * (max_w / w))
        img = img.resize((max_w, new_h), Image.LANCZOS)
    name = f"thumb_page_{page_number}.png"
    out = os.path.join(THUMB_DIR, name)
    img.save(out, "PNG")
    return name

# =============================
# Page to PIL image
# =============================
def load_page_image(pdf_path: str, page_number: int) -> Image.Image:
    imgs = convert_from_path(
        pdf_path,
        poppler_path=POPPLER_PATH,
        first_page=page_number,
        last_page=page_number,
        dpi=PDF_DPI,
    )
    if not imgs:
        raise RuntimeError("No page image.")
    return imgs[0]

# =============================
# Enhance cropped image
# =============================
def enhance_image(image_path: str, output_dir: str = ENHANCED_DIR) -> str:
    os.makedirs(output_dir, exist_ok=True
    )
    try:
        img = Image.open(image_path)
        if UPSCALE_FACTOR and abs(UPSCALE_FACTOR - 1.0) > 1e-6:
            img = img.resize(
                (int(img.width * UPSCALE_FACTOR), int(img.height * UPSCALE_FACTOR)),
                Image.LANCZOS
            )
        out_path = os.path.join(output_dir, os.path.basename(image_path))
        img.save(out_path, "PNG")
        return out_path
    except Exception as e:
        print(f"Enhance error: {e}")
        return image_path

# =============================
# Boxes
# =============================
def expand_box(box, img_w, img_h, chart_type: Optional[str] = None):
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]

    pad_l_pct = REGION_PAD_LEFT_PCT
    pad_b_pct = REGION_PAD_BOTTOM_PCT
    # bias left/bottom pads for line/bar to better capture axes/ticks
    if chart_type in ("line", "bar"):
        pad_l_pct += AXIS_PAD_BOOST
        pad_b_pct += AXIS_PAD_BOOST

    pad_l = max(int(w * pad_l_pct),   REGION_MIN_PAD_ABS_PX)
    pad_r = max(int(w * REGION_PAD_RIGHT_PCT),  REGION_MIN_PAD_ABS_PX)
    pad_t = max(int(h * REGION_PAD_TOP_PCT),    REGION_MIN_PAD_ABS_PX)
    pad_b = max(int(h * pad_b_pct), REGION_MIN_PAD_ABS_PX)
    x0 = max(0, x - pad_l); y0 = max(0, y - pad_t)
    x1 = min(img_w, x + w + pad_r); y1 = min(img_h, y + h + pad_b)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}

# =============================
# Upstage parsing
# =============================
def _upstage_cache_path(pdf_path: str) -> str:
    base = os.path.basename(pdf_path)
    return os.path.join(UPSTAGE_CACHE_DIR, base + ".upstage.json")

def upstage_parse_document(pdf_path: str) -> Optional[Dict[str, Any]]:
    if not (USE_UPSTAGE_DETECTOR and UPSTAGE_API_KEY):
        return None

    cache_path = _upstage_cache_path(pdf_path)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    try:
        headers = {"Authorization": f"Bearer {UPSTAGE_API_KEY}"}
        files = {"document": open(pdf_path, "rb")}
        data = {
            "model": UPSTAGE_MODEL,
            "ocr": UPSTAGE_FORCE_OCR,                 # "auto" or "force"
            "output_formats": "['html']",
            "coordinates": "true"
        }
        resp = requests.post(UPSTAGE_URL, headers=headers, files=files, data=data, timeout=240)
        if not resp.ok:
            print("Upstage error:", resp.status_code, resp.text[:400])
            return None
        parsed = resp.json()
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f)
        return parsed
    except Exception as e:
        print("Upstage exception:", e)
        return None

def _parse_chart_type_from_html(html: str) -> Optional[str]:
    if not html: return None
    m = re.search(r"Chart Type:\s*([a-zA-Z_\- ]+)</p>", html, re.I)
    if m:
        return m.group(1).strip().lower()
    return None

def _parse_series_names_from_html(html: str) -> Optional[List[str]]:
    if not html:
        return None
    m = re.search(r"<thead>.*?<tr>(.*?)</tr>.*?</thead>", html, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    row = m.group(1)
    cells = re.findall(r"<t[dh]>(.*?)</t[dh]>", row, re.IGNORECASE | re.DOTALL)
    names = []
    for c in cells:
        txt = re.sub("<.*?>", "", c).strip()
        if txt:
            names.append(txt)
    if names and (names[0] == "" or names[0].lower() in ("", "series", "item", "name")):
        names = names[1:]
    return names or None

AX_LHS = re.compile(r"\(LHS\)", re.I)
AX_RHS = re.compile(r"\(RHS\)", re.I)

def _parse_axis_labels_from_html(html: str):
    if not html:
        return {"x": None, "y_left": None, "y_right": None, "has_dual": False}

    parens = re.findall(r"\(([^()]{1,60})\)", html)
    y_left = None; y_right = None

    left_candidates  = set()
    right_candidates = set()
    for token in re.findall(r">([^<]{1,60})<", html):
        t = token.strip()
        if AX_LHS.search(t):
            left_candidates.add(re.sub(AX_LHS, "", t).strip(" :"))
        if AX_RHS.search(t):
            right_candidates.add(re.sub(AX_RHS, "", t).strip(" :"))

    if left_candidates:
        y_left = sorted(left_candidates, key=len)[0]
    if right_candidates:
        y_right = sorted(right_candidates, key=len)[0]

    if not y_left or not y_right:
        units_like = [p for p in parens if len(p) <= 30 and not AX_LHS.search(p) and not AX_RHS.search(p)]
        if (not y_left) and len(units_like) >= 1:
            y_left = y_left or units_like[0].strip()
        if (not y_right) and len(units_like) >= 2:
            y_right = y_right or units_like[1].strip()

    return {"x": None, "y_left": y_left, "y_right": y_right, "has_dual": bool(y_left and y_right)}

def _parse_series_axis_hints_from_html(html: str):
    left, right = set(), set()
    if not html:
        return {"left": left, "right": right}
    for token in re.findall(r">([^<]{1,60})<", html):
        t = token.strip()
        name = re.sub(r"\((LHS|RHS)\)", "", t, flags=re.I).strip(" :")
        if not name:
            continue
        if AX_LHS.search(t): left.add(name)
        if AX_RHS.search(t): right.add(name)
    return {"left": left, "right": right}

def upstage_regions_for_page(up_json: dict, page_idx0: int, page_img_w: int, page_img_h: int) -> List[Dict[str, Any]]:
    regions: List[Dict[str, Any]] = []
    els = up_json.get("elements", [])
    for el in els:
        try:
            if int(el.get("page", 1)) - 1 != page_idx0:
                continue
        except Exception:
            continue

        cat = (el.get("category") or "").lower()
        if cat not in ("chart", "graph", "figure", "table"):
            continue

        content_html = (((el.get("content") or {}).get("html")) or "")
        chart_type = _parse_chart_type_from_html(content_html) if cat in ("chart","graph","figure") else None
        series_hints = _parse_series_names_from_html(content_html)
        axes = _parse_axis_labels_from_html(content_html)
        axis_map = _parse_series_axis_hints_from_html(content_html)

        coords = el.get("coordinates")
        def add_box(x0,y0,x1,y1):
            w,h = max(0,x1-x0), max(0,y1-y0)
            if w>=10 and h>=10:
                regions.append({
                    "x":x0,"y":y0,"w":w,"h":h,
                    "category":cat,"chart_type":chart_type,
                    "series_hints":series_hints,
                    "axes": axes,
                    "axis_map": axis_map
                })

        if isinstance(coords,list) and len(coords)>=4 and all("x" in c and "y" in c for c in coords):
            xs = [float(c["x"]) for c in coords]; ys = [float(c["y"]) for c in coords]
            add_box(int(max(0.0,min(xs))*page_img_w), int(max(0.0,min(ys))*page_img_h),
                    int(min(1.0,max(xs))*page_img_w), int(min(1.0,max(ys))*page_img_h))
            continue

        m = re.search(r"top-left:\((\d+),\s*(\d+)\)\s*;\s*bottom-right:\((\d+),\s*(\d+)\)", content_html)
        if m:
            X0,Y0,X1,Y1 = map(int,m.groups())
            add_box(min(X0,X1),min(Y0,Y1),max(X0,X1),max(Y0,Y1))

    regions.sort(key=lambda r: (r["y"], r["x"]))
    print(f"[UPSTAGE] page {page_idx0+1} regions: {len(regions)}")
    return regions

# =============================
# OpenCV fallback detection
# =============================
def detect_graph_regions_opencv(pdf_path: str, page_number: int) -> List[Dict[str, int]]:
    page_img = load_page_image(pdf_path, page_number)
    cv_img = cv2.cvtColor(np.array(page_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dil = cv2.dilate(edges, kernel, iterations=2)
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = gray.shape
    regions: List[Dict[str, Any]] = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < (W * H) * 0.02:
            continue
        if w < 150 or h < 120:
            continue
        if w > 0.95 * W and h > 0.95 * H:
            continue
        regions.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "category": "unknown", "chart_type": None,
            "series_hints": [], "axes": {"y_left": None,"y_right": None,"has_dual": False},
            "axis_map": {"left": set(), "right": set()}
        })
    regions.sort(key=lambda r: (r["y"], r["x"]))
    return regions

def detect_regions(pdf_path: str, page_number: int, page_img_w: int, page_img_h: int) -> List[Dict[str, Any]]:
    if USE_UPSTAGE_DETECTOR and UPSTAGE_API_KEY:
        up_json = upstage_parse_document(pdf_path)
        if up_json:
            regs = upstage_regions_for_page(up_json, page_number - 1, page_img_w, page_img_h)
            if regs:
                try:
                    _draw_overlay(pdf_path, page_number, regs)
                except Exception:
                    pass
                return regs
    print("Falling back to OpenCV regions.")
    regs = detect_graph_regions_opencv(pdf_path, page_number)
    try:
        _draw_overlay(pdf_path, page_number, regs)
    except Exception:
        pass
    return regs

def _draw_overlay(pdf_path: str, page_number: int, regions: List[Dict[str, Any]]):
    img = load_page_image(pdf_path, page_number).copy()
    dr = ImageDraw.Draw(img)
    for i, r in enumerate(regions):
        x0, y0 = r["x"], r["y"]
        x1, y1 = x0 + r["w"], y0 + r["h"]
        dr.rectangle([x0, y0, x1, y1], outline="red", width=3)
        dr.text((x0+4, y0+4), f"{i+1}", fill="yellow")
    path = os.path.join(DEBUG_DIR, f"page_{page_number:03d}_overlay.png")
    img.save(path, "PNG")
    print(f"[DEBUG] overlay -> {path}")

# =============================
# LLM extraction via OpenRouter
# =============================
def _strict_rules(chart_type: Optional[str], want_dual: bool,
                  y_left_label: Optional[str], y_right_label: Optional[str]) -> str:
    # Hard, unambiguous instructions to reduce flips and weird headers
    base = [
        "You are converting a chart image into CSV for data analysis.",
        "Return ONLY CSV. No commentary, no code fences, no extra rows.",
        "X is always the horizontal axis (categories/dates). Y is always the vertical axis (numeric values). Do NOT swap axes.",
        "All Y values must be pure numbers. Strip %, currency, commas, mn/bn suffixes.",
        "Use the X-axis tick labels as the first column values.",
    ]
    if chart_type:
        base.append(f"The original chart type is '{chart_type}'. Keep that orientation.")
    if want_dual:
        base.append("Output EXACTLY THREE columns: X,Y_left,Y_right (in that order).")
        if y_left_label:  base.append(f"Left axis unit/label: {y_left_label}.")
        if y_right_label: base.append(f"Right axis unit/label: {y_right_label}.")
        base.append("If multiple series per axis, choose the primary series for each axis. Ignore averages, bands, guides, benchmarks.")
    else:
        base.append("Output EXACTLY TWO columns: X,Y (in that order).")
        base.append("If multiple series exist, choose the most representative (Current/Price/Index). Ignore averages, bands, guides, benchmarks.")
    return " ".join(base)

def extract_csv_from_image(image_path: str, model: Optional[str] = None,
                           chart_type: Optional[str] = None,
                           series_hints: Optional[List[str]] = None,
                           want_dual: bool = False,
                           y_left_label: Optional[str] = None,
                           y_right_label: Optional[str] = None,
                           fixup_pass: bool = False) -> Optional[str]:
    if not OPENROUTER_API_KEY:
        print("Missing OPENROUTER_API_KEY")
        return None

    model_id = model or LLM_MODEL
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_SITE_URL,
        "X-Title": OPENROUTER_APP_NAME,
    }

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    rules_text = _strict_rules(chart_type, want_dual, y_left_label, y_right_label)
    if fixup_pass:
        # A short, targeted re-ask to correct mistakes
        rules_text += " IMPORTANT: Previous attempt had schema/axis issues. Correct them strictly per instructions."

    payload = {
        "model": model_id,
        "max_tokens": 400,
        "temperature": 0.0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": rules_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
        }]
    }

    print(f"[LLM] Using model: {model_id} (fallback={LLM_MODEL_FALLBACK}) for {os.path.basename(image_path)}")

    def call(p):
        try:
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                 headers=headers, json=p, timeout=120)
        except Exception as e:
            print("LLM network exception:", e)
            return None, None
        if not resp.ok:
            print("LLM HTTP error:", resp.status_code, resp.text[:300])
            return None, resp
        try:
            j = resp.json()
            content = j["choices"][0]["message"].get("content")
            # debug dump
            try:
                dbg_path = os.path.join(DEBUG_DIR, f"raw_{('fix_' if fixup_pass else '')}{os.path.basename(image_path)}.txt")
                with open(dbg_path, "w", encoding="utf-8") as f:
                    f.write(content or "")
                print(f"[LLM] wrote raw to {dbg_path} (len={len(content or '')})")
            except Exception:
                pass
            return (content.strip() if content else None), resp
        except Exception:
            print("Unexpected LLM response:", resp.text[:300])
            return None, resp

    out, r = call(payload)
    if (not out) and LLM_MODEL_FALLBACK:
        print(f"Retrying with fallback model: {LLM_MODEL_FALLBACK}")
        payload["model"] = LLM_MODEL_FALLBACK
        out, _ = call(payload)
    return out

# =============================
# CSV -> DataFrame
# =============================
def csv_to_df(raw_text: str) -> Optional[pd.DataFrame]:
    try:
        if not raw_text or not raw_text.strip():
            return None
        text = raw_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:csv|tsv)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        sample = "\n".join(text.splitlines()[:10])
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            delim = dialect.delimiter
        except Exception:
            if "\t" in sample: delim = "\t"
            elif "," in sample: delim = ","
            else:
                text = re.sub(r"[ ]{2,}", ",", text); delim = ","
        if delim == ",": text = text.replace("\t", ",")
        df = pd.read_csv(io.StringIO(text), delimiter=delim)
        if df.empty: return None
        return df
    except Exception as e:
        print("CSV parse error:", e)
        return None

# =============================
# Normalization / Validation
# =============================
def _looks_date_like(s: pd.Series) -> bool:
    return _date_like_fraction(s) >= DATE_PARSE_THRESHOLD

def fix_axis_order(df: pd.DataFrame, chart_type: Optional[str]) -> Tuple[pd.DataFrame, bool]:
    """
    Heuristic to correct X/Y flips:
    - Prefer X to be non-numeric or date-like categories; Y to be numeric.
    - If first col is numeric and the second is non-numeric/date-like -> swap.
    - For line/bar, if col1 looks like date/categories and col0 is numeric -> swap.
    Returns: (df, swapped: bool)
    """
    try:
        cols = list(df.columns)
        if len(cols) < 2:
            return df, False
        c0, c1 = cols[0], cols[1]
        s0, s1 = df[c0], df[c1]
        c0_num = _is_numeric_series(s0)
        c1_num = _is_numeric_series(s1)
        c0_date = _looks_date_like(s0.astype(str))
        c1_date = _looks_date_like(s1.astype(str))

        should_swap = False
        # primary rule: X should not be numeric if Y is numeric
        if c0_num and not c1_num:
            should_swap = True
        # date hint: for line/bar charts, X is commonly dates/categories
        if chart_type in ("line", "bar"):
            if (c1_date and not c0_date) or (not c0_num and c1_num):
                should_swap = True

        if should_swap:
            df = df[[c1, c0] + cols[2:]]
            return df, True
    except Exception:
        pass
    return df, False

def force_headers(df: pd.DataFrame, want_dual: bool, axes_meta: Dict[str, Any]) -> pd.DataFrame:
    cols = list(df.columns)
    if want_dual:
        # canon: X, Y_left, Y_right
        new_cols = ["X"]
        if len(cols) >= 2:
            new_cols.append("Y_left")
        if len(cols) >= 3:
            new_cols.append("Y_right")
        new_cols += cols[len(new_cols):]  # append any extras just in case
        df.columns = new_cols[:len(cols)]
    else:
        # canon: X, Y
        new_cols = ["X"]
        if len(cols) >= 2:
            new_cols.append("Y")
        new_cols += cols[len(new_cols):]
        df.columns = new_cols[:len(cols)]
    return df

def coerce_numeric_y(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns[1:]:
        out[c] = out[c].map(_to_number)
    return out

def sort_by_x(df: pd.DataFrame) -> pd.DataFrame:
    x = df["X"]
    # try numeric then date
    try:
        xn = x.map(_to_number)
        if xn.notna().mean() >= 0.7:
            out = df.copy()
            out["X"] = xn
            return out.sort_values("X")
    except Exception:
        pass
    try:
        xd = pd.to_datetime(x, errors="coerce", infer_datetime_format=True)
        if xd.notna().mean() >= 0.7:
            out = df.copy()
            out["X"] = xd.dt.strftime("%Y-%m-%d")
            return out.sort_values("X")
    except Exception:
        pass
    # fallback: lexical
    return df.sort_values("X")

def dedup_by_x(df: pd.DataFrame, want_dual: bool) -> Tuple[pd.DataFrame, int]:
    """
    Ensure one Y per X using DEDUP_POLICY.
    Returns: (df_deduped, dropped_rows)
    """
    if df.empty or "X" not in df.columns:
        return df, 0
    before = len(df)
    if DEDUP_POLICY == "first":
        df2 = df.drop_duplicates(subset=["X"], keep="first")
    elif DEDUP_POLICY == "last":
        df2 = df.drop_duplicates(subset=["X"], keep="last")
    else:  # mean
        # Aggregate numeric columns by mean; keep first X ordering
        aggs = {c: "mean" for c in df.columns if c != "X"}
        df2 = df.groupby("X", as_index=False).agg(aggs)
    return df2, before - len(df2)

def validate_dataframe(df: pd.DataFrame, want_dual: bool) -> Tuple[bool, List[str]]:
    issues = []
    cols = list(df.columns)

    # column count
    expected = 3 if want_dual else 2
    if len(cols) < expected:
        issues.append(f"expected {expected} columns, got {len(cols)}")

    # required headers
    need = ["X", "Y_left", "Y_right"] if want_dual else ["X", "Y"]
    for h in need:
        if h not in cols:
            issues.append(f"missing column '{h}'")

    # numeric Y
    for c in (["Y"] if not want_dual else ["Y_left", "Y_right"]):
        if c in df.columns:
            if not _is_numeric_series(df[c]):
                issues.append(f"non-numeric values in {c}")

    # X uniqueness
    if "X" in df.columns:
        dups = df["X"].duplicated().sum()
        if dups > 0:
            issues.append(f"{dups} duplicate X values")

    return (len(issues) == 0), issues

def pretty_headers(df: pd.DataFrame, axes_meta: Dict[str, Any]) -> Dict[str, str]:
    """
    Return a mapping for pretty display names while keeping canonical keys.
    """
    pretty = {"X": "X"}
    if "Y" in df.columns:
        pretty["Y"] = axes_meta.get("y_left") or "Y"
    if "Y_left" in df.columns:
        pretty["Y_left"] = axes_meta.get("y_left") or "Y_left"
    if "Y_right" in df.columns:
        pretty["Y_right"] = axes_meta.get("y_right") or "Y_right"
    return pretty

# =============================
# Page pipeline
# =============================
def process_pdf_page(pdf_path: str, page_number: int) -> Dict[str, Any]:
    try:
        run_id = uuid.uuid4().hex[:8]
        page_img = load_page_image(pdf_path, page_number)
        page_w, page_h = page_img.width, page_img.height

        regions = detect_regions(pdf_path, page_number, page_w, page_h)
        if not regions:
            print(f"[WARN] No regions detected on page {page_number}. Check UPSTAGE_API_KEY or OpenCV thresholds.")
        else:
            print(f"Using {len(regions)} region(s) on page {page_number}")

        results: List[Dict[str, Any]] = []
        debug_raw: List[Dict[str, Any]] = []

        for idx, r in enumerate(regions):
            chart_type = r.get("chart_type")
            axes_meta = r.get("axes") or {"y_left": None, "y_right": None, "has_dual": False}
            axis_map  = r.get("axis_map") or {"left": set(), "right": set()}
            want_dual = bool(axes_meta.get("has_dual"))

            # Crop with axis-biased padding
            exp = expand_box(r, page_w, page_h, chart_type=chart_type)
            crop = page_img.crop((exp["x"], exp["y"], exp["x"] + exp["w"], exp["y"] + exp["h"]))
            crop_name = f"page_{page_number}_region_{idx}_{run_id}_crop.png"
            crop_path = os.path.join(TEMP_DIR, crop_name)
            crop.save(crop_path, "PNG")
            display_path = enhance_image(crop_path, ENHANCED_DIR)

            # --- LLM pass 1
            raw = extract_csv_from_image(
                display_path,
                chart_type=chart_type,
                series_hints=r.get("series_hints"),
                want_dual=want_dual,
                y_left_label=axes_meta.get("y_left"),
                y_right_label=axes_meta.get("y_right"),
                fixup_pass=False
            )

            debug_entry = {
                "page": page_number,
                "image": os.path.basename(display_path),
                "raw": (raw or "")[:1000]
            }

            # Parse + normalize
            records = []
            note_parts: List[str] = []
            confidence = "high"
            swapped = False
            rows_dropped = 0
            validated_ok = False

            def postprocess(raw_text: Optional[str]) -> Tuple[Optional[pd.DataFrame], List[str], bool, int, bool]:
                notes = []
                if not raw_text:
                    notes.append("No model output.")
                    return None, notes, False, 0, False
                df = csv_to_df(raw_text)
                if df is None:
                    notes.append("Parsed CSV empty or invalid.")
                    return None, notes, False, 0, False

                # Trim to first N columns (avoid any model leakage)
                df = df.iloc[:, : (3 if want_dual else 2)]

                # Attempt X/Y normalization
                df, swapped_here = fix_axis_order(df, chart_type)
                if swapped_here:
                    notes.append("Auto-corrected flipped X/Y.")
                df = force_headers(df, want_dual, axes_meta)
                df = coerce_numeric_y(df)
                df = sort_by_x(df)

                # Dedup X
                df, dropped = dedup_by_x(df, want_dual)
                if dropped > 0:
                    notes.append(f"Deduplicated X ({dropped} rows dropped via policy '{DEDUP_POLICY}').")

                # Remove rows with null Y
                if want_dual:
                    before = len(df)
                    df = df[~(df["Y_left"].isna() & df["Y_right"].isna())]
                    null_drop = before - len(df)
                    if null_drop > 0:
                        notes.append(f"Dropped {null_drop} rows with null Y values.")
                else:
                    before = len(df)
                    df = df[~df["Y"].isna()]
                    null_drop = before - len(df)
                    if null_drop > 0:
                        notes.append(f"Dropped {null_drop} rows with null Y values.")

                # Final validation
                ok, issues = validate_dataframe(df, want_dual)
                if not ok:
                    notes.append("Validation issues: " + "; ".join(issues))
                return df, notes, swapped_here, dropped + (null_drop if 'null_drop' in locals() else 0), ok

            df, notes1, swapped1, dropped1, ok1 = postprocess(raw)

            # If validation failed, optionally re-ask once
            if (not ok1) and VALIDATION_RETRY:
                raw_fix = extract_csv_from_image(
                    display_path,
                    chart_type=chart_type,
                    series_hints=r.get("series_hints"),
                    want_dual=want_dual,
                    y_left_label=axes_meta.get("y_left"),
                    y_right_label=axes_meta.get("y_right"),
                    fixup_pass=True
                )
                debug_entry["raw_fix"] = (raw_fix or "")[:1000]
                df2, notes2, swapped2, dropped2, ok2 = postprocess(raw_fix)
                # choose the better of the two
                if ok2 or (df2 is not None and (df is None or len(df2) >= len(df))):
                    df, notes1, swapped1, dropped1, ok1 = df2, notes2, swapped2, dropped2, ok2

            debug_raw.append(debug_entry)

            if df is not None:
                swapped = swapped1
                rows_dropped = dropped1
                validated_ok = ok1
                # compute confidence
                if not validated_ok:
                    confidence = "low"
                elif swapped or rows_dropped > 0:
                    confidence = "medium"
                else:
                    confidence = "high"

                # Prepare records for API
                pretty_map = pretty_headers(df, axes_meta)
                df_out = df.copy()
                # we keep canonical keys in data; frontend can choose to show pretty labels if desired
                df_out = df_out.replace([np.inf, -np.inf], np.nan).where(pd.notnull(df_out), None)
                records = deep_sanitize(df_out.to_dict(orient="records"))
                note_parts.extend(notes1)
                if confidence != "high":
                    note_parts.append(f"confidence={confidence}")
            else:
                confidence = "low"
                note_parts.extend(notes1)

            results.append({
                "page": page_number,
                "region": idx,
                "image": os.path.basename(display_path),
                "data": records,
                "note": ("; ".join(note_parts)) or None,
                "confidence": confidence,
                "chart_type": chart_type,
                "category": r.get("category"),
                "series_hints": r.get("series_hints"),
            })

        payload = {"tables": results, "debug_raw": debug_raw}
        return deep_sanitize(payload)
    except Exception as e:
        return {"error": str(e)}

# Whole-PDF (unused by UI, but handy)
def process_pdf_to_data(pdf_path: str) -> Dict[str, Any]:
    n = get_pdf_page_count(pdf_path)
    combined = {"tables": [], "debug_raw": []}
    for p in range(1, n + 1):
        r = process_pdf_page(pdf_path, p)
        combined["tables"].extend(r.get("tables", []))
        combined["debug_raw"].extend(r.get("debug_raw", []))
    return deep_sanitize(combined)

if __name__ == "__main__":
    test_pdf = r"C:\path\to\your.pdf"
    print(process_pdf_page(test_pdf, 1))
