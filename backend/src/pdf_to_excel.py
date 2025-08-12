import uuid
import os
import io
import re
import csv
import json
import math
import base64
from typing import Dict, List, Any, Optional

import numpy as np
import fitz  # PyMuPDF
import cv2
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
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

# small padding to ensure axes/ticks are included
REGION_PAD_LEFT_PCT   = _env_float("REGION_PAD_LEFT_PCT", 0.08)
REGION_PAD_RIGHT_PCT  = _env_float("REGION_PAD_RIGHT_PCT", 0.06)
REGION_PAD_TOP_PCT    = _env_float("REGION_PAD_TOP_PCT", 0.06)
REGION_PAD_BOTTOM_PCT = _env_float("REGION_PAD_BOTTOM_PCT", 0.10)
REGION_MIN_PAD_ABS_PX = _env_int("REGION_MIN_PAD_ABS_PX", 12)

UPSCALE_FACTOR = _env_float("UPSCALE_FACTOR", 1.0)  # keep 1.0 by default (no resize)

# Upstage
USE_UPSTAGE_DETECTOR = _env_bool("USE_UPSTAGE_DETECTOR", True)
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY", "")
UPSTAGE_URL = os.getenv("UPSTAGE_URL", "https://api.upstage.ai/v1/document-digitization")
UPSTAGE_FORCE_OCR = os.getenv("UPSTAGE_FORCE_OCR", "auto")  # "auto" or "force"

# Filtering knobs
PREFERRED_SERIES_RE = re.compile(os.getenv("PREFERRED_SERIES_REGEX", r"^(current|close|price|index|value|actual|series|y)$"), re.I)
EXCLUDED_SERIES_RE  = re.compile(os.getenv("EXCLUDED_SERIES_REGEX", r"(^avg$|average|\+?\d+sd|\-?\d+sd|sd$|band|upper|lower|guide|benchmark|target)"), re.I)
ONLY_ONE_SERIES     = os.getenv("ONLY_ONE_SERIES", "false").strip().lower() in ("1","true","yes","y","on")

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
    os.makedirs(output_dir, exist_ok=True)
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
def expand_box(box, img_w, img_h):
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    pad_l = max(int(w * REGION_PAD_LEFT_PCT),   REGION_MIN_PAD_ABS_PX)
    pad_r = max(int(w * REGION_PAD_RIGHT_PCT),  REGION_MIN_PAD_ABS_PX)
    pad_t = max(int(h * REGION_PAD_TOP_PCT),    REGION_MIN_PAD_ABS_PX)
    pad_b = max(int(h * REGION_PAD_BOTTOM_PCT), REGION_MIN_PAD_ABS_PX)
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
            "model": "document-parse",
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
                # debug overlay
                try:
                    _draw_overlay(pdf_path, page_number, regs)
                except Exception:
                    pass
                return regs
    print("Falling back to OpenCV regions.")
    return detect_graph_regions_opencv(pdf_path, page_number)

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
def extract_csv_from_image(image_path: str, model: Optional[str] = None,
                           chart_type: Optional[str] = None,
                           series_hints: Optional[List[str]] = None,
                           want_dual: bool = False,
                           y_left_label: Optional[str] = None,
                           y_right_label: Optional[str] = None) -> Optional[str]:
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

    rules = [
        "Use the X-axis tick labels as the first column header.",
        "Strip %, currency, 'mn'/'bn', and commas from numbers: output pure numbers.",
        "No commentary, no code fences, CSV only."
    ]
    if chart_type:
        rules.insert(0, f"The original chart type is '{chart_type}'. Do not change chart type.")
    if want_dual:
        rules.insert(1, "Return CSV with EXACTLY THREE columns: X, Y_left (left axis), Y_right (right axis).")
        if y_left_label:  rules.append(f"Left axis label (unit): {y_left_label}.")
        if y_right_label: rules.append(f"Right axis label (unit): {y_right_label}.")
        rules.append("If multiple series exist per axis, choose the primary series on each axis.")
        rules.append("Ignore averages, bands, benchmarks, or ±SD guides.")
    else:
        rules.insert(1, "Return CSV with EXACTLY TWO columns: X and Y (primary series only).")
        if series_hints:
            rules.append("If multiple series exist, choose the most representative (e.g., Current/Price/Index).")
        rules.append("Ignore averages, bands, benchmarks, or ±SD guides.")

    payload = {
        "model": model_id,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": " ".join(rules)},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
        }]
    }

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
            content = j["choices"][0]["message"]["content"]
            return (content.strip() if content else None), resp
        except Exception:
            print("Unexpected LLM response:", resp.text[:300])
            return None, resp

    out, r = call(payload)
    if (not out) and LLM_MODEL_FALLBACK:
        # retry on image-input unsupported / transient
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
# X/Y selection (dual-axis aware)
# =============================
def _pick_primary_y(y_cols, series_hints, df):
    if not y_cols:
        return None
    y = [c for c in y_cols if not EXCLUDED_SERIES_RE.search(str(c))]
    y = y or y_cols[:]
    if series_hints:
        norm = [h.strip().lower() for h in series_hints if isinstance(h,str)]
        for c in y:
            if str(c).strip().lower() in norm and PREFERRED_SERIES_RE.search(str(c)):
                return c
    for c in y:
        if PREFERRED_SERIES_RE.search(str(c)):
            return c
    best_c, best_std = None, -1.0
    for c in y:
        s = pd.Series([_to_number(v) for v in df[c]]).dropna()
        st = float(s.std()) if not s.empty else 0.0
        if st > best_std:
            best_c, best_std = c, st
    return best_c or (y_cols[0] if y_cols else None)

def filter_df_to_axes(df: pd.DataFrame,
                      series_hints: Optional[List[str]],
                      axes_meta: Dict[str, Any],
                      axis_map: Dict[str, set]) -> pd.DataFrame:
    if df is None or df.empty: return df
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    def is_numeric_col(c): 
        return df[c].apply(lambda x: _to_number(x) is not None).mean() > 0.85
    year_like = [c for c in df.columns if re.search(r"^(year|date|month|time)$", str(c), re.I)]
    x_key = year_like[0] if year_like else None
    if not x_key:
        non_num = [c for c in df.columns if not is_numeric_col(c)]
        x_key = non_num[0] if non_num else df.columns[0]

    for c in df.columns:
        if c != x_key:
            df[c] = df[c].map(_to_number)

    y_candidates = [c for c in df.columns if c != x_key]
    left_pool  = [c for c in y_candidates if (str(c) in axis_map.get("left", set()) or not axis_map.get("right"))]
    right_pool = [c for c in y_candidates if (str(c) in axis_map.get("right", set()))]

    if not left_pool:
        left_pool = [c for c in y_candidates if not EXCLUDED_SERIES_RE.search(str(c))]
    if not right_pool and axes_meta.get("has_dual"):
        right_pool = [c for c in y_candidates if not EXCLUDED_SERIES_RE.search(str(c)) and c not in left_pool]

    y_left  = _pick_primary_y(left_pool,  series_hints, df) if left_pool else None
    y_right = _pick_primary_y(right_pool, series_hints, df) if right_pool else None

    keep = [x_key]
    if y_left:  keep.append(y_left)
    if axes_meta.get("has_dual") and y_right and y_right != y_left:
        keep.append(y_right)

    if ONLY_ONE_SERIES and len(keep) > 2:
        keep = keep[:2]  # force single Y if desired

    df = df[keep]

    # rename Y columns for clarity
    cols_map = {x_key: x_key}
    if len(keep) >= 2:
        cols_map[keep[1]] = axes_meta.get("y_left") or "Y_left"
    if len(keep) == 3:
        cols_map[keep[2]] = axes_meta.get("y_right") or "Y_right"
    df = df.rename(columns=cols_map)

    try:
        x_num = df[x_key].map(_to_number)
        if x_num.notna().sum() >= len(df) * 0.7:
            df[x_key] = x_num
            df = df.sort_values(by=x_key)
    except:
        pass
    return df

def fix_headers(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    if any((c is None) or (str(c).strip() == "") or str(c).lower().startswith("unnamed") for c in cols):
        new_cols = []
        y_idx = 1
        for i, c in enumerate(cols):
            s = "" if c is None else str(c).strip()
            if i == 0 and s and not s.lower().startswith("unnamed"):
                new_cols.append(s); continue
            if s == "" or s.lower().startswith("unnamed"):
                if i == 0: new_cols.append("X")
                else:
                    new_cols.append(f"Y{y_idx}"); y_idx += 1
            else:
                new_cols.append(s)
        df.columns = new_cols
    return df

# =============================
# Page pipeline
# =============================
def process_pdf_page(pdf_path: str, page_number: int) -> Dict[str, Any]:
    try:
        run_id = uuid.uuid4().hex[:8]
        page_img = load_page_image(pdf_path, page_number)
        page_w, page_h = page_img.width, page_img.height

        regions = detect_regions(pdf_path, page_number, page_w, page_h)
        print(f"Using {len(regions)} region(s) on page {page_number}")

        results: List[Dict[str, Any]] = []
        debug_raw: List[Dict[str, Any]] = []

        for idx, r in enumerate(regions):
            # Tight crop from Upstage, plus small padding for axes
            exp = expand_box(r, page_w, page_h)
            crop = page_img.crop((exp["x"], exp["y"], exp["x"] + exp["w"], exp["y"] + exp["h"]))
            crop_name = f"page_{page_number}_region_{idx}_{run_id}_crop.png"
            crop_path = os.path.join(TEMP_DIR, crop_name)
            crop.save(crop_path, "PNG")
            display_path = enhance_image(crop_path, ENHANCED_DIR)

            axes_meta = r.get("axes") or {"y_left": None,"y_right": None,"has_dual": False}
            axis_map  = r.get("axis_map") or {"left": set(), "right": set()}
            want_dual = bool(axes_meta.get("has_dual"))

            raw = extract_csv_from_image(
                display_path,
                chart_type=r.get("chart_type"),
                series_hints=r.get("series_hints"),
                want_dual=want_dual,
                y_left_label=axes_meta.get("y_left"),
                y_right_label=axes_meta.get("y_right"),
            )
            debug_raw.append({
                "page": page_number,
                "image": os.path.basename(display_path),
                "raw": (raw or "")[:1000]
            })

            records = []
            note = None
            if raw:
                df = csv_to_df(raw)
                if df is not None:
                    df = filter_df_to_axes(df, r.get("series_hints"), axes_meta, axis_map)
                    df = fix_headers(df)
                    df = df.replace([np.inf, -np.inf], np.nan)
                    df = df.where(pd.notnull(df), None)
                    records = deep_sanitize(df.to_dict(orient="records"))
                else:
                    note = "Parsed CSV empty or invalid."
            else:
                note = "No model output."

            results.append({
                "page": page_number,
                "region": idx,
                "image": os.path.basename(display_path),
                "data": records,
                "note": note,
                "chart_type": r.get("chart_type"),
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
