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
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
LLM_MODEL_FALLBACK = os.getenv("LLM_MODEL_FALLBACK", "").strip() or None
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://localhost:5173")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "pdf-extractor")

POPPLER_PATH = os.getenv("POPPLER_PATH") or None
PDF_DPI = _env_int("PDF_DPI", 350)

# Axis-aware padding (capture axes & ticks)
REGION_PAD_LEFT_PCT   = _env_float("REGION_PAD_LEFT_PCT", 0.08)
REGION_PAD_RIGHT_PCT  = _env_float("REGION_PAD_RIGHT_PCT", 0.06)
REGION_PAD_TOP_PCT    = _env_float("REGION_PAD_TOP_PCT", 0.06)
REGION_PAD_BOTTOM_PCT = _env_float("REGION_PAD_BOTTOM_PCT", 0.10)
REGION_MIN_PAD_ABS_PX = _env_int("REGION_MIN_PAD_ABS_PX", 12)
AXIS_PAD_BOOST        = _env_float("AXIS_PAD_BOOST", 0.04)

UPSCALE_FACTOR = _env_float("UPSCALE_FACTOR", 1.0)  # 1.0 = no resize

# Upstage detector
USE_UPSTAGE_DETECTOR = _env_bool("USE_UPSTAGE_DETECTOR", True)
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY", "")
UPSTAGE_URL = os.getenv("UPSTAGE_URL", "https://api.upstage.ai/v1/document-digitization")
UPSTAGE_MODEL = os.getenv("UPSTAGE_MODEL", "document-parse")
UPSTAGE_FORCE_OCR = os.getenv("UPSTAGE_FORCE_OCR", "auto")  # "auto" or "force"

# Filtering knobs (overlay/guide series removal)
PREFERRED_SERIES_RE = re.compile(os.getenv(
    "PREFERRED_SERIES_REGEX",
    r"^(price|index|value|actual|series|y)$"
), re.I)

EXCLUDED_SERIES_RE = re.compile(os.getenv(
    "EXCLUDED_SERIES_REGEX",
    r"(?i)\b(avg|average|mean|median|guide|benchmark|target|upper|lower|band|"
    r"ci|conf(?:idence)?\s*interval|\+?1sd|\+?2sd|-?1sd|-?2sd|std|stdev|"
    r"current(?:\s*\(\d+(\.\d+)?%?\))?)\b"
), re.I)

# Accuracy controls
DEDUP_POLICY = _env_choice("DEDUP_POLICY", ["first", "last", "mean"], "first")
VALIDATION_RETRY = _env_bool("VALIDATION_RETRY", True)
DATE_PARSE_THRESHOLD = _env_float("DATE_PARSE_THRESHOLD", 0.6)
NUMERIC_THRESHOLD = _env_float("NUMERIC_THRESHOLD", 0.70)

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
    if v is None:
        return None
    s = str(v)
    s = re.sub(r"[,\s]", "", s)
    s = re.sub(r"(usd|mn|bn|%|€|£|\$)", "", s, flags=re.I)
    if s in ("", "-", ".", "--"):
        return None
    try:
        return float(s)
    except Exception:
        return None

def _is_numeric_series(s: pd.Series) -> bool:
    return s.apply(lambda x: _to_number(x) is not None).mean() >= NUMERIC_THRESHOLD

def _tries_parse_date(x: Any) -> bool:
    try:
        pd.to_datetime(str(x), errors="raise")
        return True
    except Exception:
        return False

def _date_like_fraction(col: pd.Series) -> float:
    vals = col.dropna().astype(str).tolist()
    if not vals:
        return 0.0
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
    """
    Always write a copy into ENHANCED_DIR (or output_dir) and return that path.
    Never return the TEMP_DIR path. Guarantees file existence in served folder.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(image_path))
    try:
        img = Image.open(image_path)
        if UPSCALE_FACTOR and abs(UPSCALE_FACTOR - 1.0) > 1e-6:
            img = img.resize(
                (int(img.width * UPSCALE_FACTOR), int(img.height * UPSCALE_FACTOR)),
                Image.LANCZOS
            )
        img.save(out_path, "PNG")
        return out_path
    except Exception as e:
        # Fallback: raw copy to ensure the server can serve something
        try:
            with open(image_path, "rb") as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            print(f"[enhance_image] PIL failed ({e}); copied to {out_path}")
            return out_path
        except Exception as e2:
            print(f"[enhance_image] FATAL: could not write to {out_path}: {e2}")
            return out_path

# =============================
# Boxes
# =============================
def expand_box(box, img_w, img_h, chart_type: Optional[str] = None):
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]

    pad_l_pct = REGION_PAD_LEFT_PCT
    pad_b_pct = REGION_PAD_BOTTOM_PCT
    if chart_type in ("line", "bar", "stacked_bar"):
        pad_l_pct += AXIS_PAD_BOOST
        pad_b_pct += AXIS_PAD_BOOST

    pad_l = max(int(w * pad_l_pct), REGION_MIN_PAD_ABS_PX)
    pad_r = max(int(w * REGION_PAD_RIGHT_PCT), REGION_MIN_PAD_ABS_PX)
    pad_t = max(int(h * REGION_PAD_TOP_PCT), REGION_MIN_PAD_ABS_PX)
    pad_b = max(int(h * pad_b_pct), REGION_MIN_PAD_ABS_PX)

    x0 = max(0, x - pad_l)
    y0 = max(0, y - pad_t)
    x1 = min(img_w, x + w + pad_r)
    y1 = min(img_h, y + h + pad_b)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}

# =============================
# Upstage parsing (optional)
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
            "ocr": UPSTAGE_FORCE_OCR,
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

AX_LHS = re.compile(r"\(LHS\)", re.I)
AX_RHS = re.compile(r"\(RHS\)", re.I)

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
                    "x": x0,"y": y0,"w": w,"h": h,
                    "category": cat,"chart_type": chart_type,
                    "series_hints": series_hints,
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
# LLM extraction via OpenRouter (JSON)
# =============================
def _strict_rules_json(chart_type: Optional[str], want_dual: bool,
                       y_left_label: Optional[str], y_right_label: Optional[str]) -> str:
    rules = [
        "You are extracting axis-based data from a chart image into a strict JSON schema.",
        "Return ONLY valid JSON (no prose, no code fences).",
        # include
        "Include ONLY true axis series: bars or plotted lines that vary across X.",
        "If stacked bars: provide one series per stack component.",
        "If pie: provide each slice as a series named by its label, with 'values' as percentages (0–100).",
        # exclude
        "EXCLUDE overlays and guides: averages, ±SD bands, 'Current' markers/lines/dots, benchmarks/targets, confidence bands, and any horizontal reference lines.",
        "EXCLUDE annotations/labels/titles/legends unless they are axis names or series names.",
        # order
        "Preserve the visual order of X (left→right or earliest→latest).",
        # schema
        (
            "Schema:\n"
            "{\n"
            '  "type": "bar|line|stacked_bar|pie|dual_axis|combo|other",\n'
            '  "x_axis": {"label": "<string>", "values": [<strings>]},\n'
            '  "series": [\n'
            '    {"name": "<string>", "axis": "left|right|none", "render": "bar|line", "values": [<numbers>]}\n'
            "  ]\n"
            "}\n"
        ),
    ]
    if chart_type:
        rules.append(f"Detected chart type hint: '{chart_type}'. Use the closest matching 'type' value.")
    if want_dual:
        lbl_l = (y_left_label or "Left Axis").replace('"', "'")
        lbl_r = (y_right_label or "Right Axis").replace('"', "'")
        rules.append(f"This appears dual-axis. Tag series with axis='left' (e.g., {lbl_l}) or axis='right' (e.g., {lbl_r}).")
    return " ".join(rules)

def extract_json_from_image(image_path: str, model: Optional[str] = None,
                            chart_type: Optional[str] = None,
                            series_hints: Optional[List[str]] = None,
                            want_dual: bool = False,
                            y_left_label: Optional[str] = None,
                            y_right_label: Optional[str] = None,
                            fixup_pass: bool = False) -> Optional[dict]:
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

    rules_text = _strict_rules_json(chart_type, want_dual, y_left_label, y_right_label)
    if fixup_pass:
        rules_text += " IMPORTANT: Previous attempt violated the schema or included non-axis data. Correct strictly."

    payload = {
        "model": model_id,
        "max_tokens": 1200,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": rules_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}" }}
            ]
        }]
    }

    def call(p):
        try:
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                 headers=headers, json=p, timeout=120)
        except Exception as e:
            print("LLM network exception:", e)
            return None
        if not resp.ok:
            print("LLM HTTP error:", resp.status_code, resp.text[:300])
            return None
        try:
            j = resp.json()
            content = j["choices"][0]["message"].get("content")
            return json.loads(content) if content else None
        except Exception:
            print("Unexpected LLM response:", resp.text[:300])
            return None

    out = call(payload)
    if (not out) and LLM_MODEL_FALLBACK:
        payload["model"] = LLM_MODEL_FALLBACK
        out = call(payload)
    return out

# =============================
# JSON → DataFrame
# =============================
def json_to_df(j: dict) -> Optional[pd.DataFrame]:
    try:
        if not j or not isinstance(j, dict):
            return None
        x_axis = j.get("x_axis") or {}
        x_vals = x_axis.get("values") or []
        series = j.get("series") or []
        if not x_vals or not series:
            return None
        data = {"X": list(map(lambda x: str(x), x_vals))}
        for s in series:
            name = (s.get("name") or "Y").strip() or "Y"
            vals = s.get("values") or []
            if len(vals) < len(x_vals):
                vals = vals + [None] * (len(x_vals) - len(vals))
            if len(vals) > len(x_vals):
                vals = vals[: len(x_vals)]
            data[name] = [(_to_number(v)) for v in vals]
        df = pd.DataFrame(data)
        if df.empty:
            return None
        return df
    except Exception as e:
        print("json_to_df error:", e)
        return None

# =============================
# Normalization / Validation
# =============================
def sort_by_x(df: pd.DataFrame) -> pd.DataFrame:
    x = df["X"]
    # try numeric then date (no infer_datetime_format to avoid warnings)
    try:
        xn = x.map(_to_number)
        if xn.notna().mean() >= 0.7:
            out = df.copy()
            out["X"] = xn
            return out.sort_values("X")
    except Exception:
        pass
    try:
        xd = pd.to_datetime(x, errors="coerce")
        if xd.notna().mean() >= 0.7:
            out = df.copy()
            out["X"] = xd.dt.strftime("%Y-%m-%d")
            return out.sort_values("X")
    except Exception:
        pass
    return df  # keep model order (already visual order)

def dedup_by_x(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    if df.empty or "X" not in df.columns:
        return df, 0
    before = len(df)
    if DEDUP_POLICY == "first":
        df2 = df.drop_duplicates(subset=["X"], keep="first")
    elif DEDUP_POLICY == "last":
        df2 = df.drop_duplicates(subset=["X"], keep="last")
    else:  # mean
        aggs = {c: "mean" for c in df.columns if c != "X"}
        df2 = df.groupby("X", as_index=False).agg(aggs)
    return df2, before - len(df2)

def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    issues = []
    cols = list(df.columns)
    if "X" not in cols:
        issues.append("missing column 'X'")
    numeric_cols = [c for c in cols if c != "X" and _is_numeric_series(df[c])]
    if not numeric_cols:
        issues.append("no numeric series columns detected")
    if "X" in df.columns:
        dups = df["X"].duplicated().sum()
        if dups > 0:
            issues.append(f"{dups} duplicate X values")
    return (len(issues) == 0), issues

def prune_non_axis_series(df: pd.DataFrame) -> pd.DataFrame:
    """Drop series that are overlays/guides or constant across X."""
    if df is None or df.empty or "X" not in df.columns:
        return df
    keep = ["X"]
    for c in df.columns:
        if c == "X":
            continue
        name = str(c)
        # drop by name (Avg, ±SD, Current, bands, target, etc.)
        if EXCLUDED_SERIES_RE.search(name):
            continue
        s = df[c]
        # drop constant series (horizontal reference)
        vals = pd.to_numeric(s, errors="coerce")
        if vals.notna().sum() > 0 and vals.nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    return df[keep]

def ensure_x_first(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure X is the first (leftmost) column."""
    if df is None or df.empty or "X" not in df.columns:
        return df
    cols = ["X"] + [c for c in df.columns if c != "X"]
    return df[cols]

def extract_series_meta(j: dict) -> List[Dict[str, Any]]:
    """Pull rendering/axis meta from the LLM JSON so the UI can draw combo charts."""
    meta: List[Dict[str, Any]] = []
    for s in (j or {}).get("series", []) or []:
        meta.append({
            "name": s.get("name"),
            "axis": (s.get("axis") or "left").lower(),
            "render": (s.get("render") or "line").lower(),
        })
    return meta

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
            want_dual = bool(axes_meta.get("has_dual"))

            # Crop with axis-biased padding
            exp = expand_box(r, page_w, page_h, chart_type=chart_type)
            crop = page_img.crop((exp["x"], exp["y"], exp["x"] + exp["w"], exp["y"] + exp["h"]))
            crop_name = f"page_{page_number}_region_{idx}_{run_id}_crop.png"
            crop_path = os.path.join(TEMP_DIR, crop_name)
            crop.save(crop_path, "PNG")
            display_path = enhance_image(crop_path, ENHANCED_DIR)
            if not os.path.exists(display_path):
                try:
                    with open(crop_path, "rb") as src, open(display_path, "wb") as dst:
                        dst.write(src.read())
                    print(f"[process_pdf_page] Backfilled missing enhanced image -> {display_path}")
                except Exception as e:
                    print(f"[process_pdf_page] Could not backfill enhanced image: {e}")

            # --- LLM pass 1 (JSON)
            raw_json = extract_json_from_image(
                display_path,
                chart_type=chart_type,
                series_hints=r.get("series_hints"),
                want_dual=want_dual,
                y_left_label=axes_meta.get("y_left"),
                y_right_label=axes_meta.get("y_right"),
                fixup_pass=False,
            )

            debug_entry = {
                "page": page_number,
                "image": os.path.basename(display_path),
                "raw": json.dumps(raw_json, ensure_ascii=False)[:2000] if raw_json else "",
            }

            # Parse + normalize
            records = []
            note_parts: List[str] = []
            confidence = "high"
            rows_dropped = 0
            validated_ok = False
            inferred_chart_type = None
            series_meta: List[Dict[str, Any]] = []

            def postprocess(j: Optional[dict]) -> Tuple[Optional[pd.DataFrame], List[str], int, bool, Optional[str], List[Dict[str, Any]]]:
                notes: List[str] = []
                if not j:
                    notes.append("No model output.")
                    return None, notes, 0, False, None, []
                df = json_to_df(j)
                if df is None:
                    notes.append("Parsed JSON empty or invalid.")
                    return None, notes, 0, False, j.get("type"), extract_series_meta(j)

                # Sort, dedup, prune overlays/constant, ensure X first
                df = sort_by_x(df)
                df, dropped = dedup_by_x(df)
                if dropped > 0:
                    notes.append(f"Deduplicated X ({dropped} rows dropped via policy '{DEDUP_POLICY}').")

                df = prune_non_axis_series(df)
                df = ensure_x_first(df)

                ok, issues = validate_dataframe(df)
                if not ok:
                    notes.append("Validation issues: " + "; ".join(issues))

                return df, notes, dropped, ok, j.get("type"), extract_series_meta(j)

            df, notes1, dropped1, ok1, t1, meta1 = postprocess(raw_json)

            # Retry once if invalid
            if (not ok1) and VALIDATION_RETRY:
                raw_fix = extract_json_from_image(
                    display_path,
                    chart_type=chart_type,
                    series_hints=r.get("series_hints"),
                    want_dual=want_dual,
                    y_left_label=axes_meta.get("y_left"),
                    y_right_label=axes_meta.get("y_right"),
                    fixup_pass=True,
                )
                debug_entry["raw_fix"] = json.dumps(raw_fix, ensure_ascii=False)[:2000] if raw_fix else ""
                df2, notes2, dropped2, ok2, t2, meta2 = postprocess(raw_fix)
                if ok2 or (df2 is not None and (df is None or len(df2) >= len(df))):
                    df, notes1, dropped1, ok1, t1, meta1 = df2, notes2, dropped2, ok2, t2, meta2

            debug_raw.append(debug_entry)

            if df is not None:
                rows_dropped = dropped1
                validated_ok = ok1
                inferred_chart_type = t1 or chart_type
                series_meta = meta1 or []
                if not validated_ok:
                    confidence = "low"
                elif rows_dropped > 0:
                    confidence = "medium"
                else:
                    confidence = "high"

                df_out = df.copy()
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
                "chart_type": inferred_chart_type or chart_type,
                "category": r.get("category"),
                "series_hints": r.get("series_hints"),
                "series_meta": series_meta,   # <-- for combo/dual axis rendering
            })

        payload = {"tables": results, "debug_raw": debug_raw}
        return deep_sanitize(payload)
    except Exception as e:
        return {"error": str(e)}

# Whole-PDF (optional)
def process_pdf_to_data(pdf_path: str) -> Dict[str, Any]:
    n = get_pdf_page_count(pdf_path)
    combined = {"tables": [], "debug_raw": []}
    for p in range(1, n + 1):
        r = process_pdf_page(pdf_path, p)
        combined["tables"].extend(r.get("tables", []))
        combined["debug_raw"].extend(r.get("debug_raw", []))
    return deep_sanitize(combined)

# Optional cache helper if you want a /reset endpoint
_cache: dict = {}
def reset_cache():
    try:
        _cache.clear()
    except Exception:
        pass

if __name__ == "__main__":
    test_pdf = r"C:\path\to\your.pdf"
    print(process_pdf_page(test_pdf, 1))
