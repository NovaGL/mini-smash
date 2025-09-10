# -*- coding: utf-8 -*-
import os
import re
import json
import zipfile
import tempfile
from datetime import datetime, date
from typing import List, Tuple, Any

import streamlit as st
import pandas as pd
import requests
import camelot
from PIL import Image
from io import BytesIO
import mimetypes

# =========================================================
# Streamlit configuration
# =========================================================
st.set_page_config(page_title="Coupa PO Finder", page_icon="📄", layout="wide")

# =========================================================
# Constants & defaults
# =========================================================
DEFAULT_UOM_CODE = "EA"           # fallback for quantity lines
PRICE_DECIMALS = 2
QTY_DECIMALS = 2

# Internal storage columns (canonical schema we keep in session)
BASE_COLS = [
    "line_num", "inv_type", "description", "price", "quantity", "uom_code",
    "account_id", "commodity_name",
    "po_number", "order_header_num", "order_line_id", "order_line_num",
    "order_header_id", "source_part_num",
    "delete"
]

# Editor show order (hide order_header_num, order_line_id, and account_id in editor)
# ⬇️ PO # first, then PO Line # (as requested)
EDITOR_COL_ORDER = [
    "line_num",
    "inv_type",
    "description",
    "source_part_num",   # Supplier Part #
    "uom_code",
    "commodity_name",
    "po_number",         # --> PO # first
    "order_line_num",    # --> then PO Line #
    "quantity",
    "price",
    "delete"
]

# =========================================================
# Small utilities
# =========================================================
def _po_prefix() -> str:
    return (st.session_state.get("po_prefix") or "RCH").strip()

def _to_decimal_str(value, precision=2) -> str:
    try:
        return f"{float(value):.{precision}f}"
    except Exception:
        return f"{0:.{precision}f}"

def _numeric_from_po_string(po_str: str) -> int:
    """
    Derive the numeric PO number from any string like 'RCH000123'.
    Uses the trailing digit run; falls back to all digits; else 0.
    """
    if not po_str:
        return 0
    m = re.search(r'(\d+)$', str(po_str))
    if m:
        return int(m.group(1).lstrip('0') or '0')
    digits = re.findall(r'\d+', str(po_str))
    if digits:
        return int(digits[-1].lstrip('0') or '0')
    return 0

def _numeric_po_id(scanned_po: str) -> str:
    """Used for GET /purchase_orders/:id – strips the configured prefix and zeros."""
    prefix = _po_prefix()
    if scanned_po and scanned_po.startswith(prefix):
        return scanned_po[len(prefix):].lstrip('0') or '0'
    return (scanned_po or '').lstrip('0') or '0'

def _get_any(d: dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default

def _unwrap_first(obj, key_list):
    if not isinstance(obj, dict):
        return obj
    for key in key_list:
        if key in obj:
            node = obj[key]
            if isinstance(node, list):
                return node[0] if node else {}
            if isinstance(node, dict):
                return node
    return obj

def normalize_po_json(po_raw):
    if isinstance(po_raw, list):
        po_raw = po_raw[0] if po_raw else {}
    po = _unwrap_first(po_raw, ["order-headers", "order_header", "order-header"])
    return po

def extract_order_lines(po):
    if not isinstance(po, dict):
        return []
    lines = po.get("order-lines") or po.get("order_lines") or []
    flat = []
    for item in lines:
        if isinstance(item, dict) and "order-line" in item:
            flat.append(item["order-line"])
        else:
            flat.append(item)
    return flat

def is_debug() -> bool:
    return bool(st.session_state.get("debug_enabled", False))

def _redact_headers(h: dict) -> dict:
    if not h:
        return {}
    redacted = dict(h)
    if "Authorization" in redacted:
        redacted["Authorization"] = "Bearer ********"
    return redacted

def _log_request(method, url, headers=None, params=None, json_body=None, files=None, group_title=None):
    if not is_debug():
        return
    title = group_title or f"{method} {url}"
    with st.expander(f"🔍 Request – {title}", expanded=False):
        st.write("**Headers**")
        st.code(json.dumps(_redact_headers(headers or {}), indent=2))
        if params:
            st.write("**Params**")
            st.code(json.dumps(params, indent=2))
        if json_body is not None:
            st.write("**Body**")
            st.code(json.dumps(json_body, indent=2))
        if files:
            st.write("**Files**")
            st.code(str(list(files.keys())))

def _log_response(title: str, resp: requests.Response, compact_fields: List[str] = None):
    if not is_debug():
        return
    with st.expander(f"📥 Response – {title} (HTTP {resp.status_code})", expanded=False):
        st.write("**Headers**")
        st.code(json.dumps(dict(resp.headers), indent=2))
        try:
            data = resp.json()
        except Exception:
            data = resp.text

        if isinstance(data, dict) and compact_fields:
            compact = {k: data.get(k) for k in compact_fields}
            st.write("**Summary**")
            st.code(json.dumps(compact, indent=2))
            with st.expander("Full JSON", expanded=False):
                st.code(json.dumps(data, indent=2))
        else:
            if isinstance(data, (dict, list)):
                st.code(json.dumps(data, indent=2))
            else:
                st.code(str(data))

def _log_ocr_debug(image_info: str, po_numbers: List[str]):
    """Log OCR debug information when debug mode is enabled"""
    if not is_debug():
        return
    
    with st.expander(f"🔍 OCR Debug – {image_info}", expanded=False):
        st.write("**PO Number Detection:**")
        prefix = _po_prefix()
        po_pattern = r'\b' + re.escape(prefix) + r'\d{6}\b'
        st.code(f"Pattern used: {po_pattern}")
        
        if po_numbers:
            st.success(f"Found {len(po_numbers)} PO number(s): {', '.join(po_numbers)}")
        else:
            st.warning("No PO numbers found - OCR functionality has been disabled")

# =========================================================
# OCR (disabled - returns empty results with warning)
# =========================================================
def _ocr_text_from_image_bytes(img_bytes: bytes, filename: str = "image") -> str:
    """
    OCR functionality has been removed. Returns empty string.
    """
    if is_debug():
        _log_ocr_debug(filename, [])
        st.warning("⚠️ **OCR Disabled**: Image text extraction has been removed from this version.")
    return ""

def _extract_po_numbers_from_text(text: str) -> List[str]:
    """Find PO matches based on configured prefix + 6 digits."""
    if not text:
        return []
    prefix = _po_prefix()
    # Make pattern more flexible - allow for spaces or other characters between prefix and numbers
    po_pattern = r'\b' + re.escape(prefix) + r'\s*\d{6}\b'
    matches = re.findall(po_pattern, text, re.IGNORECASE)
    # Clean up matches by removing any spaces
    cleaned_matches = [re.sub(r'\s+', '', match.upper()) for match in matches]
    return list(set(cleaned_matches))

def extract_po_numbers_from_image(uploaded_file) -> List[str]:
    """Extract PO numbers from uploaded image file - OCR disabled"""
    if is_debug():
        st.warning(f"⚠️ **OCR Disabled**: Cannot extract text from image {uploaded_file.name}")
    return []

# =========================================================
# PDF extraction (improved with debug info)
# =========================================================
def extract_po_numbers_from_single_pdf(file_path, filename):
    po_numbers = []
    debug_info = {"tables_found": 0, "pages_processed": 0, "extraction_methods": []}
    
    try:
        # Try camelot for table extraction
        if is_debug():
            debug_info["extraction_methods"].append("camelot")
        
        tables = camelot.read_pdf(file_path, pages='all')
        debug_info["tables_found"] = len(tables)
        
        prefix = _po_prefix()
        po_pattern = r'\b' + re.escape(prefix) + r'\s*\d{6}\b'
        
        for i, table in enumerate(tables):
            df = table.df
            for _, row in df.iterrows():
                for cell_value in row:
                    cell_str = str(cell_value)
                    matches = re.findall(po_pattern, cell_str, re.IGNORECASE)
                    for match in matches:
                        clean_match = re.sub(r'\s+', '', match.upper())
                        po_numbers.append(clean_match)
        
        # Try PyPDF2 for text extraction
        try:
            import PyPDF2
            if is_debug():
                debug_info["extraction_methods"].append("PyPDF2")
                
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                debug_info["pages_processed"] = len(reader.pages)
                
                all_text = ""
                for page in reader.pages:
                    text = page.extract_text() or ""
                    all_text += text + "\n"
                    matches = re.findall(po_pattern, text, re.IGNORECASE)
                    for match in matches:
                        clean_match = re.sub(r'\s+', '', match.upper())
                        po_numbers.append(clean_match)
                
                if is_debug():
                    _log_ocr_debug(f"PDF: {filename}", list(set(po_numbers)))
                    
        except ImportError:
            if is_debug():
                st.warning("PyPDF2 not available. Install with: `pip install PyPDF2`")
        except Exception as e:
            if is_debug():
                st.warning(f"PyPDF2 extraction failed: {str(e)}")
        
        # Try pdfplumber as alternative
        try:
            import pdfplumber
            if is_debug():
                debug_info["extraction_methods"].append("pdfplumber")
                
            with pdfplumber.open(file_path) as pdf:
                all_text = ""
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    all_text += text + "\n"
                    matches = re.findall(po_pattern, text, re.IGNORECASE)
                    for match in matches:
                        clean_match = re.sub(r'\s+', '', match.upper())
                        po_numbers.append(clean_match)
                        
        except ImportError:
            if is_debug():
                st.info("pdfplumber not available (optional). Install with: `pip install pdfplumber`")
        except Exception as e:
            if is_debug():
                st.warning(f"pdfplumber extraction failed: {str(e)}")
                
    except Exception as e:
        if is_debug():
            st.error(f"PDF processing error for {filename}: {str(e)}")
    
    # Log debug info
    if is_debug():
        with st.expander(f"📄 PDF Debug – {filename}", expanded=False):
            st.json(debug_info)
            if po_numbers:
                st.success(f"Found PO numbers: {list(set(po_numbers))}")
            else:
                st.warning("No PO numbers found in PDF")
    
    return list(set(po_numbers))

def extract_po_numbers_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    try:
        return extract_po_numbers_from_single_pdf(tmp_path, uploaded_file.name)
    except Exception as e:
        if is_debug():
            st.error(f"Error processing PDF {uploaded_file.name}: {str(e)}")
        return []
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

def process_zip_file(uploaded_file):
    all_po_numbers = []
    file_data_map = {}
    processing_summary = {"files_processed": 0, "files_failed": 0, "file_types": {}}
    
    try:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            members = [f for f in zip_ref.namelist() if not f.startswith('__MACOSX')]
            
            for member in members:
                try:
                    content = zip_ref.read(member)
                    file_ext = member.lower().split('.')[-1] if '.' in member else 'unknown'
                    processing_summary["file_types"][file_ext] = processing_summary["file_types"].get(file_ext, 0) + 1
                    
                    # PDFs
                    if member.lower().endswith('.pdf'):
                        file_data_map[member] = content
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                            tmp.write(content)
                            tmp_path = tmp.name
                        found_pos = extract_po_numbers_from_single_pdf(tmp_path, member)
                        all_po_numbers.extend(found_pos)
                        os.unlink(tmp_path)
                        
                    # Images (OCR disabled)
                    elif member.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')):
                        # OCR functionality removed - just store the file data
                        file_data_map[member] = content
                        if is_debug():
                            st.warning(f"⚠️ Skipping OCR for image: {member} (OCR disabled)")
                    
                    processing_summary["files_processed"] += 1
                    
                except Exception as e:
                    processing_summary["files_failed"] += 1
                    if is_debug():
                        st.warning(f"Failed to process {member}: {str(e)}")
        
        if is_debug():
            with st.expander(f"📦 ZIP Processing Summary", expanded=False):
                st.json(processing_summary)
                st.write(f"Total unique PO numbers found: {len(set(all_po_numbers))}")
                
        return list(set(all_po_numbers)), file_data_map
    except Exception as e:
        if is_debug():
            st.error(f"ZIP processing error: {str(e)}")
        return [], {}

# =========================================================
# Coupa API helpers
# =========================================================
def verify_po_in_coupa(po_id, headers, coupa_instance):
    """
    Verify PO and normalize to flat 'order-lines' list.
    """
    try:
        numeric_po_id = _numeric_po_id(po_id)
        fields_spec = [
            "id", "number", "status", "total",
            {"supplier": ["id", "name"]},
            {"currency": ["code"]},
            {"ship-to-address": ["id", "name", "street1", "city", "state", "postal-code", "country"]},
            {"bill-to-address": ["id", "name", "street1", "city", "state", "postal-code", "country"]},
            {"order-lines": [
                "id", "description", "price", "quantity",
                "line-num", "order-header-id", "source-part-num",
                "service-type",
                {"uom": ["code"]},
                {"item": ["id"]},
                {"account": ["id"]},
                {"commodity": ["name"]}
            ]}
        ]

        def _fetch(params):
            url = f"https://{coupa_instance}.coupahost.com/api/purchase_orders/{numeric_po_id}"
            _log_request("GET", url, headers=headers, params=params)
            r = requests.get(url, headers=headers, params=params, timeout=30)
            _log_response(f"GET Purchase Order {numeric_po_id}", r)
            return r

        resp = _fetch({"fields": json.dumps(fields_spec)})
        if resp.status_code != 200:
            return {'exists': False, 'error': f'HTTP {resp.status_code}'}

        po_raw = resp.json()
        po = normalize_po_json(po_raw)
        lines = extract_order_lines(po)

        if not lines:
            resp_full = _fetch(None)
            if resp_full.status_code != 200:
                return {'exists': False, 'error': f'HTTP {resp_full.status_code}'}
            po = normalize_po_json(resp_full.json())
            lines = extract_order_lines(po)

        po["order-lines"] = lines
        return {'exists': True, 'po_data': po}

    except Exception as e:
        return {'exists': False, 'error': str(e)}

def _guess_mime(fname: str) -> str:
    mime, _ = mimetypes.guess_type(fname or "")
    return mime or "application/octet-stream"

def upload_original_scan(invoice_id, file_bytes, upload_filename, headers, coupa_instance):
    """
    PUT /api/invoices/:id/image_scan with the original filename.
    Accepts PDFs or images (PNG/JPG/TIFF).
    """
    try:
        files = {'file': (upload_filename, file_bytes, _guess_mime(upload_filename))}
        upload_headers = {k: v for k, v in headers.items() if k.lower() != "content-type"}
        upload_headers["Accept"] = "application/json"
        upload_url = f"https://{coupa_instance}.coupahost.com/api/invoices/{invoice_id}/image_scan"
        _log_request("PUT", upload_url, headers=upload_headers, files={"file": upload_filename})
        resp = requests.put(upload_url, headers=upload_headers, files=files, timeout=60)
        _log_response("PUT image_scan", resp, compact_fields=["id", "status"])
        return (resp.status_code in [200, 201]), (resp.text if resp.text else f"HTTP {resp.status_code}")
    except Exception as e:
        return False, str(e)

# =========================================================
# Editor building / sanitation
# =========================================================
def build_preview_df(po_id: str, po_data: dict) -> pd.DataFrame:
    numeric_po = int(_numeric_po_id(po_id))
    rows = []
    for i, line in enumerate(extract_order_lines(po_data), 1):
        line_num_from_po = _get_any(line, "line-num", "line_num")
        order_header_id = _get_any(line, "order-header-id", "order_header_id")
        source_part_num = _get_any(line, "source-part-num", "source_part_num")
        qty = float(_get_any(line, "quantity", "qty", default=0) or 0)
        price = float(_get_any(line, "price", default=0) or 0)
        acct_id = _get_any(_get_any(line, "account", default={}), "id")
        commodity_name = _get_any(_get_any(line, "commodity", default={}), "name")
        uom_code = _get_any(_get_any(line, "uom", default={}), "code")
        service_type = _get_any(line, "service-type", "service_type")

        inv_type = "InvoiceAmountLine" if (service_type and str(service_type).strip()) else "InvoiceQuantityLine"

        rows.append({
            "line_num": i,
            "inv_type": inv_type,
            "description": _get_any(line, "description", default=""),
            "price": price,
            "quantity": qty,
            "uom_code": uom_code or (DEFAULT_UOM_CODE if inv_type == "InvoiceQuantityLine" else None),
            "account_id": acct_id,
            "commodity_name": commodity_name,
            "po_number": po_id,                            # default per line
            "order_header_num": numeric_po,
            "order_line_id": _get_any(line, "id"),
            "order_line_num": str(line_num_from_po) if line_num_from_po is not None else str(i),
            "order_header_id": order_header_id,
            "source_part_num": source_part_num,
            "delete": False
        })

    df = pd.DataFrame(rows)
    for col in BASE_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    for c in ["line_num", "quantity", "price", "order_header_num", "order_line_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _default_account_id_from(df: pd.DataFrame):
    vals = df["account_id"].dropna()
    return int(vals.iloc[0]) if not vals.empty else None

def resequence_lines(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(by=["line_num"], kind="stable")
    df["line_num"] = range(1, len(df) + 1)
    df["order_line_num"] = df["line_num"].astype(str)
    return df

def sanitize_editor_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "delete" in df.columns:
        df = df[~df["delete"].astype(bool)].copy()

    for idx, r in df.iterrows():
        inv_type = str(r.get("inv_type") or "InvoiceQuantityLine").strip()
        df.at[idx, "inv_type"] = inv_type
        df.at[idx, "price"] = float(_to_decimal_str(r.get("price", 0) or 0, PRICE_DECIMALS))
        qty_val = r.get("quantity", 0) or 0
        df.at[idx, "quantity"] = float(_to_decimal_str(qty_val, QTY_DECIMALS))
        # keep provided or default UOM on qty lines
        if inv_type == "InvoiceQuantityLine":
            uom_code = (r.get("uom_code") or "").strip() or DEFAULT_UOM_CODE
            df.at[idx, "uom_code"] = uom_code
        else:
            df.at[idx, "uom_code"] = None
            if df.at[idx, "quantity"] == 0:
                df.at[idx, "quantity"] = 0.00
        # ensure line has some po_number string
        df.at[idx, "po_number"] = (str(r.get("po_number") or "").strip() or str(df.at[idx, "po_number"] or "")) or ""

    df = resequence_lines(df)
    for c in ["order_header_num", "order_line_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# =========================================================
# Invoice creation
# =========================================================
def create_invoice_from_po(
    context_po_id: str,
    invoice_number: str,
    invoice_date: date,
    original_file_data: bytes,
    upload_filename: str,
    po_data: dict,
    edited_df: pd.DataFrame,
    header_tax_rate: float
) -> Tuple[bool, Any, bool, str]:
    """
    POST /api/invoices, then PUT image_scan.
    Returns (invoice_success, invoice_resp|error, scan_success, scan_message).
    """
    current_env = st.session_state.get("environment", "Test")
    env_suffix = "_PROD" if current_env == "Production" else ""
    IDENTIFIER = os.environ.get(f"PO_IDENTIFIER{env_suffix}")
    SECRET = os.environ.get(f"PO_SECRET{env_suffix}")
    COUPA_INSTANCE = os.environ.get(f"COUPA_INSTANCE{env_suffix}")

    token_url = f"https://{COUPA_INSTANCE}.coupahost.com/oauth2/token"
    token_data = {"grant_type": "client_credentials", "scope": "core.invoice.write"}

    # ---- Auth
    try:
        _log_request("POST", token_url, headers={"Authorization":"Bearer ********"}, json_body=token_data)
        response = requests.post(
            token_url,
            auth=(IDENTIFIER, SECRET),
            data=token_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30
        )
        _log_response("POST OAuth Token (Invoice Write)", response, compact_fields=["access_token", "token_type", "expires_in"])
        response.raise_for_status()
        access_token = response.json().get("access_token")
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    except Exception as e:
        return False, f"Auth error: {e}", False, "N/A"

    # ---- Build header & lines
    
    supplier_obj = po_data.get("supplier") or {}
    currency_obj = po_data.get("currency") or {}

    subtotal = float(edited_df["price"].fillna(0).astype(float).mul(edited_df["quantity"].fillna(0).astype(float)).sum())
    tax_rate_used = float(header_tax_rate or 0.0)
    tax_amount = round(subtotal * (tax_rate_used / 100.0), 2) if tax_rate_used > 0 else 0.0

    invoice_data = {
        "invoice-number": invoice_number,
        "invoice-date": invoice_date.strftime("%Y-%m-%d"),
        "supplier": {"id": supplier_obj.get("id")} if supplier_obj.get("id") else None,
        "currency": {"code": currency_obj.get("code")} if currency_obj.get("code") else None,
        "ship-to-address": po_data.get("ship-to-address"),
        "bill-to-address": po_data.get("bill-to-address"),
        "line-level-taxation": False if tax_amount > 0 else None,
        "tax-rate": tax_rate_used if tax_amount > 0 else None,
        "tax-amount": float(_to_decimal_str(tax_amount, PRICE_DECIMALS)) if tax_amount > 0 else None,
        "invoice-lines": []
    }
    invoice_data = {k: v for k, v in invoice_data.items() if v is not None}

    for _, r in edited_df.iterrows():
        inv_type = (r.get("inv_type") or "InvoiceQuantityLine").strip()
        is_qty = inv_type == "InvoiceQuantityLine"
        line_po_number = str(r.get("po_number") or context_po_id).strip() or context_po_id
        order_header_num = _numeric_from_po_string(line_po_number)

        line = {
            "type": inv_type,
            "line-num": int(r["line_num"]) if pd.notna(r["line_num"]) else None,
            "description": str(r.get("description", "")),
            "price": _to_decimal_str(r.get("price", 0) or 0, PRICE_DECIMALS),
            "po-number": line_po_number,
            "order-header-num": int(order_header_num),
            "order-line-id": int(r["order_line_id"]) if pd.notna(r["order_line_id"]) else None,
            "order-line-num": str(r["order_line_num"]) if pd.notna(r["order_line_num"]) else None,
        }
        if is_qty:
            line["quantity"] = _to_decimal_str(r.get("quantity", 0) or 0, QTY_DECIMALS)
            uom_code = (r.get("uom_code") or "").strip() or DEFAULT_UOM_CODE
            line["uom"] = {"code": uom_code}

        if pd.notna(r.get("source_part_num")) and str(r.get("source_part_num")).strip():
            line["source-part-num"] = str(r.get("source_part_num")).strip()
        if pd.notna(r.get("account_id")) and str(r.get("account_id")).strip():
            line["account"] = {"id": int(r.get("account_id"))}
        if pd.notna(r.get("commodity_name")) and str(r.get("commodity_name")).strip():
            line["commodity"] = {"name": str(r.get("commodity_name")).strip()}

        invoice_data["invoice-lines"].append({k: v for k, v in line.items() if v is not None})

    # ---- Create invoice
    try:
        create_url = f"https://{COUPA_INSTANCE}.coupahost.com/api/invoices"
        _log_request("POST", create_url, headers=headers, json_body=invoice_data)
        resp = requests.post(create_url, headers=headers, json=invoice_data, timeout=45)
        _log_response("POST /api/invoices", resp, compact_fields=["id", "status", "total", "tax-amount", "tax-rate"])
        if resp.status_code not in [200, 201]:
            return False, f"HTTP {resp.status_code} - {resp.text}", False, "N/A"

        invoice_id = resp.json().get("id")
    except Exception as e:
        return False, str(e), False, "N/A"

    # ---- Upload original scan (PDF or Image)
    try:
        scan_ok, scan_msg = upload_original_scan(
            invoice_id=invoice_id,
            file_bytes=original_file_data,
            upload_filename=upload_filename,
            headers=headers,
            coupa_instance=COUPA_INSTANCE
        )
        return True, {"id": invoice_id}, scan_ok, scan_msg
    except Exception as e:
        return True, {"id": invoice_id}, False, str(e)

# =========================================================
# Streamlit App
# =========================================================
def main():
    # SAFE defaults (do not overwrite widget-managed keys after they're created)
    st.session_state.setdefault("verification_complete", False)
    st.session_state.setdefault("invoice_lines_preview", {})  # {po_id: DataFrame}
    st.session_state.setdefault("po_context", {})             # {po_id: {...ui state...}}
    st.session_state.setdefault("create_results", [])         # list of batch results
    st.session_state.setdefault("selected_for_batch", {})     # {po_id: bool}
    st.session_state.setdefault("po_prefix", "RCH")

    # Sidebar (Environment, Debug, PO Prefix)
    with st.sidebar:
        st.header("Environment")
        st.selectbox(
            "Select Environment:",
            ["Test", "Production"],
            index=0 if st.session_state.get("environment", "Test") == "Test" else 1,
            key="environment"
        )
        st.text_input("PO Prefix", key="po_prefix", value=st.session_state.get("po_prefix", "RCH"))

        # Environment quick checks & caution banner
        env_suffix = "_PROD" if st.session_state.get("environment", "Test") == "Production" else ""
        instance = os.environ.get(f"COUPA_INSTANCE{env_suffix}", "")
        ident = os.environ.get(f"PO_IDENTIFIER{env_suffix}", "")
        secret = os.environ.get(f"PO_SECRET{env_suffix}", "")
        if st.session_state.get("environment") == "Production":
            st.error("🛑 **PRODUCTION – Use with caution**")
        if not instance or not ident or not secret:
            st.warning("⚠️ Missing one or more required environment variables for the selected environment (COUPA_INSTANCE, PO_IDENTIFIER, PO_SECRET).")
        st.caption(f"Instance: {instance or 'Not set'}")

        st.header("Debug")
        st.toggle(
            "Debug Mode",
            value=st.session_state.get('debug_enabled', False),
            key="debug_enabled",
            help="Show grouped API requests/responses and OCR debug info (auth redacted)"
        )

    # Main header + production banner (top-level)
    st.markdown("Upload **PDF or ZIP** to find PO numbers, verify in Coupa, **edit lines**, and create invoices")
    st.warning("⚠️ **OCR Disabled**: Image text extraction has been removed. Only PDF text extraction is available.")
    if st.session_state.get("environment") == "Production":
        st.error("🛑 **You are in Production. Use with caution.**")

    # File upload (PDF or ZIP only - images no longer supported for OCR)
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "zip"],
        help="Upload a PDF or ZIP containing PDFs (image OCR has been disabled)"
    )

    if uploaded_file:
        file_lower = uploaded_file.name.lower()
        is_zip = file_lower.endswith('.zip')
        is_pdf = file_lower.endswith('.pdf')

        with st.spinner(f"Processing {'ZIP' if is_zip else 'PDF'} file..."):
            if is_zip:
                found_pos, file_data_map = process_zip_file(uploaded_file)
                st.session_state.file_data_map = file_data_map   # may include PDFs
                st.session_state.original_file_data = None
                st.session_state.original_file_name = None
            elif is_pdf:
                found_pos = extract_po_numbers_from_pdf(uploaded_file)
                st.session_state.original_file_data = uploaded_file.getvalue()
                st.session_state.original_file_name = uploaded_file.name
                st.session_state.file_data_map = {}
            else:
                found_pos = []

        if found_pos:
            unique_pos = list(set(found_pos))
            st.success(f"Found {len(unique_pos)} unique PO(s): {', '.join(unique_pos)}")

            # Verify
            if st.button("🔍 Verify POs in Coupa", type="primary"):
                env_suffix = "_PROD" if st.session_state.get("environment", "Test") == "Production" else ""
                IDENTIFIER = os.environ.get(f"PO_IDENTIFIER{env_suffix}")
                SECRET = os.environ.get(f"PO_SECRET{env_suffix}")
                COUPA_INSTANCE = os.environ.get(f"COUPA_INSTANCE{env_suffix}")
                if not IDENTIFIER or not SECRET or not COUPA_INSTANCE:
                    st.error("Please set environment variables in the selected environment.")
                    return

                # Auth for PO read
                token_url = f"https://{COUPA_INSTANCE}.coupahost.com/oauth2/token"
                token_data = {"grant_type": "client_credentials", "scope": "core.purchase_order.read"}
                try:
                    _log_request("POST", token_url, headers={"Authorization":"Bearer ********"}, json_body=token_data)
                    with st.spinner("🔑 Authenticating..."):
                        response = requests.post(
                            token_url,
                            auth=(IDENTIFIER, SECRET),
                            data=token_data,
                            headers={"Content-Type": "application/x-www-form-urlencoded"},
                            timeout=30
                        )
                    _log_response("POST OAuth Token (PO Read)", response, compact_fields=["access_token", "token_type", "expires_in"])
                    response.raise_for_status()
                    access_token = response.json().get("access_token")
                    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
                except Exception as e:
                    st.error(f"Auth failed: {str(e)}")
                    return

                # Verify POs
                with st.spinner("🔍 Verifying POs in Coupa..."):
                    verified_pos = []
                    for po_id in unique_pos:
                        result = verify_po_in_coupa(po_id, headers, COUPA_INSTANCE)
                        if result['exists']:
                            verified_pos.append({'po_id': po_id, 'po_data': result['po_data']})

                st.session_state.verified_pos = verified_pos
                st.session_state.verification_complete = True
                st.session_state.create_results = []       # reset batch results
                st.session_state.selected_for_batch = {}   # reset selection
        else:
            if is_debug():
                st.warning("No PO numbers found. Check the debug information above for details.")
            else:
                st.warning("No PO numbers found. Enable Debug Mode to see extraction details.")

    # Verified POs – Invoice Generation editors (no create buttons here)
    if st.session_state.get("verification_complete") and st.session_state.get("verified_pos"):
        st.markdown("---")
        st.subheader("✅ Verified POs – Invoice Generation")

        for po_info in st.session_state.verified_pos:
            po_id = po_info['po_id']
            po_data = po_info['po_data']

            # Document source resolution
            doc_name = None
            doc_bytes = None
            if hasattr(st.session_state, 'file_data_map') and st.session_state.file_data_map:
                if len(st.session_state.file_data_map) == 1:
                    doc_name = next(iter(st.session_state.file_data_map.keys()))
                    doc_bytes = st.session_state.file_data_map[doc_name]
            else:
                if st.session_state.get('original_file_data'):
                    doc_bytes = st.session_state.original_file_data
                    doc_name = st.session_state.get('original_file_name', 'invoice.pdf')

            # store per-PO context
            st.session_state.po_context.setdefault(po_id, {
                "doc_bytes": doc_bytes,
                "doc_name": doc_name,
                "invoice_number": "",
                "invoice_date": datetime.today().date(),
                "tax_rate": 10.0  # default GST-A 10%
            })

            # Default batch selection (True) set once
            st.session_state.selected_for_batch.setdefault(po_id, True)

            with st.expander(f"📄 {po_id}", expanded=True):
                top_cols = st.columns([1.2, 1.2, 1.2, 2.4])
                with top_cols[0]:
                    st.session_state.po_context[po_id]["invoice_number"] = st.text_input(
                        "Invoice Number", key=f"inv_num_{po_id}", placeholder="Enter invoice number",
                        value=st.session_state.po_context[po_id]["invoice_number"]
                    )
                with top_cols[1]:
                    st.session_state.po_context[po_id]["invoice_date"] = st.date_input(
                        "Invoice Date", key=f"inv_date_{po_id}",
                        value=st.session_state.po_context[po_id]["invoice_date"],
                        format="DD/MM/YYYY"
                    )
                with top_cols[2]:
                    st.session_state.po_context[po_id]["tax_rate"] = st.number_input(
                        "Header Tax Rate (%)", key=f"tax_rate_{po_id}",
                        value=float(st.session_state.po_context[po_id]["tax_rate"]),
                        step=0.01
                    )
                with top_cols[3]:
                    st.write("**PO Info:**")
                    st.write(f"PO Number: {po_data.get('number') or po_id}")
                    st.write(f"Status: {po_data.get('status', 'Unknown')}")
                    supplier_name = (po_data.get('supplier') or {}).get('name', 'Unknown')
                    st.write(f"Supplier: {supplier_name}")

                # Header Tax subtitle under heading
                st.caption("Header tax is applied at the invoice header (subtotal × rate).")

                # Build preview df once
                if po_id not in st.session_state.invoice_lines_preview:
                    st.session_state.invoice_lines_preview[po_id] = build_preview_df(po_id, po_data)

                preview_df = st.session_state.invoice_lines_preview[po_id]

                # Determine default account id to inherit for new lines
                default_acct = _default_account_id_from(preview_df)

                # Add line buttons
                add_cols = st.columns([1, 1, 4])
                with add_cols[0]:
                    if st.button("➕ Add Quantity Line", key=f"add_qty_{po_id}"):
                        new_row = {
                            "line_num": len(preview_df) + 1,
                            "inv_type": "InvoiceQuantityLine",
                            "description": "",
                            "price": 0.00,
                            "quantity": 1.00,
                            "uom_code": DEFAULT_UOM_CODE,
                            "account_id": default_acct,        # inherit
                            "commodity_name": None,
                            "po_number": po_id,                 # default per line (editable)
                            "order_header_num": int(_numeric_po_id(po_id)),
                            "order_line_id": None,
                            "order_line_num": str(len(preview_df) + 1),
                            "order_header_id": None,
                            "source_part_num": None,
                            "delete": False
                        }
                        preview_df = pd.concat([preview_df, pd.DataFrame([new_row])], ignore_index=True)
                        preview_df = resequence_lines(preview_df)
                        st.session_state.invoice_lines_preview[po_id] = preview_df
                with add_cols[1]:
                    if st.button("➕ Add Amount Line", key=f"add_amt_{po_id}"):
                        new_row = {
                            "line_num": len(preview_df) + 1,
                            "inv_type": "InvoiceAmountLine",
                            "description": "",
                            "price": 0.00,
                            "quantity": 0.00,
                            "uom_code": None,
                            "account_id": default_acct,        # inherit
                            "commodity_name": None,
                            "po_number": po_id,                 # default per line (editable)
                            "order_header_num": int(_numeric_po_id(po_id)),
                            "order_line_id": None,
                            "order_line_num": str(len(preview_df) + 1),
                            "order_header_id": None,
                            "source_part_num": None,
                            "delete": False
                        }
                        preview_df = pd.concat([preview_df, pd.DataFrame([new_row])], ignore_index=True)
                        preview_df = resequence_lines(preview_df)
                        st.session_state.invoice_lines_preview[po_id] = preview_df

                # Editor config (labels: "PO #" and "PO Line #")
                col_cfg = {
                    "line_num": st.column_config.NumberColumn("Line #", disabled=True, width="small"),
                    "inv_type": st.column_config.SelectboxColumn("Type", options=["InvoiceQuantityLine", "InvoiceAmountLine"], required=True),
                    "description": st.column_config.TextColumn("Description", required=True, width="medium"),
                    "source_part_num": st.column_config.TextColumn("Supplier Part #"),
                    "uom_code": st.column_config.TextColumn("UOM"),
                    "commodity_name": st.column_config.TextColumn("Commodity"),
                    "po_number": st.column_config.TextColumn("PO #"),
                    "order_line_num": st.column_config.TextColumn("PO Line #"),
                    "quantity": st.column_config.NumberColumn("Quantity", min_value=0.0, step=0.01, format="%.2f"),
                    "price": st.column_config.NumberColumn("Price", min_value=0.0, step=0.01, format="%.2f"),
                    "delete": st.column_config.CheckboxColumn("Delete")
                }
                editor_order = [c for c in EDITOR_COL_ORDER if c in preview_df.columns]

                edited_df = st.data_editor(
                    preview_df,
                    column_config=col_cfg,
                    column_order=editor_order,
                    hide_index=True,
                    use_container_width=True,
                    key=f"editor_{po_id}"
                )

                # Sanitize and store back
                edited_df = sanitize_editor_rows(edited_df)
                st.session_state.invoice_lines_preview[po_id] = edited_df

                # Totals row – rename to Net Total
                subtotal = float(edited_df["price"].fillna(0).astype(float).mul(edited_df["quantity"].fillna(0).astype(float)).sum())
                est_tax = round(subtotal * (float(st.session_state.po_context.setdefault(po_id, {}).get("tax_rate", 10)) / 100.0), 2)
                net_total = subtotal + est_tax
                mt = st.columns(3)
                with mt[0]: st.metric("Subtotal", f"${subtotal:,.2f}")
                with mt[1]: st.metric("Header Tax", f"${est_tax:,.2f}")
                with mt[2]: st.metric("Net Total", f"${net_total:,.2f}")

    # =====================================================
    # Batch Create section (bottom) – Include selection here
    # =====================================================
    verified_pos = st.session_state.get("verified_pos", [])
    if verified_pos:
        st.markdown("---")
        st.subheader("🚀 Create Invoices (Batch)")

        # Build selection table (with Include checkboxes)
        summary_rows = []
        for po_info in verified_pos:
            po_id = po_info['po_id']
            ctx = st.session_state.po_context.get(po_id, {})
            include = bool(st.session_state.selected_for_batch.get(po_id, True))
            inv_num = ctx.get("invoice_number") or ""
            inv_date_val = ctx.get("invoice_date")
            inv_date_str = inv_date_val.strftime("%d/%m/%Y") if isinstance(inv_date_val, (datetime, date)) else ""
            tax_rate = ctx.get("tax_rate")
            edited_df = st.session_state.invoice_lines_preview.get(po_id, pd.DataFrame())
            subtotal = float(edited_df["price"].fillna(0).astype(float).mul(edited_df["quantity"].fillna(0).astype(float)).sum()) if not edited_df.empty else 0
            est_tax = round(subtotal * (float(tax_rate or 0)/100.0), 2) if include else 0
            summary_rows.append({
                "Include": include,
                "PO": po_id,
                "Invoice #": inv_num,
                "Date (dd/mm/yyyy)": inv_date_str,
                "Lines": 0 if edited_df is None else len(edited_df),
                "Subtotal": f"${subtotal:,.2f}",
                "Header Tax": f"${est_tax:,.2f}"
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_editor = st.data_editor(
            summary_df,
            hide_index=True,
            column_config={
                "Include": st.column_config.CheckboxColumn("Include"),
                "PO": st.column_config.TextColumn("PO"),
                "Invoice #": st.column_config.TextColumn("Invoice #"),
                "Date (dd/mm/yyyy)": st.column_config.TextColumn("Date (dd/mm/yyyy)", disabled=True),
                "Lines": st.column_config.NumberColumn("Lines", disabled=True),
                "Subtotal": st.column_config.TextColumn("Subtotal", disabled=True),
                "Header Tax": st.column_config.TextColumn("Header Tax", disabled=True),
            },
            use_container_width=True,
            key="batch_table"
        )

        # Persist Include selections & count
        includes_true = 0
        for _, row in summary_editor.iterrows():
            inc = bool(row["Include"])
            st.session_state.selected_for_batch[row["PO"]] = inc
            if inc:
                includes_true += 1

        # Production acknowledgement
        prod_ok = True
        if st.session_state.get("environment") == "Production":
            prod_ok = st.checkbox("I understand I am creating invoices in **Production** (use with caution).", key="prod_ack", value=False)

        # Only show the Create button if there is at least 1 included row AND prod is acknowledged (if applicable)
        if includes_true == 0:
            st.info("Select at least one row to include before creating invoices.")
        elif st.session_state.get("environment") == "Production" and not prod_ok:
            st.warning("Please acknowledge Production use before creating invoices.")
        else:
            if st.button("Create Invoices", type="primary", help="Creates all checked invoices and uploads scans"):
                st.session_state.create_results = []
                with st.spinner("Creating invoices..."):
                    for po_info in verified_pos:
                        po_id = po_info['po_id']
                        po_data = po_info['po_data']
                        include = bool(st.session_state.selected_for_batch.get(po_id, True))
                        ctx = st.session_state.po_context.get(po_id, {})
                        if not include:
                            st.session_state.create_results.append({
                                "PO": po_id, "Invoice #": ctx.get("invoice_number"),
                                "Invoice ID": "", "Invoice Status": "Skipped",
                                "Scan Status": "Skipped", "Link": ""
                            })
                            continue

                        inv_num = ctx.get("invoice_number")
                        inv_date_val = ctx.get("invoice_date")
                        if not inv_num or not inv_date_val:
                            st.session_state.create_results.append({
                                "PO": po_id, "Invoice #": inv_num or "",
                                "Invoice ID": "", "Invoice Status": "Failed (missing number/date)",
                                "Scan Status": "N/A", "Link": ""
                            })
                            continue

                        edited_df = st.session_state.invoice_lines_preview.get(po_id)
                        doc_bytes = ctx.get("doc_bytes")
                        upload_filename = ctx.get("doc_name") or "invoice.pdf"

                        if edited_df is None or edited_df.empty:
                            st.session_state.create_results.append({
                                "PO": po_id, "Invoice #": inv_num,
                                "Invoice ID": "", "Invoice Status": "Failed (no lines)",
                                "Scan Status": "N/A", "Link": ""
                            })
                            continue
                        if not doc_bytes:
                            st.session_state.create_results.append({
                                "PO": po_id, "Invoice #": inv_num,
                                "Invoice ID": "", "Invoice Status": "Failed (no file)",
                                "Scan Status": "N/A", "Link": ""
                            })
                            continue

                        inv_ok, inv_resp, scan_ok, scan_resp = create_invoice_from_po(
                            context_po_id=po_id,
                            invoice_number=inv_num,
                            invoice_date=inv_date_val,
                            original_file_data=doc_bytes,
                            upload_filename=upload_filename,
                            po_data=po_data,
                            edited_df=edited_df,
                            header_tax_rate=float(ctx.get("tax_rate") or 0)
                        )

                        if inv_ok and isinstance(inv_resp, dict):
                            invoice_id = inv_resp.get("id")
                            env_suffix = "_PROD" if st.session_state.get("environment", "Test") == "Production" else ""
                            coupa_instance = os.environ.get(f"COUPA_INSTANCE{env_suffix}")
                            link = f"https://{coupa_instance}.coupahost.com/invoices/{invoice_id}" if invoice_id else ""
                            st.session_state.create_results.append({
                                "PO": po_id, "Invoice #": inv_num,
                                "Invoice ID": invoice_id or "",
                                "Invoice Status": "Success",
                                "Scan Status": ("Success" if scan_ok else f"Failed ({scan_resp})"),
                                "Link": link
                            })
                        else:
                            st.session_state.create_results.append({
                                "PO": po_id, "Invoice #": inv_num,
                                "Invoice ID": "",
                                "Invoice Status": f"Failed ({inv_resp})",
                                "Scan Status": "N/A",
                                "Link": ""
                            })

        # Single, pretty Results with big metrics and one table
        if st.session_state.create_results:
            st.markdown("---")
            st.subheader("📊 Results")

            def emoji_status(s: str) -> str:
                s = str(s or "").lower()
                if s.startswith("success"):
                    return "✅ Success"
                if s.startswith("failed"):
                    return "❌ Failed"
                if s.startswith("skipped"):
                    return "⭐️ Skipped"
                return s

            # Build results table + counts
            rows = []
            succ = fail = skip = 0
            for r in st.session_state.create_results:
                inv_stat = emoji_status(r["Invoice Status"])
                scan_stat = emoji_status(r["Scan Status"])
                if inv_stat.startswith("✅"):
                    succ += 1
                elif inv_stat.startswith("❌"):
                    fail += 1
                elif inv_stat.startswith("⭐️"):
                    skip += 1
                rows.append({
                    "PO": r["PO"],
                    "Invoice #": r["Invoice #"],
                    "Invoice ID": r["Invoice ID"],
                    "Invoice Status": inv_stat,
                    "Scan Status": scan_stat,
                    "Link": r["Link"]
                })

            # Big metrics row (like subtotal/metrics)
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("✅ Success", succ)
            with m2: st.metric("❌ Failed", fail)
            with m3: st.metric("⭐️ Skipped", skip)

            # Results table (make link clickable if possible)
            link_col_cfg = {}
            try:
                # Streamlit supports LinkColumn in newer versions
                link_col_cfg["Link"] = st.column_config.LinkColumn("Open", display_text="Open")
            except Exception:
                pass

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                column_config=link_col_cfg
            )

            # Buttons to open successful invoices (in case LinkColumn not available)
            success_links = [r for r in st.session_state.create_results if r.get("Link")]
            if success_links:
                st.markdown("**Open Invoices:**")
                for r in success_links:
                    if r["Link"]:
                        st.link_button(f"Open {r['Invoice ID']}", r["Link"])

if __name__ == "__main__":
    main()
