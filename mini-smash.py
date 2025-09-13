# -*- coding: utf-8 -*-
import os
import re
import json
import zipfile
import tempfile
import mimetypes
import warnings
import time
import hashlib
from datetime import datetime, date
from typing import List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import io

import streamlit as st
import pandas as pd
import requests
import numpy as np
from PIL import Image
from io import BytesIO

# Cloud-friendly optional imports with fallbacks
try:
    import camelot
    HAS_CAMELOT = True
except ImportError:
    HAS_CAMELOT = False
    if st.session_state.get('show_import_warnings', True):
        st.sidebar.warning("⚠️ Camelot not available - PDF table extraction disabled")

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False
    if st.session_state.get('show_import_warnings', True):
        st.sidebar.warning("⚠️ EasyOCR not available - Image OCR disabled")

# Optional import for tags functionality
try:
    import streamlit_tags as st_tags
except ImportError:
    st_tags = None

# Optional imports - will be handled gracefully if missing
try:
    import torch
except ImportError:
    torch = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# =========================================================
# Streamlit configuration
# =========================================================
st.set_page_config(page_title="miniSmash Invoice Creator", page_icon="📄", layout="wide")

# =========================================================
# Constants & defaults
# =========================================================
DEFAULT_UOM_CODE = "EA"           # fallback for quantity lines
PRICE_DECIMALS = 2
QTY_DECIMALS = 2

# Default invoice patterns for detection
DEFAULT_INVOICE_PATTERNS = [
    'INVOICE NUMBER:', 'INVOICE NO:', 'INVOICE#', 'INVOICE #:', 
    'INV:', 'INVOICE:', 'TAX INVOICE NO.'
]

# OAuth scopes
OAUTH_SCOPE_PO_READ = "core.purchase_order.read"
OAUTH_SCOPE_INVOICE_WRITE_READ = "core.invoice.write core.invoice.read"
OAUTH_SCOPE_ALL = "core.purchase_order.read core.invoice.write core.invoice.read"  # Combined scope

# Internal storage columns (canonical schema we keep in session)
BASE_COLS = [
    "line_num", "inv_type", "description", "price", "quantity", "uom_code",
    "account_id", "commodity_name", "currency_code",
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
    "quantity",          # --> Quantity back near beginning
    "price",             # --> Price back near beginning
    "uom_code",
    "source_part_num",   # Supplier Part #
    "commodity_name",
    "po_number",         # --> PO # 
    "order_line_num",    # --> then PO Line #
    "delete"
]

# =========================================================
# Small utilities
# =========================================================
def _po_prefix() -> List[str]:
    """Return list of PO prefixes to search for"""
    prefix_input = (st.session_state.get("po_prefix") or "PO").strip()
    # Split by comma and clean up each prefix
    prefixes = [p.strip().upper() for p in prefix_input.split(',') if p.strip()]
    return prefixes if prefixes else ["PO"]

def _to_decimal_str(value, precision=2) -> str:
    try:
        return f"{float(value):.{precision}f}"
    except Exception:
        return f"{0:.{precision}f}"

def _numeric_from_po_string(po_str: str) -> int:
    """
    Derive the numeric PO number from any string like 'PO000123'.
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
    """Used for GET /purchase_orders/:id – strips any configured prefix and zeros."""
    prefixes = _po_prefix()
    for prefix in prefixes:
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
                # Limit large responses to prevent browser crashes
                json_str = json.dumps(data, indent=2)
                if len(json_str) > 10000:  # 10KB limit
                    st.write(f"**Response too large ({len(json_str)} chars), showing first 5000 characters:**")
                    st.code(json_str[:5000] + "\n... (truncated)")
                else:
                    st.code(json_str)
        else:
            if isinstance(data, (dict, list)):
                # Limit large responses to prevent browser crashes
                json_str = json.dumps(data, indent=2)
                if len(json_str) > 10000:  # 10KB limit
                    st.write(f"**Response too large ({len(json_str)} chars), showing essential info only:**")
                    if isinstance(data, dict):
                        essential = {k: v for k, v in data.items() if k in ["id", "status", "invoice-number", "total", "errors"]}
                        st.code(json.dumps(essential, indent=2))
                        st.write(f"*Full response has {len(data)} fields*")
                    else:
                        st.code(json_str[:5000] + "\n... (truncated)")
                else:
                    st.code(json_str)
            else:
                # Limit text responses too
                text_str = str(data)
                if len(text_str) > 10000:
                    st.write(f"**Response too large ({len(text_str)} chars), truncated:**")
                    st.code(text_str[:5000] + "\n... (truncated)")
                else:
                    st.code(text_str)

def _log_ocr_debug(image_info: str, po_numbers: List[str], gpu_info: dict = None, invoice_numbers: List[str] = None, extracted_dates: List[date] = None):
    """Log OCR debug information when debug mode is enabled"""
    if not is_debug():
        return
    
    with st.expander(f"🔍 OCR Debug – {image_info}", expanded=False):
        # GPU/Processing Info
        if gpu_info:
            st.write("**Processing Setup:**")
            if gpu_info.get('torch_available'):
                st.write(f"**PyTorch**: Available")
                st.write(f"**GPU**: {'Available' if gpu_info.get('gpu_available') else 'Not Available'}")
                st.write(f"**Using**: {'GPU' if gpu_info.get('gpu_available') else 'CPU'}")
            else:
                st.write(f"**PyTorch**: Not Available")
                st.write(f"**GPU**: Not Available")
                st.write(f"**Using**: CPU")
            st.write("")
            
        st.write("**PO Number Detection:**")
        prefixes = _po_prefix()
        patterns = [r'\b' + re.escape(prefix) + r'\d{6}\b' for prefix in prefixes]
        st.code(f"Prefixes: {', '.join(prefixes)}")
        st.code(f"Patterns used: {patterns}")
        if po_numbers:
            st.success(f"Found PO numbers: {', '.join(po_numbers)}")
        else:
            st.warning("No PO numbers found")
            
        # Invoice Number Detection
        if st.session_state.get('auto_invoice_detect', True):
            st.write("**Invoice Number Detection:**")
            invoice_patterns = st.session_state.get('invoice_patterns', DEFAULT_INVOICE_PATTERNS)
            st.code(f"Patterns: {', '.join(invoice_patterns)}")
            if invoice_numbers:
                st.success(f"Found invoice numbers: {', '.join(invoice_numbers)}")
                if len(invoice_numbers) > 1:
                    # Show smart selection logic
                    best_invoice = max(invoice_numbers, key=len)
                    st.info(f"Smart selection (longest): {best_invoice}")
                    st.info(f"All detected: {', '.join(invoice_numbers)}")
                else:
                    st.info(f"Auto-populated invoice number: {invoice_numbers[0]}")
            else:
                st.warning("No invoice numbers found")
        
        # Date Detection
        if st.session_state.get('auto_date_detect', True):
            st.write("**Date Detection:**")
            if extracted_dates:
                date_strs = [d.strftime("%d/%m/%Y") for d in extracted_dates]
                st.success(f"Found dates: {', '.join(date_strs)}")
                most_recent = max(extracted_dates)
                st.info(f"Auto-selected date: {most_recent.strftime('%d/%m/%Y')}")
            else:
                st.warning("No dates found")

# =========================================================
# Helper functions
# =========================================================
def get_env_config():
    """Get environment configuration based on current environment setting."""
    current_env = st.session_state.get("environment", "Test")
    env_suffix = "_PROD" if current_env == "Production" else ""
    
    return {
        "env_suffix": env_suffix,
        "instance": os.environ.get(f"COUPA_INSTANCE{env_suffix}"),
        "identifier": os.environ.get(f"PO_IDENTIFIER{env_suffix}"),
        "secret": os.environ.get(f"PO_SECRET{env_suffix}")
    }

def get_oauth_token(scope: str):
    """Get OAuth token for given scope with caching. Returns (success, token_or_error)."""
    import time
    
    # Initialize token cache if not exists
    if 'oauth_token_cache' not in st.session_state:
        st.session_state.oauth_token_cache = {}
    
    # Check if we have a valid cached token for this scope
    cache_key = scope
    cached_token = st.session_state.oauth_token_cache.get(cache_key)
    
    if is_debug():
        debug_expander = st.expander("🔑 Debug: Authentication", expanded=False)
    
    if cached_token:
        token, expires_at = cached_token
        # Check if token is still valid (with 30 second buffer)
        if time.time() < (expires_at - 30):
            if is_debug():
                with debug_expander:
                    st.write("**Token Status**: ✅ Using cached token")
                    st.write(f"**Scope**: {scope}")
                    remaining_time = int(expires_at - time.time())
                    st.write(f"**Expires in**: {remaining_time} seconds")
            return True, token
        else:
            # Token expired, remove from cache
            if is_debug():
                with debug_expander:
                    st.write("**Token Status**: ⏰ Token expired, requesting new token")
                    st.write(f"**Scope**: {scope}")
            del st.session_state.oauth_token_cache[cache_key]
    
    # No valid cached token, request new one
    config = get_env_config()
    
    if not all([config["instance"], config["identifier"], config["secret"]]):
        return False, "Missing required environment variables"
    
    token_url = f"https://{config['instance']}.coupahost.com/oauth2/token"
    token_data = {"grant_type": "client_credentials", "scope": scope}
    
    try:
        # Use existing debug_expander or create one if it doesn't exist
        if is_debug():
            if 'debug_expander' not in locals():
                debug_expander = st.expander("🔑 Debug: Authentication", expanded=False)
            
            with debug_expander:
                # Create standardized Request/Response tabs
                req_tab, resp_tab = st.tabs(["📡 API Request", "📨 API Response"])
                
                with req_tab:
                    st.write("**🔑 OAuth Authentication Request:**")
                    request_data = {
                        "Field": ["URL", "Method", "Purpose", "Scope", "Grant Type", "Instance", "Auth Type"],
                        "Value": [
                            token_url,
                            "POST",
                            "Get OAuth access token",
                            scope,
                            "client_credentials",
                            config['instance'],
                            "Basic Authentication"
                        ]
                    }
                    st.dataframe(pd.DataFrame(request_data), use_container_width=True, hide_index=True)
                    
                    st.write("**📋 Request Headers:**")
                    headers_data = {
                        "Header": ["Content-Type", "Authorization"],
                        "Value": ["application/x-www-form-urlencoded", "Basic (Client ID:Secret)"]
                    }
                    st.dataframe(pd.DataFrame(headers_data), use_container_width=True, hide_index=True)
        
        response = requests.post(
            token_url,
            auth=(config["identifier"], config["secret"]),
            data=token_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30
        )
        response.raise_for_status()
        
        token_data = response.json()
        access_token = token_data.get("access_token")
        expires_in = token_data.get("expires_in", 3600)  # Default 1 hour
        
        # Cache the token with expiry time
        expires_at = time.time() + expires_in
        st.session_state.oauth_token_cache[cache_key] = (access_token, expires_at)
        
        if is_debug():
            with debug_expander:
                with resp_tab:
                    st.write("**🔑 OAuth Authentication Response:**")
                    response_data = {
                        "Field": ["HTTP Status", "Content Type", "Response Time", "Cache Status"],
                        "Value": [
                            f"{response.status_code} ({'Success' if response.status_code == 200 else 'Error'})",
                            response.headers.get('Content-Type', 'Unknown'),
                            "< 1 second",
                            "Token cached for future use" if response.status_code == 200 else "No caching"
                        ]
                    }
                    st.dataframe(pd.DataFrame(response_data), use_container_width=True, hide_index=True)

                    st.write("**📋 Response Headers:**")
                    headers_data = []
                    for key, value in response.headers.items():
                        headers_data.append({"Header": key, "Value": value})
                    st.dataframe(pd.DataFrame(headers_data), use_container_width=True, hide_index=True)

                    if response.status_code == 200:
                        token_data = response.json()
                        st.write("**📄 Response Body (Sanitized):**")
                        # Show sanitized response
                        sanitized_response = {
                            "token_type": token_data.get('token_type', 'Bearer'),
                            "expires_in": token_data.get('expires_in', 3600),
                            "scope": token_data.get('scope', 'N/A'),
                            "access_token": f"{token_data.get('access_token', '')[:20]}...{token_data.get('access_token', '')[-10:]}" if token_data.get('access_token') else 'N/A'
                        }
                        st.code(json.dumps(sanitized_response, indent=2))

                        st.write("**✅ Authentication Result:**")
                        result_data = {
                            "Metric": ["Status", "Token Obtained", "Cache Duration", "Expires At"],
                            "Value": [
                                "✅ Success",
                                "✅ Yes",
                                f"{token_data.get('expires_in', 3600)} seconds",
                                time.strftime('%H:%M:%S', time.localtime(expires_at))
                            ]
                        }
                        st.dataframe(pd.DataFrame(result_data), use_container_width=True, hide_index=True)
                    else:
                        st.write("**❌ Error Response:**")
                        error_summary = {
                            "Field": ["Status", "Result", "Issue"],
                            "Value": [
                                f"HTTP {response.status_code}",
                                "❌ Authentication Failed",
                                "Check credentials and network"
                            ]
                        }
                        st.dataframe(pd.DataFrame(error_summary), use_container_width=True, hide_index=True)

                        try:
                            error_data = response.json()
                            st.write("**📄 Error Details:**")
                            st.code(json.dumps(error_data, indent=2))
                        except:
                            st.write("**📄 Raw Error Response:**")
                            st.code(response.text[:1000] + "..." if len(response.text) > 1000 else response.text)
        
        return True, access_token
    except Exception as e:
        if is_debug() and 'debug_expander' in locals():
            with debug_expander:
                with resp_tab:
                    st.write("**🔑 OAuth Authentication Response:**")
                    error_data = {
                        "Field": ["Status", "Result", "Error Type", "Error Message"],
                        "Value": [
                            "Exception Occurred",
                            "❌ Authentication Failed",
                            "Network/Request Error",
                            str(e)
                        ]
                    }
                    st.dataframe(pd.DataFrame(error_data), use_container_width=True, hide_index=True)
        return False, str(e)

def clear_oauth_token_cache():
    """Clear all cached OAuth tokens. Useful when switching environments or on auth errors."""
    if 'oauth_token_cache' in st.session_state:
        st.session_state.oauth_token_cache.clear()

def get_invoice_status(invoice_id: str) -> str:
    """Get current status of an invoice from Coupa API. Returns user-friendly status."""
    if not invoice_id:
        return "Unknown"
    
    try:
        success, token_or_error = get_oauth_token(OAUTH_SCOPE_ALL)
        if not success:
            return "API Error"
        
        config = get_env_config()
        if not config["instance"]:
            return "Config Error"
        
        headers = {"Authorization": f"Bearer {token_or_error}", "Accept": "application/json"}
        url = f"https://{config['instance']}.coupahost.com/api/invoices/{invoice_id}"
        
        if is_debug():
            debug_expander = st.expander(f"🔍 Debug: Invoice Status Check - {invoice_id}", expanded=False)
            with debug_expander:
                # Create standardized Request/Response tabs
                req_tab, resp_tab = st.tabs(["📡 API Request", "📨 API Response"])

                with req_tab:
                    st.write("**📊 Invoice Status API Request:**")
                    request_data = {
                        "Field": ["URL", "Method", "Invoice ID", "Purpose", "Timeout"],
                        "Value": [
                            url,
                            "GET",
                            invoice_id,
                            "Retrieve current invoice status",
                            "10 seconds"
                        ]
                    }
                    st.dataframe(pd.DataFrame(request_data), use_container_width=True, hide_index=True)

                    st.write("**📋 Request Headers:**")
                    headers_data = []
                    redacted_headers = _redact_headers(headers)
                    for key, value in redacted_headers.items():
                        headers_data.append({"Header": key, "Value": str(value)})
                    st.dataframe(pd.DataFrame(headers_data), use_container_width=True, hide_index=True)

        response = requests.get(url, headers=headers, timeout=10)

        if is_debug():
            with debug_expander:
                with resp_tab:
                    st.write("**📊 Invoice Status API Response:**")
                    response_summary = {
                        "Field": ["HTTP Status", "Content Type", "Response Size", "Status Result"],
                        "Value": [
                            f"{response.status_code} ({'Success' if response.status_code == 200 else 'Error'})",
                            response.headers.get('Content-Type', 'Unknown'),
                            f"{len(response.content)} bytes",
                            "✅ Status retrieved" if response.status_code == 200 else "❌ Status unavailable"
                        ]
                    }
                    st.dataframe(pd.DataFrame(response_summary), use_container_width=True, hide_index=True)

                    st.write("**📋 Response Headers:**")
                    headers_data = []
                    for key, value in response.headers.items():
                        headers_data.append({"Header": key, "Value": str(value)})
                    st.dataframe(pd.DataFrame(headers_data), use_container_width=True, hide_index=True)

                    if response.status_code == 200:
                        try:
                            data = response.json()
                            status = data.get("status", "Unknown")

                            st.write("**✅ Status Check Result:**")
                            result_data = {
                                "Metric": ["Result", "Invoice Status", "Response Fields", "Data Quality"],
                                "Value": [
                                    "✅ Status retrieved successfully",
                                    status,
                                    f"{len(data)} fields returned",
                                    "✅ Complete" if status != "Unknown" else "⚠️ Incomplete"
                                ]
                            }
                            st.dataframe(pd.DataFrame(result_data), use_container_width=True, hide_index=True)

                            st.write("**📄 Response Body (Essential Fields):**")
                            # Show only essential fields for status check
                            essential = {k: v for k, v in data.items() if k in ["id", "status", "invoice-number", "supplier", "total", "created-at", "updated-at"]}
                            st.code(json.dumps(essential, indent=2))
                            if len(data) > len(essential):
                                st.info(f"Full response contains {len(data)} fields - showing {len(essential)} essential fields only")
                        except Exception as e:
                            st.write("**❌ Response Parsing Error:**")
                            error_data = {
                                "Issue": ["JSON Parse Failed", "Raw Response Length", "Error Message"],
                                "Details": [
                                    "Could not parse response as JSON",
                                    f"{len(response.text)} characters",
                                    str(e)
                                ]
                            }
                            st.dataframe(pd.DataFrame(error_data), use_container_width=True, hide_index=True)
                            st.code(response.text[:500] + "..." if len(response.text) > 500 else response.text)
                    else:
                        st.write("**❌ Status Check Error:**")
                        error_summary = {
                            "Field": ["Status", "Result", "Issue"],
                            "Value": [
                                f"HTTP {response.status_code}",
                                "❌ Invoice not found or inaccessible",
                                "Check invoice ID and permissions"
                            ]
                        }
                        st.dataframe(pd.DataFrame(error_summary), use_container_width=True, hide_index=True)

                        st.write("**📄 Error Response:**")
                        st.code(response.text[:500] + "..." if len(response.text) > 500 else response.text)
        
        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "Unknown")
            
            # Map Coupa status to user-friendly status
            status_map = {
                "pending": "Draft",
                "Pending Approval": "Pending Approval",
                "approved": "Approved", 
                "paid": "Paid",
                "on_hold": "On Hold",
                "rejected": "Rejected",
                "pending_receipt": "Pending Receipt",
                "exported": "Exported"
            }
            
            return status_map.get(status.lower(), status.title().replace('_', ' '))
        else:
            return "Not Found"
    except Exception:
        return "API Error"

# =========================================================
# OCR (enabled with EasyOCR) - Cached Reader for Performance
# =========================================================
@st.cache_resource
def _get_ocr_reader():
    """
    Get cached EasyOCR reader for better performance.
    Only initializes once per session. Cloud-friendly with fallback.
    """
    if not HAS_EASYOCR:
        if is_debug():
            st.warning("EasyOCR not available - OCR functionality disabled")
        return None, False
        
    try:
        # Try GPU first, fall back to CPU if not available
        if torch is not None:
            try:
                gpu_available = torch.cuda.is_available()
            except Exception:
                gpu_available = False
        else:
            gpu_available = False
        
        # Initialize EasyOCR reader with adaptive GPU setting and suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = easyocr.Reader(['en'], gpu=gpu_available, verbose=False)
        
        return reader, gpu_available
    except Exception as e:
        if is_debug():
            st.error(f"❌ OCR Reader initialization failed: {str(e)}")
        return None, False

@st.cache_data(ttl=3600, show_spinner=False)  # Cache OCR results for 1 hour, hide spinner
def _cached_ocr_text_from_image_bytes(img_bytes_hash: str, img_bytes: bytes, filename: str = "image") -> str:
    """
    Cached OCR text extraction to avoid re-processing identical images.
    Uses hash of image bytes as cache key for content-based caching.
    Cloud-friendly with fallback.
    """
    if not HAS_EASYOCR:
        return ""  # Return empty string if OCR not available
        
    try:
        # Get cached OCR reader
        reader, gpu_available = _get_ocr_reader()
        
        if reader is None:
            return ""  # Return empty string if reader initialization failed
        
        # Convert bytes to PIL Image
        image = Image.open(BytesIO(img_bytes))
        
        # Convert PIL Image to numpy array for EasyOCR
        image_array = np.array(image)
        
        # Perform OCR
        results = reader.readtext(image_array)
        
        # Extract text from results
        extracted_text = ' '.join([result[1] for result in results])
        
        return extracted_text
        
    except Exception as e:
        if is_debug():
            st.error(f"❌ Cached OCR Error for {filename}: {str(e)}")
        return ""

def _ocr_text_from_image_bytes(img_bytes: bytes, filename: str = "image") -> str:
    """
    Extract text from image bytes using cached OCR for better performance.
    """
    try:
        # Create hash of image bytes for content-based caching
        img_hash = hashlib.md5(img_bytes).hexdigest()
        
        # Use cached OCR extraction
        extracted_text = _cached_ocr_text_from_image_bytes(img_hash, img_bytes, filename)
        
        if not extracted_text:
            return ""
        
        # Get GPU info for debug logging
        _, gpu_available = _get_ocr_reader()
        
        # Find PO numbers for debug logging
        po_numbers = _extract_po_numbers_from_text(extracted_text)
        
        # Extract invoice numbers and store in session state for auto-population
        invoice_numbers = _extract_invoice_numbers_from_text(extracted_text)
        if invoice_numbers and st.session_state.get('auto_invoice_detect', True):
            # Store all detected invoice numbers
            st.session_state.detected_invoice_numbers = invoice_numbers
            # Smart selection: prefer longer numbers over single digits
            best_invoice = max(invoice_numbers, key=len) if len(invoice_numbers) > 1 else invoice_numbers[0]
            st.session_state.detected_invoice_number = best_invoice
        
        # Extract dates and store in session state for auto-population
        extracted_dates = extract_dates_from_text(extracted_text)
        if extracted_dates and st.session_state.get('auto_date_detect', True):
            st.session_state.detected_dates = extracted_dates
            st.session_state.detected_invoice_date = extracted_dates[0]  # Use the priority date (invoice date label preferred)
        
        if is_debug():
            gpu_info = {
                'torch_available': 'torch' in locals(),
                'gpu_available': gpu_available,
                'using': 'GPU' if gpu_available else 'CPU'
            }
            _log_ocr_debug(filename, po_numbers, gpu_info=gpu_info, invoice_numbers=invoice_numbers, extracted_dates=extracted_dates)
            
        return extracted_text
        
    except Exception as e:
        if is_debug():
            st.error(f"❌ OCR Error for {filename}: {str(e)}")
            _, gpu_available = _get_ocr_reader()
            gpu_info = {
                'torch_available': 'torch' in locals(),
                'gpu_available': gpu_available,
                'using': 'GPU' if gpu_available else 'CPU'
            }
            _log_ocr_debug(filename, [], gpu_info=gpu_info)
        return ""

def _extract_po_numbers_from_text(text: str) -> List[str]:
    """Find PO matches based on configured prefixes + 6 digits with early stopping optimization."""
    if not text:
        return []
        
    prefixes = _po_prefix()
    all_matches = []
    max_matches_per_prefix = 10  # Early stopping - rarely need more than 10 POs per prefix
    
    for prefix in prefixes:
        # Make pattern more flexible - allow for spaces or other characters between prefix and numbers
        po_pattern = r'\b' + re.escape(prefix) + r'\s*\d{6}\b'
        matches = re.findall(po_pattern, text, re.IGNORECASE)
        
        # Clean up matches by removing any spaces and apply early stopping
        cleaned_matches = []
        for match in matches[:max_matches_per_prefix]:  # Early stopping per prefix
            cleaned_match = re.sub(r'\s+', '', match.upper())
            cleaned_matches.append(cleaned_match)
            
        all_matches.extend(cleaned_matches)
    
    # Remove duplicates and apply overall limit for OCR performance
    unique_matches = list(set(all_matches))
    return unique_matches[:50]  # Early stopping - limit total PO numbers for performance

def _extract_invoice_numbers_from_text(text: str) -> List[str]:
    """Extract invoice numbers from text using configured patterns with early stopping."""
    if not st.session_state.get('auto_invoice_detect', True):
        return []
        
    invoice_numbers = []
    patterns = st.session_state.get('invoice_patterns', DEFAULT_INVOICE_PATTERNS)
    max_invoices = 5  # Early stopping - OCR images typically have 1-2 invoice numbers
    
    # First try standard same-line patterns
    for pattern in patterns:
        if len(invoice_numbers) >= max_invoices:  # Early stopping across patterns
            break
            
        # Clean the pattern and create regex
        clean_pattern = pattern.strip()
        
        # Create regex pattern that matches the full pattern followed by the invoice number
        # Allow up to 20 characters for longer invoice numbers like 11100033246
        if clean_pattern.endswith(':'):
            regex_pattern = rf'{re.escape(clean_pattern)}\s*([A-Za-z0-9\-_]{{1,20}})'
        elif clean_pattern.endswith('#'):
            regex_pattern = rf'{re.escape(clean_pattern)}\s*([A-Za-z0-9\-_]{{1,20}})'
        else:
            # For patterns without delimiters, add optional colon or hash
            regex_pattern = rf'{re.escape(clean_pattern)}[#:\s]*([A-Za-z0-9\-_]{{1,20}})'
        
        matches = re.findall(regex_pattern, text, re.IGNORECASE)
        
        # Clean up matches - stop at whitespace, tab, newline, or other delimiters with early stopping
        for match in matches[:3]:  # Limit matches per pattern for performance
            # Split on common delimiters and take the first part
            clean_match = re.split(r'[\s\t\n\r,;|]', match)[0]
            if clean_match and len(clean_match) <= 20:
                invoice_numbers.append(clean_match)
                if len(invoice_numbers) >= max_invoices:  # Early stopping
                    break
    
    # If no matches found, try multi-line patterns (common in PDFs)
    if not invoice_numbers:
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if len(invoice_numbers) >= max_invoices:
                break
                
            line_lower = line.strip().lower()
            
            # Look for patterns that indicate the next line might contain the invoice number
            for pattern in patterns:
                clean_pattern = pattern.strip().lower().rstrip(':').rstrip('#')
                
                if clean_pattern in line_lower:
                    # Check the next few lines for invoice numbers
                    for j in range(i + 1, min(i + 4, len(lines))):  # Check next 3 lines
                        next_line = lines[j].strip()
                        
                        # Look for typical invoice number patterns
                        invoice_patterns = [
                            r'^([A-Z]{2,}-\d+)$',  # INV-003, BILL-123
                            r'^([A-Z]+\d+)$',      # INV003, BILL123
                            r'^(\d{3,})$',         # 12345
                            r'^([A-Z]+[-_]\d+)$',  # INV_003, BILL_123
                        ]
                        
                        for inv_pattern in invoice_patterns:
                            match = re.match(inv_pattern, next_line, re.IGNORECASE)
                            if match:
                                invoice_num = match.group(1)
                                if len(invoice_num) <= 20:
                                    invoice_numbers.append(invoice_num)
                                    break
                        
                        if invoice_numbers:
                            break
                    
                    if invoice_numbers:
                        break
    
    return list(set(invoice_numbers))  # Remove duplicates

def extract_dates_from_text(text: str) -> List[date]:
    """
    Extract potential dates from text using various common date formats with early stopping.
    Returns a list of unique date objects.
    Prioritizes DD/MM/YYYY format for Australian documents.
    Enhanced with multi-line date detection for patterns like:
    Invoice Date
    12 September 2025
    """
    if not text:
        return []
    
    dates = []
    text_clean = text.strip()
    max_dates = 10  # Early stopping - OCR images typically have 1-3 relevant dates
    
    # Basic date patterns with different separators
    date_patterns = [
        # Standard formats: DD/MM/YYYY, MM/DD/YYYY, YYYY/MM/DD
        r'\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})\b',
        r'\b(\d{4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,2})\b',
        # Two-digit year formats
        r'\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2})\b',
        # Space-separated dates
        r'\b(\d{1,2})\s+(\d{1,2})\s+(\d{4})\b',
        # Compact formats without separators
        r'\b(\d{2})(\d{2})(\d{4})\b',
        r'\b(\d{4})(\d{2})(\d{2})\b',
    ]
    
    # Month name patterns
    month_patterns = [
        # Month DD, YYYY format
        r'\b(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)\s+(\d{1,2}),?\s+(\d{4})\b',
        # DD Month YYYY format  
        r'\b(\d{1,2})\s+(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December),?\s+(\d{4})\b',
    ]
    
    month_map = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
        'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6,
        'jul': 7, 'july': 7, 'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
        'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
    }
    
    # Process numeric date patterns with early stopping
    for pattern in date_patterns:
        if len(dates) >= max_dates:  # Early stopping across patterns
            break
            
        matches = re.finditer(pattern, text_clean, re.IGNORECASE)
        for match in matches:
            if len(dates) >= max_dates:  # Early stopping within pattern
                break
                
            try:
                groups = match.groups()
                if len(groups) == 3:
                    num1, num2, num3 = groups
                    
                    # Handle different year positions
                    if len(num1) == 4:  # YYYY/MM/DD format
                        year, month, day = int(num1), int(num2), int(num3)
                        if 1 <= month <= 12 and 1 <= day <= 31:
                            try:
                                dates.append(date(year, month, day))
                            except ValueError:
                                pass
                    
                    elif len(num3) == 4:  # DD/MM/YYYY or MM/DD/YYYY format
                        year = int(num3)
                        
                        # Try DD/MM/YYYY first (Australian format)
                        try:
                            day, month = int(num1), int(num2)
                            if 1 <= month <= 12 and 1 <= day <= 31:
                                dates.append(date(year, month, day))
                                continue  # Skip MM/DD/YYYY if DD/MM/YYYY worked
                        except ValueError:
                            pass
                        
                        # Try MM/DD/YYYY if DD/MM/YYYY failed
                        try:
                            month, day = int(num1), int(num2)
                            if 1 <= month <= 12 and 1 <= day <= 31:
                                dates.append(date(year, month, day))
                        except ValueError:
                            pass
                    
                    elif len(num3) == 2:  # YY format
                        year = 2000 + int(num3) if int(num3) < 50 else 1900 + int(num3)
                        
                        # Try DD/MM/YY first
                        try:
                            day, month = int(num1), int(num2)
                            if 1 <= month <= 12 and 1 <= day <= 31:
                                dates.append(date(year, month, day))
                                continue
                        except ValueError:
                            pass
                        
                        # Try MM/DD/YY
                        try:
                            month, day = int(num1), int(num2)
                            if 1 <= month <= 12 and 1 <= day <= 31:
                                dates.append(date(year, month, day))
                        except ValueError:
                            pass
                            
            except (ValueError, OverflowError):
                continue
    
    # Process month name patterns with early stopping
    for pattern in month_patterns:
        if len(dates) >= max_dates:  # Early stopping across patterns
            break
            
        matches = re.finditer(pattern, text_clean, re.IGNORECASE)
        for match in matches:
            if len(dates) >= max_dates:  # Early stopping within pattern
                break
                
            try:
                groups = match.groups()
                if len(groups) == 3:
                    # Determine if first group is month name or day
                    if groups[0].isdigit():
                        # DD Month YYYY format
                        day_str, month_name, year_str = groups
                        day = int(day_str)
                        month = month_map.get(month_name.lower())
                        year = int(year_str)
                    else:
                        # Month DD YYYY format
                        month_name, day_str, year_str = groups
                        month = month_map.get(month_name.lower())
                        day = int(day_str)
                        year = int(year_str)
                    
                    if month and 1 <= day <= 31 and 1900 <= year <= 2100:
                        try:
                            dates.append(date(year, month, day))
                        except ValueError:
                            continue
            except (ValueError, OverflowError):
                continue
    
    # Multi-line date detection - look for date labels followed by dates on next lines
    if len(dates) < max_dates:
        lines = text.split('\n')
        date_label_patterns = [
            'invoice date', 'date', 'bill date', 'tax invoice date', 
            'issue date', 'document date', 'payment date', 'due date'
        ]
        
        for i, line in enumerate(lines):
            if len(dates) >= max_dates:
                break
                
            line_lower = line.strip().lower()
            
            # Look for date labels
            for label_pattern in date_label_patterns:
                if label_pattern in line_lower and len(line_lower) <= 20:  # Avoid matching in long text
                    # Check the next few lines for dates
                    for j in range(i + 1, min(i + 4, len(lines))):
                        next_line = lines[j].strip()
                        if not next_line or len(next_line) > 50:  # Skip empty lines or very long lines
                            continue
                            
                        # Try to extract date from this line using existing patterns
                        line_dates = []
                        
                        # Try month name patterns first (more specific)
                        for pattern in month_patterns:
                            matches = re.finditer(pattern, next_line, re.IGNORECASE)
                            for match in matches:
                                try:
                                    groups = match.groups()
                                    if len(groups) == 3:
                                        # Determine if first group is month name or day
                                        if groups[0].isdigit():
                                            # DD Month YYYY format
                                            day_str, month_name, year_str = groups
                                            day = int(day_str)
                                            month = month_map.get(month_name.lower())
                                            year = int(year_str)
                                        else:
                                            # Month DD YYYY format
                                            month_name, day_str, year_str = groups
                                            month = month_map.get(month_name.lower())
                                            day = int(day_str)
                                            year = int(year_str)
                                        
                                        if month and 1 <= day <= 31 and 1900 <= year <= 2100:
                                            try:
                                                line_dates.append(date(year, month, day))
                                            except ValueError:
                                                continue
                                except (ValueError, OverflowError):
                                    continue
                        
                        # If no month names found, try numeric patterns
                        if not line_dates:
                            for pattern in date_patterns:
                                matches = re.finditer(pattern, next_line, re.IGNORECASE)
                                for match in matches:
                                    try:
                                        groups = match.groups()
                                        if len(groups) == 3:
                                            num1, num2, num3 = groups
                                            
                                            # Handle different year positions
                                            if len(num1) == 4:  # YYYY/MM/DD format
                                                year, month, day = int(num1), int(num2), int(num3)
                                                if 1 <= month <= 12 and 1 <= day <= 31:
                                                    try:
                                                        line_dates.append(date(year, month, day))
                                                    except ValueError:
                                                        pass
                                            
                                            elif len(num3) == 4:  # DD/MM/YYYY or MM/DD/YYYY format
                                                year = int(num3)
                                                
                                                # Try DD/MM/YYYY first (Australian format)
                                                try:
                                                    day, month = int(num1), int(num2)
                                                    if 1 <= month <= 12 and 1 <= day <= 31:
                                                        line_dates.append(date(year, month, day))
                                                        continue  # Skip MM/DD/YYYY if DD/MM/YYYY worked
                                                except ValueError:
                                                    pass
                                                
                                                # Try MM/DD/YYYY if DD/MM/YYYY failed
                                                try:
                                                    month, day = int(num1), int(num2)
                                                    if 1 <= month <= 12 and 1 <= day <= 31:
                                                        line_dates.append(date(year, month, day))
                                                except ValueError:
                                                    pass
                                            
                                            elif len(num3) == 2:  # YY format
                                                year = 2000 + int(num3) if int(num3) < 50 else 1900 + int(num3)
                                                
                                                # Try DD/MM/YY first
                                                try:
                                                    day, month = int(num1), int(num2)
                                                    if 1 <= month <= 12 and 1 <= day <= 31:
                                                        line_dates.append(date(year, month, day))
                                                        continue
                                                except ValueError:
                                                    pass
                                                
                                                # Try MM/DD/YY
                                                try:
                                                    month, day = int(num1), int(num2)
                                                    if 1 <= month <= 12 and 1 <= day <= 31:
                                                        line_dates.append(date(year, month, day))
                                                except ValueError:
                                                    pass
                                                    
                                    except (ValueError, OverflowError):
                                        continue
                        
                        # If we found dates on this line after a date label, add them with priority
                        if line_dates:
                            dates.extend(line_dates[:2])  # Limit to 2 dates per line
                            break  # Stop checking further lines for this label
                    
                    if len(dates) >= max_dates:
                        break
    
    # Remove duplicates and sort dates by preference
    unique_dates = list(set(dates))
    unique_dates.sort()
    
    # Prioritize dates found near "invoice date" labels
    if unique_dates:
        # Look for dates that were found near invoice date labels
        priority_dates = []
        regular_dates = []
        
        # Re-scan text to identify dates near invoice date labels
        lines = text.split('\n')
        invoice_date_patterns = ['invoice date', 'tax invoice date', 'bill date', 'document date']
        
        for i, line in enumerate(lines):
            line_lower = line.strip().lower()
            
            # Check if this line contains an invoice date label
            for pattern in invoice_date_patterns:
                if pattern in line_lower and len(line_lower) <= 30:
                    # Check this line and next few lines for dates
                    for j in range(i, min(i + 3, len(lines))):
                        check_line = lines[j].strip()
                        
                        # Find dates in this line
                        for d in unique_dates:
                            date_str = d.strftime('%d/%m/%Y')
                            alt_date_str = d.strftime('%d %B %Y')
                            alt_date_str2 = d.strftime('%B %d, %Y')
                            
                            if (date_str in check_line or 
                                alt_date_str.lower() in check_line.lower() or
                                alt_date_str2.lower() in check_line.lower()):
                                if d not in priority_dates:
                                    priority_dates.append(d)
                    break
        
        # Separate remaining dates
        for d in unique_dates:
            if d not in priority_dates:
                regular_dates.append(d)
        
        # Return prioritized list: invoice date labeled dates first, then others
        final_dates = priority_dates + regular_dates
        
        # Limit to most relevant dates
        return final_dates[:5] if len(final_dates) > 5 else final_dates
    
    return unique_dates

def extract_po_numbers_from_image(uploaded_file) -> List[str]:
    """Extract PO numbers from uploaded image file using OCR"""
    try:
        # Read the uploaded file bytes
        img_bytes = uploaded_file.read()
        
        # Extract text using OCR
        text = _ocr_text_from_image_bytes(img_bytes, uploaded_file.name)
        
        # Extract PO numbers from the text
        po_numbers = _extract_po_numbers_from_text(text)
        
        return po_numbers
        
    except Exception as e:
        if is_debug():
            st.error(f"❌ Error processing image {uploaded_file.name}: {str(e)}")
        return []

# =========================================================
# PDF extraction (improved with debug info and caching)
# =========================================================
@st.cache_data(ttl=1800, show_spinner=False)  # Cache PDF results for 30 minutes, hide spinner
def _cached_extract_pdf_tables(pdf_hash: str, file_path: str, filename: str):
    """
    Cached PDF table extraction to avoid re-processing identical PDFs.
    Uses hash of PDF content as cache key. Cloud-friendly with fallback.
    """
    if not HAS_CAMELOT:
        if is_debug():
            st.warning("Camelot not available - skipping table extraction")
        return []
        
    try:
        tables = camelot.read_pdf(file_path, pages='all')
        return tables
    except Exception as e:
        if is_debug():
            st.error(f"❌ PDF Table Extraction Error for {filename}: {str(e)}")
        return []

def extract_po_numbers_from_single_pdf(file_path, filename):
    po_numbers = []
    invoice_numbers = []
    debug_info = {"tables_found": 0, "pages_processed": 0, "extraction_methods": []}
    
    try:
        # Create hash of PDF file for caching
        with open(file_path, 'rb') as f:
            pdf_content = f.read()
            pdf_hash = hashlib.md5(pdf_content).hexdigest()
        
        # Try cached camelot table extraction first
        if is_debug():
            debug_info["extraction_methods"].append("camelot (cached)")
        
        tables = _cached_extract_pdf_tables(pdf_hash, file_path, filename)
        debug_info["tables_found"] = len(tables)
        
        prefixes = _po_prefix()
        
        for i, table in enumerate(tables):
            df = table.df
            for _, row in df.iterrows():
                for cell_value in row:
                    cell_str = str(cell_value)
                    # Extract PO numbers
                    for prefix in prefixes:
                        po_pattern = r'\b' + re.escape(prefix) + r'\s*\d{6}\b'
                        matches = re.findall(po_pattern, cell_str, re.IGNORECASE)
                        for match in matches:
                            clean_match = re.sub(r'\s+', '', match.upper())
                            po_numbers.append(clean_match)
                    
                    # Extract invoice numbers
                    if st.session_state.get('auto_invoice_detect', True):
                        found_invoices = _extract_invoice_numbers_from_text(cell_str)
                        invoice_numbers.extend(found_invoices)
        
        # Check if we got invoice numbers from tables - if so, we can skip text extraction
        table_found_invoices = len(set(invoice_numbers)) > 0
        
        # Only proceed with text extraction if tables didn't find invoice numbers
        all_text = ""
        text_extracted = False
        
        if not table_found_invoices or not st.session_state.get('auto_invoice_detect', True):
            # Try PyPDF2 for text extraction first
            try:
                import PyPDF2
                if is_debug():
                    debug_info["extraction_methods"].append("PyPDF2")
                    
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    debug_info["pages_processed"] = len(reader.pages)
                    
                    for page in reader.pages:
                        text = page.extract_text() or ""
                        all_text += text + "\n"
                        # Extract PO numbers
                        for prefix in prefixes:
                            po_pattern = r'\b' + re.escape(prefix) + r'\s*\d{6}\b'
                            matches = re.findall(po_pattern, text, re.IGNORECASE)
                            for match in matches:
                                clean_match = re.sub(r'\s+', '', match.upper())
                                po_numbers.append(clean_match)
                        
                        # Extract invoice numbers
                        if st.session_state.get('auto_invoice_detect', True):
                            found_invoices = _extract_invoice_numbers_from_text(text)
                            invoice_numbers.extend(found_invoices)
                    
                    # Check if we got meaningful text (more than just whitespace and basic characters)
                    meaningful_text = re.sub(r'\s+', '', all_text)
                    if len(meaningful_text) > 50:  # At least 50 non-whitespace characters
                        text_extracted = True
                        
            except ImportError:
                if is_debug():
                    st.warning("PyPDF2 not available. Install with: `pip install PyPDF2`")
            except Exception as e:
                if is_debug():
                    st.warning(f"PyPDF2 extraction failed: {str(e)}")

            # Try pdfplumber as alternative if PyPDF2 didn't work well
            if not text_extracted:
                try:
                    import pdfplumber
                    if is_debug():
                        debug_info["extraction_methods"].append("pdfplumber")
                        
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            text = page.extract_text() or ""
                            all_text += text + "\n"
                            for prefix in prefixes:
                                po_pattern = r'\b' + re.escape(prefix) + r'\s*\d{6}\b'
                                matches = re.findall(po_pattern, text, re.IGNORECASE)
                                for match in matches:
                                    clean_match = re.sub(r'\s+', '', match.upper())
                                    po_numbers.append(clean_match)
                        
                        # Check if we got meaningful text
                        meaningful_text = re.sub(r'\s+', '', all_text)
                        if len(meaningful_text) > 50:
                            text_extracted = True
                            
                except ImportError:
                    if is_debug():
                        st.info("pdfplumber not available (optional). Install with: `pip install pdfplumber`")
                except Exception as e:
                    if is_debug():
                        st.warning(f"pdfplumber extraction failed: {str(e)}")

            # If no meaningful text was extracted, try OCR on PDF pages (cloud-friendly)
            if not text_extracted:
                try:
                    import pdf2image
                    if is_debug():
                        debug_info["extraction_methods"].append("pdf2image + OCR")
                    
                    # Convert PDF pages to images and run OCR
                    images = pdf2image.convert_from_path(file_path)
                    debug_info["pages_processed"] = len(images)
                    
                    for i, image in enumerate(images):
                        # Convert PIL image to bytes
                        img_buffer = io.BytesIO()
                        image.save(img_buffer, format='PNG')
                        img_bytes = img_buffer.getvalue()
                        
                        # Run OCR on the image
                        ocr_text = _ocr_text_from_image_bytes(img_bytes, f"{filename}_page_{i+1}")
                        all_text += ocr_text + "\n"
                        
                        # Extract PO numbers from OCR text
                        for prefix in prefixes:
                            po_pattern = r'\b' + re.escape(prefix) + r'\s*\d{6}\b'
                            matches = re.findall(po_pattern, ocr_text, re.IGNORECASE)
                            for match in matches:
                                clean_match = re.sub(r'\s+', '', match.upper())
                                po_numbers.append(clean_match)
                        
                        # Extract invoice numbers from OCR text
                        if st.session_state.get('auto_invoice_detect', True):
                            found_invoices = _extract_invoice_numbers_from_text(ocr_text)
                            invoice_numbers.extend(found_invoices)
                            
                except ImportError:
                    if is_debug():
                        st.warning("⚠️ pdf2image not available - PDF to image OCR disabled (cloud-friendly)")
                except Exception as e:
                    if is_debug():
                        st.warning(f"PDF to image OCR failed: {str(e)} (this is normal in cloud environments)")
        
        # Process extracted data (from any method that succeeded)
        if all_text:
            # Store detected invoice number in session state
            unique_invoices = list(set(invoice_numbers))
            if unique_invoices and st.session_state.get('auto_invoice_detect', True):
                # Store all detected invoice numbers
                st.session_state.detected_invoice_numbers = unique_invoices
                # Smart selection: prefer longer numbers over single digits
                best_invoice = max(unique_invoices, key=len) if len(unique_invoices) > 1 else unique_invoices[0]
                st.session_state.detected_invoice_number = best_invoice
            
            # Extract dates from all text and store in session state (improved logic)
            extracted_dates = extract_dates_from_text(all_text)
            if extracted_dates and st.session_state.get('auto_date_detect', True):
                st.session_state.detected_dates = extracted_dates
                # Filter out dates that might be from filename (like today's date)
                today = date.today()
                # Priority dates (near invoice date labels) come first, filter appropriately
                invoice_dates = [d for d in extracted_dates if d != today and d < today]
                if invoice_dates:
                    # Use the first valid date (priority date if available)
                    st.session_state.detected_invoice_date = invoice_dates[0]
                else:
                    # Fallback to first date if no past dates found
                    st.session_state.detected_invoice_date = extracted_dates[0]
                    
    except Exception as e:
        if is_debug():
            st.error(f"PDF processing error for {filename}: {str(e)}")
    
    # Log debug info
    if is_debug():
        with st.expander(f"📄 PDF Debug – {filename}", expanded=False):
            st.json(debug_info)
    
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
    file_extraction_data = {}  # Store extracted data per file
    po_invoice_combinations = []  # NEW: Track PO+Invoice combinations instead of just POs
    processing_summary = {"files_processed": 0, "files_failed": 0, "file_types": {}}
    
    try:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            members = [f for f in zip_ref.namelist() if not f.startswith('__MACOSX')]
            
            for member in members:
                try:
                    content = zip_ref.read(member)
                    file_ext = member.lower().split('.')[-1] if '.' in member else 'unknown'
                    processing_summary["file_types"][file_ext] = processing_summary["file_types"].get(file_ext, 0) + 1
                    
                    # Initialize file extraction data
                    file_extraction_data[member] = {
                        'pos': [],
                        'invoice_numbers': [],
                        'dates': []
                    }
                    
                    # PDFs
                    if member.lower().endswith('.pdf'):
                        file_data_map[member] = content
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                            tmp.write(content)
                            tmp_path = tmp.name
                        
                        # Extract POs, invoice numbers, and dates from PDF
                        found_pos = extract_po_numbers_from_single_pdf(tmp_path, member)
                        
                        # Also extract invoice numbers and dates from PDF text
                        try:
                            import PyPDF2
                            with open(tmp_path, 'rb') as f:
                                reader = PyPDF2.PdfReader(f)
                                text = ""
                                for page in reader.pages:
                                    text += page.extract_text() + " "
                                
                                # Extract invoice numbers and dates
                                invoice_numbers = _extract_invoice_numbers_from_text(text)
                                extracted_dates = extract_dates_from_text(text)
                                
                                file_extraction_data[member]['pos'] = found_pos
                                file_extraction_data[member]['invoice_numbers'] = invoice_numbers
                                file_extraction_data[member]['dates'] = extracted_dates
                                
                                # Create PO+Invoice combinations for multi-invoice support
                                if found_pos and invoice_numbers:
                                    for po in found_pos:
                                        for inv in invoice_numbers:
                                            combo_key = f"{po}::{inv}"  # Unique key for PO+Invoice
                                            po_invoice_combinations.append({
                                                'po_id': po,
                                                'invoice_number': inv,
                                                'source_file': member,
                                                'combo_key': combo_key,
                                                'dates': extracted_dates
                                            })
                                else:
                                    # Fallback: just add POs without invoice numbers
                                    for po in found_pos:
                                        combo_key = f"{po}::NO_INVOICE"
                                        po_invoice_combinations.append({
                                            'po_id': po,
                                            'invoice_number': '',
                                            'source_file': member,
                                            'combo_key': combo_key,
                                            'dates': extracted_dates
                                        })
                                        
                        except Exception:
                            file_extraction_data[member]['pos'] = found_pos
                            # Fallback: just add POs without invoice numbers
                            for po in found_pos:
                                combo_key = f"{po}::NO_INVOICE"
                                po_invoice_combinations.append({
                                    'po_id': po,
                                    'invoice_number': '',
                                    'source_file': member,
                                    'combo_key': combo_key,
                                    'dates': []
                                })
                        
                        all_po_numbers.extend(found_pos)
                        os.unlink(tmp_path)
                        
                    # Images (OCR enabled)
                    elif member.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')):
                        # Extract text using OCR
                        try:
                            text = _ocr_text_from_image_bytes(content, member)
                            found_pos = _extract_po_numbers_from_text(text)
                            
                            # Extract invoice numbers and dates from OCR text
                            invoice_numbers = _extract_invoice_numbers_from_text(text)
                            extracted_dates = extract_dates_from_text(text)
                            
                            file_extraction_data[member]['pos'] = found_pos
                            file_extraction_data[member]['invoice_numbers'] = invoice_numbers
                            file_extraction_data[member]['dates'] = extracted_dates
                            
                            # Create PO+Invoice combinations
                            if found_pos and invoice_numbers:
                                for po in found_pos:
                                    for inv in invoice_numbers:
                                        combo_key = f"{po}::{inv}"
                                        po_invoice_combinations.append({
                                            'po_id': po,
                                            'invoice_number': inv,
                                            'source_file': member,
                                            'combo_key': combo_key,
                                            'dates': extracted_dates
                                        })
                            else:
                                for po in found_pos:
                                    combo_key = f"{po}::NO_INVOICE"
                                    po_invoice_combinations.append({
                                        'po_id': po,
                                        'invoice_number': '',
                                        'source_file': member,
                                        'combo_key': combo_key,
                                        'dates': extracted_dates
                                    })
                            
                            all_po_numbers.extend(found_pos)
                            file_data_map[member] = content
                        except Exception as e:
                            if is_debug():
                                st.warning(f"⚠️ OCR failed for image: {member} - {str(e)}")
                            file_data_map[member] = content
                    
                    processing_summary["files_processed"] += 1
                    
                except Exception as e:
                    processing_summary["files_failed"] += 1
                    if is_debug():
                        st.warning(f"Failed to process {member}: {str(e)}")
        
        # Store the file extraction data and PO combinations in session state
        st.session_state.zip_file_extraction_data = file_extraction_data
        st.session_state.po_invoice_combinations = po_invoice_combinations  # NEW: Store combinations
        
        # For ZIP files with multiple invoices per PO, return PO+Invoice combinations as separate "PO IDs"
        if po_invoice_combinations:
            # Create combined PO IDs that include invoice numbers for multi-invoice support
            combined_pos = []
            for combo in po_invoice_combinations:
                if combo['invoice_number']:
                    # Create a combined PO ID like "PO039761@INV-002" to make it unique
                    combined_po_id = f"{combo['po_id']}@{combo['invoice_number']}"
                    combined_pos.append(combined_po_id)
                else:
                    combined_pos.append(combo['po_id'])
            return combined_pos, file_data_map
        else:
            # Fallback to original behavior
            return list(set(all_po_numbers)), file_data_map
    except Exception as e:
        if is_debug():
            st.error(f"ZIP processing error: {str(e)}")
        return [], {}

# =========================================================
# Error Message Utilities
# =========================================================
def clean_error_message(error_msg: str) -> str:
    """
    Clean up repetitive and verbose error messages for better display.
    """
    if not error_msg or not isinstance(error_msg, str):
        return "Unknown error"
    
    # Remove the "O " prefix if present
    msg = error_msg
    if msg.startswith("O "):
        msg = msg[2:]
    
    # Extract PO number if present at the start
    po_prefix = ""
    if ":" in msg and len(msg.split(":", 1)[0]) < 20:
        po_prefix = msg.split(":", 1)[0] + ": "
        msg = msg.split(":", 1)[1].strip()
    
    # Special case: If it contains "Invoice # has already been used for this supplier", just show that
    if "Invoice # has already been used for this supplier" in msg:
        return po_prefix + "Invoice # has already been used for this supplier"
    
    # Clean up common technical prefixes for better readability
    if "invoice-header:" in msg:
        # Remove the technical prefix and just show the user-friendly message
        parts = msg.split("invoice-header:", 1)
        if len(parts) > 1:
            msg = parts[1].strip()
    
    # Split by semicolons and count duplicates
    parts = [part.strip() for part in msg.split(';') if part.strip()]
    
    if not parts:
        return po_prefix + "Unknown error"
    
    # Count occurrences of each message
    message_counts = {}
    for part in parts:
        if part in message_counts:
            message_counts[part] += 1
        else:
            message_counts[part] = 1
    
    # Build summary with most important messages first
    sorted_messages = sorted(message_counts.items(), key=lambda x: (-x[1], x[0]))
    
    summary_parts = []
    for msg, count in sorted_messages[:2]:  # Show top 2 message types
        if count > 1:
            summary_parts.append(f"{msg} (×{count})")
        else:
            summary_parts.append(msg)
    
    # Combine and limit length
    result = po_prefix + "; ".join(summary_parts)
    if len(result) > 150:
        result = result[:147] + "..."
    
    return result

# =========================================================
# Coupa API helpers
# =========================================================
def check_invoice_duplicate(invoice_number: str, supplier_id: int, headers: dict, coupa_instance: str, supplier_name: str = None) -> Tuple[bool, str, bool, str]:
    """
    Check if an invoice number already exists for the given supplier.
    Returns (is_duplicate, message, has_error, invoice_id).
    """
    try:
        # Search for existing invoices using Coupa's standard format - only fetch minimal fields
        search_url = f"https://{coupa_instance}.coupahost.com/api/invoices"
        
        if supplier_name:
            # Use supplier-name approach
            params = {
                "supplier-name": supplier_name,
                "invoice-number": invoice_number,
                "filter": "default_invoices_filter",                
                "limit": "1"
            }
        else:
            # Fallback to supplier[id] approach
            params = {
                "supplier[id]": str(supplier_id),
                "invoice-number": invoice_number,
                "filter": "default_invoices_filter",                
                "limit": "1"
            }
        
        if is_debug():
            debug_expander = st.expander(f"🔍 Debug: Invoice Duplicate Check - {invoice_number}", expanded=False)
            with debug_expander:
                # Create standardized Request/Response tabs
                req_tab, resp_tab = st.tabs(["📡 API Request", "📨 API Response"])
                
                with req_tab:
                    st.write("**🔍 Invoice Duplicate Check Request:**")
                    request_data = {
                        "Field": ["URL", "Method", "Invoice Number", "Supplier ID", "Purpose", "Timeout", "Filter"],
                        "Value": [
                            search_url,
                            "GET",
                            invoice_number,
                            str(supplier_id),
                            "Check for duplicate invoices",
                            "30 seconds",
                            "default_invoices_filter"
                        ]
                    }
                    st.dataframe(pd.DataFrame(request_data), use_container_width=True, hide_index=True)
                    
                    st.write("**📋 Request Parameters:**")
                    params_df_data = []
                    for key, value in params.items():
                        params_df_data.append({"Parameter": key, "Value": str(value)})
                    st.dataframe(pd.DataFrame(params_df_data), use_container_width=True, hide_index=True)
        
        response = requests.get(search_url, headers=headers, params=params, timeout=30)

        if is_debug():
            with debug_expander:
                # Show API status inside debug expander
                if response.status_code == 200:
                    st.success("✅ **API Call Successful**")
                else:
                    st.error(f"❌ **API Call Failed** (HTTP {response.status_code})")

                with resp_tab:
                    # Response Summary
                    st.write("**📊 Response Summary:**")
                    response_summary = {
                        "Metric": ["HTTP Status", "Response Format", "Content Length", "Processing Time"],
                        "Value": [
                            f"{response.status_code} ({'Success' if response.status_code == 200 else 'Error'})",
                            "JSON",
                            f"{len(response.text)} characters" if hasattr(response, 'text') else "Unknown",
                            "< 30 seconds"
                        ]
                    }
                    st.dataframe(pd.DataFrame(response_summary), use_container_width=True, hide_index=True)
                    
                    if response.status_code == 200:
                        invoices = response.json()
                        
                        # Duplicate Check Results
                        st.write("**🔍 Duplicate Check Results:**")
                        is_duplicate = isinstance(invoices, list) and len(invoices) > 0
                        result_data = {
                            "Check": ["Duplicates Found", "Response Type", "Invoice Count", "Result Status"],
                            "Value": [
                                "⚠️ YES - Duplicate exists!" if is_duplicate else "✅ NO - Unique invoice",
                                "List" if isinstance(invoices, list) else type(invoices).__name__,
                                str(len(invoices)) if isinstance(invoices, list) else "1 object",
                                "❌ BLOCKED" if is_duplicate else "✅ APPROVED"
                            ]
                        }
                        st.dataframe(pd.DataFrame(result_data), use_container_width=True, hide_index=True)
                        
                        # Response Data Preview
                        st.write("**📄 Response Data:**")
                        json_str = json.dumps(invoices, indent=2)
                        if len(json_str) > 5000:
                            preview_data = {
                                "Info": ["Response Size", "Preview", "Full Data"],
                                "Details": [
                                    f"{len(json_str)} characters (large response)",
                                    "Showing first 3000 characters below",
                                    "Truncated for display performance"
                                ]
                            }
                            st.dataframe(pd.DataFrame(preview_data), use_container_width=True, hide_index=True)
                            st.code(json_str[:3000] + "\n... (truncated for performance)")
                        else:
                            st.code(json_str)
                    else:
                        # Error Analysis
                        st.write("**❌ Error Details:**")
                        error_data = {
                            "Field": ["HTTP Status", "Error Type", "Likely Cause", "Suggested Action"],
                            "Value": [
                                str(response.status_code),
                                "Authentication" if response.status_code == 401 else "Permission" if response.status_code == 403 else "Client Error" if response.status_code < 500 else "Server Error",
                                "Invalid token" if response.status_code == 401 else "Insufficient permissions" if response.status_code == 403 else "Bad request" if response.status_code < 500 else "Server issue",
                                "Check authentication" if response.status_code == 401 else "Verify permissions" if response.status_code == 403 else "Check request format" if response.status_code < 500 else "Retry later"
                            ]
                        }
                        st.dataframe(pd.DataFrame(error_data), use_container_width=True, hide_index=True)
                        
                        # Error Response
                        try:
                            error_response = response.json()
                            st.code(json.dumps(error_response, indent=2))
                        except:
                            st.code(response.text[:1000] + "..." if len(response.text) > 1000 else response.text)
        
        if response.status_code == 200:
            invoices = response.json()
            
            # Check if response is a list with invoices
            if isinstance(invoices, list) and len(invoices) > 0:
                existing_invoice = invoices[0]
                invoice_id = existing_invoice.get('id', '')
                supplier_info = existing_invoice.get("supplier", {})
                supplier_name = supplier_info.get("name", "Unknown Supplier") if isinstance(supplier_info, dict) else "Unknown Supplier"
                return True, f"Invoice #{invoice_number} already exists for {supplier_name} (ID: {invoice_id})", False, str(invoice_id)
            else:
                return False, "No duplicate found", False, ""
        elif response.status_code == 401:
            clear_oauth_token_cache()  # Clear cache on auth failure
            return False, f"Authentication failed (401) - check permissions", True, ""
        elif response.status_code == 403:
            clear_oauth_token_cache()  # Clear cache on auth failure
            return False, f"Access forbidden (403) - insufficient permissions", True, ""
        else:
            # Log the full error response for debugging
            error_text = response.text[:500] if hasattr(response, 'text') else str(response)
            return False, f"API error: HTTP {response.status_code} - {error_text}", True, ""
            
    except Exception as e:
        return False, f"Error checking duplicates: {str(e)}", True, ""

def _extract_real_po_id(combined_po_id: str) -> str:
    """
    Extract real PO ID from combined PO@Invoice format.
    e.g., "PO039761@INV-002" -> "PO039761"
    """
    if '@' in combined_po_id:
        return combined_po_id.split('@')[0]
    return combined_po_id

def _extract_invoice_from_combined(combined_po_id: str) -> str:
    """
    Extract invoice number from combined PO@Invoice format.
    e.g., "PO039761@INV-002" -> "INV-002"
    """
    if '@' in combined_po_id:
        return combined_po_id.split('@')[1]
    return ""

def verify_po_in_coupa(po_id, headers, coupa_instance):
    """
    Verify PO and normalize to flat 'order-lines' list.
    Handles combined PO IDs like "PO039761@INV-002" for multi-invoice support.
    """
    try:
        # Extract real PO ID for API calls (remove @Invoice suffix if present)
        real_po_id = _extract_real_po_id(po_id)
        numeric_po_id = _numeric_po_id(real_po_id)
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
            
            if is_debug():
                debug_expander = st.expander(f"🔍 Debug: PO Data Loading - {numeric_po_id}", expanded=False)
                with debug_expander:
                    st.write("**API Request:**")
                    st.write(f"- **URL**: {url}")
                    st.write(f"- **Method**: GET")
                    st.write(f"- **Purpose**: Load PO details from Coupa")
                    st.write("- **Headers**:")
                    st.code(json.dumps(_redact_headers(headers), indent=2))
                    st.write("- **Params**:")
                    st.code(json.dumps(params, indent=2))
            
            r = requests.get(url, headers=headers, params=params, timeout=30)
            
            if is_debug():
                with debug_expander:
                    st.write("**API Response:**")
                    st.write(f"- **Status**: {r.status_code}")
                    st.write("- **Response Headers**:")
                    st.code(json.dumps(dict(r.headers), indent=2))
                    
                    if r.status_code == 200:
                        st.write("- **Result**: ✅ PO data loaded successfully")
                        try:
                            response_data = r.json()
                            st.write("- **Response Body**:")
                            # Limit response size for display
                            json_str = json.dumps(response_data, indent=2)
                            if len(json_str) > 10000:
                                st.write(f"**Response too large ({len(json_str)} chars), showing first 5000 characters:**")
                                st.code(json_str[:5000] + "\n... (truncated)")
                            else:
                                st.code(json_str)
                        except Exception as e:
                            st.write(f"- **Response Body**: Could not parse JSON: {str(e)}")
                            st.code(r.text[:1000] + "..." if len(r.text) > 1000 else r.text)
                    else:
                        st.write(f"- **Result**: ❌ PO not found or inaccessible")
                        st.write("- **Error Response**:")
                        try:
                            error_data = r.json()
                            st.code(json.dumps(error_data, indent=2))
                        except:
                            st.code(r.text[:1000] + "..." if len(r.text) > 1000 else r.text)
                    
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

        if is_debug():
            debug_expander = st.expander(f"🔍 Debug: File Upload - {upload_filename}", expanded=False)
            with debug_expander:
                # Create standardized Request/Response tabs
                req_tab, resp_tab = st.tabs(["📡 API Request", "📨 API Response"])

                with req_tab:
                    file_size_mb = len(file_bytes) / (1024 * 1024)
                    st.write("**📄 File Upload API Request:**")
                    request_data = {
                        "Field": ["URL", "Method", "Invoice ID", "Filename", "File Size", "MIME Type", "Purpose", "Timeout"],
                        "Value": [
                            upload_url,
                            "PUT",
                            str(invoice_id),
                            upload_filename,
                            f"{file_size_mb:.2f} MB",
                            _guess_mime(upload_filename),
                            "Upload file to invoice image_scan endpoint",
                            f"{'180s (large)' if file_size_mb > 20 else '120s (medium)' if file_size_mb > 10 else '60s (normal)'}"
                        ]
                    }
                    st.dataframe(pd.DataFrame(request_data), use_container_width=True, hide_index=True)

                    st.write("**📋 Request Headers:**")
                    headers_data = []
                    redacted_headers = _redact_headers(upload_headers)
                    for key, value in redacted_headers.items():
                        headers_data.append({"Header": key, "Value": str(value)})
                    st.dataframe(pd.DataFrame(headers_data), use_container_width=True, hide_index=True)

                    st.write("**📄 File Information:**")
                    file_info = {
                        "Property": ["Original Name", "Content Type", "Size (bytes)", "Size Category"],
                        "Value": [
                            upload_filename,
                            _guess_mime(upload_filename),
                            f"{len(file_bytes):,}",
                            "Large (>20MB)" if file_size_mb > 20 else "Medium (>10MB)" if file_size_mb > 10 else "Normal"
                        ]
                    }
                    st.dataframe(pd.DataFrame(file_info), use_container_width=True, hide_index=True)
        
        # Adjust timeout based on file size
        file_size_mb = len(file_bytes) / (1024 * 1024)
        if file_size_mb > 20:
            timeout = 180  # 3 minutes for very large files
        elif file_size_mb > 10:
            timeout = 120  # 2 minutes for large files
        else:
            timeout = 60   # 1 minute for normal files
            
        resp = requests.put(upload_url, headers=upload_headers, files=files, timeout=timeout)
        
        if is_debug():
            with debug_expander:
                with resp_tab:
                    st.write("**📄 File Upload API Response:**")
                    response_summary = {
                        "Field": ["HTTP Status", "Content Type", "Response Size", "Upload Result"],
                        "Value": [
                            f"{resp.status_code} ({'Success' if resp.status_code in [200, 201] else 'Error'})",
                            resp.headers.get('Content-Type', 'Unknown'),
                            f"{len(resp.content)} bytes",
                            "✅ Upload completed" if resp.status_code in [200, 201] else "❌ Upload failed"
                        ]
                    }
                    st.dataframe(pd.DataFrame(response_summary), use_container_width=True, hide_index=True)

                    st.write("**📋 Response Headers:**")
                    headers_data = []
                    for key, value in resp.headers.items():
                        headers_data.append({"Header": key, "Value": str(value)})
                    st.dataframe(pd.DataFrame(headers_data), use_container_width=True, hide_index=True)

                    if resp.status_code in [200, 201]:
                        st.write("**✅ Upload Success:**")
                        try:
                            response_data = resp.json()
                            upload_result = {
                                "Metric": ["Result", "Response Fields", "Invoice ID", "Processing Status"],
                                "Value": [
                                    "✅ File uploaded successfully",
                                    f"{len(response_data)} fields returned",
                                    str(response_data.get('id', 'Unknown')),
                                    response_data.get('status', 'Unknown')
                                ]
                            }
                            st.dataframe(pd.DataFrame(upload_result), use_container_width=True, hide_index=True)

                            st.write("**📄 Response Body (Essential Fields):**")
                            json_str = json.dumps(response_data, indent=2)
                            if len(json_str) > 8000:
                                # Show essential fields for large responses
                                essential = {k: v for k, v in response_data.items() if k in ["id", "status", "invoice-number", "total", "errors", "image-scan"]}
                                st.code(json.dumps(essential, indent=2))
                                st.info(f"Full response contains {len(response_data)} fields - showing essential fields only")
                            else:
                                st.code(json_str)
                        except Exception as e:
                            st.write("**❌ Response Parsing Error:**")
                            error_data = {
                                "Issue": ["JSON Parse Failed", "Raw Response Length", "Error Message"],
                                "Details": [
                                    "Could not parse response as JSON",
                                    f"{len(resp.text)} characters",
                                    str(e)
                                ]
                            }
                            st.dataframe(pd.DataFrame(error_data), use_container_width=True, hide_index=True)
                            st.code(resp.text[:500] + "..." if len(resp.text) > 500 else resp.text)
                    else:
                        st.write("**❌ Upload Error:**")
                        error_summary = {
                            "Field": ["Status", "Result", "Response Size"],
                            "Value": [
                                f"HTTP {resp.status_code}",
                                "❌ File upload failed",
                                f"{len(resp.content)} bytes"
                            ]
                        }
                        st.dataframe(pd.DataFrame(error_summary), use_container_width=True, hide_index=True)

                        try:
                            error_data = resp.json()
                            st.write("**📄 Error Details:**")
                            st.code(json.dumps(error_data, indent=2))
                        except:
                            st.write("**📄 Raw Error Response:**")
                            st.code(resp.text[:1000] + "..." if len(resp.text) > 1000 else resp.text)
        
        return (resp.status_code in [200, 201]), (resp.text if resp.text else f"HTTP {resp.status_code}")
    except Exception as e:
        if is_debug():
            st.error(f"❌ Image upload exception: {str(e)}")
        return False, str(e)

def verify_pos_concurrent(po_ids: List[str], headers: dict, coupa_instance: str, max_workers: int = 5) -> dict:
    """
    Verify multiple POs concurrently to speed up the process.
    Returns dict with po_id as key and verification result as value.
    """
    results = {}
    total_pos = len(po_ids)
    
    def verify_single_po(po_id):
        try:
            result = verify_po_in_coupa(po_id, headers, coupa_instance)
            return po_id, result
        except Exception as e:
            return po_id, {'exists': False, 'error': str(e)}
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    completed = 0
    
    # Use ThreadPoolExecutor for concurrent API calls
    with ThreadPoolExecutor(max_workers=min(max_workers, len(po_ids))) as executor:
        # Submit all tasks
        future_to_po = {executor.submit(verify_single_po, po_id): po_id for po_id in po_ids}
        
        # Collect results as they complete
        for future in as_completed(future_to_po):
            po_id, result = future.result()
            results[po_id] = result
            
            # Update progress
            completed += 1
            progress = completed / total_pos
            progress_bar.progress(progress)
            status_text.text(f"Verified {completed}/{total_pos} POs...")
    
    # Clean up progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return results

# =========================================================
# Editor building / sanitation
# =========================================================
def build_preview_df(po_id: str, po_data: dict) -> pd.DataFrame:
    # Extract real PO ID in case this is a combined PO@Invoice format
    real_po_id = _extract_real_po_id(po_id)
    numeric_po = int(_numeric_po_id(real_po_id))
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
        
        # Simple line type detection logic
        # If there is a UOM then it's qty, if not then it's amount
        if uom_code and str(uom_code).strip():
            # UOM present = quantity line
            inv_type = "InvoiceQuantityLine"
        else:
            # No UOM = amount line
            inv_type = "InvoiceAmountLine"

        rows.append({
            "line_num": i,
            "inv_type": inv_type,
            "description": _get_any(line, "description", default=""),
            "price": price,
            "quantity": qty if inv_type == "InvoiceQuantityLine" else None,  # No qty for amount based lines
            "uom_code": uom_code or (DEFAULT_UOM_CODE if inv_type == "InvoiceQuantityLine" else None),
            "account_id": acct_id,
            "commodity_name": commodity_name,
            "currency_code": _get_any(po_data.get("currency", {}), "code"),  # Get currency from PO
            "po_number": real_po_id,                       # Use clean PO ID without @Invoice suffix
            "order_header_num": int(_numeric_po_id(_extract_real_po_id(po_id))),
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

def _default_commodity_from(df: pd.DataFrame):
    """Get the most common commodity name from existing lines."""
    vals = df["commodity_name"].dropna()
    if vals.empty:
        return None
    # Return the most common commodity, or first one if all are unique
    return vals.mode().iloc[0] if not vals.mode().empty else vals.iloc[0]

def _get_po_defaults_for_new_line(po_id: str, existing_df: pd.DataFrame):
    """Get default values for new lines from PO data and existing lines."""
    defaults = {
        "account_id": _default_account_id_from(existing_df),
        "commodity_name": _default_commodity_from(existing_df),
        "po_number": _extract_real_po_id(po_id),
        "order_header_num": int(_numeric_po_id(_extract_real_po_id(po_id))),
        "order_header_id": None,
        "source_part_num": None,
        "currency_code": None
    }

    # Try to get additional defaults from existing lines
    if not existing_df.empty:
        # Get the most common order_header_id if available
        header_ids = existing_df["order_header_id"].dropna()
        if not header_ids.empty:
            defaults["order_header_id"] = header_ids.mode().iloc[0] if not header_ids.mode().empty else header_ids.iloc[0]

        # Get currency from existing lines (should be consistent across all lines)
        if "currency_code" in existing_df.columns:
            currency_codes = existing_df["currency_code"].dropna()
            if not currency_codes.empty:
                defaults["currency_code"] = currency_codes.iloc[0]  # All lines should have same currency

    return defaults

def resequence_lines(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(by=["line_num"], kind="stable")
    # Only update the display line_num for internal processing, preserve original PO line numbers
    df["line_num"] = range(1, len(df) + 1)
    # Do NOT change order_line_num - it should preserve the original PO line numbers
    # df["order_line_num"] = df["line_num"].astype(str)  # REMOVED - preserve original PO line numbers
    return df

def get_next_available_line_num(df: pd.DataFrame) -> str:
    """Get the next available PO line number that doesn't conflict with existing ones."""
    existing_line_nums = set()
    for val in df["order_line_num"].dropna():
        try:
            existing_line_nums.add(int(val))
        except (ValueError, TypeError):
            pass

    # Start from the highest existing line number + 1, or 1 if no existing lines
    if existing_line_nums:
        next_num = max(existing_line_nums) + 1
    else:
        next_num = 1

    # Make sure we don't conflict with any existing line numbers
    while next_num in existing_line_nums:
        next_num += 1

    return str(next_num)

def sanitize_editor_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Note: We keep delete-marked rows visible in the editor and only filter them
    # out when creating the actual invoice, so users can see what they've marked for deletion
    
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
    header_tax_rate: float,
    approval_option: str = "Save as Draft",
    stage_callback=None
) -> Tuple[bool, Any, bool, str]:
    """
    POST /api/invoices, then PUT image_scan.
    Returns (invoice_success, invoice_resp|error, scan_success, scan_message).
    stage_callback: Optional function to call with progress messages
    """
    # ---- Auth
    if stage_callback:
        stage_callback("Stage 1/3: Authenticating...")
    success, token_or_error = get_oauth_token(OAUTH_SCOPE_ALL)
    if not success:
        clear_oauth_token_cache()  # Clear cache on auth failure
        return False, f"Auth error: {token_or_error}", False, "N/A"
    
    config = get_env_config()
    headers = {
        "Authorization": f"Bearer {token_or_error}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    # ---- Build header & lines
    
    # Check file size for potential issues early
    file_size_mb = len(original_file_data) / (1024 * 1024)
    if is_debug():
        with st.expander("🔍 Invoice Creation Debug Info", expanded=False):
            st.write("**File Processing:**")
            st.write(f"**Original file**: {upload_filename}")
            st.write(f"**File size**: {file_size_mb:.2f} MB")
            if file_size_mb > 20:
                st.error("⚠️ Very large file (>20MB) - may cause upload timeouts")
            elif file_size_mb > 10:
                st.warning("⚠️ Large file (>10MB) - upload may be slow")
            else:
                st.success("✅ File size is reasonable for upload")
    
    supplier_obj = po_data.get("supplier") or {}
    currency_obj = po_data.get("currency") or {}

    subtotal = float(edited_df["price"].fillna(0).astype(float).mul(edited_df["quantity"].fillna(0).astype(float)).sum())
    tax_rate_used = float(header_tax_rate or 0.0)
    tax_amount = round(subtotal * (tax_rate_used / 100.0), 2) if tax_rate_used > 0 else 0.0

    # Debug log to verify the process
    if is_debug():
        print(f"Debug - Creating invoice with approval_option: '{approval_option}'")
        print("Debug - Step 1: Creating invoice (no status specified - will default to draft)")
        print("Debug - Step 2: Will upload image scan")
        if approval_option == "Submit for Approval":
            print("Debug - Step 3: Will submit for approval after image upload")
    
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

    # Filter out rows marked for deletion before creating invoice lines
    if "delete" in edited_df.columns:
        edited_df = edited_df[~edited_df["delete"].astype(bool)].copy()

    for _, r in edited_df.iterrows():
        inv_type = (r.get("inv_type") or "InvoiceQuantityLine").strip()
        is_qty = inv_type == "InvoiceQuantityLine"
        line_po_number = str(r.get("po_number") or context_po_id).strip() or context_po_id
        # Extract real PO ID from combined PO@Invoice format for order header num
        real_line_po = _extract_real_po_id(line_po_number)
        order_header_num = _numeric_from_po_string(real_line_po)

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
        if pd.notna(r.get("currency_code")) and str(r.get("currency_code")).strip():
            line["currency"] = {"code": str(r.get("currency_code")).strip()}

        invoice_data["invoice-lines"].append({k: v for k, v in line.items() if v is not None})

    # ---- Create invoice
    try:
        if stage_callback:
            stage_callback("Stage 2/3: Creating invoice record...")
        
        create_url = f"https://{config['instance']}.coupahost.com/api/invoices"
        
        # Add filter parameter to limit response size and prevent crashes
        params = {
            "filter": "default_invoices_filter"
        }
        
        if is_debug():
            debug_expander = st.expander(f"🔍 Debug: Invoice Creation - {invoice_number}", expanded=False)
            with debug_expander:
                # Create standardized Request/Response tabs
                req_tab, resp_tab = st.tabs(["📡 API Request", "📨 API Response"])

                with req_tab:
                    st.write("**💼 Invoice Creation API Request:**")
                    request_data = {
                        "Field": ["URL", "Method", "Invoice Number", "Invoice Date", "Line Items", "Purpose", "Timeout"],
                        "Value": [
                            create_url,
                            "POST",
                            invoice_number,
                            str(invoice_date),
                            f"{len(invoice_data.get('invoice-lines', []))} lines",
                            "Create new invoice in Coupa",
                            "45 seconds"
                        ]
                    }
                    st.dataframe(pd.DataFrame(request_data), use_container_width=True, hide_index=True)

                    st.write("**📋 Request Headers:**")
                    headers_data = []
                    redacted_headers = _redact_headers(headers)
                    for key, value in redacted_headers.items():
                        headers_data.append({"Header": key, "Value": str(value)})
                    st.dataframe(pd.DataFrame(headers_data), use_container_width=True, hide_index=True)

                    if params:
                        st.write("**🔗 Request Parameters:**")
                        params_data = []
                        for key, value in params.items():
                            params_data.append({"Parameter": key, "Value": str(value)})
                        st.dataframe(pd.DataFrame(params_data), use_container_width=True, hide_index=True)

                    st.write("**📄 Request Body:**")
                    # Truncate large request bodies
                    json_str = json.dumps(invoice_data, indent=2)
                    if len(json_str) > 8000:
                        st.write(f"**Request body is large ({len(json_str)} chars), showing essential fields:**")
                        essential = {k: v for k, v in invoice_data.items() if k in ["invoice-number", "invoice-date", "total", "supplier", "currency", "invoice-lines"]}
                        if "invoice-lines" in essential and len(essential["invoice-lines"]) > 3:
                            essential["invoice-lines"] = essential["invoice-lines"][:3] + [{"...": f"and {len(essential['invoice-lines']) - 3} more lines"}]
                        st.code(json.dumps(essential, indent=2))
                    else:
                        st.code(json_str)

        resp = requests.post(create_url, headers=headers, json=invoice_data, params=params, timeout=45)

        if is_debug():
            with debug_expander:
                with resp_tab:
                    st.write("**💼 Invoice Creation API Response:**")
                    response_summary = {
                        "Field": ["HTTP Status", "Content Type", "Response Size", "Processing Time"],
                        "Value": [
                            f"{resp.status_code} ({'Success' if resp.status_code in [200, 201] else 'Error'})",
                            resp.headers.get('Content-Type', 'Unknown'),
                            f"{len(resp.content)} bytes",
                            "< 45 seconds"
                        ]
                    }
                    st.dataframe(pd.DataFrame(response_summary), use_container_width=True, hide_index=True)

                    st.write("**📋 Response Headers:**")
                    headers_data = []
                    for key, value in resp.headers.items():
                        headers_data.append({"Header": key, "Value": str(value)})
                    st.dataframe(pd.DataFrame(headers_data), use_container_width=True, hide_index=True)

                    if resp.status_code in [200, 201]:
                        st.write("**✅ Success Result:**")
                        try:
                            response_data = resp.json()
                            invoice_id = response_data.get('id', 'Unknown')

                            result_data = {
                                "Metric": ["Result", "Invoice ID", "Status", "Response Fields"],
                                "Value": [
                                    "✅ Invoice created successfully",
                                    str(invoice_id),
                                    response_data.get('status', 'Unknown'),
                                    f"{len(response_data)} fields returned"
                                ]
                            }
                            st.dataframe(pd.DataFrame(result_data), use_container_width=True, hide_index=True)

                            st.write("**📄 Response Body (Essential Fields):**")
                            json_str = json.dumps(response_data, indent=2)
                            if len(json_str) > 8000:
                                # Show essential fields for large responses
                                essential = {k: v for k, v in response_data.items() if k in ["id", "status", "invoice-number", "total", "errors", "supplier", "currency"]}
                                st.code(json.dumps(essential, indent=2))
                                st.info(f"Full response contains {len(response_data)} fields - showing essential fields only")
                            else:
                                st.code(json_str)
                        except Exception as e:
                            st.write("**❌ Response Parsing Error:**")
                            error_data = {
                                "Issue": ["JSON Parse Failed", "Raw Response Length", "Error Message"],
                                "Details": [
                                    "Could not parse response as JSON",
                                    f"{len(resp.text)} characters",
                                    str(e)
                                ]
                            }
                            st.dataframe(pd.DataFrame(error_data), use_container_width=True, hide_index=True)
                            st.code(resp.text[:500] + "..." if len(resp.text) > 500 else resp.text)
                    else:
                        st.write("**❌ Error Response:**")
                        error_summary = {
                            "Field": ["Status", "Result", "Response Size"],
                            "Value": [
                                f"HTTP {resp.status_code}",
                                "❌ Invoice creation failed",
                                f"{len(resp.content)} bytes"
                            ]
                        }
                        st.dataframe(pd.DataFrame(error_summary), use_container_width=True, hide_index=True)

                        try:
                            error_data = resp.json()
                            st.write("**📄 Error Details:**")
                            st.code(json.dumps(error_data, indent=2))
                        except:
                            st.write("**📄 Raw Error Response:**")
                            st.code(resp.text[:2000] + "..." if len(resp.text) > 2000 else resp.text)
        
        if resp.status_code not in [200, 201]:
            # Try to parse detailed error messages from API response
            error_msg = f"HTTP {resp.status_code}"
            try:
                error_data = resp.json()
                if "errors" in error_data:
                    detailed_errors = []
                    errors = error_data["errors"]
                    
                    # Extract and deduplicate error messages
                    error_categories = {}  # Group similar errors together

                    for field, messages in errors.items():
                        if field != "warnings" and messages:
                            if isinstance(messages, list):
                                for msg in messages:
                                    # Group by the actual error message, count field occurrences
                                    if msg not in error_categories:
                                        error_categories[msg] = []
                                    error_categories[msg].append(field)
                            else:
                                if messages not in error_categories:
                                    error_categories[messages] = []
                                error_categories[messages].append(field)

                    # Build deduplicated error messages
                    for error_msg_text, fields in error_categories.items():
                        if len(fields) == 1:
                            # Single field error
                            detailed_errors.append(f"{fields[0]}: {error_msg_text}")
                        elif len(fields) <= 3:
                            # Few fields - list them
                            field_list = ", ".join(fields)
                            detailed_errors.append(f"{field_list}: {error_msg_text}")
                        else:
                            # Many fields - show count
                            detailed_errors.append(f"{len(fields)} fields: {error_msg_text}")

                    if detailed_errors:
                        error_msg = "; ".join(detailed_errors)
                    else:
                        error_msg = f"HTTP {resp.status_code} - {resp.text}"
                else:
                    error_msg = f"HTTP {resp.status_code} - {resp.text}"
            except:
                error_msg = f"HTTP {resp.status_code} - {resp.text}"
            
            return False, error_msg, False, "N/A"

        invoice_id = resp.json().get("id")
        invoice_response = resp.json()
    except Exception as e:
        return False, str(e), False, "N/A"

    # ---- Upload original scan (PDF or Image) first
    try:
        if stage_callback:
            # Show file info and warnings for better UX
            file_size_mb = len(original_file_data) / (1024 * 1024)
            
            # Warn about large files that might be slow
            if file_size_mb > 10:
                stage_callback(f"Stage 3/3: Uploading large image ({file_size_mb:.1f} MB) - This may take a while...")
                if is_debug():
                    st.warning(f"⚠️ Large file detected: {file_size_mb:.1f} MB. Upload may be slow.")
            else:
                stage_callback(f"Stage 3/3: Uploading image scan ({file_size_mb:.1f} MB)...")
            
        scan_ok, scan_msg = upload_original_scan(
            invoice_id=invoice_id,
            file_bytes=original_file_data,
            upload_filename=upload_filename,
            headers=headers,
            coupa_instance=config['instance']
        )
        
        # Additional debug for slow uploads
        if not scan_ok and is_debug():
            file_size_mb = len(original_file_data) / (1024 * 1024)
            if file_size_mb > 5:
                st.error(f"💾 Upload failed for {file_size_mb:.1f} MB file. Large files may timeout or be rejected by the server.")
                st.info("💡 Consider reducing file size or using a different format if uploads continue to fail.")
            
    except Exception as e:
        scan_ok, scan_msg = False, str(e)
        if is_debug():
            st.error(f"🚨 Image upload exception: {str(e)}")
            # Show file info to help with troubleshooting
            file_size_mb = len(original_file_data) / (1024 * 1024)
            st.write(f"**File size**: {file_size_mb:.2f} MB")
            st.write(f"**File name**: {upload_filename}")
            st.write(f"**MIME type**: {_guess_mime(upload_filename)}")

    # ---- Submit for approval ONLY if requested and AFTER image upload
    if approval_option == "Submit for Approval" and invoice_id:
        try:
            if stage_callback:
                stage_callback("Submitting for approval...")
                
            submit_url = f"https://{config['instance']}.coupahost.com/api/invoices/{invoice_id}/submit"
            
            if is_debug():
                debug_expander = st.expander(f"🔍 Debug: Invoice Submission - {invoice_id}", expanded=False)
                with debug_expander:
                    st.write("**API Request:**")
                    st.write(f"- **URL**: {submit_url}")
                    st.write(f"- **Method**: PUT")
                    st.write(f"- **Action**: Submit for approval")
                    st.write("- **Headers**:")
                    st.code(json.dumps(_redact_headers(headers), indent=2))
            
            submit_resp = requests.put(submit_url, headers=headers, timeout=30)
            
            if is_debug():
                with debug_expander:
                    st.write("**API Response:**")
                    st.write(f"- **Status**: {submit_resp.status_code}")
                    st.write("- **Response Headers**:")
                    st.code(json.dumps(dict(submit_resp.headers), indent=2))
                    
                    if submit_resp.status_code in [200, 201]:
                        st.write("- **Result**: ✅ Invoice submitted for approval")
                        try:
                            response_data = submit_resp.json()
                            st.write("- **Response Body**:")
                            st.code(json.dumps(response_data, indent=2))
                        except Exception as e:
                            st.write(f"- **Response Body**: Could not parse JSON: {str(e)}")
                            st.code(submit_resp.text[:500] + "..." if len(submit_resp.text) > 500 else submit_resp.text)
                    else:
                        st.write("- **Result**: ❌ Invoice submission failed")
                        st.write("- **Error Response**:")
                        try:
                            error_data = submit_resp.json()
                            st.code(json.dumps(error_data, indent=2))
                        except:
                            st.code(submit_resp.text[:1000] + "..." if len(submit_resp.text) > 1000 else submit_resp.text)
            
            if submit_resp.status_code not in [200, 201]:
                # If submit fails, log the error but continue - invoice was still created
                if is_debug():
                    print(f"Debug - Submit failed with status {submit_resp.status_code}: {submit_resp.text}")
                # Update the invoice response to indicate submit failure
                invoice_response["submit_error"] = f"Submit failed: HTTP {submit_resp.status_code}"
            else:
                # Update the invoice response with submit success
                if is_debug():
                    print("Debug - Invoice successfully submitted for approval")
                invoice_response["submitted"] = True
                
        except Exception as e:
            if is_debug():
                print(f"Debug - Submit exception: {str(e)}")
            invoice_response["submit_error"] = f"Submit exception: {str(e)}"

    return True, invoice_response, scan_ok, scan_msg

# =========================================================
# Streamlit App
# =========================================================
def main():
    # Add comprehensive CSS to remove borders from all streamlit-tags
    st.markdown("""
    <style>
    /* Global tag styling - remove all borders */
    span[data-baseweb="tag"],
    .st-emotion-cache-* span[data-baseweb="tag"],
    div[data-testid*="element-container"] span[data-baseweb="tag"] {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    /* Streamlit-tags specific selectors */
    .streamlit-tags .ReactTags__tag,
    .streamlit-tags .tag,
    .stTags .tag,
    div[class*="streamlit-tags"] .tag,
    div[class*="stTags"] .tag {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    /* React-tags component selectors */
    .ReactTags__tag,
    .ReactTags__remove {
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Base-web tag component */
    [data-baseweb="tag"] {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    /* Fallback - target any element with "tag" in the class */
    div[class*="tag"],
    span[class*="tag"] {
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Target the iframe container instead of iframe content */
    .st-emotion-cache-8atqhb,
    .stCustomComponentV1,
    iframe[title*="streamlit_tags"] {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    /* Target the wrapper div around the iframe */
    div[class*="st-emotion-cache"] iframe[title*="streamlit_tags"] {
        border: none !important;
    }
    
    /* Remove border from the emotion cache wrapper */
    .st-emotion-cache-8atqhb {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }
    
    /* Enable text wrapping in dataframe cells */
    .stDataFrame [data-testid="stDataFrameResizable"] td {
        white-space: normal !important;
        word-wrap: break-word !important;
        max-width: 300px;
    }
    
    /* Specifically target Status column (5th column) for better word wrapping */
    .stDataFrame [data-testid="stDataFrameResizable"] td:nth-child(5) {
        white-space: normal !important;
        word-wrap: break-word !important;
        word-break: break-word !important;
        max-width: 400px;
        overflow-wrap: anywhere !important;
    }
    
    /* Input field styling */
    .streamlit-tags input,
    .stTags input,
    .ReactTags__tagInput {
        border: 1px solid #ddd !important;
        border-radius: 4px !important;
        outline: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # SAFE defaults (do not overwrite widget-managed keys after they're created)
    st.session_state.setdefault("verification_complete", False)
    st.session_state.setdefault("invoice_lines_preview", {})  # {po_id: DataFrame}
    st.session_state.setdefault("po_context", {})             # {po_id: {...ui state...}}
    st.session_state.setdefault("create_results", [])         # list of batch results
    st.session_state.setdefault("selected_for_batch", {})     # {po_id: bool}
    st.session_state.setdefault("po_prefix", "PO")

    # Sidebar (Environment, Debug, PO Prefix)
    with st.sidebar:
        st.header("Environment")
        st.selectbox(
            "Select Environment:",
            ["Test", "Production"],
            index=0 if st.session_state.get("environment", "Test") == "Test" else 1,
            key="environment"
        )
        
        # Show instance for current environment
        env_suffix = "_PROD" if st.session_state.get("environment", "Test") == "Production" else ""
        instance = os.environ.get(f"COUPA_INSTANCE{env_suffix}", "")
        st.caption(f"Instance: {instance or 'Not set'}")
        
        # Show production warning immediately
        if st.session_state.get("environment") == "Production":
            st.error("🛑 **PRODUCTION – Use with caution**")
        
        st.header("Document Processing")
        # Try to use st_tags if available, otherwise fall back to text input with visual tags
        try:
            if st_tags is not None:
                # Use sidebar-specific tags input widget
                current_prefixes = st_tags.st_tags_sidebar(
                label='PO Prefixes',
                text='Press enter to add more',
                value=_po_prefix() if 'po_prefix' in st.session_state else ['PO'],
                suggestions=['PO', 'PO', 'INV', 'REQ', 'ORD', 'PUR'],
                maxtags=10,
                key='po_prefix_tags'
            )
            
            # Update session state with the tag values
            if current_prefixes:
                st.session_state.po_prefix = ','.join(current_prefixes)
            else:
                st.session_state.po_prefix = 'PO'
                
        except ImportError:
            # Fall back to regular text input with tags displayed below
            st.text_input(
                "PO Prefixes", 
                key="po_prefix",
                help="Comma-separated list of PO prefixes to search for (e.g., REQ,PO,INV)",
                placeholder="PO,PO,INV"
            )
            
            # Display current prefixes as visual tags (no border)
            current_prefixes = _po_prefix()
            if current_prefixes:
                st.write("**Active Prefixes:**")
                # Create HTML tags for each prefix (borderless)
                tag_html = ""
                for prefix in current_prefixes:
                    tag_html += f'<span style="display: inline-block; background-color: #ff6b6b; color: white; padding: 4px 10px; margin: 2px; border-radius: 15px; font-size: 12px; font-weight: bold; border: none;">{prefix}</span>'
                st.markdown(tag_html, unsafe_allow_html=True)
                st.caption("💡 Install streamlit-tags for better tag input: `pip install streamlit-tags`")
        
        # Invoice Number Auto-Detection Settings
        st.toggle(
            "Auto-detect Invoice Numbers",
            value=st.session_state.get('auto_invoice_detect', True),
            key="auto_invoice_detect",
            help="Automatically extract invoice numbers from text patterns"
        )
        
        if st.session_state.get('auto_invoice_detect', True):
            try:
                if st_tags is not None:
                    invoice_patterns = st_tags.st_tags_sidebar(
                    label='Invoice Number Patterns',
                    text='Press enter to add patterns',
                    value=st.session_state.get('invoice_patterns', DEFAULT_INVOICE_PATTERNS),
                    suggestions=DEFAULT_INVOICE_PATTERNS + ['BILL NUMBER:', 'REFERENCE:'],
                    maxtags=10,
                    key='invoice_patterns_tags'
                )
                
                st.session_state.invoice_patterns = invoice_patterns
                
            except ImportError:
                st.text_input(
                    "Invoice Patterns", 
                    key="invoice_patterns_input",
                    value=','.join(DEFAULT_INVOICE_PATTERNS),
                    help="Comma-separated patterns to search for invoice numbers"
                )
                st.session_state.invoice_patterns = [p.strip() for p in st.session_state.get('invoice_patterns_input', '').split(',') if p.strip()]

        # Date Auto-Detection Settings
        st.toggle(
            "Auto-detect Dates",
            value=st.session_state.get('auto_date_detect', True),
            key="auto_date_detect",
            help="Automatically extract dates from document text and provide as options"
        )
        
        st.header("Invoice Creation")
        approval_option = st.selectbox(
            "Default Status",
            ["Save as Draft", "Submit for Approval"],
            help="Default status for created invoices",
            key="approval_option"
        )
                
        # Environment quick checks & caution banner
        env_suffix = "_PROD" if st.session_state.get("environment", "Test") == "Production" else ""
        instance = os.environ.get(f"COUPA_INSTANCE{env_suffix}", "")
        ident = os.environ.get(f"INV_IDENTIFIER{env_suffix}", "")
        secret = os.environ.get(f"INV_SECRET{env_suffix}", "")
        if not instance or not ident or not secret:
            st.warning("⚠️ Missing one or more required environment variables for the selected environment (COUPA_INSTANCE, INV_IDENTIFIER, INV_SECRET).")

        st.header("Debug")
        st.toggle(
            "Debug Mode",
            value=st.session_state.get('debug_enabled', False),
            key="debug_enabled",
            help="Show grouped API requests/responses and OCR debug info (auth redacted)"
        )

    # Main header + production banner (top-level)
    st.title("📄 miniSmash Invoice Creator")
    st.markdown("""
       
    **Automate your Coupa invoice creation process in 3 simple steps:**
    
    **1. 📤 Upload Files** → Upload PDF invoices, images, or ZIP files containing your invoice documents  
    **2. 🔍 Smart Processing** → Advanced OCR technology automatically finds PO numbers, extracts line items, and matches invoice data  
    **3. ✅ Verify & Create** → Real-time Coupa verification ensures accuracy before creating official invoices with complete audit trail
    
    This tool streamlines the entire invoice processing workflow, from document upload to final Coupa invoice creation, 
    with intelligent data extraction and comprehensive error checking to ensure accuracy and compliance.
    """)
    if st.session_state.get("environment") == "Production":
        st.error("🛑 **You are in Production. Use with caution.**")

    # Setup requirements dropdown
    with st.expander("📋 **Step 1: Setup Requirements** - Click to expand", expanded=False):
        st.markdown("""
        **Environment Variables Required:**
        
        *Test Environment:*
        - `COUPA_INSTANCE` - Your Coupa instance name (e.g., "mycompany")
        - `INV_IDENTIFIER` - Your OAuth client identifier
        - `INV_SECRET` - Your OAuth client secret
        
        *Production Environment:*
        - `COUPA_INSTANCE_PROD` - Production instance name
        - `INV_IDENTIFIER_PROD` - Production OAuth client identifier  
        - `INV_SECRET_PROD` - Production OAuth client secret
        
        **Required OAuth Scopes (Client Credentials Grant):**
        Your OAuth application in Coupa must have these scopes:
        - `core.purchase_order.read` - Read PO information
        - `core.invoice.read` - Check for duplicate invoices
        - `core.invoice.write` - Create new invoices
        
        **Python Packages:**
        
        **☁️ Cloud Deployment (Streamlit Cloud, Heroku, etc.):**
        ```
        pip install -r requirements-cloud.txt
        ```
        
        **💻 Full Local Development:**
        ```
        pip install -r requirements-full.txt
        ```
        
        **📦 Individual Packages:**
        ```
        # Core (cloud-friendly)
        pip install streamlit requests pandas pillow streamlit-tags
        
        # Advanced (local only - requires system dependencies)
        pip install camelot-py[cv] easyocr pdf2image
        ```
        
        **OCR:** Uses optimized EasyOCR with caching (models download automatically on first use - subsequent processing is much faster)
        
        **Sidebar Configuration:**
        Use the sidebar (left panel) to:
        - Switch between Test and Production environments
        - Configure PO prefixes (searches for prefix + 6 digits, e.g., PO123456, PO789012)
        - Enable/disable auto-detection for invoice numbers (uses patterns like "INVOICE NUMBER:", "INV:")
        - Enable/disable auto-detection for dates from document text
        - Turn on Debug Mode for detailed API logging, OCR troubleshooting, and performance information
        """)
        
        # Show current environment status
        st.markdown("")  # Add some space
        st.markdown("**Current Environment Status:**")
        current_env = st.session_state.get("environment", "Test")
        env_suffix = "_PROD" if current_env == "Production" else ""
        
        instance = os.environ.get(f"COUPA_INSTANCE{env_suffix}", "")
        identifier = os.environ.get(f"INV_IDENTIFIER{env_suffix}", "")
        secret = os.environ.get(f"INV_SECRET{env_suffix}", "")
        
        # Overall status message above the columns
        if not instance or not identifier or not secret:
            st.error("❌ **Setup incomplete!** Please set all required environment variables before proceeding.")
        else:
            st.success("✅ **Setup complete!** All required environment variables are configured.")
        
    
    
        status_cols = st.columns(3)
        with status_cols[0]:
            status_text = "✅ Set" if instance else "❌ Not Set"
            st.markdown(f"**Instance**")
            st.markdown(f"### {status_text}")
        with status_cols[1]:
            status_text = "✅ Set" if identifier else "❌ Not Set"  
            st.markdown(f"**Identifier**")
            st.markdown(f"### {status_text}")
        with status_cols[2]:
            status_text = "✅ Set" if secret else "❌ Not Set"
            st.markdown(f"**Secret**")
            st.markdown(f"### {status_text}")
            st.markdown("")  # Add some space
    
    # Usage guide dropdown
    with st.expander("📖 **Step 2: How to Use** - Click to expand", expanded=False):
        st.markdown("""
        **Step-by-Step Usage Guide:**
        
        **1. Configure Settings (Sidebar)** ⚙️
        - **Environment**: Select Test or Production environment
        - **PO Prefixes**: Set prefixes to search for (looks for prefix + 6 digits, e.g., PO123456, PO456789)
        - **Auto-detect Invoice Numbers**: Toggle automatic extraction using patterns like:
          - "INVOICE NUMBER:", "INVOICE NO:", "INVOICE#", "INVOICE #:", "INV:", "INVOICE:"
          - Looks for numbers immediately following these text patterns
        - **Auto-detect Dates**: Toggle automatic date extraction supporting formats like:
          - DD/MM/YYYY, MM/DD/YYYY, YYYY/MM/DD (with /, -, or . separators)
          - Month names: "January 15, 2024", "15 January 2024"
          - Compact formats: DDMMYYYY, YYYYMMDD
        - **Debug Mode**: Enable detailed API logging and OCR troubleshooting information
        
        **2. Upload Your Document** 📄
        - Upload a PDF invoice, image, or ZIP file containing multiple documents
        - **Performance**: OCR models are cached after first use - subsequent processing is much faster
        - **File Size**: Files >10MB will show warnings and may process slower
        - The system will automatically extract PO numbers and invoice details using optimized OCR
        - Supported formats: PDF, PNG, JPG, JPEG, TIF, TIFF, BMP, GIF, ZIP
        
        **3. Load PO Data from Coupa** �
        - Click "Load PO Data from Coupa" to retrieve PO details and validate the found PO numbers
        - **Fast Processing**: Uses concurrent verification for faster results with progress tracking
        - The system will automatically detect duplicate invoices in parallel
        - Review the verification results and any duplicate warnings
        
        **4. Edit Invoice Details** ✏️
        - Each verified PO will show an expandable section with status indicators
        - **Auto-populated fields**: Invoice numbers and dates are automatically detected from documents
        - **Manual editing**: Adjust invoice number, date, and tax rate in the input fields
        - **Line item editing**: Use the data editor table to modify quantities, prices, descriptions
        - **Add lines**: Use "➕ Add Qty" or "➕ Add Amt" buttons to add quantity or amount-based lines
        - **Delete lines**: Check the 🗑️ checkbox column in any row to delete it immediately
        - **Restore data**: Use "↩️ Restore All" to reset back to original extracted data
        
        **5. Review Batch Summary** 📊
        - The batch summary table shows all POs ready for invoice creation
        - **Include/exclude**: Use the "Include" checkbox column to select which invoices to create
        - **Automatic exclusions**: Duplicate invoices and those with errors are automatically unchecked for safety
        - **Edit inline**: You can modify invoice numbers directly in the summary table
        - Review totals, line counts, and ensure everything looks correct
        
        **6. Create Invoices** 🚀
        - Production users must check the acknowledgment box before proceeding
        - Click "Create Invoices" to process all selected (checked) POs
        - **Enhanced Progress Tracking**: Real-time progress with detailed stages:
          - Overall progress bar: "Processing X/Y: PO XXXXX"
          - Individual stages: "Authenticating...", "Creating invoice record...", "Uploading image scan (X.X MB)..."
          - **File Size Warnings**: Large files (>10MB) will show warnings and extended timeouts
        - **Smart Error Handling**: Detailed error messages with troubleshooting suggestions
        - Success/failure status displayed with comprehensive results for each invoice
        
        ### Key Features & Safety
        - **� High-Performance OCR**: Cached models for faster processing after first use
        - **📊 Real-Time Progress**: Detailed stage-by-stage progress tracking with file size warnings
        - **�🔄 Auto-detection**: Extracts invoice numbers, dates, and PO numbers from document text
        - **🔍 Smart Duplicate Prevention**: Concurrent duplicate checking with clickable links to existing invoices
        - **✏️ Full editing control**: Modify all invoice details before creation
        - **📊 Optimized Batch Processing**: Create multiple invoices efficiently with individual progress tracking  
        - **🛡️ Enhanced Safety Features**: Production warnings, duplicate detection, and confirmation steps
        - **🐛 Comprehensive Debug Mode**: Detailed API logging, OCR processing info, and performance metrics
        
        ### Performance & Troubleshooting Tips
        - **First-time setup**: OCR model loading takes 30-60 seconds initially, then cached for fast reuse
        - **File sizes**: Keep documents under 10MB for best performance - larger files show warnings
        - **Debug mode**: Enable for detailed processing information, performance metrics, and troubleshooting
        - **Use clear documents**: High-quality PDFs and images give better text extraction results
        - **Check auto-detected data**: Always verify extracted invoice numbers and dates
        - **Test first**: Use Test environment before switching to Production
        - **Review duplicates**: Pay attention to duplicate warnings with clickable links to existing invoices
        - **Use Restore**: If you make mistakes editing lines, use "↩️ Restore All" to start over
        - **Monitor progress**: Watch detailed stage messages to understand what's happening during processing
        """)

    # File upload (PDF, images, or ZIP supported with OCR)
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "zip", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif"],
        help="Upload a PDF, image, or ZIP containing PDFs/images"
    )

    if uploaded_file:
        # Create content-based hash for better caching (same content = same results)
        file_content = uploaded_file.getvalue()
        content_hash = hashlib.md5(file_content).hexdigest()
        file_id = f"{uploaded_file.name}_{len(file_content)}_{content_hash[:8]}"
        
        # Check if we've already processed this content (content-based caching)
        if (st.session_state.get("last_processed_file_id") == file_id and 
            "processed_po_numbers" in st.session_state):
            # Use cached results
            found_pos = st.session_state.processed_po_numbers
            if is_debug():
                with st.expander(f"🔍 Cache Debug – {uploaded_file.name}", expanded=False):
                    st.write("**Cache Status**: ✅ Content Hit (same file content)")
                    st.write(f"**Content Hash**: {content_hash[:12]}...")
                    st.write(f"**Cached PO Count**: {len(found_pos)}")
                    st.write("**Performance**: Skipped all OCR/PDF processing")
        else:
            # Process the file (reset file pointer after reading for hash)
            uploaded_file.seek(0)
            file_lower = uploaded_file.name.lower()
            is_zip = file_lower.endswith('.zip')
            is_pdf = file_lower.endswith('.pdf')
            is_image = file_lower.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'))

            with st.spinner(f"Processing {'ZIP' if is_zip else 'PDF' if is_pdf else 'Image'} file..."):
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
                elif is_image:
                    found_pos = extract_po_numbers_from_image(uploaded_file)
                    st.session_state.original_file_data = uploaded_file.getvalue()
                    st.session_state.original_file_name = uploaded_file.name
                    st.session_state.file_data_map = {}
                else:
                    found_pos = []
            
            # Cache the results
            st.session_state.last_processed_file_id = file_id
            st.session_state.processed_po_numbers = found_pos

        if found_pos:
            unique_pos = list(set(found_pos))
            
            # Show found information in expandable section with dataframe
            with st.expander(f"📄 Found Information ({len(unique_pos)} unique PO{'s' if len(unique_pos) != 1 else ''})", expanded=False):
                # Create dataframe with found information
                # Get detected data
                detected_invoice_numbers = st.session_state.get("detected_invoice_numbers", [])
                detected_invoice = st.session_state.get("detected_invoice_number", "")
                detected_dates = st.session_state.get("detected_dates", [])
                detected_date = st.session_state.get("detected_invoice_date")
                
                # Get filename (handle different file types)
                filename = "Unknown"
                if hasattr(st.session_state, 'current_filename'):
                    filename = st.session_state.current_filename
                elif 'uploaded_file' in locals() and uploaded_file:
                    filename = uploaded_file.name
                
                # Create rows for the table
                rows = []
                
                # Check if this is ZIP file data (per-file extraction)
                zip_data = st.session_state.get("zip_file_extraction_data", {})
                if zip_data:
                    # ZIP file: show data per file within the ZIP
                    for filename, data in zip_data.items():
                        file_pos = data.get('pos', [])
                        file_invoices = data.get('invoice_numbers', [])
                        file_dates = data.get('dates', [])
                        
                        # Show all POs found in this file (don't filter by unique_pos)
                        if file_pos:
                            for po in file_pos:
                                # Best invoice number for this file
                                invoice_display = ""
                                if file_invoices:
                                    best_invoice = file_invoices[0]  # Take first invoice
                                    invoice_display = best_invoice
                                    if len(file_invoices) > 1:
                                        invoice_display += f" (+ {len(file_invoices)-1} more)"
                                
                                # Best date for this file  
                                date_display = ""
                                if file_dates:
                                    best_date = file_dates[0]  # Take first date (now priority invoice date)
                                    date_display = best_date.strftime("%d/%m/%Y")
                                    if len(file_dates) > 1:
                                        date_display += f" (+ {len(file_dates)-1} more)"
                                
                                rows.append({
                                    "📁 File": filename,
                                    "📦 PO #": po,
                                    "📄 Invoice #": invoice_display if invoice_display else "Not detected",
                                    "📅 Invoice Date": date_display if date_display else "Not detected"
                                })
                        else:
                            # Show file even if no POs found
                            rows.append({
                                "📁 File": filename,
                                "📦 PO #": "No POs found",
                                "📄 Invoice #": "N/A",
                                "📅 Invoice Date": "N/A"
                            })
                else:
                    # Single file: use global detected data
                    detected_invoice_numbers = st.session_state.get("detected_invoice_numbers", [])
                    detected_invoice = st.session_state.get("detected_invoice_number", "")
                    detected_dates = st.session_state.get("detected_dates", [])
                    detected_date = st.session_state.get("detected_invoice_date")
                    
                    # Get filename (handle different file types)
                    filename = "Unknown"
                    if hasattr(st.session_state, 'current_filename'):
                        filename = st.session_state.current_filename
                    elif 'uploaded_file' in locals() and uploaded_file:
                        filename = uploaded_file.name
                    
                    for po in unique_pos:
                        # Determine best invoice number for this row
                        invoice_display = ""
                        if detected_invoice:
                            invoice_display = detected_invoice
                            if len(detected_invoice_numbers) > 1:
                                invoice_display += f" (selected from {len(detected_invoice_numbers)})"
                        elif detected_invoice_numbers:
                            invoice_display = detected_invoice_numbers[0]
                        
                        # Determine best date for this row
                        date_display = ""
                        if detected_date:
                            date_display = detected_date.strftime("%d/%m/%Y")
                            if len(detected_dates) > 1:
                                date_display += f" (most recent of {len(detected_dates)})"
                        
                        rows.append({
                            "📁 File": filename,
                            "📦 PO #": po,
                            "📄 Invoice #": invoice_display if invoice_display else "Not detected",
                            "📅 Invoice Date": date_display if date_display else "Not detected"
                        })
                
                # Display the dataframe
                if rows:
                    df = pd.DataFrame(rows)
                    st.dataframe(
                        df,
                        width='stretch',
                        hide_index=True,
                        column_config={
                            "📁 File": st.column_config.TextColumn("📁 File", width="medium"),
                            "📦 PO #": st.column_config.TextColumn("📦 PO #", width="medium"),
                            "📄 Invoice #": st.column_config.TextColumn("📄 Invoice #", width="large"),
                            "📅 Invoice Date": st.column_config.TextColumn("📅 Invoice Date", width="medium")
                        }
                    )
                else:
                    st.info("No data to display")

            # Verify
            if st.button("Find Matching POs in Coupa", type="primary"):
                # Auth for all operations (PO read + invoice read/write)
                success, token_or_error = get_oauth_token(OAUTH_SCOPE_ALL)
                if not success:
                    clear_oauth_token_cache()  # Clear cache on auth failure
                    st.error(f"Auth failed: {token_or_error}")
                    return
                
                config = get_env_config()
                headers = {"Authorization": f"Bearer {token_or_error}", "Accept": "application/json"}

                # Load PO data concurrently (faster!)
                with st.spinner(f"🔍 Verifying {len(unique_pos)} POs in Coupa..."):
                    # Use concurrent verification for speed
                    verification_results = verify_pos_concurrent(unique_pos, headers, config['instance'])
                    
                    verified_pos = []
                    zip_data = st.session_state.get("zip_file_extraction_data", {})
                    
                    for po_id, result in verification_results.items():
                        if result['exists']:
                            po_entry = {'po_id': po_id, 'po_data': result['po_data']}
                            
                            # Check if this is a combined PO ID (PO@Invoice format)
                            real_po_id = _extract_real_po_id(po_id)
                            invoice_from_combined = _extract_invoice_from_combined(po_id)
                            
                            # For ZIP files, find which file this PO came from and attach the file-specific data
                            if zip_data:
                                # Look for the combination data we stored during ZIP processing
                                po_combinations = st.session_state.get('po_invoice_combinations', [])
                                
                                # Find the specific combination for this combined PO ID
                                matching_combo = None
                                for combo in po_combinations:
                                    if combo['po_id'] == real_po_id and combo['invoice_number'] == invoice_from_combined:
                                        matching_combo = combo
                                        break
                                
                                if matching_combo:
                                    # Use data from the specific file for this combination
                                    source_file = matching_combo['source_file']
                                    po_entry['source_file'] = source_file
                                    po_entry['file_invoice_numbers'] = [matching_combo['invoice_number']]  # Single invoice per combination
                                    po_entry['file_dates'] = matching_combo['dates']
                                    po_entry['file_bytes'] = st.session_state.file_data_map.get(source_file)
                                    
                                    if is_debug():
                                        # Extract clean IDs for display
                                        display_po_id = po_id.replace('@', ' - ')
                                        with st.expander(f"🔍 Debug: File Processing & PO Matching - {display_po_id}", expanded=False):
                                            st.write("**File-to-PO Matching Results:**")
                                            
                                            # Format dates for display
                                            formatted_dates = []
                                            if matching_combo['dates']:
                                                for date_str in matching_combo['dates']:
                                                    try:
                                                        # Try to parse and format common date formats
                                                        if isinstance(date_str, str) and len(date_str) >= 8:
                                                            # Handle common formats like YYYY-MM-DD, MM/DD/YYYY, etc.
                                                            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y%m%d', '%m-%d-%Y']:
                                                                try:
                                                                    parsed_date = datetime.strptime(date_str, fmt)
                                                                    formatted_dates.append(parsed_date.strftime('%Y-%m-%d'))
                                                                    break
                                                                except:
                                                                    continue
                                                            else:
                                                                formatted_dates.append(date_str)  # Keep original if parsing fails
                                                        else:
                                                            formatted_dates.append(str(date_str))
                                                    except:
                                                        formatted_dates.append(str(date_str))
                                            
                                            # Create clean DataFrame
                                            matching_data = {
                                                "Field": ["PO Number", "Invoice Number", "Source PDF", "Extracted Dates", "Match Status"],
                                                "Value": [
                                                    real_po_id,
                                                    matching_combo['invoice_number'],
                                                    source_file,
                                                    ", ".join(formatted_dates) if formatted_dates else "No dates found",
                                                    "✅ Matched Successfully"
                                                ]
                                            }
                                            st.dataframe(pd.DataFrame(matching_data), use_container_width=True, hide_index=True)
                                else:
                                    # Fallback: search through all files for this real PO
                                    for filename, file_data in zip_data.items():
                                        if real_po_id in file_data.get('pos', []):
                                            po_entry['source_file'] = filename
                                            po_entry['file_invoice_numbers'] = file_data.get('invoice_numbers', [])
                                            po_entry['file_dates'] = file_data.get('dates', [])
                                            po_entry['file_bytes'] = st.session_state.file_data_map.get(filename)
                                            if is_debug():
                                                debug_po_display = po_id.replace('@', ' - ')
                                                with st.expander(f"🔍 Debug: Fallback PO Search - {debug_po_display}", expanded=False):
                                                    st.write("**Fallback Search Results:**")
                                                    
                                                    # Format dates for display
                                                    file_dates = file_data.get('dates', [])
                                                    formatted_dates = []
                                                    if file_dates:
                                                        for date_str in file_dates:
                                                            try:
                                                                if isinstance(date_str, str) and len(date_str) >= 8:
                                                                    for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y%m%d', '%m-%d-%Y']:
                                                                        try:
                                                                            parsed_date = datetime.strptime(date_str, fmt)
                                                                            formatted_dates.append(parsed_date.strftime('%Y-%m-%d'))
                                                                            break
                                                                        except:
                                                                            continue
                                                                    else:
                                                                        formatted_dates.append(date_str)
                                                                else:
                                                                    formatted_dates.append(str(date_str))
                                                            except:
                                                                formatted_dates.append(str(date_str))
                                                    
                                                    # Create clean DataFrame
                                                    fallback_data = {
                                                        "Field": ["PO Number", "Found in PDF", "Invoice Numbers", "Extracted Dates", "Search Type"],
                                                        "Value": [
                                                            real_po_id,
                                                            filename,
                                                            ", ".join(file_data.get('invoice_numbers', [])) if file_data.get('invoice_numbers') else "None found",
                                                            ", ".join(formatted_dates) if formatted_dates else "No dates found",
                                                            "🔍 Fallback Search"
                                                        ]
                                                    }
                                                    st.dataframe(pd.DataFrame(fallback_data), use_container_width=True, hide_index=True)
                                            break
                            
                            verified_pos.append(po_entry)

                # Check for duplicate invoices for verified POs (reuse same token)
                if verified_pos:
                    # Reuse the same token (has all required permissions)
                    if success:
                        # Safety check - limit number of items to prevent browser crashes
                        max_duplicate_checks = 50  # Reasonable limit for browser stability
                        if len(verified_pos) > max_duplicate_checks:
                            st.warning(f"⚠️ Large batch detected ({len(verified_pos)} POs). Duplicate checking limited to first {max_duplicate_checks} items for browser stability.")
                            st.info("💡 Consider processing in smaller batches for better performance.")
                        
                        # Use spinner for the duplicate check message and put the logic inside
                        with st.spinner(f"🔍 Checking {min(len(verified_pos), max_duplicate_checks)} PO{'s' if len(verified_pos) != 1 else ''} for duplicate invoices..."):
                            # Create progress tracking for duplicate checks
                            dup_progress = st.progress(0)
                            dup_status = st.empty()
                                
                            headers_dup = {"Authorization": f"Bearer {token_or_error}", "Accept": "application/json"}
                        
                        # Collect invoice data for all verified POs - use CACHED invoice numbers (with deduplication)
                        invoice_data_list = []
                        max_duplicate_checks = 50  # Safety limit
                        checked_invoices = {}  # Deduplicate same invoice numbers
                        
                        for po_entry in verified_pos[:max_duplicate_checks]:  # Apply limit here
                            po_id = po_entry['po_id']
                            po_data = po_entry['po_data']
                            
                            # Get cached invoice number from PO context (the fix we made)
                            cached_invoice = st.session_state.po_context.get(po_id, {}).get("invoice_number", "")
                            
                            # Fallback to detected or file invoice numbers if no cached value
                            if not cached_invoice:
                                po_invoice_numbers = po_entry.get('file_invoice_numbers', [])
                                if not po_invoice_numbers:
                                    po_invoice_numbers = [st.session_state.get("detected_invoice_number", "")]
                                cached_invoice = po_invoice_numbers[0] if po_invoice_numbers else ""
                            
                            supplier_id = po_data.get("supplier", {}).get("id")
                            supplier_name = po_data.get("supplier", {}).get("name")
                            
                            # Add to check list if we have required data
                            if cached_invoice and cached_invoice.strip() and supplier_id:
                                invoice_key = f"{cached_invoice.strip()}_{supplier_id}"
                                
                                # Only check each unique invoice+supplier combination once
                                if invoice_key not in checked_invoices:
                                    invoice_data_list.append({
                                        "po_id": po_id,
                                        "invoice_number": cached_invoice.strip(),
                                        "supplier_id": supplier_id,
                                        "supplier_name": supplier_name,
                                        "invoice_key": invoice_key
                                    })
                                    checked_invoices[invoice_key] = po_id
                        
                        # Check for duplicates with progress tracking (optimized for browser stability)
                        if invoice_data_list:
                            duplicate_results = {}
                            total_items = len(invoice_data_list)
                            
                            for idx, data in enumerate(invoice_data_list):
                                po_id = data["po_id"]
                                invoice_number = data["invoice_number"]
                                supplier_id = data["supplier_id"]
                                supplier_name = data.get("supplier_name")
                                
                                # Update progress less frequently to prevent browser overload
                                if idx % max(1, total_items // 10) == 0 or idx == total_items - 1:
                                    progress = (idx + 1) / total_items
                                    dup_progress.progress(progress)
                                    dup_status.text(f"Checking {idx + 1}/{total_items}: PO {po_id}")
                                    # Small delay after UI updates to prevent browser overload
                                    time.sleep(0.1)
                                
                                if not invoice_number or not supplier_id:
                                    duplicate_results[po_id] = {
                                        "is_duplicate": False,
                                        "message": "Skipped - missing invoice number or supplier ID",
                                        "has_error": False
                                    }
                                    continue
                                
                                try:
                                    is_duplicate, message, has_error, existing_invoice_id = check_invoice_duplicate(
                                        invoice_number, supplier_id, headers_dup, config['instance'], supplier_name
                                    )
                                    
                                    duplicate_results[po_id] = {
                                        "is_duplicate": is_duplicate,
                                        "message": message,
                                        "has_error": has_error,
                                        "existing_invoice_id": existing_invoice_id
                                    }
                                except Exception as e:
                                    # Handle API errors gracefully to prevent crashes and endless loops
                                    duplicate_results[po_id] = {
                                        "is_duplicate": False,
                                        "message": f"Check failed: {str(e)}",
                                        "has_error": True,
                                        "needs_recheck": False  # CRITICAL: Prevent endless loop on API failure
                                    }
                                
                                # Minimal delay between API calls
                                time.sleep(0.05)
                                
                            # Store duplicate info with each PO (handle deduplication)
                            for po_entry in verified_pos:
                                po_id = po_entry['po_id']
                                po_data = po_entry['po_data']
                                
                                # Get this PO's invoice info to find matching result
                                cached_invoice = st.session_state.po_context.get(po_id, {}).get("invoice_number", "")
                                if not cached_invoice:
                                    cached_invoice = st.session_state.get("detected_invoice_number", "")
                                
                                supplier_id = po_data.get("supplier", {}).get("id")
                                invoice_key = f"{cached_invoice.strip()}_{supplier_id}" if cached_invoice and supplier_id else None
                                
                                # Find result by matching checked invoice key or direct PO ID
                                result = None
                                if invoice_key and invoice_key in checked_invoices:
                                    # Use result from the PO that was actually checked
                                    checked_po_id = checked_invoices[invoice_key]
                                    result = duplicate_results.get(checked_po_id)
                                
                                # Fallback to direct lookup or default
                                if not result:
                                    result = duplicate_results.get(po_id, {
                                        "is_duplicate": False, 
                                        "message": "Not checked - duplicate invoice/supplier", 
                                        "has_error": False,
                                        "needs_recheck": False  # Prevent loops for non-checked items
                                    })
                                
                                po_entry['duplicate_check'] = result
                            
                            # Cleanup progress indicators
                            time.sleep(0.5)  # Brief pause to show completion
                            dup_progress.empty()
                            dup_status.empty()
                        else:
                            st.info("✅ No invoice numbers found for duplicate checking")
                    else:
                        # Add empty duplicate check info when auth fails - PREVENT ENDLESS LOOPS
                        for po_entry in verified_pos:
                            po_entry['duplicate_check'] = {
                                "is_duplicate": False, 
                                "message": f"Check failed: Authentication error", 
                                "has_error": True,
                                "needs_recheck": False  # CRITICAL: Prevent endless loop on auth failure
                            }

                st.session_state.verified_pos = verified_pos
                st.session_state.verification_complete = True
                st.session_state.create_results = []       # reset batch results
                st.session_state.selected_for_batch = {}   # reset selection
                
                # Now set proper defaults based on duplicate status
                for po_entry in verified_pos:
                    po_id = po_entry['po_id']
                    duplicate_info = po_entry.get('duplicate_check', {})
                    is_duplicate = duplicate_info.get('is_duplicate', False)
                    has_error = duplicate_info.get('has_error', False)
                    
                    # Check if PO has invoice number
                    po_context = st.session_state.po_context.get(po_id, {})
                    invoice_num = po_context.get("invoice_number", "")
                    has_invoice = bool(invoice_num and invoice_num.strip())
                    
                    # Auto-untick duplicates and errors, but allow invoices with valid data
                    if has_invoice and not is_duplicate and not has_error:
                        st.session_state.selected_for_batch[po_id] = True
                    else:
                        st.session_state.selected_for_batch[po_id] = False
        else:
            if is_debug():
                st.warning("No PO numbers found. Check the debug information above for details.")
            else:
                st.warning("No PO numbers found. Enable Debug Mode to see extraction details.")
    else:
        # Clear cache when no file is uploaded
        if "last_processed_file_id" in st.session_state:
            del st.session_state.last_processed_file_id
        if "processed_po_numbers" in st.session_state:
            del st.session_state.processed_po_numbers

    # Verified POs – Invoice Generation editors (no create buttons here)
    if st.session_state.get("verification_complete") and st.session_state.get("verified_pos"):
        st.markdown("---")
        st.subheader("✅ Verified POs – Invoice Generation")

        # Handle any needed duplicate re-checks FIRST (before showing PO sections)
        pos_needing_recheck = []
        for po_info in st.session_state.verified_pos:
            po_id = po_info['po_id']
            duplicate_info = po_info.get('duplicate_check', {})
            needs_recheck = duplicate_info.get('needs_recheck', False)
            current_inv_num = st.session_state.po_context.get(po_id, {}).get("invoice_number", "")
            
            if needs_recheck and current_inv_num and current_inv_num.strip():
                pos_needing_recheck.append(po_info)
        
        # Batch process duplicate re-checks with spinner
        if pos_needing_recheck:
            recheck_spinner = st.empty()
            with recheck_spinner.container():
                with st.spinner(f"🔍 Re-checking {len(pos_needing_recheck)} invoice number{'s' if len(pos_needing_recheck) != 1 else ''} for duplicates..."):
                    time.sleep(0.1)  # Brief pause to show spinner
            
            # Create progress tracking for duplicate re-checks
            recheck_progress = st.progress(0)
            recheck_status = st.empty()
            
            try:
                # Get auth token for duplicate check
                success_dup, token_dup_or_error = get_oauth_token(OAUTH_SCOPE_ALL)
                config = get_env_config()
                
                if success_dup:
                    headers_dup = {"Authorization": f"Bearer {token_dup_or_error}", "Accept": "application/json"}
                    
                    for idx, po_info in enumerate(pos_needing_recheck):
                        po_id = po_info['po_id']
                        po_data = po_info['po_data']
                        current_inv_num = st.session_state.po_context.get(po_id, {}).get("invoice_number", "")
                        
                        # Update progress
                        progress = (idx + 1) / len(pos_needing_recheck)
                        recheck_progress.progress(progress)
                        recheck_status.text(f"Checking PO {po_id}: {current_inv_num}")
                        
                        # Get supplier info for this PO
                        supplier_id = po_data.get("supplier", {}).get("id") if po_data else None
                        supplier_name = po_data.get("supplier", {}).get("name") if po_data else None
                        
                        if supplier_id and current_inv_num:
                            try:
                                is_duplicate_new, message_new, has_error_new, existing_invoice_id_new = check_invoice_duplicate(
                                    current_inv_num, supplier_id, headers_dup, config['instance'], supplier_name
                                )
                                
                                # Update the duplicate check results
                                duplicate_info = {
                                    "is_duplicate": is_duplicate_new,
                                    "message": message_new,
                                    "has_error": has_error_new,
                                    "existing_invoice_id": existing_invoice_id_new,
                                    "needs_recheck": False
                                }
                                po_info['duplicate_check'] = duplicate_info
                                
                                # Update selection based on new results
                                if is_duplicate_new or has_error_new:
                                    st.session_state.selected_for_batch[po_id] = False
                                else:
                                    st.session_state.selected_for_batch[po_id] = True
                                    
                            except Exception as func_e:
                                duplicate_info = {
                                    "is_duplicate": False,
                                    "message": f"Re-check failed: {str(func_e)}",
                                    "has_error": True,
                                    "needs_recheck": False
                                }
                                po_info['duplicate_check'] = duplicate_info
                        else:
                            duplicate_info = {
                                "is_duplicate": False,
                                "message": "No supplier ID or invoice number for duplicate check",
                                "has_error": True,
                                "needs_recheck": False
                            }
                            po_info['duplicate_check'] = duplicate_info
                else:
                    # Mark all as failed to check
                    for po_info in pos_needing_recheck:
                        duplicate_info = {
                            "is_duplicate": False,
                            "message": "API credentials not available",
                            "has_error": True,
                            "needs_recheck": False
                        }
                        po_info['duplicate_check'] = duplicate_info
                        
                # Complete and clean up progress indicators
                recheck_progress.progress(1.0)
                recheck_status.text("✅ Duplicate checking complete")
                time.sleep(1)  # Brief pause to show completion
                recheck_progress.empty()
                recheck_status.empty()
                        
            except Exception as e:
                # Mark all as failed to check
                for po_info in pos_needing_recheck:
                    duplicate_info = {
                        "is_duplicate": False,
                        "message": f"Re-check failed: {str(e)}",
                        "has_error": True,
                        "needs_recheck": False
                    }
                    po_info['duplicate_check'] = duplicate_info
                    
                # Clean up progress indicators on error
                recheck_progress.empty()
                recheck_status.empty()

        for po_info in st.session_state.verified_pos:
            po_id = po_info['po_id']
            po_data = po_info['po_data']

            # Document source resolution - handle ZIP files with per-PO file mapping
            doc_name = None
            doc_bytes = None
            
            # Check if this PO has specific file data (from ZIP processing)
            po_source_file = po_info.get('source_file')
            po_file_bytes = po_info.get('file_bytes')
            
            if po_source_file and po_file_bytes:
                    doc_name = next(iter(st.session_state.file_data_map.keys()))
                    doc_bytes = st.session_state.file_data_map[doc_name]
            else:
                # Single file upload
                if st.session_state.get('original_file_data'):
                    doc_bytes = st.session_state.original_file_data
                    doc_name = st.session_state.get('original_file_name', 'invoice.pdf')

            # Get file-specific detected data for this PO
            po_invoice_numbers = po_info.get('file_invoice_numbers', [])
            po_dates = po_info.get('file_dates', [])
            
            # Use file-specific data if available, otherwise fall back to global
            if po_invoice_numbers:            
                detected_invoice = max(po_invoice_numbers, key=len) if len(po_invoice_numbers) > 1 else po_invoice_numbers[0]
            else:
                detected_invoice = st.session_state.get("detected_invoice_number", "")
            
            # Prioritize the smart-extracted date over just picking the latest date
            detected_date = st.session_state.get("detected_invoice_date")
            if not detected_date and po_dates:
                detected_date = max(po_dates)  # Fallback to most recent date if no prioritized date
                
            default_date = detected_date if detected_date else datetime.today().date()
            
            st.session_state.po_context.setdefault(po_id, {
                "doc_bytes": doc_bytes,
                "doc_name": doc_name,
                "invoice_number": detected_invoice if detected_invoice else "",  # Auto-populate with OCR detected value
                "invoice_date": datetime.today().date(),  # Always start with today's date
                "tax_rate": 10.0,  # default GST-A 10%
                "detected_invoice": detected_invoice,  # Store for auto-population
                "detected_date": detected_date
            })

            # Default batch selection based on invoice number presence AND duplicate status
            current_inv_num = st.session_state.po_context[po_id].get("invoice_number", "")
            detected_invoice = st.session_state.get("detected_invoice_number", "")
            has_invoice = bool((current_inv_num and current_inv_num.strip()) or detected_invoice)
            
            # Check duplicate status from verification (re-checks are handled at top level)
            duplicate_info = po_info.get('duplicate_check', {})
            
            is_duplicate = duplicate_info.get('is_duplicate', False)
            has_dup_error = duplicate_info.get('has_error', False)
            
            # Default to include ONLY if has invoice AND no duplicates AND no errors
            default_include = has_invoice and not is_duplicate and not has_dup_error
            
            # Always update selection based on current duplicate status (don't just use setdefault)
            if po_id not in st.session_state.selected_for_batch:
                st.session_state.selected_for_batch[po_id] = default_include
            else:
                # If this PO was previously selected but now has duplicates/errors, auto-untick it
                current_selection = st.session_state.selected_for_batch[po_id]
                if current_selection and (is_duplicate or has_dup_error):
                    st.session_state.selected_for_batch[po_id] = False
            
            # Simple status indicator in header  
            status_icon = ""
            status = ""
            if is_duplicate:
                status = " [DUPLICATE INVOICE]"
                status_icon = "📑 "
            elif has_dup_error:
                status = " [CHECK FAILED]"
                status_icon = "❌ "
            elif duplicate_info.get('message', '') and 'No duplicate found' in duplicate_info.get('message', ''):
                status = " [INVOICE OK]"
                status_icon  = "✔️ "

            # Create clean display name for tabs: replace @ with " - "
            display_po_id = po_id.replace('@', ' - ')
            
            with st.expander(f"{status_icon}{display_po_id}{status}", expanded=True):
                # Debug information if debug mode is enabled - MOVED UP for better visibility
                if is_debug():
                    debug_po_display = po_id.replace('@', ' - ')
                    with st.expander(f"🔍 Debug: PO Processing Status - {debug_po_display}", expanded=False):
                        # Create tabs for organized debug information with Request/Response format
                        debug_tab1, debug_tab2, debug_tab3, debug_tab4 = st.tabs(["📊 Processing Status", " API Request", "📨 API Response", "🔧 Actions"])

                        with debug_tab1:
                            # Invoice Processing Status Table
                            processing_data = {
                                "Field": ["Current Invoice", "Has Invoice", "Is Duplicate", "Has Error", "Selected for Batch"],
                                "Value": [
                                    current_inv_num or "None",
                                    "✅ Yes" if has_invoice else "❌ No",
                                    "⚠️ Yes" if is_duplicate else "✅ No",
                                    "⚠️ Yes" if has_dup_error else "✅ No",
                                    st.session_state.selected_for_batch.get(po_id, 'Not set')
                                ]
                            }
                            st.dataframe(pd.DataFrame(processing_data), use_container_width=True, hide_index=True)

                            if duplicate_info.get('message', 'No message') != 'No message':
                                duplicate_msg_data = {
                                    "Type": ["Duplicate Check Message"],
                                    "Details": [duplicate_info.get('message')]
                                }
                                st.dataframe(pd.DataFrame(duplicate_msg_data), use_container_width=True, hide_index=True)

                        with debug_tab2:
                            # API Request Information in DataFrame format
                            po_entry = next((p for p in st.session_state.verified_pos if p['po_id'] == po_id), None)
                            real_po_id = _extract_real_po_id(po_id)
                            numeric_po_id = _numeric_po_id(real_po_id)

                            st.write("**🔗 PO Data Loading API Request:**")
                            api_request_data = {
                                "Field": ["URL", "Method", "PO ID", "Numeric PO ID", "Purpose", "Timeout", "Authentication"],
                                "Value": [
                                    f"https://{st.session_state.get('environment', 'sandbox-api')}.coupahost.com/api/purchase_orders/{numeric_po_id}",
                                    "GET",
                                    real_po_id,
                                    numeric_po_id,
                                    "Load PO details from Coupa",
                                    "30 seconds",
                                    "Bearer token (OAuth 2.0)"
                                ]
                            }
                            st.dataframe(pd.DataFrame(api_request_data), use_container_width=True, hide_index=True)

                            # Request Parameters
                            st.write("**📋 Request Parameters:**")
                            params_data = {
                                "Parameter": ["fields", "format", "include"],
                                "Value": [
                                    "id, number, status, total, supplier, currency, order-lines",
                                    "JSON",
                                    "supplier details, currency, order-lines with full details"
                                ]
                            }
                            st.dataframe(pd.DataFrame(params_data), use_container_width=True, hide_index=True)

                        with debug_tab3:
                            # API Response Information in DataFrame format
                            if po_entry and po_entry.get('po_data'):
                                st.success("✅ **API Response Status**: SUCCESS (200 OK)")
                                po_data = po_entry['po_data']

                                # Extract response details
                                po_number_display = po_data.get('number') or real_po_id
                                supplier = po_data.get('supplier', {})
                                currency = po_data.get('currency', {})
                                order_lines = po_data.get('order-lines', [])

                                # API Response Summary
                                st.write("**📊 Response Summary:**")
                                response_summary_data = {
                                    "Metric": ["HTTP Status", "Response Format", "PO Found", "Data Completeness", "Lines Loaded", "Supplier Info", "Currency Info"],
                                    "Value": [
                                        "200 OK",
                                        "JSON",
                                        "✅ Found" if po_data.get('id') else "❌ Not Found",
                                        "✅ Complete" if (supplier.get('name') and len(order_lines) > 0) else "⚠️ Incomplete",
                                        f"{len(order_lines)} lines",
                                        "✅ Available" if supplier.get('name') else "❌ Missing",
                                        "✅ Available" if currency.get('code') else "❌ Missing"
                                    ]
                                }
                                st.dataframe(pd.DataFrame(response_summary_data), use_container_width=True, hide_index=True)

                                # PO Details from API Response
                                st.write("**📋 PO Details from Response:**")
                                po_details_data = {
                                    "Field": ["PO Number", "Status", "Supplier Name", "Supplier ID", "Currency", "Total Amount", "Order Lines"],
                                    "Value": [
                                        po_number_display,
                                        po_data.get('status', 'Unknown'),
                                        supplier.get('name', 'Unknown'),
                                        str(supplier.get('id', 'Unknown')),
                                        currency.get('code', 'Unknown'),
                                        str(po_data.get('total', 'Unknown')),
                                        f"{len(order_lines)} lines"
                                    ]
                                }
                                st.dataframe(pd.DataFrame(po_details_data), use_container_width=True, hide_index=True)

                                # Response Quality Check
                                st.write("**✅ Data Quality Assessment:**")
                                quality_data = {
                                    "Component": ["PO ID", "Supplier Data", "Currency Data", "Line Items", "Overall Quality"],
                                    "Status": [
                                        "✅ Valid" if po_data.get('id') else "❌ Missing",
                                        "✅ Complete" if supplier.get('name') and supplier.get('id') else "⚠️ Partial",
                                        "✅ Complete" if currency.get('code') else "❌ Missing",
                                        "✅ Loaded" if len(order_lines) > 0 else "❌ Empty",
                                        "✅ Excellent" if all([po_data.get('id'), supplier.get('name'), len(order_lines) > 0]) else "⚠️ Issues Found"
                                    ]
                                }
                                st.dataframe(pd.DataFrame(quality_data), use_container_width=True, hide_index=True)

                            else:
                                st.error("❌ **API Response Status**: FAILED or NOT LOADED")

                                # Error Analysis
                                if po_entry and po_entry.get('error'):
                                    st.write("**❌ Error Details:**")
                                    error_details_data = {
                                        "Field": ["Error Type", "Error Message", "Suggested Action", "HTTP Status"],
                                        "Value": [
                                            "API Call Failure",
                                            po_entry.get('error'),
                                            "Verify PO exists in Coupa and retry",
                                            "Unknown (check network/auth)"
                                        ]
                                    }
                                    st.dataframe(pd.DataFrame(error_details_data), use_container_width=True, hide_index=True)
                                elif po_entry:
                                    st.write("**🔍 Troubleshooting Information:**")
                                    troubleshoot_data = {
                                        "Check": ["PO Entry Exists", "Has Error Info", "Data Object Present", "Likely Cause"],
                                        "Result": [
                                            "✅ Yes",
                                            "✅ No Explicit Error",
                                            "❌ Missing Data",
                                            "Silent API failure or timeout"
                                        ]
                                    }
                                    st.dataframe(pd.DataFrame(troubleshoot_data), use_container_width=True, hide_index=True)
                                else:
                                    st.write("**⚠️ Initial Status:**")
                                    status_data = {
                                        "Status": ["No PO Entry Found", "Verification Not Attempted", "Awaiting PO Load"],
                                        "Recommendation": ["Check verification process", "Ensure PO exists in system", "Try 'Load PO Data' function"]
                                    }
                                    st.dataframe(pd.DataFrame(status_data), use_container_width=True, hide_index=True)

                        with debug_tab4:
                            # Session Data and Actions
                            st.write("**💾 Session State Information:**")
                            has_original = "invoice_lines_original" in st.session_state
                            has_po_original = has_original and po_id in st.session_state.invoice_lines_original if has_original else False
                            original_count = len(st.session_state.invoice_lines_original.get(po_id, [])) if has_po_original else 0

                            # Session State Status
                            session_data = {
                                "Component": ["Original Data Cache", "PO Specific Cache", "Current Preview Data", "Invoice Context", "Clean PO ID"],
                                "Status": [
                                    "✅ Available" if has_original else "❌ Not Available",
                                    "✅ Available" if has_po_original else "❌ Not Available",
                                    "✅ Available" if po_id in st.session_state.get('invoice_lines_preview', {}) else "❌ Not Available",
                                    "✅ Available" if po_id in st.session_state.get('po_context', {}) else "❌ Not Available",
                                    _extract_real_po_id(po_id)
                                ],
                                "Details": [
                                    "Global original data store",
                                    f"{original_count} cached rows" if has_po_original else "No cached rows",
                                    f"{len(st.session_state.invoice_lines_preview.get(po_id, []))} preview rows" if po_id in st.session_state.get('invoice_lines_preview', {}) else "No preview data",
                                    "Invoice number, date, tax rate" if po_id in st.session_state.get('po_context', {}) else "No context data",
                                    "Internal format uses @ separator"
                                ]
                            }
                            st.dataframe(pd.DataFrame(session_data), use_container_width=True, hide_index=True)

                            # Debug Actions
                            st.write("**🔧 Debug Actions:**")
                            action_cols = st.columns(2)
                            with action_cols[0]:
                                if st.button("🔄 Reset Original Data", key=f"reset_original_debug_{po_id}", help="Recreate original from current data"):
                                    if po_id in st.session_state.get('invoice_lines_preview', {}):
                                        if "invoice_lines_original" not in st.session_state:
                                            st.session_state.invoice_lines_original = {}
                                        st.session_state.invoice_lines_original[po_id] = st.session_state.invoice_lines_preview[po_id].copy()
                                        st.success("✅ Original data reset successfully")
                                        st.rerun()
                                    else:
                                        st.warning("⚠️ No current data to reset from")

                            with action_cols[1]:
                                if st.button("🔍 Force PO Reload", key=f"force_reload_debug_{po_id}", help="Would trigger PO data refresh from API"):
                                    st.info("🔄 Force reload would refresh PO data from Coupa API")

                # Show status message only if there's an issue (spinner handles re-checking)
                if is_duplicate:
                    # Create link to existing invoice if we have the ID
                    existing_id = duplicate_info.get('existing_invoice_id', '')
                    if existing_id:
                        env_suffix = "_PROD" if st.session_state.get("environment", "Test") == "Production" else ""
                        coupa_instance = os.environ.get(f"COUPA_INSTANCE{env_suffix}")
                        invoice_url = f"https://{coupa_instance}.coupahost.com/invoices/{existing_id}"
                        
                        # Extract just the invoice number from the message for cleaner display
                        message = duplicate_info.get('message', 'Unknown error')
                        if 'Invoice #' in message and ' already exists' in message:
                            # Extract invoice number from "Invoice #12345 already exists for..."
                            invoice_num = message.split('Invoice #')[1].split(' already exists')[0]
                            
                            # Show error message (invoice link is now shown on right side)
                            st.error(f"Duplicate invoice number: {duplicate_info.get('message', 'Unknown error')}")
                        else:
                            st.error(f"Duplicate invoice number: {duplicate_info.get('message', 'Unknown error')}")
                            st.markdown(f"[View existing invoice]({invoice_url})")
                    else:
                        st.error(f"Duplicate invoice number: {duplicate_info.get('message', 'Unknown error')}")
                        st.info("Duplicate invoice detected")
                elif has_dup_error:
                    st.warning(f"Invoice duplicate check failed: {duplicate_info.get('message', 'Unknown error')}")
                
                # Auto-populate invoice number FIRST before validation
                # Use file-specific detected data from PO context if available
                
                # Auto-populate if no current value and we have detected values
                auto_messages = []
                
                # Get the stored detected values for this specific PO
                po_context = st.session_state.po_context.get(po_id, {})
                detected_invoice = po_context.get("detected_invoice", "")
                detected_date = po_context.get("detected_date")
                
                # Check if this invoice number was auto-populated and we should show the message
                auto_populated_invoice = (current_inv_num == detected_invoice and detected_invoice and st.session_state.get('auto_invoice_detect', True))
                message_shown_key = f"auto_message_shown_{po_id}_invoice"

                if auto_populated_invoice and not st.session_state.get(message_shown_key, False):
                    # This is an auto-populated invoice number and we haven't shown the message yet
                    auto_messages.append(f"Invoice: {detected_invoice}")
                    st.session_state[message_shown_key] = True
                elif not current_inv_num and detected_invoice and st.session_state.get('auto_invoice_detect', True):
                    # No current invoice number, but we have a detected one - populate it
                    st.session_state.po_context[po_id]["invoice_number"] = detected_invoice
                    current_inv_num = detected_invoice
                    auto_messages.append(f"Invoice: {detected_invoice}")
                    st.session_state[message_shown_key] = True
                
                # Auto-populate date if we have a detected date and auto-detect is enabled
                current_date = st.session_state.po_context[po_id].get("invoice_date")
                
                # Auto-populate from detected date - check if we should show message for auto-populated date
                date_message_shown_key = f"auto_message_shown_{po_id}_date"
                auto_populated_date = (current_date == detected_date and detected_date and st.session_state.get('auto_date_detect', True))

                if auto_populated_date and not st.session_state.get(date_message_shown_key, False):
                    # This is an auto-populated date and we haven't shown the message yet
                    auto_messages.append(f"Invoice Date: {detected_date.strftime('%d/%m/%Y')}")
                    st.session_state[date_message_shown_key] = True
                elif (detected_date and
                    st.session_state.get('auto_date_detect', True) and
                    current_date == datetime.today().date() and
                    detected_date != datetime.today().date()):  # Only if detected date is different from today

                    st.session_state.po_context[po_id]["invoice_date"] = detected_date
                    auto_messages.append(f"Invoice Date: {detected_date.strftime('%d/%m/%Y')}")
                    st.session_state[date_message_shown_key] = True
                
                # Show combined auto-population message with auto-dismiss
                if auto_messages:
                    # Store message with timestamp for auto-dismiss logic
                    current_time = time.time()
                    message_key = f"auto_popup_{po_id}"

                    # Check if we should show the message
                    if message_key not in st.session_state or (current_time - st.session_state[message_key]) > 10:
                        st.session_state[message_key] = current_time
                        # Combine all auto-populated fields in one message
                        combined_message = " and ".join(auto_messages)
                        st.success(f"✅ Auto-populated {combined_message}")
                    elif (current_time - st.session_state[message_key]) <= 10:
                        # Still within 10 second window, show the combined message
                        combined_message = " and ".join(auto_messages)
                        st.success(f"✅ Auto-populated {combined_message}")                # THEN do validation - check current invoice number
                if not current_inv_num or not current_inv_num.strip():
                    st.error("⚠️ Invoice Number is required. This PO will be skipped during batch creation.")

                # Mobile-first responsive layout
                # On mobile: stack vertically, on desktop: use columns
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Input fields in a more compact layout
                    input_cols = st.columns(3)
                    with input_cols[0]:
                        # Use a factory function to create callbacks with correct po_id closure
                        def create_update_callback(po_id):
                            def update_invoice_number():
                                # Debug: Log which PO callback is being triggered (without import conflict)
                                if is_debug():
                                    debug_po_display = po_id.replace('@', ' - ')
                                    with st.expander(f"🔄 Debug: Invoice Number Update - {debug_po_display}", expanded=False):
                                        st.write(f"**Callback triggered for PO**: {debug_po_display}")
                                        st.write(f"**Widget Key**: inv_num_{po_id}")
                                        st.write(f"**Previous Value**: {st.session_state.po_context[po_id].get('invoice_number', 'None')}")
                                        st.write(f"**New Value**: {st.session_state.get(f'inv_num_{po_id}', 'Not found')}")
                                
                                # Get the current value from the widget
                                current_value = st.session_state[f"inv_num_{po_id}"]
                                old_value = st.session_state.po_context[po_id].get("invoice_number", "")
                                
                                # Always update the context with the current value
                                st.session_state.po_context[po_id]["invoice_number"] = current_value
                                
                                # If invoice number changed, immediately trigger duplicate re-check
                                if current_value != old_value and hasattr(st.session_state, 'verified_pos'):
                                    for po_entry in st.session_state.verified_pos:
                                        if po_entry['po_id'] == po_id:
                                            if current_value and current_value.strip():
                                                # Mark for immediate re-check but don't rerun from callback
                                                po_entry['duplicate_check'] = {
                                                    "is_duplicate": False,
                                                    "message": "Re-checking invoice number...",
                                                    "has_error": False,
                                                    "needs_recheck": True
                                                }
                                            else:
                                                # Clear duplicate status if no invoice number
                                                po_entry['duplicate_check'] = {
                                                    "is_duplicate": False,
                                                    "message": "No invoice number",
                                                    "has_error": False,
                                                    "needs_recheck": False
                                                }
                                            break
                                
                                # Update include status based on invoice number
                                if current_value and current_value.strip():
                                    st.session_state.selected_for_batch[po_id] = True
                                else:
                                    st.session_state.selected_for_batch[po_id] = False
                            return update_invoice_number
                        
                        # Create the callback for this specific PO
                        update_callback = create_update_callback(po_id)
                        
                        # Invoice number input (auto-population already handled above)
                        current_invoice = st.session_state.po_context[po_id].get("invoice_number", "")
                        detected_invoice = st.session_state.get("detected_invoice_number", "")
                        
                        st.text_input(
                            "Invoice #", 
                            key=f"inv_num_{po_id}", 
                            placeholder="Auto-detected or enter manually",
                            value=current_invoice,
                            on_change=update_callback,
                            help="Invoice number auto-detected from document text" if detected_invoice and current_invoice == detected_invoice else None
                        )
                    with input_cols[1]:
                        current_date = st.session_state.po_context[po_id]["invoice_date"]
                        detected_date = st.session_state.get("detected_invoice_date")
                        
                        st.session_state.po_context[po_id]["invoice_date"] = st.date_input(
                            "Date", key=f"inv_date_{po_id}",
                            value=current_date,
                            format="DD/MM/YYYY",
                            help="Date auto-detected from document text" if detected_date and current_date == detected_date else None
                        )
                    with input_cols[2]:
                        st.session_state.po_context[po_id]["tax_rate"] = st.number_input(
                            "Tax %", key=f"tax_rate_{po_id}",
                            value=float(st.session_state.po_context[po_id]["tax_rate"]),
                            step=0.01
                        )
                
                with col2:
                    # Show invoice link only if duplicate is found
                    if is_duplicate:
                        existing_id = duplicate_info.get('existing_invoice_id', '')
                        if existing_id:
                            env_suffix = "_PROD" if st.session_state.get("environment", "Test") == "Production" else ""
                            coupa_instance = os.environ.get(f"COUPA_INSTANCE{env_suffix}")
                            invoice_url = f"https://{coupa_instance}.coupahost.com/invoices/{existing_id}"
                            
                            # Extract invoice number from message
                            message = duplicate_info.get('message', '')
                            if 'Invoice #' in message and ' already exists' in message:
                                invoice_num = message.split('Invoice #')[1].split(' already exists')[0]
                                st.markdown(f"**Invoice:** [{invoice_num}]({invoice_url})")
                    
                    # Extract real PO ID for display and numeric ID for URL
                    real_po_id = _extract_real_po_id(po_id)
                    numeric_po_id = _numeric_po_id(real_po_id)
                    displayed_po = po_data.get('number') or real_po_id
                    po_url = f"https://royalchildrens-test.coupahost.com/order_headers/{numeric_po_id}"
                    st.markdown(f"**PO:** [{displayed_po}]({po_url})")
                    st.markdown(f"**Status:** {po_data.get('status', 'Unknown')}")
                    supplier_name = (po_data.get('supplier') or {}).get('name', 'Unknown')
                    # Show full supplier name with proper wrapping
                    st.markdown(f"**Supplier:** {supplier_name}")

                # Compact header tax caption
                st.caption("Header tax = subtotal × rate")

                # Build preview df once and store original copy for restore
                if po_id not in st.session_state.invoice_lines_preview:
                    original_df = build_preview_df(po_id, po_data)
                    st.session_state.invoice_lines_preview[po_id] = original_df.copy()
                    # Initialize original data storage if needed
                    if "invoice_lines_original" not in st.session_state:
                        st.session_state.invoice_lines_original = {}
                    # Store original data for restore functionality
                    st.session_state.invoice_lines_original[po_id] = original_df.copy()
                    
                # Ensure original data exists (backup in case it was lost)
                if "invoice_lines_original" not in st.session_state:
                    st.session_state.invoice_lines_original = {}
                if po_id not in st.session_state.invoice_lines_original:
                    # Recreate original from current data if lost
                    current_df = st.session_state.invoice_lines_preview[po_id]
                    st.session_state.invoice_lines_original[po_id] = current_df.copy()

                preview_df = st.session_state.invoice_lines_preview[po_id]

                # Get all default values for new lines (includes all mandatory fields)
                line_defaults = _get_po_defaults_for_new_line(po_id, preview_df)

                # All buttons on same row ABOVE editor - 3 columns (always enabled)
                btn_col1, btn_col2, btn_col3 = st.columns(3)

                with btn_col1:
                    if st.button("➕ Add Qty", key=f"add_qty_{po_id}", help="Add Quantity Line", width='stretch'):
                        new_row = {
                            "line_num": len(preview_df) + 1,
                            "inv_type": "InvoiceQuantityLine",
                            "description": "",
                            "price": 0.00,
                            "quantity": 1.00,
                            "uom_code": DEFAULT_UOM_CODE,
                            "order_line_id": None,
                            "order_line_num": get_next_available_line_num(preview_df),
                            "delete": False,
                            **line_defaults  # Include all PO defaults (account_id, commodity_name, po_number, etc.)
                        }
                        preview_df = pd.concat([preview_df, pd.DataFrame([new_row])], ignore_index=True)
                        preview_df = resequence_lines(preview_df)
                        st.session_state.invoice_lines_preview[po_id] = preview_df
                        st.rerun()
                        
                with btn_col2:
                    if st.button("➕ Add Amt", key=f"add_amt_{po_id}", help="Add Amount Line", width='stretch'):
                        new_row = {
                            "line_num": len(preview_df) + 1,
                            "inv_type": "InvoiceAmountLine",
                            "description": "",
                            "price": 0.00,
                            "quantity": None,  # Blank for amount lines - not relevant
                            "uom_code": None,  # Blank for amount lines - not relevant
                            "order_line_id": None,
                            "order_line_num": get_next_available_line_num(preview_df),
                            "delete": False,
                            **line_defaults  # Include all PO defaults (account_id, commodity_name, po_number, etc.)
                        }
                        preview_df = pd.concat([preview_df, pd.DataFrame([new_row])], ignore_index=True)
                        preview_df = resequence_lines(preview_df)
                        st.session_state.invoice_lines_preview[po_id] = preview_df
                        st.rerun()
                        
                with btn_col3:
                    if st.button("↩️ Restore All", key=f"restore_all_{po_id}", help="Restore to original data from extraction", width='stretch'):
                        # Debug: Check what data we have
                        has_original = "invoice_lines_original" in st.session_state
                        has_po_data = has_original and po_id in st.session_state.invoice_lines_original
                        
                        if has_original and has_po_data:
                            try:
                                # Restore from original data stored when first created
                                original_df = st.session_state.invoice_lines_original[po_id].copy()
                                st.session_state.invoice_lines_preview[po_id] = original_df
                                st.success(f"✅ Restored {len(original_df)} rows to original extracted data")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error during restore: {str(e)}")
                        else:
                            # Try to recreate original from build_preview_df
                            try:
                                st.warning("⚠️ Original data not found, recreating from PO data...")
                                original_df = build_preview_df(po_id, po_data)
                                st.session_state.invoice_lines_preview[po_id] = original_df.copy()
                                # Store for future restores
                                if "invoice_lines_original" not in st.session_state:
                                    st.session_state.invoice_lines_original = {}
                                st.session_state.invoice_lines_original[po_id] = original_df.copy()
                                st.success(f"✅ Recreated and restored {len(original_df)} rows from PO data")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Could not restore data: {str(e)}")
                                st.info("💡 Try re-uploading the PDF to reset the data")

                # Editor config - more compact for mobile with dynamic UOM handling
                col_cfg = {
                    "line_num": st.column_config.NumberColumn("#", disabled=True, width="small"),
                    "inv_type": st.column_config.SelectboxColumn("Type", options=["InvoiceQuantityLine", "InvoiceAmountLine"], required=True, width="small"),
                    "description": st.column_config.TextColumn("Description", required=True, width="large"),
                    "quantity": st.column_config.NumberColumn("Qty", min_value=0.0, step=0.01, format="%.2f", width="small"),
                    "price": st.column_config.NumberColumn("Price", min_value=0.0, step=0.01, format="$%.2f", width="small"),  # Currency format
                    "uom_code": st.column_config.TextColumn("UOM", width="small", help="Unit of Measure (disabled for Amount lines)"),
                    "source_part_num": st.column_config.TextColumn("Part #", width="medium"),
                    "commodity_name": st.column_config.TextColumn("Commodity", width="medium"),
                    "po_number": st.column_config.TextColumn("PO #", width="small"),
                    "order_line_num": st.column_config.TextColumn("Line #", width="small"),
                    "delete": st.column_config.CheckboxColumn("🗑️", width="small", help="Check to immediately delete this row")
                }

                editor_order = [c for c in EDITOR_COL_ORDER if c in preview_df.columns]

                edited_df = st.data_editor(
                    preview_df,
                    column_config=col_cfg,
                    column_order=editor_order,
                    hide_index=True,
                    width='stretch',
                    key=f"editor_{po_id}"
                )

                # Immediate change detection and auto-save
                data_changed = False
                try:
                    # Get the previous state for comparison
                    prev_key = f"prev_editor_data_{po_id}"
                    if prev_key in st.session_state:
                        # Compare with previous state
                        if not edited_df.equals(st.session_state[prev_key]):
                            data_changed = True
                    else:
                        # First time - consider it changed if different from original
                        if not edited_df.equals(preview_df):
                            data_changed = True

                    # Update the previous state for next comparison
                    st.session_state[prev_key] = edited_df.copy()

                except Exception:
                    # If comparison fails, assume changed to be safe
                    data_changed = True
                    st.session_state[f"prev_editor_data_{po_id}"] = edited_df.copy()

                # Sanitize and store back - but check for deleted rows first
                edited_df = sanitize_editor_rows(edited_df)
                
                # Clean up UOM and quantity fields for amount lines
                if not edited_df.empty and "inv_type" in edited_df.columns:
                    # Clear UOM and quantity for amount lines (they're not relevant)
                    amount_line_mask = edited_df["inv_type"] == "InvoiceAmountLine"
                    if amount_line_mask.any():
                        if "uom_code" in edited_df.columns:
                            edited_df.loc[amount_line_mask, "uom_code"] = None
                        if "quantity" in edited_df.columns:
                            edited_df.loc[amount_line_mask, "quantity"] = None

                # Check if any rows are marked for deletion and remove them immediately
                if "delete" in edited_df.columns:
                    rows_to_delete = edited_df["delete"].astype(bool)
                    if rows_to_delete.any():
                        # Remove deleted rows immediately
                        edited_df = edited_df[~rows_to_delete].copy()
                        # Resequence line numbers
                        edited_df = resequence_lines(edited_df)
                        # Store the updated dataframe
                        st.session_state.invoice_lines_preview[po_id] = edited_df
                        # Show confirmation message
                        deleted_count = rows_to_delete.sum()
                        st.toast(f"🗑️ Deleted {deleted_count} row(s)", icon="✅")
                        # Trigger rerun to refresh the interface
                        st.rerun()

                st.session_state.invoice_lines_preview[po_id] = edited_df

                # Auto-save functionality - show confirmation when changes are detected
                if data_changed:
                    # Show brief auto-save confirmation
                    st.toast("💾 Changes auto-saved", icon="✅")

                # Calculate totals dynamically (all rows are active now since deleted ones are removed)
                subtotal = float(edited_df["price"].fillna(0).astype(float).mul(edited_df["quantity"].fillna(0).astype(float)).sum())
                est_tax = round(subtotal * (float(st.session_state.po_context.setdefault(po_id, {}).get("tax_rate", 10)) / 100.0), 2)
                net_total = subtotal + est_tax
                
                # Compact but readable metrics row - 3 metrics for better mobile experience
                with st.container():
                    # Always use 3-column layout (removed total lines metric)
                    mt_cols = st.columns(3)
                    with mt_cols[0]: 
                        st.metric("Subtotal", f"${subtotal:,.2f}")
                    with mt_cols[1]: 
                        st.metric("Tax", f"${est_tax:,.2f}")
                    with mt_cols[2]: 
                        st.metric("Net Total", f"${net_total:,.2f}")

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
                "Invoice #": inv_num,
                "PO": _extract_real_po_id(po_id),
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
                "Invoice #": st.column_config.TextColumn("Invoice #"),
                "PO": st.column_config.TextColumn("PO"),
                "Date (dd/mm/yyyy)": st.column_config.TextColumn("Date (dd/mm/yyyy)", disabled=True),
                "Lines": st.column_config.NumberColumn("Lines", disabled=True),
                "Subtotal": st.column_config.TextColumn("Subtotal", disabled=True),
                "Header Tax": st.column_config.TextColumn("Header Tax", disabled=True),
            },
            width='stretch',
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
            # Action buttons - primary action first, then support action
            col1, col2 = st.columns([2, 1])
            
            with col1:
                create_clicked = st.button("Create Invoices", type="primary", help="Creates invoices for all checked items")
            
            with col2:
                # Hidden for now - user doesn't want refresh duplicates button
                # if st.button("Refresh Duplicates", help="Re-check invoice numbers for duplicates after making changes"):
                #     st.rerun()
                pass
            
            # Handle Create Invoices action
            if create_clicked:
                    approval_option = st.session_state.get("approval_option", "Save as Draft")
                    
                    # Debug: Show batch creation setup and configuration
                    if is_debug():
                        with st.expander("🔧 Debug: Batch Invoice Creation Setup", expanded=False):
                            # Create tabs for organized debug information
                            setup_tab, selection_tab, validation_tab = st.tabs(["⚙️ Configuration", "📋 PO Selection", "✅ Validation"])

                            with setup_tab:
                                st.write("**🚀 Batch Creation Configuration:**")
                                config_data = {
                                    "Setting": ["Approval Option", "Expected Invoice Status", "Environment", "Total POs Available", "Auto Submit", "Create Results State"],
                                    "Value": [
                                        approval_option,
                                        "submitted" if approval_option == "Submit for Approval" else "draft/pending",
                                        st.session_state.get("environment", "Test"),
                                        f"{len(verified_pos)} POs",
                                        "✅ Yes" if approval_option == "Submit for Approval" else "❌ No",
                                        f"Initialized (currently {len(st.session_state.create_results)} items)"
                                    ]
                                }
                                st.dataframe(pd.DataFrame(config_data), use_container_width=True, hide_index=True)

                                st.write("**📋 Process Steps:**")
                                steps_data = {
                                    "Step": ["1. Create Invoice", "2. Upload Image Scan", "3. Submit for Approval"],
                                    "Status": [
                                        "✅ Always performed",
                                        "✅ Always performed (if file available)",
                                        "✅ Enabled" if approval_option == "Submit for Approval" else "❌ Disabled"
                                    ],
                                    "Notes": [
                                        "Creates invoice in draft status",
                                        "Uploads original PDF/image to invoice",
                                        "Submits invoice for approval workflow" if approval_option == "Submit for Approval" else "Invoice remains as draft"
                                    ]
                                }
                                st.dataframe(pd.DataFrame(steps_data), use_container_width=True, hide_index=True)

                            with selection_tab:
                                st.write("**📦 PO Selection Summary:**")
                                included_pos = []
                                excluded_pos = []

                                for po_info in verified_pos:
                                    po_id = po_info['po_id']
                                    include = bool(st.session_state.selected_for_batch.get(po_id, True))
                                    display_po_id = _extract_real_po_id(po_id)

                                    # Get additional context
                                    context = st.session_state.po_context.get(po_id, {})
                                    invoice_num = context.get("invoice_number", "Not set")
                                    lines_count = len(st.session_state.invoice_lines_preview.get(po_id, []))

                                    po_data = {
                                        "PO": display_po_id,
                                        "Invoice #": invoice_num,
                                        "Lines": f"{lines_count} lines",
                                        "Include": "✅ Yes" if include else "❌ No"
                                    }

                                    if include:
                                        included_pos.append(po_data)
                                    else:
                                        excluded_pos.append(po_data)

                                if included_pos:
                                    st.write(f"**✅ POs Selected for Creation ({len(included_pos)}):**")
                                    st.dataframe(pd.DataFrame(included_pos), use_container_width=True, hide_index=True)

                                if excluded_pos:
                                    st.write(f"**❌ POs Excluded from Creation ({len(excluded_pos)}):**")
                                    st.dataframe(pd.DataFrame(excluded_pos), use_container_width=True, hide_index=True)

                                if not included_pos:
                                    st.warning("⚠️ No POs selected for creation!")

                            with validation_tab:
                                st.write("**🔍 Pre-Creation Validation:**")
                                validation_issues = []
                                ready_pos = []

                                for po_info in verified_pos:
                                    po_id = po_info['po_id']
                                    include = bool(st.session_state.selected_for_batch.get(po_id, True))

                                    if include:
                                        display_po_id = _extract_real_po_id(po_id)
                                        context = st.session_state.po_context.get(po_id, {})
                                        invoice_num = context.get("invoice_number", "")
                                        lines = st.session_state.invoice_lines_preview.get(po_id, [])

                                        issues = []
                                        if not invoice_num or not invoice_num.strip():
                                            issues.append("Missing invoice number")
                                        if len(lines) == 0:
                                            issues.append("No invoice lines")

                                        validation_data = {
                                            "PO": display_po_id,
                                            "Status": "❌ Issues Found" if issues else "✅ Ready",
                                            "Issues": "; ".join(issues) if issues else "None"
                                        }

                                        if issues:
                                            validation_issues.append(validation_data)
                                        else:
                                            ready_pos.append(validation_data)

                                if ready_pos:
                                    st.write(f"**✅ Ready for Creation ({len(ready_pos)}):**")
                                    st.dataframe(pd.DataFrame(ready_pos), use_container_width=True, hide_index=True)

                                if validation_issues:
                                    st.write(f"**❌ Validation Issues ({len(validation_issues)}):**")
                                    st.dataframe(pd.DataFrame(validation_issues), use_container_width=True, hide_index=True)
                                    st.error("⚠️ Some POs have validation issues and will be skipped during creation.")

                                if not ready_pos and not validation_issues:
                                    st.info("ℹ️ No POs selected for validation.")
                    
                    st.session_state.create_results = []
                    
                    # Count POs to process for progress tracking
                    pos_to_process = []
                    for po_info in verified_pos:
                        po_id = po_info['po_id']
                        include = bool(st.session_state.selected_for_batch.get(po_id, True))
                        if include:
                            pos_to_process.append(po_info)
                    
                    total_pos = len(pos_to_process)
                    if total_pos == 0:
                        st.warning("No POs selected for processing.")
                    else:
                        # Create invoice creation with spinner and progress tracking
                        with st.spinner(f"🏭 Creating invoices for {total_pos} PO{'s' if total_pos != 1 else ''}..."):
                            # Create progress tracking
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for idx, po_info in enumerate(pos_to_process):
                                po_id = po_info['po_id']
                                po_data = po_info['po_data']
                                
                                # Update overall progress
                                progress = idx / total_pos
                                progress_bar.progress(progress)
                                status_text.text(f"Processing {idx + 1}/{total_pos}: PO {po_id}")
                                
                                # Get current invoice number from widget state (most current value)
                                widget_key = f"inv_num_{po_id}"
                                current_invoice = st.session_state.get(widget_key, "")
                                if not current_invoice:
                                    # Fallback to context if widget state not available
                                    ctx = st.session_state.po_context.get(po_id, {})
                                    current_invoice = ctx.get("invoice_number", "")
                                
                                # Get other context data
                                ctx = st.session_state.po_context.get(po_id, {})
                                
                                # Use the current invoice number we just retrieved
                                inv_num = current_invoice
                                inv_date_val = ctx.get("invoice_date")
                                if not inv_num or not inv_date_val:
                                    result_data = {
                                        "PO": po_id, "Invoice #": inv_num or "",
                                        "Invoice ID": "", "Invoice Status": "Failed (missing number/date)",
                                        "Scan Status": "N/A", "Link": ""
                                    }
                                    st.session_state.create_results.append(result_data)
                                    if is_debug():
                                        debug_po_display = po_id.replace('@', ' - ')
                                        with st.expander(f"🔍 Debug: Invoice Creation Error - {debug_po_display}", expanded=False):
                                            st.write(f"**Result**: Added MISSING DATA result")
                                            st.write(f"**PO**: {debug_po_display}")
                                            st.write(f"**Invoice Number**: '{inv_num}' (missing)" if not inv_num else f"**Invoice Number**: '{inv_num}'")
                                            st.write(f"**Invoice Date**: {inv_date_val}" if inv_date_val else "**Invoice Date**: missing")
                                            st.write(f"**Total results**: {len(st.session_state.create_results)}")
                                    continue

                                edited_df = st.session_state.invoice_lines_preview.get(po_id)
                                doc_bytes = ctx.get("doc_bytes")
                                upload_filename = ctx.get("doc_name") or "invoice.pdf"

                                if edited_df is None or edited_df.empty:
                                    result_data = {
                                        "PO": po_id, "Invoice #": inv_num,
                                        "Invoice ID": "", "Invoice Status": "Failed (no lines)",
                                        "Scan Status": "N/A", "Link": ""
                                    }
                                    st.session_state.create_results.append(result_data)
                                    if is_debug():
                                        debug_po_display = po_id.replace('@', ' - ')
                                        with st.expander(f"🔍 Debug: Invoice Creation Error - {debug_po_display}", expanded=False):
                                            st.write(f"**Result**: Added NO LINES result")
                                            st.write(f"**PO**: {debug_po_display}")
                                            st.write(f"**Issue**: No invoice lines to create")
                                            st.write(f"**Total results**: {len(st.session_state.create_results)}")
                                    continue
                                if not doc_bytes:
                                    st.session_state.create_results.append({
                                        "PO": po_id, "Invoice #": inv_num,
                                        "Invoice ID": "", "Invoice Status": "Failed (no file)",
                                        "Scan Status": "N/A", "Link": ""
                                    })
                                    continue

                                # Show detailed progress for individual stages (no wrapper spinner)
                                stage_status = st.empty()
                                stage_status.text(f"📄 Processing PO {po_id}...")
                                
                                inv_ok, inv_resp, scan_ok, scan_resp = create_invoice_from_po(
                                    context_po_id=po_id,
                                    invoice_number=inv_num,
                                    invoice_date=inv_date_val,
                                    original_file_data=doc_bytes,
                                    upload_filename=upload_filename,
                                    po_data=po_data,
                                    edited_df=edited_df,
                                    header_tax_rate=float(ctx.get("tax_rate") or 0),
                                    approval_option=approval_option,
                                    stage_callback=lambda msg: stage_status.text(msg)
                                )
                                
                                # Show error in status if failed, clear after short delay for success
                                if not inv_ok:
                                    cleaned_error = clean_error_message(str(inv_resp))
                                    stage_status.text(f"❌ PO {po_id}: {cleaned_error}")
                                    # Keep error visible briefly, then clear at end of processing
                                else:
                                    stage_status.text(f"✅ PO {po_id}: Invoice created successfully")
                                    # Will be cleared at end of loop
                                
                                # Process results from invoice creation
                                if inv_ok and isinstance(inv_resp, dict):
                                    invoice_id = inv_resp.get("id")
                                    env_suffix = "_PROD" if st.session_state.get("environment", "Test") == "Production" else ""
                                    coupa_instance = os.environ.get(f"COUPA_INSTANCE{env_suffix}")
                                    link = f"https://{coupa_instance}.coupahost.com/invoices/{invoice_id}" if invoice_id else ""
                                    
                                    # Determine the final status message - keep it clean for the table
                                    if approval_option == "Submit for Approval":
                                        if inv_resp.get("submitted"):
                                            status_msg = "Success - Submitted for Approval"
                                        elif inv_resp.get("submit_error"):
                                            status_msg = "Success - Draft (unable to submit)"
                                        else:
                                            status_msg = "Success - Submitted for Approval"
                                    else:
                                        status_msg = "Success - Draft"
                                    
                                    result_data = {
                                        "PO": po_id, "Invoice #": inv_num,
                                        "Invoice ID": invoice_id or "",
                                        "Invoice Status": status_msg,
                                        "Scan Status": ("Success" if scan_ok else f"Failed ({scan_resp})"),
                                        "Link": link
                                    }
                                    st.session_state.create_results.append(result_data)
                                    # Update main status to show success
                                    status_text.text(f"✅ PO {po_id}: Invoice created successfully")
                                else:
                                    # Clean up the error message for better display
                                    cleaned_error = clean_error_message(str(inv_resp))
                                    result_data = {
                                        "PO": po_id, "Invoice #": inv_num,
                                        "Invoice ID": "",
                                        "Invoice Status": f"Failed: {cleaned_error}",
                                        "Scan Status": "N/A",
                                        "Link": ""
                                    }
                                    st.session_state.create_results.append(result_data)
                                    # Update main status to show failure
                                    status_text.text(f"❌ PO {po_id}: {cleaned_error}")
                            
                            # Complete progress tracking for each PO
                            final_progress = (idx + 1) / total_pos
                            progress_bar.progress(final_progress)
                            
                            # Clear any remaining stage status after each PO
                            if 'stage_status' in locals():
                                stage_status.empty()

                        # Process skipped POs from the original loop
                        for po_info in verified_pos:
                            po_id = po_info['po_id']
                            include = bool(st.session_state.selected_for_batch.get(po_id, True))
                            
                            if not include:
                                # Check if this PO already has a result to avoid duplicates
                                existing_result = None
                                for existing in st.session_state.create_results:
                                    if existing.get("PO") == po_id:
                                        existing_result = existing
                                        break
                                
                                if existing_result:
                                    # Update existing result to ensure it shows "Skipped"
                                    existing_result["Invoice Status"] = "Skipped"
                                    existing_result["Scan Status"] = "Skipped"
                                else:
                                    # Get current invoice number for display
                                    widget_key = f"inv_num_{po_id}"
                                    current_invoice = st.session_state.get(widget_key, "")
                                    if not current_invoice:
                                        ctx = st.session_state.po_context.get(po_id, {})
                                        current_invoice = ctx.get("invoice_number", "")
                                    
                                    result_data = {
                                        "PO": po_id, "Invoice #": current_invoice,
                                        "Invoice ID": "", "Invoice Status": "Skipped",
                                        "Scan Status": "Skipped", "Link": ""
                                    }
                                    st.session_state.create_results.append(result_data)
                        
                        # Clean up progress indicators
                        progress_bar.empty()
                        status_text.empty()
                    
                    # Final debug after processing all POs
                    if is_debug():
                        with st.expander("🔧 Debug - Processing Complete Summary", expanded=False):
                            st.write(f"Debug - PROCESSING COMPLETE! Final result count: {len(st.session_state.create_results)}")
                            st.write("All results:")
                            for i, result in enumerate(st.session_state.create_results):
                                st.write(f"  {i+1}. PO {result.get('PO')}: Status='{result.get('Invoice Status')}', Invoice='{result.get('Invoice #')}', Link='{result.get('Link', 'No link')}'")

        # Single, pretty Results with big metrics and one table
        if st.session_state.create_results:
            
            st.markdown("---")
            st.subheader("📊 Results")

            def emoji_status(s: str) -> str:
                original_s = str(s or "")
                s_lower = original_s.lower()
                if s_lower.startswith("success"):
                    return "✅ Success"
                elif s_lower.startswith("failed"):
                    # Preserve the full cleaned error message after "failed: "
                    if ": " in original_s:
                        error_part = original_s.split(": ", 1)[1]
                    else:
                        error_part = original_s.replace('Failed', '', 1).strip()
                    # Clean the error message to remove technical prefixes
                    cleaned_error = clean_error_message(error_part)
                    return f"❌ {cleaned_error}"
                elif s_lower.startswith("skipped"):
                    return "⏭️ Skipped"
                return original_s

            # Build results table + counts - simplified columns
            rows = []
            succ = fail = skip = 0
            
            # Collect invoice IDs for batch status fetching
            invoice_ids_to_fetch = []
            for r in st.session_state.create_results:
                inv_stat = emoji_status(r["Invoice Status"])
                if inv_stat.startswith("✅") and r.get("Invoice ID"):
                    invoice_ids_to_fetch.append(r.get("Invoice ID"))
            
            # Fetch all invoice statuses in batch if there are any
            invoice_statuses = {}
            if invoice_ids_to_fetch:
                with st.spinner(f"Fetching status for {len(invoice_ids_to_fetch)} invoices..."):
                    for invoice_id in invoice_ids_to_fetch:
                        invoice_statuses[invoice_id] = get_invoice_status(invoice_id)
            
            for r in st.session_state.create_results:
                inv_stat = emoji_status(r["Invoice Status"])
                if inv_stat.startswith("✅"):
                    succ += 1
                elif inv_stat.startswith("❌"):
                    fail += 1
                elif inv_stat.startswith("⏭️"):
                    skip += 1
                
                # Get the current invoice number from context (not from stored results)
                po_id = r["PO"]
                ctx = st.session_state.po_context.get(po_id, {})
                current_invoice = ctx.get("invoice_number", r["Invoice #"])  # fallback to stored if context missing
                current_date = ctx.get("invoice_date", "")
                
                # Format date properly
                if hasattr(current_date, 'strftime'):
                    date_str = current_date.strftime('%d/%m/%Y')
                else:
                    date_str = str(current_date) if current_date else ""
                
                # Get invoice status from pre-fetched data or use error message for failures
                coupa_status = "N/A"
                if inv_stat.startswith("✅") and r.get("Invoice ID"):
                    coupa_status = invoice_statuses.get(r.get("Invoice ID"), "Unknown")
                elif inv_stat.startswith("❌"):
                    # For failures, put the error message in the status column
                    coupa_status = inv_stat.replace("❌ ", "")  # Remove the emoji prefix
                elif inv_stat.startswith("⏭️"):
                    coupa_status = "Skipped"
                
                # Create proper results table data
                rows.append({
                    "Result": "✅" if not inv_stat.startswith(("⏭️", "❌")) else ("⏭️" if inv_stat.startswith("⏭️") else "❌"),
                    "Invoice #": r.get("Link", "") if (r.get("Link") and inv_stat.startswith("✅")) else current_invoice,
                    "Invoice Display": current_invoice,  # The text to display
                    "Date": date_str,
                    "PO": po_id,
                    "Status": coupa_status,
                    "Has Link": bool(r.get("Link") and inv_stat.startswith("✅"))  # Track if it should be clickable
                })

            # Big metrics row (like subtotal/metrics)
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("✅ Success", succ)
            with m2: st.metric("❌ Failed", fail)
            with m3: st.metric("⏭️ Skipped", skip)

            # Results table - use data_editor for proper link handling
            results_df = pd.DataFrame(rows)
            
            if is_debug():
                with st.expander("🔧 Debug - DataFrame Creation", expanded=False):
                    st.write(f"Total rows created: {len(rows)}")
                    st.write(f"Results DataFrame empty: {results_df.empty}")
                    if rows:
                        st.write("First row example:")
                        st.json(rows[0])
            
            if not results_df.empty:
                # Create display with proper data for data_editor
                display_rows = []
                for _, row in results_df.iterrows():
                    display_rows.append({
                        "Result": row["Result"],
                        "Invoice #": row["Invoice Display"],
                        "Date": row["Date"],
                        "PO": row["PO"],
                        "Status": row["Status"],
                        "Link": row["Invoice #"] if row.get("Has Link", False) else ""
                    })
                
                display_df = pd.DataFrame(display_rows)
                
                st.data_editor(
                    display_df,
                    width='stretch',
                    column_config={
                        "Result": st.column_config.TextColumn("Result", width="small", disabled=True),
                        "Invoice #": st.column_config.TextColumn("Invoice #", width="medium", disabled=True), 
                        "Date": st.column_config.TextColumn("Date", width="medium", disabled=True),
                        "PO": st.column_config.TextColumn("PO", width="medium", disabled=True),
                        "Status": st.column_config.TextColumn("Status", width="large", disabled=True),
                        "Link": st.column_config.LinkColumn(
                            "🔗", 
                            width="medium", 
                            help="Click to view invoice in Coupa",
                            display_text="View Invoice"
                        )
                    },
                    hide_index=True,
                    disabled=True  # Make the entire table read-only
                )
            elif len(rows) > 0:
                # Fallback: if DataFrame appears empty but rows exist, show them anyway
                if is_debug():
                    st.warning("DataFrame appears empty but rows exist - showing fallback table")
                display_df = pd.DataFrame(rows)
                st.dataframe(display_df, width='stretch')
            else:
                st.info("No results to display")

if __name__ == "__main__":
    main()
