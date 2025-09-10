# -*- coding: utf-8 -*-
"""
miniSmash Invoice Creator
A Streamlit application for processing PDF invoices with OCR and Coupa API integration.
"""

# Standard library imports
import json
import mimetypes
import os
import re
import tempfile
import zipfile
from datetime import datetime, date
from io import BytesIO
from typing import List, Tuple, Any

# Third-party imports
import streamlit as st
import pandas as pd
import requests
import camelot
import easyocr
import numpy as np
from PIL import Image

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

try:
    from streamlit_tags import st_tags_sidebar
    STREAMLIT_TAGS_AVAILABLE = True
except ImportError:
    STREAMLIT_TAGS_AVAILABLE = False

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
    prefix_input = (st.session_state.get("po_prefix") or "RCH").strip()
    # Split by comma and clean up each prefix
    prefixes = [p.strip().upper() for p in prefix_input.split(',') if p.strip()]
    return prefixes if prefixes else ["RCH"]

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

def extract_dates_from_text(text: str) -> List[date]:
    """
    Extract potential dates from text using various common date formats.
    Returns a list of unique date objects.
    Prioritizes DD/MM/YYYY format for Australian documents.
    """
    if not text:
        return []
    
    dates = []
    text_clean = text.strip()
    
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
    
    # Month name patterns (more comprehensive)
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
    
    # Process numeric date patterns
    for pattern in date_patterns:
        matches = re.finditer(pattern, text_clean, re.IGNORECASE)
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
    
    # Process month name patterns
    for pattern in month_patterns:
        matches = re.finditer(pattern, text_clean, re.IGNORECASE)
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
                            dates.append(date(year, month, day))
                        except ValueError:
                            continue
            except (ValueError, OverflowError):
                continue
    
    # Remove duplicates and sort
    unique_dates = list(set(dates))
    unique_dates.sort()
    
    return unique_dates

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

def _log_ocr_debug(image_info: str, po_numbers: List[str], gpu_info: dict = None, invoice_numbers: List[str] = None, extracted_dates: List[date] = None, extracted_text: str = ""):
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
            invoice_patterns = st.session_state.get('invoice_patterns', ['INVOICE NUMBER:', 'INVOICE#', 'INV:', 'INVOICE:'])
            st.code(f"Patterns: {', '.join(invoice_patterns)}")
            if invoice_numbers:
                st.success(f"Found invoice numbers: {', '.join(invoice_numbers)}")
                if invoice_numbers:
                    st.info(f"Auto-populated invoice number: {invoice_numbers[0]}")
            else:
                st.warning("No invoice numbers found")
                
        # Date Detection
        st.write("**Date Detection:**")
        if extracted_dates:
            date_strs = [d.strftime("%d/%m/%Y") for d in extracted_dates]
            st.success(f"Found dates: {', '.join(date_strs)}")
            if st.session_state.get('auto_date_detect', True) and extracted_dates:
                most_recent = max(extracted_dates)
                st.info(f"Auto-selected date: {most_recent.strftime('%d/%m/%Y')}")
        else:
            st.warning("No dates found")
            
        # Show extracted text for debugging
        if st.checkbox("Show extracted text", key=f"show_text_{image_info}"):
            with st.expander("📝 Raw Extracted Text", expanded=False):
                st.text_area("OCR Text:", value=extracted_text, height=150, key=f"text_{image_info}")
                
                # Manual date test
                if st.button("🔍 Test Date Extraction", key=f"test_dates_{image_info}"):
                    test_text = st.session_state.get(f"text_{image_info}", extracted_text)
                    if test_text:
                        test_dates = extract_dates_from_text(test_text)
                        if test_dates:
                            test_date_strs = [d.strftime("%d/%m/%Y") for d in test_dates]
                            st.success(f"Manual test found dates: {', '.join(test_date_strs)}")
                        else:
                            st.warning("Manual test found no dates")
                    else:
                        st.warning("No text available for testing")

# =========================================================
# OCR (enabled with EasyOCR)
# =========================================================
def _ocr_text_from_image_bytes(img_bytes: bytes, filename: str = "image") -> str:
    """
    Extract text from image bytes using EasyOCR.
    """
    try:
        # Try GPU first, fall back to CPU if not available
        if torch is not None:
            try:
                gpu_available = torch.cuda.is_available()
            except Exception:
                gpu_available = False
        else:
            gpu_available = False
        
        # Initialize EasyOCR reader with adaptive GPU setting
        reader = easyocr.Reader(['en'], gpu=gpu_available)
        
        # Convert bytes to PIL Image
        image = Image.open(BytesIO(img_bytes))
        
        # Convert PIL Image to numpy array for EasyOCR
        image_array = np.array(image)
        
        # Perform OCR
        results = reader.readtext(image_array)
        
        # Extract text from results
        extracted_text = ' '.join([result[1] for result in results])
        
        # Find PO numbers for debug logging
        po_numbers = _extract_po_numbers_from_text(extracted_text)
        
        # Extract invoice numbers and store in session state for auto-population
        invoice_numbers = _extract_invoice_numbers_from_text(extracted_text)
        if invoice_numbers and st.session_state.get('auto_invoice_detect', True):
            st.session_state.detected_invoice_number = invoice_numbers[0]  # Use the first match
        
        # Extract dates and store in session state for date picker options
        extracted_dates = extract_dates_from_text(extracted_text)
        
        # Debug: Always show what we extracted (even if debug is off)
        if extracted_text and len(extracted_text.strip()) > 0:
            # Show a small non-intrusive message about OCR success
            pass  # We can add a success indicator here if needed
        
        # Debug: Show OCR text and date extraction results
        if is_debug():
            with st.expander("🔍 OCR & Date Debug", expanded=False):
                st.text_area("Raw OCR Text:", value=extracted_text, height=150, key="debug_ocr_text")
                st.markdown(f"**Text Length:** {len(extracted_text)} characters")
                
                # Test date extraction on the actual OCR text
                if extracted_text:
                    test_dates = extract_dates_from_text(extracted_text)
                    if test_dates:
                        date_strs = [d.strftime("%d/%m/%Y (%A, %B %d, %Y)") for d in test_dates]
                        st.success(f"✅ Found {len(test_dates)} date(s): {', '.join(date_strs)}")
                    else:
                        st.warning("⚠️ No dates found in OCR text")
                        
                        # Show potential date-like patterns for debugging
                        patterns = re.findall(r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b', extracted_text)
                        if patterns:
                            st.info(f"Found date-like patterns: {patterns}")
                        else:
                            st.info("No date-like patterns found in OCR text")
        
        # Always store detected dates regardless of auto-detect setting
        if extracted_dates:
            st.session_state.detected_dates = extracted_dates
            # Set the most recent date as default if auto-detect is enabled
            if st.session_state.get('auto_date_detect', True):
                st.session_state.detected_invoice_date = extracted_dates[-1]  # Use the most recent date
        else:
            # Clear previous dates if none found in current document
            st.session_state.detected_dates = []
            if 'detected_invoice_date' in st.session_state:
                del st.session_state.detected_invoice_date
        
        if is_debug():
            gpu_info = {
                'torch_available': 'torch' in locals(),
                'gpu_available': gpu_available,
                'using': 'GPU' if gpu_available else 'CPU'
            }
            _log_ocr_debug(filename, po_numbers, gpu_info=gpu_info, invoice_numbers=invoice_numbers, extracted_dates=extracted_dates, extracted_text=extracted_text)
            
        return extracted_text
        
    except Exception as e:
        if is_debug():
            st.error(f"❌ OCR Error for {filename}: {str(e)}")
            gpu_info = {
                'torch_available': 'torch' in locals(),
                'gpu_available': gpu_available,
                'using': 'GPU' if gpu_available else 'CPU'
            }
            _log_ocr_debug(filename, [], gpu_info=gpu_info)
        return ""

def _extract_po_numbers_from_text(text: str) -> List[str]:
    """Find PO matches based on configured prefixes + 6 digits."""
    if not text:
        return []
    prefixes = _po_prefix()
    all_matches = []
    
    for prefix in prefixes:
        # Make pattern more flexible - allow for spaces or other characters between prefix and numbers
        po_pattern = r'\b' + re.escape(prefix) + r'\s*\d{6}\b'
        matches = re.findall(po_pattern, text, re.IGNORECASE)
        # Clean up matches by removing any spaces
        cleaned_matches = [re.sub(r'\s+', '', match.upper()) for match in matches]
        all_matches.extend(cleaned_matches)
    
    return list(set(all_matches))

def _extract_invoice_numbers_from_text(text: str) -> List[str]:
    """Extract invoice numbers from text using configured patterns."""
    if not st.session_state.get('auto_invoice_detect', True):
        return []
        
    invoice_numbers = []
    patterns = st.session_state.get('invoice_patterns', ['INVOICE NUMBER:', 'INVOICE NO:', 'INVOICE#', 'INV:', 'INVOICE:'])
    
    for pattern in patterns:
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
        
        # Clean up matches - stop at whitespace, tab, newline, or other delimiters
        for match in matches:
            # Split on common delimiters and take the first part
            clean_match = re.split(r'[\s\t\n\r,;|]', match)[0]
            if clean_match and len(clean_match) <= 20:
                invoice_numbers.append(clean_match)
    
    return list(set(invoice_numbers))  # Remove duplicates

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
# PDF extraction (improved with debug info)
# =========================================================
def extract_po_numbers_from_single_pdf(file_path, filename):
    po_numbers = []
    invoice_numbers = []
    debug_info = {"tables_found": 0, "pages_processed": 0, "extraction_methods": []}
    
    try:
        # Try camelot for table extraction
        if is_debug():
            debug_info["extraction_methods"].append("camelot")
        
        tables = camelot.read_pdf(file_path, pages='all')
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
        
        # Try PyPDF2 for text extraction
        if PyPDF2 is not None:
            try:
                if is_debug():
                    debug_info["extraction_methods"].append("PyPDF2")
                    
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    debug_info["pages_processed"] = len(reader.pages)
                
                all_text = ""
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
                
                # Store detected invoice number in session state
                unique_invoices = list(set(invoice_numbers))
                if unique_invoices and st.session_state.get('auto_invoice_detect', True):
                    st.session_state.detected_invoice_number = unique_invoices[0]
                
                if is_debug():
                    _log_ocr_debug(f"PDF: {filename}", list(set(po_numbers)), invoice_numbers=unique_invoices)
                    
            except Exception as e:
                if is_debug():
                    st.warning(f"PyPDF2 extraction failed: {str(e)}")
        else:
            if is_debug():
                st.warning("PyPDF2 not available. Install with: `pip install PyPDF2`")
        
        # Try pdfplumber as alternative
        if pdfplumber is not None:
            try:
                if is_debug():
                    debug_info["extraction_methods"].append("pdfplumber")
                    
                with pdfplumber.open(file_path) as pdf:
                    all_text = ""
                    for page in pdf.pages:
                        text = page.extract_text() or ""
                        all_text += text + "\n"
                        for prefix in prefixes:
                            po_pattern = r'\b' + re.escape(prefix) + r'\s*\d{6}\b'
                            matches = re.findall(po_pattern, text, re.IGNORECASE)
                            for match in matches:
                                clean_match = re.sub(r'\s+', '', match.upper())
                                po_numbers.append(clean_match)
                        
            except Exception as e:
                if is_debug():
                    st.warning(f"pdfplumber extraction failed: {str(e)}")
        else:
            if is_debug():
                st.info("pdfplumber not available (optional). Install with: `pip install pdfplumber`")
                
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
                        
                    # Images (OCR enabled)
                    elif member.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')):
                        # Extract text using OCR
                        try:
                            text = _ocr_text_from_image_bytes(content, member)
                            found_pos = _extract_po_numbers_from_text(text)
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

    # No need to filter deleted rows since they are removed immediately when checked
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
        # Try to use st_tags if available, otherwise fall back to text input with visual tags
        if STREAMLIT_TAGS_AVAILABLE:
            # Use sidebar-specific tags input widget
            current_prefixes = st_tags_sidebar(
                label='PO Prefixes',
                text='Press enter to add more',
                value=_po_prefix() if 'po_prefix' in st.session_state else ['RCH'],
                suggestions=['RCH', 'PO', 'INV', 'REQ', 'ORD', 'PUR'],
                maxtags=10,
                key='po_prefix_tags'
            )
            
            # Update session state with the tag values
            if current_prefixes:
                st.session_state.po_prefix = ','.join(current_prefixes)
            else:
                st.session_state.po_prefix = 'RCH'
                
        else:
            # Fall back to regular text input with tags displayed below
            st.text_input(
                "PO Prefixes", 
                key="po_prefix",
                help="Comma-separated list of PO prefixes to search for (e.g., RCH,PO,INV)",
                placeholder="RCH,PO,INV"
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
            if STREAMLIT_TAGS_AVAILABLE:
                invoice_patterns = st_tags_sidebar(
                    label='Invoice Number Patterns',
                    text='Press enter to add patterns',
                    value=st.session_state.get('invoice_patterns', ['INVOICE NUMBER:', 'INVOICE NO:', 'INVOICE#', 'INV:', 'INVOICE:']),
                    suggestions=['INVOICE NUMBER:', 'INVOICE NO:', 'INVOICE#', 'INV:', 'INVOICE:', 'BILL NUMBER:', 'REFERENCE:'],
                    maxtags=10,
                    key='invoice_patterns_tags'
                )
                
                st.session_state.invoice_patterns = invoice_patterns
                
            else:
                st.text_input(
                    "Invoice Patterns", 
                    key="invoice_patterns_input",
                    value="INVOICE NUMBER:,INVOICE NO:,INVOICE#,INV:,INVOICE:",
                    help="Comma-separated patterns to search for invoice numbers"
                )
                st.session_state.invoice_patterns = [p.strip() for p in st.session_state.get('invoice_patterns_input', '').split(',') if p.strip()]
        
        # Date Auto-Detection Settings
        st.header("Date Detection")
        st.toggle(
            "Auto-detect Dates",
            value=st.session_state.get('auto_date_detect', True),
            key="auto_date_detect",
            help="Automatically extract dates from document text and provide as options"
        )
        
        # Show detected dates in sidebar for easy access
        if "detected_dates" in st.session_state and st.session_state.detected_dates:
            detected_dates = st.session_state.detected_dates
            st.success(f"📅 Found {len(detected_dates)} date(s) in document")
            
            # Create selectbox for detected dates
            date_options = {d.strftime("%d/%m/%Y (%A)"): d for d in detected_dates}
            selected_sidebar_date = st.selectbox(
                "📄 Dates from Document:",
                options=list(date_options.keys()),
                key="sidebar_selected_date",
                index=len(date_options)-1 if date_options else 0,  # Default to most recent
                help="Click to auto-fill invoice date fields"
            )
            
            # Store selected date for auto-population
            if selected_sidebar_date:
                st.session_state.selected_document_date = date_options[selected_sidebar_date]
                if st.button("📌 Apply to All Invoices", key="apply_date_all", type="secondary"):
                    # Apply selected date to all PO contexts
                    for po_id in st.session_state.get("po_context", {}):
                        st.session_state.po_context[po_id]["invoice_date"] = date_options[selected_sidebar_date]
                    st.success("✅ Date applied to all invoices!")
                    st.rerun()
        elif st.session_state.get('auto_date_detect', True):
            st.info("📄 Upload a document to detect dates")
        else:
            st.info("🔍 Date detection disabled")
        
        # Debug: Date Extraction Test
        if is_debug():
            st.markdown("---")
            st.subheader("🔍 Date Detection Test")
            test_text = st.text_area(
                "Test Text for Date Extraction:",
                value="Invoice Date: 15/09/2025\nDue Date: 30 September 2025\nAmount: $1000",
                height=100,
                key="date_test_text"
            )
            
            if st.button("Test Date Extraction", key="test_date_extraction"):
                test_dates = extract_dates_from_text(test_text)
                if test_dates:
                    date_strs = [d.strftime("%d/%m/%Y") for d in test_dates]
                    st.success(f"Found dates: {', '.join(date_strs)}")
                    # Update session state for testing
                    st.session_state.detected_dates = test_dates
                    st.session_state.detected_invoice_date = test_dates[-1] if test_dates else None
                else:
                    st.warning("No dates found in test text")
            
            # Show current detected dates
            if "detected_dates" in st.session_state:
                detected = st.session_state.detected_dates
                if detected:
                    date_strs = [d.strftime("%d/%m/%Y") for d in detected]
                    st.info(f"Currently detected dates: {', '.join(date_strs)}")
                else:
                    st.info("No dates currently detected")
            else:
                st.info("No date detection has run yet")

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
    st.markdown("Upload **PDF, Images, or ZIP** to find PO numbers, verify in Coupa, **edit lines**, and create invoices")
    if st.session_state.get("environment") == "Production":
        st.error("🛑 **You are in Production. Use with caution.**")

    # File upload (PDF, images, or ZIP supported with OCR)
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "zip", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif"],
        help="Upload a PDF, image, or ZIP containing PDFs/images"
    )

    if uploaded_file:
        # Create a unique identifier for the uploaded file (name + size + type)
        file_id = f"{uploaded_file.name}_{uploaded_file.size}_{uploaded_file.type}"
        
        # Check if we've already processed this exact file
        if (st.session_state.get("last_processed_file_id") == file_id and 
            "processed_po_numbers" in st.session_state):
            # Use cached results
            found_pos = st.session_state.processed_po_numbers
            if is_debug():
                with st.expander(f"🔍 Cache Debug – {uploaded_file.name}", expanded=False):
                    st.write("**Cache Status**: Hit")
                    st.write(f"**File ID**: {file_id}")
                    st.write(f"**Cached PO Count**: {len(found_pos)}")
                    st.write("**Status**: Using previously processed results (no re-processing needed)")
        else:
            # Process the file
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
            detected_date = st.session_state.get("detected_invoice_date") if st.session_state.get('auto_date_detect', True) else None
            default_date = detected_date if detected_date else datetime.today().date()
            
            st.session_state.po_context.setdefault(po_id, {
                "doc_bytes": doc_bytes,
                "doc_name": doc_name,
                "invoice_number": "",
                "invoice_date": default_date,
                "tax_rate": 10.0  # default GST-A 10%
            })

            # Default batch selection (True) set once
            st.session_state.selected_for_batch.setdefault(po_id, True)

            with st.expander(f"📄 {po_id}", expanded=True):
                # Auto-populate invoice number FIRST before validation
                current_inv_num = st.session_state.po_context[po_id].get("invoice_number", "")
                detected_invoice = st.session_state.get("detected_invoice_number", "")
                
                # Auto-populate if no current value and we have a detected value
                if not current_inv_num and detected_invoice and st.session_state.get('auto_invoice_detect', True):
                    st.session_state.po_context[po_id]["invoice_number"] = detected_invoice
                    current_inv_num = detected_invoice
                    st.success(f"✅ Auto-populated invoice number: {detected_invoice}")
                
                # THEN do validation - check current invoice number
                if not current_inv_num or not current_inv_num.strip():
                    st.error("⚠️ Invoice Number is required. This PO will be skipped during batch creation.")

                # Mobile-first responsive layout with split columns for key fields
                # Split the form into logical sections
                form_col1, form_col2 = st.columns([1, 1])
                
                with form_col1:
                    st.markdown("**📄 Invoice Information**")
                    # Use on_change callback to properly update state
                    def update_invoice_number():
                        # Get the current value from the widget
                        current_value = st.session_state[f"inv_num_{po_id}"]
                        st.session_state.po_context[po_id]["invoice_number"] = current_value
                        # Update include status based on invoice number
                        if current_value and current_value.strip():
                            st.session_state.selected_for_batch[po_id] = True
                        else:
                            st.session_state.selected_for_batch[po_id] = False
                    
                    # Invoice number input (auto-population already handled above)
                    current_invoice = st.session_state.po_context[po_id].get("invoice_number", "")
                    detected_invoice = st.session_state.get("detected_invoice_number", "")
                    
                    st.text_input(
                        "Invoice Number", 
                        key=f"inv_num_{po_id}", 
                        placeholder="Auto-detected or enter manually",
                        value=current_invoice,
                        on_change=update_invoice_number,
                        help="Invoice number auto-detected from document text" if detected_invoice and current_invoice == detected_invoice else None
                    )
                
                with form_col2:
                    st.markdown("**📅 Date & Tax**")
                    # Date and tax in sub-columns
                    date_tax_cols = st.columns([2, 1])
                    with date_tax_cols[0]:
                        # Simple date input with auto-population from sidebar
                        current_date = st.session_state.po_context[po_id]["invoice_date"]
                        selected_doc_date = st.session_state.get("selected_document_date")
                        
                        # Auto-populate from sidebar selection if available
                        if selected_doc_date and current_date == date.today():
                            st.session_state.po_context[po_id]["invoice_date"] = selected_doc_date
                            current_date = selected_doc_date
                        
                        st.session_state.po_context[po_id]["invoice_date"] = st.date_input(
                            "Invoice Date", 
                            key=f"inv_date_{po_id}",
                            value=current_date,
                            format="DD/MM/YYYY",
                            help="💡 Use sidebar to select from detected dates" if st.session_state.get("detected_dates") else None
                        )
                    with date_tax_cols[1]:
                        st.session_state.po_context[po_id]["tax_rate"] = st.number_input(
                            "Tax %", key=f"tax_rate_{po_id}",
                            value=float(st.session_state.po_context[po_id]["tax_rate"]),
                            step=0.01
                        )
                
                # PO Information section
                st.markdown("**📋 Purchase Order Information**")
                info_col1, info_col2 = st.columns([1, 1])
                with info_col1:
                    st.markdown(f"**PO Number:** {po_data.get('number') or po_id}")
                    st.markdown(f"**Status:** {po_data.get('status', 'Unknown')}")
                with info_col2:
                    supplier_name = (po_data.get('supplier') or {}).get('name', 'Unknown')
                    st.markdown(f"**Supplier:** {supplier_name}")
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

                # Determine default account id to inherit for new lines
                default_acct = _default_account_id_from(preview_df)

                # Action buttons - 3 columns (removed Remove button)
                btn_col1, btn_col2, btn_col3 = st.columns(3)
                
                with btn_col1:
                    if st.button("➕ Add Qty", key=f"add_qty_{po_id}", help="Add Quantity Line", width="stretch"):
                        new_row = {
                            "line_num": len(preview_df) + 1,
                            "inv_type": "InvoiceQuantityLine",
                            "description": "",
                            "price": 0.00,
                            "quantity": 1.00,
                            "uom_code": DEFAULT_UOM_CODE,
                            "account_id": default_acct,
                            "commodity_name": None,
                            "po_number": po_id,
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
                        st.rerun()
                        
                with btn_col2:
                    if st.button("➕ Add Amt", key=f"add_amt_{po_id}", help="Add Amount Line", use_container_width=True):
                        new_row = {
                            "line_num": len(preview_df) + 1,
                            "inv_type": "InvoiceAmountLine",
                            "description": "",
                            "price": 0.00,
                            "quantity": 0.00,
                            "uom_code": None,
                            "account_id": default_acct,
                            "commodity_name": None,
                            "po_number": po_id,
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
                        st.rerun()
                        
                with btn_col3:
                    if st.button("↩️ Restore All", key=f"restore_all_{po_id}", help="Restore to original data from extraction", use_container_width=True):
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

                # Editor config - responsive column widths
                col_cfg = {
                    "line_num": st.column_config.NumberColumn("#", disabled=True, width="small"),
                    "inv_type": st.column_config.SelectboxColumn("Type", options=["InvoiceQuantityLine", "InvoiceAmountLine"], required=True, width="small"),
                    "description": st.column_config.TextColumn("Description", required=True, width="medium"),  # Changed from large to medium
                    "quantity": st.column_config.NumberColumn("Qty", min_value=0.0, step=0.01, format="%.2f", width="small"),
                    "price": st.column_config.NumberColumn("Price", min_value=0.0, step=0.01, format="%.2f", width="medium"),  # Changed from small to medium
                    "uom_code": st.column_config.TextColumn("UOM", width="small"),
                    "source_part_num": st.column_config.TextColumn("Part #", width="medium"),
                    "commodity_name": st.column_config.TextColumn("Commodity", width="small"),  # Changed from medium to small
                    "po_number": st.column_config.TextColumn("PO #", width="small"),
                    "order_line_num": st.column_config.TextColumn("Line #", width="small"),
                    "delete": st.column_config.CheckboxColumn("🗑️", width="small", help="Check to immediately delete this row")  # Shortened header
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

                # Sanitize and store back - but check for deleted rows first
                edited_df = sanitize_editor_rows(edited_df)
                
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

                # Calculate totals dynamically (all rows are active now since deleted ones are removed)
                subtotal = float(edited_df["price"].fillna(0).astype(float).mul(edited_df["quantity"].fillna(0).astype(float)).sum())
                est_tax = round(subtotal * (float(st.session_state.po_context.setdefault(po_id, {}).get("tax_rate", 10)) / 100.0), 2)
                net_total = subtotal + est_tax
                
                # Show line count
                total_lines = len(edited_df)
                
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

                # Debug section (expandable)
                with st.expander("🔧 Debug Info", expanded=False):
                    debug_cols = st.columns(3)
                    with debug_cols[0]:
                        st.caption("Session State Status:")
                        has_original = "invoice_lines_original" in st.session_state
                        has_po_original = has_original and po_id in st.session_state.invoice_lines_original if has_original else False
                        original_count = len(st.session_state.invoice_lines_original.get(po_id, [])) if has_po_original else 0
                        
                        st.write(f"✅ Original data exists: {has_original}")
                        st.write(f"✅ PO original data: {has_po_original}")
                        st.write(f"📊 Original rows: {original_count}")
                    
                    with debug_cols[1]:
                        st.caption("Current Data:")
                        st.write(f"📊 Current rows: {len(edited_df)}")
                        st.write(f"🆔 PO ID: {po_id}")
                        
                    with debug_cols[2]:
                        st.caption("Actions:")
                        if st.button("🔄 Reset Original", key=f"reset_original_{po_id}", help="Recreate original from current data"):
                            if "invoice_lines_original" not in st.session_state:
                                st.session_state.invoice_lines_original = {}
                            st.session_state.invoice_lines_original[po_id] = edited_df.copy()
                            st.success("✅ Original data reset to current state")
                            st.rerun()

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
                    return "⏭️ Skipped"
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
                elif inv_stat.startswith("⏭️"):
                    skip += 1
                rows.append({
                    "PO": r["PO"],
                    "Invoice #": r["Invoice #"],
                    "Invoice ID": r["Invoice ID"],
                    "Invoice Status": inv_stat,
                    "Scan Status": scan_stat
                })

            # Big metrics row (like subtotal/metrics)
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("✅ Success", succ)
            with m2: st.metric("❌ Failed", fail)
            with m3: st.metric("⏭️ Skipped", skip)

            # Results table - don't use LinkColumn, just show the data
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True
            )

            # Buttons to view successful invoices (in case LinkColumn not available)
            success_links = [r for r in st.session_state.create_results if r.get("Link")]
            if success_links:
                st.markdown("**View Created Invoices:**")
                for r in success_links:
                    if r["Link"]:
                        invoice_num = r.get("Invoice #", r.get("Invoice ID", ""))
                        st.link_button(f"View Invoice {invoice_num}", r["Link"])

if __name__ == "__main__":
    main()
