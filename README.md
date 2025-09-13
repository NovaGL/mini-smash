# 📄 miniSmash Invoice Creator

A powerful Streamlit application for automated invoice processing and creation in Coupa. Extract PO numbers from PDFs and images using optimized OCR, verify them in Coupa, edit invoice details, and create invoices in batch with built-in duplicate detection and real-time progress tracking.

## ✨ Features

- **⚡ Fast OCR Processing**: Extract text from PDFs and images with smart caching
- **📦 ZIP File Support**: Process multiple documents at once
- **🔍 PO Verification**: Verify purchase orders in Coupa with duplicate detection
- **✏️ Easy Editing**: Edit invoice details with auto-save and smart validation
- **💰 Currency Formatting**: Professional price display and field inheritance
- **🤖 Auto-Population**: Automatically detects and fills invoice numbers and dates
- **📊 Batch Processing**: Create multiple invoices at once
- **🛡️ Production Safety**: Built-in warnings and confirmation steps
- **🔧 Debug Mode**: Comprehensive troubleshooting information
- **📱 Mobile Friendly**: Works on both desktop and mobile devices

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Coupa instance with OAuth application configured
- Required environment variables (see Setup section)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd miniSmash-Invoice-Creator
```

2. Install required packages:
```bash
pip install streamlit requests pandas pillow easyocr pdf2image streamlit-tags
```

3. Set up environment variables (see Configuration section)

4. Run the application:
```bash
streamlit run pdf_to_csv.py
```

## ⚙️ Configuration

### Environment Variables

Set the following environment variables for your target environments:

**Test Environment:**
- `COUPA_INSTANCE` - Your Coupa instance name (e.g., "mycompany")
- `PO_IDENTIFIER` - Your OAuth client identifier
- `PO_SECRET` - Your OAuth client secret

**Production Environment:**
- `COUPA_INSTANCE_PROD` - Production instance name
- `PO_IDENTIFIER_PROD` - Production OAuth client identifier  
- `PO_SECRET_PROD` - Production OAuth client secret

### Required OAuth Scopes (Client Credentials Grant)

Your OAuth application in Coupa must have these scopes:
- `core.purchase_order.read` - Read PO information
- `core.invoice.read` - Check for duplicate invoices
- `core.invoice.write` - Create new invoices

## 📖 How to Use

### 1. Configure Settings (Sidebar)
- **Environment**: Select Test or Production environment
- **PO Prefixes**: Set prefixes to search for (looks for prefix + 6 digits, e.g., RCH123456, PO456789)
- **Auto-detect Invoice Numbers**: Toggle automatic extraction using patterns like:
  - "INVOICE NUMBER:", "INVOICE NO:", "INVOICE#", "INVOICE #:", "INV:", "INVOICE:"
  - Looks for numbers immediately following these text patterns
- **Auto-detect Dates**: Toggle automatic date extraction supporting formats like:
  - DD/MM/YYYY, MM/DD/YYYY, YYYY/MM/DD (with /, -, or . separators)
  - Month names: "January 15, 2024", "15 January 2024"
  - Compact formats: DDMMYYYY, YYYYMMDD
- **Debug Mode**: Enable detailed API logging and OCR troubleshooting information

### 2. Upload Your Document
- Upload a PDF, image, or ZIP file with your invoices
- The app will find PO numbers and invoice details automatically
- Works with: PDF, PNG, JPG, JPEG, TIF, TIFF, BMP, GIF, ZIP
- Large files (over 10MB) will show warnings but still work

### 3. Verify POs in Coupa
- Click "Verify POs in Coupa" to check your PO numbers are real
- The app will also look for duplicate invoices at the same time
- You'll see progress bars and results for each PO

### 4. Edit Invoice Details
- Each PO gets its own section where you can make changes
- Invoice numbers and dates are filled in automatically
- Edit quantities, prices, and descriptions in the table
- Changes auto-save when you click away (no Enter key needed)
- Use ➕ buttons to add new lines or 🗑️ checkboxes to delete lines
- "↩️ Restore All" button brings back the original data if you make mistakes

### 5. Review What You're Creating
- The summary table shows all invoices ready to create
- Use checkboxes in the "Include" column to pick which ones you want
- Problem invoices (duplicates, errors) are automatically unchecked for safety
- You can still edit invoice numbers directly in this table
- Check the totals and line counts look right

### 6. Create Your Invoices
- Production users must check a safety box first
- Click "Create Invoices" to process your selected invoices
- You'll see detailed progress bars showing each step:
  - "Processing 1/5: PO 123456" with percentage complete
  - Individual stages like "Creating invoice..." and "Uploading file..."
  - Warnings if files are large and might be slow
- Results show success/failure with helpful error messages

## 🔧 How It Works

- Uses OCR to extract PO numbers, invoice numbers, and dates from documents
- Verifies PO numbers in Coupa and checks for duplicate invoices
- Provides easy editing interface with auto-save functionality
- Creates invoices in batch with progress tracking

## 🛡️ Safety Features

- **Production Warnings**: Big red warnings when you're working with real data
- **Duplicate Detection**: Finds existing invoices and warns you before creating duplicates  
- **Confirmation Steps**: Makes you confirm before doing anything important

## 🐛 Troubleshooting

### Turn On Debug Mode
Enable Debug Mode in the sidebar to see what's happening:
- **File Processing**: See how long things take and what was found in your documents
- **API Calls**: Watch the app communicate with Coupa (passwords hidden for security)
- **Errors**: Get clear explanations of what went wrong and how to fix it
- **Performance**: See if your computer is using GPU acceleration for faster processing

### Common Issues

- **PO numbers not found**: Check PO prefixes and document quality
- **Environment errors**: Verify Coupa credentials and OAuth setup
- **Large files**: Files over 10MB process slower, over 20MB may timeout
- **Performance**: First run takes 30-60s for OCR setup, then much faster

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Enable Debug Mode for detailed error information
3. Review the environment variable configuration
4. Check your OAuth application setup in Coupa

## 🎯 Tips for Best Results

- Use clear, readable documents (PDFs work better than blurry images)
- Always verify auto-filled invoice numbers and dates
- Test in Test mode before using Production
- Watch for duplicate warnings and keep files under 10MB for best performance

---

Made with ❤️ using Streamlit and EasyOCR
