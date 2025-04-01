# TermSheet Validation AI

A Python-based tool for extracting and validating trade details from term sheets using Google Cloud Vision AI and Tesseract OCR.

## Features

- Process term sheets from various sources:
  - Local files (PDF, Word, images)
  - Email attachments
- Multiple OCR engines:
  - Primary: Google Cloud Vision AI
  - Fallback: Tesseract OCR
- Support for multiple file formats:
  - PDF documents
  - Word documents (.docx)
  - Images (PNG, JPEG, TIFF, BMP, GIF)
- Export results to:
  - Excel (.xlsx)
  - JSON (optional)
- Detailed metadata and confidence scores
- Email integration with attachment handling

## Project Structure

```
.
├── main.py              # Main entry point and CLI interface
├── document_processor.py # Document processing and OCR handling
├── fetchEmails.py       # Email fetching and attachment handling
├── vision_ocr.py        # Google Cloud Vision AI integration
├── requirements.txt     # Python dependencies
├── .env                # Environment variables
└── GeminiAPIClient.json # Google Cloud Vision credentials
```

## Prerequisites

- Python 3.8 or higher
- Google Cloud Vision API credentials
- Tesseract OCR (for fallback)
- Poppler (for PDF processing)

### Installing Prerequisites

1. Install Tesseract OCR:

   ```bash
   # macOS
   brew install tesseract

   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr

   # Windows
   # Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
   ```

2. Install Poppler:

   ```bash
   # macOS
   brew install poppler

   # Ubuntu/Debian
   sudo apt-get install poppler-utils

   # Windows
   # Download and install from: http://blog.alivate.com.au/poppler-windows/
   ```

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd term-sheet-validation
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your Google Cloud Vision credentials path:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=path/to/GeminiAPIClient.json
   ```

## Usage

### Command Line Interface

1. Process a local file:

   ```bash
   python main.py --file sample_term_sheet.pdf --trade-id TEST_001
   ```

2. Process from email:

   ```bash
   python main.py --email --trade-id TEST_001
   ```

3. With Google Vision credentials:

   ```bash
   python main.py --file sample_term_sheet.pdf --trade-id TEST_001 --vision-credentials path/to/credentials.json
   ```

4. Save results as JSON:
   ```bash
   python main.py --file sample_term_sheet.pdf --trade-id TEST_001 --save-json
   ```

### Command Line Arguments

- `--email`: Process from email (mutually exclusive with --file)
- `--file`: Path to local file to process
- `--trade-id`: Trade ID to process (required)
- `--email-user`: Email username (if using email mode)
- `--reference`: Path to reference data for validation
- `--output`: Output Excel file path (default: extracted_trades.xlsx)
- `--save-json`: Save results as JSON in addition to Excel
- `--vision-credentials`: Path to Google Cloud Vision credentials JSON file

### Email Processing

For email processing, you'll need to:

1. Enable IMAP in your email account
2. For Gmail:
   - Enable 2-factor authentication
   - Generate an App Password
   - Use the App Password instead of your regular password

## Output

The tool generates two types of output:

1. Excel file (`extracted_trades.xlsx`):

   - Contains extracted text and metadata
   - One row per processed document

2. JSON file (if --save-json is used):
   - Contains detailed information including:
     - Extracted text
     - Document metadata
     - Processing method
     - Confidence scores
     - Email information (if applicable)

## Error Handling

- The tool automatically falls back to Tesseract OCR if Google Cloud Vision fails
- Temporary files are cleaned up after processing
- Detailed error messages are provided for troubleshooting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Cloud Vision AI for OCR capabilities
- Tesseract OCR for fallback processing
- All contributors and maintainers
