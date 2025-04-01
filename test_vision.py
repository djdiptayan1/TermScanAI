#!/usr/bin/env python3
"""
Test script for TermSheet Validation with Google Cloud Vision integration
"""
import os
import json
from vision_ocr import VisionOCR
from fetchEmails import extract_trade_details, validate_trade_details, export_to_excel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def process_file_with_vision(file_path: str, trade_id: str, reference_file: str = None):
    """Process a file using Google Cloud Vision and extract trade details."""
    print(f"\nProcessing file: {file_path}")
    print("-" * 50)

    try:
        # Initialize Vision OCR
        vision_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not vision_credentials:
            print(
                "Warning: No Google Cloud Vision credentials found in environment variables."
            )
            print("Please set GOOGLE_APPLICATION_CREDENTIALS environment variable.")
            return None

        ocr = VisionOCR(vision_credentials)

        # Extract text using Vision OCR
        print("\nExtracting text using Google Cloud Vision...")
        extracted_text = ocr.extract_text(file_path)
        print(f"Extracted text length: {len(extracted_text)}")
        print("\nFirst 500 characters of extracted text:")
        print("-" * 50)
        print(
            extracted_text[:500] + "..."
            if len(extracted_text) > 500
            else extracted_text
        )

        # Get document properties
        print("\nDetecting document properties...")
        properties = ocr.detect_document_properties(file_path)
        print("\nDocument properties:")
        print("-" * 50)
        print(json.dumps(properties, indent=2))

        # Extract trade details
        print("\nExtracting trade details...")
        trade_details = extract_trade_details(extracted_text)
        trade_details["trade_id"] = trade_id

        # Export to Excel
        excel_path = export_to_excel(trade_details, f"{trade_id}_extracted.xlsx")
        print(f"\nTrade details exported to: {excel_path}")

        # Validate if reference file provided
        if reference_file and os.path.exists(reference_file):
            print("\nValidating against reference data...")
            validation_results = validate_trade_details(trade_details, reference_file)
            print("\nValidation results:")
            print("-" * 50)
            print(json.dumps(validation_results, indent=2))

        return {
            "trade_details": trade_details,
            "document_properties": properties,
            "excel_path": excel_path,
        }

    except Exception as e:
        print(f"\nError processing document: {str(e)}")
        return None


def main():
    """Main function to test the integration."""
    # Test files
    test_files = {
        "PDF": "sample_term_sheet.pdf",
        "Image": "sample_term_sheet.png",  # You'll need to create this
    }

    # Reference file for validation
    reference_file = "reference_data.xlsx"

    # Process each file
    for file_type, file_path in test_files.items():
        if os.path.exists(file_path):
            print(f"\nProcessing {file_type} file...")
            result = process_file_with_vision(
                file_path, f"TEST_{file_type}", reference_file
            )

            if result:
                print(f"\nSuccessfully processed {file_type} file!")
                print(f"Results saved to: {result['excel_path']}")
        else:
            print(f"\nSkipping {file_type} file: {file_path} (file not found)")


if __name__ == "__main__":
    main()
