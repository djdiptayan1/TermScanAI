#!/usr/bin/env python3
"""
Email fetching and attachment handling module
"""
import imaplib
import email
import os
import re
import json
import pandas as pd
from email.header import decode_header
import PyPDF2
import docx
import pytesseract
from PIL import Image
import cv2
import numpy as np
from vision_ocr import VisionOCR
from typing import Optional, Dict, List, Tuple

# Initialize Google Cloud Vision client
vision_ocr = None


def init_vision_ocr(credentials_path: Optional[str] = None):
    """Initialize Google Cloud Vision OCR"""
    global vision_ocr
    try:
        vision_ocr = VisionOCR(credentials_path)
        print("Google Cloud Vision OCR initialized successfully")
    except Exception as e:
        print(f"Failed to initialize Google Cloud Vision OCR: {str(e)}")
        print("Falling back to Tesseract OCR")


def fetch_email_by_trade_id(
    trade_id: str, email_credentials: Tuple[str, str]
) -> Optional[Dict]:
    """
    Connect to email server and fetch the latest email containing the trade_id

    Args:
        trade_id: The trade ID to search for
        email_credentials: Tuple of (username, password)

    Returns:
        Dictionary containing email data and attachments, or None if not found
    """
    username, password = email_credentials

    try:
        # Connect to the server
        mail = imaplib.IMAP4_SSL(
            "imap.gmail.com"
        )  # Change based on your email provider
        mail.login(username, password)
        mail.select("inbox")

        # Search for emails containing the trade ID
        status, messages = mail.search(None, f'SUBJECT "{trade_id}"')

        if not messages[0]:
            return None

        # Get the latest email
        latest_email_id = messages[0].split()[-1]
        status, msg_data = mail.fetch(latest_email_id, "(RFC822)")

        email_message = email.message_from_bytes(msg_data[0][1])

        # Process attachments
        attachments = []
        for part in email_message.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get("Content-Disposition") is None:
                continue

            filename = part.get_filename()

            if filename:
                # Decode filename if needed
                if decode_header(filename)[0][1] is not None:
                    filename = decode_header(filename)[0][0].decode(
                        decode_header(filename)[0][1]
                    )

                # Save attachment
                filepath = os.path.join("temp_attachments", filename)
                os.makedirs("temp_attachments", exist_ok=True)
                with open(filepath, "wb") as f:
                    f.write(part.get_payload(decode=True))
                attachments.append({"filename": filename, "path": filepath})

        return {
            "subject": email_message["subject"],
            "from": email_message["from"],
            "date": email_message["date"],
            "attachments": attachments,
        }

    except Exception as e:
        print(f"Error fetching email: {str(e)}")
        return None

    finally:
        try:
            mail.close()
            mail.logout()
        except:
            pass


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF using Google Cloud Vision (primary) or PyPDF2 (fallback)
    """
    text = ""

    # Try Google Cloud Vision first
    if vision_ocr:
        try:
            text = vision_ocr.extract_text_from_pdf(pdf_path)
            if text:
                return text
        except Exception as e:
            print(f"Google Cloud Vision PDF processing failed: {str(e)}")
            print("Falling back to PyPDF2")

    # Fallback to PyPDF2
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
    except Exception as e:
        print(f"PyPDF2 processing failed: {str(e)}")

    return text


def extract_text_from_docx(docx_path):
    """Extract text from Word document"""
    text = ""
    try:
        doc = docx.Document(docx_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error extracting text from Word doc: {e}")
    return text


def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from image using Google Cloud Vision (primary) or Tesseract (fallback)
    """
    text = ""

    # Try Google Cloud Vision first
    if vision_ocr:
        try:
            text = vision_ocr.extract_text_from_image(image_path)
            if text:
                return text
        except Exception as e:
            print(f"Google Cloud Vision OCR failed: {str(e)}")
            print("Falling back to Tesseract OCR")

    # Fallback to Tesseract
    try:
        # Use OpenCV to improve image quality
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Use pytesseract
        text = pytesseract.image_to_string(img)
    except Exception as e:
        print(f"Tesseract OCR failed: {str(e)}")

    return text


def extract_trade_details(text):
    """
    Extract key trade details from text using regex patterns
    Returns a dictionary of extracted fields
    """
    trade_details = {}

    # Define regex patterns for key fields
    patterns = {
        "trade_date": r"(?:Trade Date|Trade\s*Date)[:\s]+([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})",
        "value_date": r"(?:Value Date|Settlement Date|Value\s*Date)[:\s]+([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})",
        "currency": r"Currency:\s*([A-Z]{3})",
        "underlying_currency": r"(?:Underlying|Base)\s*Currency[:\s]+([A-Z]{3})",
        "counter_currency": r"(?:Counter|Quote)\s*Currency[:\s]+([A-Z]{3})",
        "spot_price": r"(?:Spot Price|Spot Rate|Spot)[:\s]+([\d\.]+)",
        "fixing_level": r"(?:Fixing Level|Fixing Rate|Fixing)[:\s]+([\d\.]+)",
    }

    # Apply patterns to extract data
    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            trade_details[field] = match.group(1).strip()
        else:
            trade_details[field] = None

    return trade_details


def process_attachments(attachments):
    """
    Process each attachment to extract text based on file type
    """
    extracted_text = ""

    for attachment in attachments:
        file_path = attachment["path"]
        file_name = attachment["filename"].lower()

        if file_name.endswith(".pdf"):
            extracted_text += extract_text_from_pdf(file_path)
        elif file_name.endswith(".docx") or file_name.endswith(".doc"):
            extracted_text += extract_text_from_docx(file_path)
        elif file_name.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
            extracted_text += extract_text_from_image(file_path)

    return extracted_text


def export_to_excel(trade_details, output_path="extracted_trades.xlsx"):
    """
    Export the extracted trade details to Excel
    """
    # Convert dict to DataFrame
    df = pd.DataFrame([trade_details])

    # Export to Excel
    df.to_excel(output_path, index=False)
    print(f"Trade details exported to {output_path}")

    return output_path


def validate_trade_details(extracted_details, reference_file=None):
    """
    Basic validation of trade details
    If reference file is provided, validate against it
    """
    validation_results = {"is_valid": True, "missing_fields": [], "mismatch_fields": []}

    # Check for missing fields
    for field, value in extracted_details.items():
        if value is None or value == "":
            validation_results["missing_fields"].append(field)
            validation_results["is_valid"] = False

    # Validate against reference file if provided
    if reference_file and os.path.exists(reference_file):
        try:
            ref_df = pd.read_excel(reference_file)
            ref_data = ref_df.iloc[
                0
            ].to_dict()  # Assuming first row has the reference data

            for field in extracted_details:
                if field in ref_data:
                    # Normalize values for comparison
                    extracted_val = extracted_details[field]
                    reference_val = ref_data[field]

                    # Convert numeric values to float for comparison if possible
                    if (
                        field in ["spot_price", "fixing_level"]
                        and extracted_val
                        and reference_val
                    ):
                        try:
                            extracted_val = float(extracted_val)
                            reference_val = float(reference_val)
                        except (ValueError, TypeError):
                            pass  # If conversion fails, compare as strings

                    # Compare values
                    if extracted_val != reference_val:
                        validation_results["mismatch_fields"].append(
                            {
                                "field": field,
                                "extracted": extracted_details[field],
                                "reference": ref_data[field],
                            }
                        )
                        validation_results["is_valid"] = False
        except Exception as e:
            print(f"Error validating against reference file: {e}")

    return validation_results


def process_trade(trade_id, email_credentials, reference_file=None):
    """
    Main function to process a trade by ID
    """
    print(f"Processing trade ID: {trade_id}")

    # Fetch email with attachments
    email_data = fetch_email_by_trade_id(trade_id, email_credentials)
    if not email_data:
        return {"error": f"No email found for trade ID: {trade_id}"}

    print(f"Email found. Processing {len(email_data['attachments'])} attachments...")

    # Process attachments to extract text
    extracted_text = process_attachments(email_data["attachments"])

    # Extract trade details from text
    trade_details = extract_trade_details(extracted_text)
    trade_details["trade_id"] = trade_id  # Add trade ID to details

    # Export to Excel
    excel_path = export_to_excel(trade_details)

    # Validate extracted data
    validation_results = validate_trade_details(trade_details, reference_file)

    # Clean up temporary files (optional)
    # for attachment in email_data['attachments']:
    #     if os.path.exists(attachment["path"]):
    #         os.remove(attachment["path"])

    return {
        "trade_details": trade_details,
        "validation": validation_results,
        "excel_path": excel_path,
    }


def cleanup_attachments(attachments: List[Dict]) -> None:
    """
    Clean up temporary attachment files

    Args:
        attachments: List of attachment dictionaries containing file paths
    """
    for attachment in attachments:
        try:
            if os.path.exists(attachment["path"]):
                os.remove(attachment["path"])
        except Exception as e:
            print(f"Error cleaning up attachment {attachment['path']}: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Replace with your email credentials
    EMAIL_USER = "your_email@gmail.com"
    EMAIL_PASSWORD = "your_password"  # For Gmail, use app password

    # Get trade ID from user
    trade_id = input("Enter the trade ID to process: ")

    # Optional: Path to reference Excel file for validation
    reference_file = input(
        "Enter path to reference file (optional, press Enter to skip): "
    )
    if reference_file.strip() == "":
        reference_file = None

    # Process the trade
    result = process_trade(trade_id, (EMAIL_USER, EMAIL_PASSWORD), reference_file)

    # Display results
    if "error" in result:
        print(result["error"])
    else:
        print("\nExtracted Trade Details:")
        for field, value in result["trade_details"].items():
            print(f"{field}: {value}")

        print("\nValidation Results:")
        if result["validation"]["is_valid"]:
            print("All fields valid and match reference data.")
        else:
            if result["validation"]["missing_fields"]:
                print(
                    f"Missing fields: {', '.join(result['validation']['missing_fields'])}"
                )
            if result["validation"]["mismatch_fields"]:
                print("Mismatched fields:")
                for mismatch in result["validation"]["mismatch_fields"]:
                    print(
                        f"  {mismatch['field']}: Extracted={mismatch['extracted']}, Reference={mismatch['reference']}"
                    )

        print(f"\nTrade details exported to: {result['excel_path']}")
