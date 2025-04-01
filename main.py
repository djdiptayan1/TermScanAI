#!/usr/bin/env python3
"""
TermSheet Validation AI Prototype

This script provides a command-line interface for extracting and validating
trade details from term sheets in various formats (PDF, Word, images).
"""
import argparse
import os
import json
import getpass
import sys
from typing import Dict, Any, Optional
from fetchEmails import fetch_email_by_trade_id, cleanup_attachments
from document_processor import DocumentProcessor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="TermSheet Validation Tool")

    # Main operation modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--email", action="store_true", help="Extract from email")
    group.add_argument(
        "--file", type=str, help="Path to local file to process (PDF, Word, image)"
    )

    # Required arguments
    parser.add_argument(
        "--trade-id", type=str, required=True, help="Trade ID to process"
    )

    # Optional arguments
    parser.add_argument(
        "--email-user", type=str, help="Email username (if using email mode)"
    )
    parser.add_argument(
        "--reference", type=str, help="Path to reference data for validation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="extracted_trades.xlsx",
        help="Output Excel file path (default: extracted_trades.xlsx)",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save results as JSON in addition to Excel",
    )
    parser.add_argument(
        "--vision-credentials",
        type=str,
        help="Path to Google Cloud Vision credentials JSON file",
    )

    return parser.parse_args()


def process_trade(
    trade_id: str,
    file_path: Optional[str] = None,
    email_credentials: Optional[tuple] = None,
    reference_file: Optional[str] = None,
    output_file: str = "extracted_trades.xlsx",
    vision_credentials: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Process a trade from either email or local file"""
    try:
        # Initialize document processor
        processor = DocumentProcessor(vision_credentials)

        # Get document content
        if email_credentials:
            # Fetch from email
            email_data = fetch_email_by_trade_id(trade_id, email_credentials)
            if not email_data:
                print(f"No email found for trade ID: {trade_id}")
                return None

            print(f"Processing {len(email_data['attachments'])} attachments...")

            # Process each attachment
            results = []
            for attachment in email_data["attachments"]:
                try:
                    result = processor.process_document(attachment["path"])
                    results.append(result)
                except Exception as e:
                    print(
                        f"Error processing attachment {attachment['filename']}: {str(e)}"
                    )

            # Clean up attachments
            cleanup_attachments(email_data["attachments"])

            # Combine results
            combined_text = "\n\n".join(r["text"] for r in results)
            combined_metadata = {
                "type": "email",
                "attachments": len(results),
                "email_subject": email_data["subject"],
                "email_from": email_data["from"],
                "email_date": email_data["date"],
            }

            result = {"text": combined_text, "metadata": combined_metadata}

        else:
            # Process local file
            if not os.path.exists(file_path):
                print(f"Error: File {file_path} not found.")
                return None

            result = processor.process_document(file_path)

        # Add trade ID to result
        result["trade_id"] = trade_id

        # Export to Excel
        import pandas as pd

        df = pd.DataFrame([result])
        df.to_excel(output_file, index=False)
        print(f"\nTrade details exported to: {output_file}")

        # Save as JSON if requested
        if args.save_json:
            json_file = f"{trade_id}_results.json"
            with open(json_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to JSON file: {json_file}")

        return result

    except Exception as e:
        print(f"Error processing trade: {str(e)}")
        return None


def display_results(result: Dict[str, Any]) -> None:
    """Display the processing results"""
    if not result:
        return

    print("\nExtracted Trade Details:")
    print("-" * 50)
    print(f"Trade ID: {result['trade_id']}")
    print(f"Document Type: {result['metadata']['type']}")
    print(
        f"Processing Method: {result['metadata'].get('processing_method', 'primary')}"
    )

    if result["metadata"]["type"] == "email":
        print(f"Email Subject: {result['metadata']['email_subject']}")
        print(f"From: {result['metadata']['email_from']}")
        print(f"Date: {result['metadata']['email_date']}")

    print("\nExtracted Text (first 500 chars):")
    print("-" * 50)
    print(result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"])

    print("\nMetadata:")
    print("-" * 50)
    print(json.dumps(result["metadata"], indent=2))


def main():
    """Main function"""
    global args
    args = parse_arguments()

    # Process based on mode
    if args.email:
        # Email processing mode
        if not args.email_user:
            args.email_user = input("Enter email username: ")
        email_password = getpass.getpass("Enter email password: ")

        result = process_trade(
            args.trade_id,
            email_credentials=(args.email_user, email_password),
            reference_file=args.reference,
            output_file=args.output,
            vision_credentials=args.vision_credentials,
        )
    else:
        # Local file processing mode
        result = process_trade(
            args.trade_id,
            file_path=args.file,
            reference_file=args.reference,
            output_file=args.output,
            vision_credentials=args.vision_credentials,
        )

    if not result:
        sys.exit(1)

    # Display results
    display_results(result)


if __name__ == "__main__":
    main()
