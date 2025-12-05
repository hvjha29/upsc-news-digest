import pdfplumber
import csv
import os
from pathlib import Path

def extract_tables_from_pdf(pdf_path):
    """Extract tables from a PDF file"""
    all_tables = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table in tables:
                all_tables.append({
                    'page_number': page_num + 1,  # 1-indexed
                    'table_data': table
                })
    
    return all_tables

def clean_text(text):
    """Clean text by removing extra whitespace and newlines"""
    if text is None:
        return ""
    # Replace newlines with spaces and clean up extra whitespace
    cleaned = " ".join(str(text).split())
    return cleaned

def is_notes_table(table):
    """
    Determine if a table appears to be a notes table based on:
    - Has 2 columns
    - Left column has topic names (non-empty)
    - Right column has explanations
    """
    if not table or len(table) == 0:
        return False
    
    # Check if the table has 2 columns with meaningful content
    two_column_rows = [row for row in table if len(row) >= 2]
    if len(two_column_rows) == 0:
        return False
    
    # Consider it a notes table if it has at least 2 rows with non-empty left column
    valid_rows = 0
    for row in two_column_rows:
        left_col = clean_text(row[0]) if row[0] else ""
        right_col = clean_text(row[1]) if len(row) > 1 and row[1] else ""
        
        # Check if left column has a topic (contains numbers or topic-like content)
        if left_col and ('.' in left_col or 'Topic' in left_col or 'Rights' in left_col or 
                         any(c.isdigit() for c in left_col) or len(right_col) > len(left_col)):
            valid_rows += 1
    
    return valid_rows >= 2

def parse_pdf_to_csv(pdf_path, output_csv_path):
    """Parse PDF and write to CSV in the required format"""
    print(f"Parsing: {pdf_path}")
    
    # Extract tables from PDF
    all_tables = extract_tables_from_pdf(pdf_path)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # Get paper name - use specific name for polity, generic for others
    if "Polity" in pdf_path:
        paper_name = "Indian Polity & Governance Notes"
    else:
        filename = Path(pdf_path).stem
        paper_name = filename.replace('-', ' ').replace('_', ' ') + " Notes"
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow([
            'id', 'year', 'paper', 'question_no', 'question_text', 
            'word_limit', 'marks', 'topic_hint', 'source_url'
        ])
        
        row_id = 1
        
        for table_info in all_tables:
            table_data = table_info['table_data']
            
            # Check if this looks like a notes table
            if is_notes_table(table_data):
                for row in table_data:
                    if len(row) >= 2:  # Ensure the row has at least 2 columns
                        left_col = clean_text(row[0]) if row[0] else ""
                        right_col = clean_text(row[1]) if len(row) > 1 and row[1] else ""
                        
                        # Skip if left column is empty and doesn't look like a topic
                        if not left_col or not right_col:
                            continue
                        
                        # Check if this row looks like a proper topic-explanation pair
                        # Topic should have a number or keyword indicating it's a topic
                        if '.' in left_col or any(c.isdigit() for c in left_col) or len(right_col) > len(left_col):
                            # Extract a short summary for topic_hint (first 100 characters of explanation)
                            topic_hint = right_col[:1000] + "..." if len(right_col) > 100 else right_col

                            # If left column starts with a number, try to extract just the topic name
                            topic_name = left_col
                            if left_col.count('.') >= 1:  # If it has number.topic format
                                # Extract part after the number
                                parts = left_col.split('.', 1)
                                if len(parts) > 1 and any(c.isdigit() for c in parts[0]):
                                    topic_name = parts[1].strip()
                                else:
                                    topic_name = left_col

                            # Write row to CSV
                            writer.writerow([
                                row_id,                                    # id
                                "2022-2023",                              # year (from cover page)
                                paper_name,                                # paper
                                row_id,                                    # question_no
                                topic_hint,                                # question_text
                                "",                                        # word_limit
                                "",                                        # marks
                                topic_name,                                # topic_hint
                                f"file://{os.path.abspath(pdf_path)}"      # source_url
                            ])
                            
                            row_id += 1
    
    print(f"Successfully parsed {pdf_path} and wrote to {output_csv_path}")

def main():
    # Define the PDF files to parse
    pdf_files = [
        "docs/Polity-PT730_v3.pdf",
        "docs/Geography-PT-730.pdf", 
        "docs/History_PT730_v2.pdf",
        "docs/International-Relations-PT730.pdf",
        "docs/Science-and-Technology_v5-1.pdf",
        "docs/Environment_PT730_v2.pdf"
    ]
    
    # Process each PDF file
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            # Create output filename based on source PDF - for the main request, create the polity one
            if "Polity" in pdf_file:
                output_path = "data/upsc_polity_notes.csv"
            else:
                base_name = os.path.basename(pdf_file).replace('.pdf', '').replace('-', '_').replace(' ', '_')
                output_path = f"data/{base_name}_notes.csv"
            parse_pdf_to_csv(pdf_file, output_path)
        else:
            print(f"Warning: {pdf_file} not found, skipping...")

if __name__ == "__main__":
    main()