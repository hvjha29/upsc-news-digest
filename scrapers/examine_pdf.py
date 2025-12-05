import pdfplumber

# Examine more pages to understand the actual table format
with pdfplumber.open("docs/Polity-PT730_v3.pdf") as pdf:
    # Look at pages that likely contain the actual notes tables
    for i in range(3, min(10, len(pdf.pages))):  # pages 4-9
        page = pdf.pages[i]
        print(f"\n--- Page {i+1} ---")
        text = page.extract_text()
        print("Text snippet:", text[:500])
        
        # Check for tables on the page
        tables = page.extract_tables()
        if tables:
            print(f"Found {len(tables)} table(s) on page {i+1}")
            for j, table in enumerate(tables):
                print(f"Table {j+1} has {len(table)} rows:")
                for k, row in enumerate(table):
                    if k < 5:  # Show first 5 rows
                        print(f"  Row {k+1}: {row}")
                    else:
                        print("  ...")
                        break