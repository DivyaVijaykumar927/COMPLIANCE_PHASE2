import csv
# File paths
input_file = "master_clauses.csv"
output_file = "cleaned_file.csv"

# List of required columns
required_columns = [
    "Filename", "Document Name-Answer", "Parties", "Agreement Date", 
    "Effective Date-Answer", "Renewal Term", "Notice Period To Terminate Renewal", "Expiration Date-Answer",
    "Governing Law-Answer", "Exclusivity", "Post-Termination Services"
]

# Read and filter rows
valid_rows = []
with open(input_file, "r", encoding="utf-8") as infile:
    reader = csv.reader(infile)
    
    # Read the header and get indices of required columns
    header = next(reader)
    column_indices = [header.index(col) for col in required_columns if col in header]
    
    # Append only the required columns to the valid_rows list
    valid_rows.append([header[i] for i in column_indices])  # Add filtered header
    
    for row in reader:
        # Filter row to keep only required columns
        filtered_row = [row[i] for i in column_indices]
        
        # Check if all cells in the row are non-empty
        if all(cell.strip() for cell in filtered_row):
            valid_rows.append(filtered_row)
        
        if len(valid_rows) > 101:  # Stop after collecting 100 valid rows + header
            break

# Save the filtered rows to a new CSV
with open(output_file, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerows(valid_rows)

print("First 100 rows with complete data and specific columns saved to cleaned_file.csv.")
