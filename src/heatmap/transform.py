from PyPDF2 import PdfReader, PdfWriter

def merge_pdfs_side_by_side(pdf1_path, pdf2_path, output_path):
    # Initialize PDF reader for both input files
    pdf1 = PdfReader(pdf1_path)
    pdf2 = PdfReader(pdf2_path)
    
    # Get the first page of each PDF
    page1 = pdf1.pages[0]
    page2 = pdf2.pages[0]
    
    # Get dimensions of the input pages
    page1_width = float(page1.mediabox.width)
    page1_height = float(page1.mediabox.height)
    page2_width = float(page2.mediabox.width)
    page2_height = float(page2.mediabox.height)
    
    # Define output page size (A4: 595 x 842 points)
    output_width = 462 * 2
    output_height = 600
    
    # Calculate scaling factors to fit both pages side by side
    scale1 = min((output_width - 4) / 2 / page1_width, output_height / page1_height)
    scale2 = min((output_width - 4) / 2 / page2_width, output_height / page2_height)
    
    # Create a new PDF writer
    writer = PdfWriter()
    
    # Create a blank page
    new_page = writer.add_blank_page(width=output_width, height=output_height)
    
    # Merge first page (left side)
    new_page.mergeScaledTranslatedPage(
        page2=page1,
        scale=scale1,
        tx=1,  # No translation in x (left side)
        ty=(output_height - page1_height * scale1) / 2,  # Center vertically
        expand=False
    )
    
    # Merge second page (right side)
    new_page.mergeScaledTranslatedPage(
        page2=page2,
        scale=scale2,
        tx=output_width / 2 + 1,  # Move to right half
        ty=(output_height - page2_height * scale2) / 2,  # Center vertically
        expand=False
    )
    
    # Write the output PDF
    with open(output_path, 'wb') as output_file:
        writer.write(output_file)

# Example usage
if __name__ == "__main__":
    merge_pdfs_side_by_side('2010_no_legend_flat.pdf', '2020_legend_flat.pdf', 'output.pdf')