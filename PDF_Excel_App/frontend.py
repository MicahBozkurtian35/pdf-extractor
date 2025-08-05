import streamlit as st
import os
import uuid
import shutil
import PyPDF2  # For counting PDF pages
from TestGeneratedGraph import process_pdf_to_excel  # Your backend function

# Inject some CSS to try to style checked checkboxes with a green background.
st.markdown(
    """
    <style>
    /* This targets the div following the checkbox input when checked.
       Note: Styling of built-in Streamlit widgets is limited and may change between versions. */
    div[data-testid="stCheckbox"] input:checked + div {
        background-color: green !important;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set up directories for uploads and outputs
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.title("PDF to Excel Graph Data Extractor")

# File uploader widget
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# --- Page Selector Function ---
def page_selector(num_pages, columns_per_row=10):
    """
    Displays a grid of checkboxes (one per page) where none are checked by default.
    Also displays a "Toggle All" button that selects/deselects all pages.
    Returns a sorted list of selected page numbers.
    """
    # Initialize session state for selected pages if not already set.
    if "selected_pages" not in st.session_state:
        st.session_state["selected_pages"] = set()  # default: none selected

    # Toggle All button: if clicked, select all pages if not all are selected; otherwise clear.
    if st.button("Toggle All"):
        if len(st.session_state["selected_pages"]) < num_pages:
            st.session_state["selected_pages"] = set(range(1, num_pages + 1))
        else:
            st.session_state["selected_pages"] = set()
        st.experimental_rerun()  # Force immediate update

    # Build the grid of checkboxes
    selected = st.session_state["selected_pages"]
    for start in range(1, num_pages + 1, columns_per_row):
        cols = st.columns(min(columns_per_row, num_pages - start + 1))
        for i, col in enumerate(cols):
            page_number = start + i
            # The checkbox is checked if the page number is in the set.
            current = page_number in selected
            if col.checkbox(f"{page_number}", value=current, key=f"page_{page_number}"):
                # If the box is checked, add to set
                selected.add(page_number)
            else:
                # If unchecked, remove it from the set (if it exists)
                if page_number in selected:
                    selected.remove(page_number)
    st.write("Pages selected:", ", ".join(map(str, sorted(selected))))
    return sorted(selected)

# --- Main App Logic ---
if uploaded_file is not None:
    # Save the uploaded PDF to disk with a unique name.
    unique_id = uuid.uuid4().hex
    pdf_filename = f"{unique_id}.pdf"
    pdf_path = os.path.join(UPLOAD_DIR, pdf_filename)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File uploaded successfully!")

    # Count pages using PyPDF2.
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        num_pages = len(pdf_reader.pages)
        st.write(f"This PDF has **{num_pages}** pages.")
    except Exception as e:
        st.error(f"Error reading PDF page count: {e}")
        num_pages = 0

    # Display the page selector if there are pages.
    if num_pages > 0:
        st.write("Select pages to process:")
        selected_pages = page_selector(num_pages)
    else:
        selected_pages = None

    # Create progress bar and status placeholders.
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Progress callback function.
    def progress_callback(percent):
        progress_bar.progress(int(percent))
        status_text.text(f"Processing: {int(percent)}%")

    if st.button("Convert"):
        excel_filename = f"{unique_id}.xlsx"
        excel_path = os.path.join(OUTPUT_DIR, excel_filename)
        try:
            # Pass the selected_pages list to your backend function.
            process_pdf_to_excel(pdf_path, excel_path, selected_pages=selected_pages, progress_callback=progress_callback)
            st.success("Conversion complete!")
            # Provide a download button for the Excel file.
            with open(excel_path, "rb") as f:
                excel_data = f.read()
            st.download_button(
                label="Download Excel File",
                data=excel_data,
                file_name=excel_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Error during conversion: {e}")
