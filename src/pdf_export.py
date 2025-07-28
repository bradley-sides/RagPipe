import markdown2
from xhtml2pdf import pisa

def render_text_to_pdf(markdown_text, output_path="output.pdf"):
    html = markdown2.markdown(markdown_text)
    with open(output_path, "w+b") as result_file:
        pisa.CreatePDF(html, dest=result_file)
    return output_path