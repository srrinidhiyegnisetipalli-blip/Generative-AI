import gradio as gr
from pypdf import PdfReader

reader = PdfReader("the_nestle_hr_policy_pdf_2012.pdf")

pdf_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        pdf_text += text + "\n"

def answer_question(question):
    question = question.lower()

    if question in pdf_text.lower():
        return "Information found in the document."

    lines = pdf_text.split("\n")

    for line in lines:
        if any(word in line.lower() for word in question.split()):
            return line

    return "No relevant information found."

app = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Ask a question"),
    outputs="text",
    title="Nestle HR Assistant (Free Version)"
)
app.launch()