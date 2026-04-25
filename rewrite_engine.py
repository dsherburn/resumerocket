"""
Resume Rewrite Engine
Triggered by webhook from Tally form after Stripe payment confirmation.
"""

import os
import json
import base64
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from io import BytesIO
from pathlib import Path

import re

import anthropic
import httpx
from fpdf import FPDF

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
FROM_EMAIL = "onboarding@resend.dev"

RESUME_SYSTEM_PROMPT = """You are an expert resume writer with 15 years of experience helping professionals land roles at top companies. You write in clear, concise, achievement-focused language with strong action verbs and quantified results wherever possible.

Your task: Rewrite the provided resume to maximize interview callbacks for the stated target role.

Rules:
- Keep all facts accurate — never invent achievements or numbers
- Lead every bullet with a strong action verb (Delivered, Drove, Built, Reduced, Increased, Led, Designed, etc.)
- Quantify impact where numbers exist in the original (revenue, %, headcount, time saved, etc.)
- Remove filler phrases: "responsible for", "worked on", "helped with", "assisted in"
- Optimize for ATS keyword matching for the target role — include relevant keywords naturally
- Output format: clean markdown mirroring standard resume structure (Name, Contact, Summary, Experience, Education, Skills)
- Max length: 1 page for <10 years experience, 2 pages for 10+ years
- Write a 3-sentence summary at the top that positions the candidate for the target role

Target role: {target_role}
Career level: {career_level}
Key achievement to highlight: {key_achievement}"""

LINKEDIN_SYSTEM_PROMPT = """You are a LinkedIn profile expert. Rewrite the provided LinkedIn headline and About section to attract recruiters for the target role.

Rules:
- Headline: max 220 characters, keyword-rich, value-focused (not just job title)
- About: 3-5 short paragraphs, first-person, conversational but professional
- End the About section with a clear call to action
- Optimize for LinkedIn search keywords for the target role

Target role: {target_role}"""


def extract_text_from_resume(file_content: bytes, filename: str) -> str:
    """Extract text from uploaded resume file using Claude's file understanding."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    if filename.lower().endswith(".pdf"):
        media_type = "application/pdf"
        encoded = base64.standard_b64encode(file_content).decode("utf-8")
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Extract all text from this resume exactly as written, preserving structure. Output only the extracted text, no commentary."
                    }
                ],
            }],
        )
    else:
        # Plain text or DOCX — treat as text for now
        try:
            return file_content.decode("utf-8", errors="ignore")
        except Exception:
            return file_content.decode("latin-1", errors="ignore")

    return message.content[0].text


def rewrite_resume(
    resume_text: str,
    target_role: str,
    career_level: str,
    key_achievement: str,
) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    system = RESUME_SYSTEM_PROMPT.format(
        target_role=target_role,
        career_level=career_level,
        key_achievement=key_achievement,
    )

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=system,
        messages=[{
            "role": "user",
            "content": f"Here is the resume to rewrite:\n\n{resume_text}"
        }],
    )

    return message.content[0].text


def optimize_linkedin(
    linkedin_url: str,
    target_role: str,
    resume_text: str,
) -> str:
    """Generate optimized LinkedIn headline + About section."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    system = LINKEDIN_SYSTEM_PROMPT.format(target_role=target_role)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system,
        messages=[{
            "role": "user",
            "content": (
                f"LinkedIn profile URL: {linkedin_url}\n\n"
                f"Here is the candidate's resume for context:\n\n{resume_text}\n\n"
                "Please write an optimized LinkedIn headline and About section."
            )
        }],
    )

    return message.content[0].text


_UNICODE_MAP = {
    '—': '--', '–': '-', '‒': '-',
    '‘': "'", '’': "'", '‚': "'",
    '“': '"', '”': '"', '„': '"',
    '…': '...', '·': '-', '•': '-',
}

def _plain(text: str) -> str:
    """Strip markdown markers and normalize to latin-1 safe text for fpdf2."""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    for ch, rep in _UNICODE_MAP.items():
        text = text.replace(ch, rep)
    return text.encode('latin-1', 'replace').decode('latin-1')


def markdown_to_pdf(md: str) -> bytes:
    """Convert markdown resume text to a formatted PDF using fpdf2."""
    pdf = FPDF(format="Letter")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=19)

    L = 19        # left margin ~0.75"
    W = pdf.w - 2 * L

    in_name_block = True

    for line in md.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("# "):
            pdf.set_font("Helvetica", "B", 20)
            pdf.set_x(L)
            pdf.cell(W, 9, _plain(stripped[2:].strip()), align="C",
                     new_x="LMARGIN", new_y="NEXT")
            in_name_block = True

        elif stripped.startswith("## "):
            in_name_block = False
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 10.5)
            pdf.set_x(L)
            pdf.cell(W, 6, _plain(stripped[3:].strip()).upper(),
                     new_x="LMARGIN", new_y="NEXT")
            y = pdf.get_y()
            pdf.line(L, y, L + W, y)
            pdf.ln(1)

        elif stripped.startswith("### "):
            in_name_block = False
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_x(L)
            pdf.cell(W, 5, _plain(stripped[4:].strip()),
                     new_x="LMARGIN", new_y="NEXT")

        elif stripped.startswith(("- ", "* ")):
            in_name_block = False
            pdf.set_font("Helvetica", "", 9.5)
            pdf.set_x(L + 4)
            pdf.multi_cell(W - 4, 4.5, f"- {_plain(stripped[2:].strip())}")

        elif in_name_block:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_x(L)
            pdf.cell(W, 4.5, _plain(stripped), align="C",
                     new_x="LMARGIN", new_y="NEXT")

        else:
            pdf.set_font("Helvetica", "", 9.5)
            pdf.set_x(L)
            pdf.multi_cell(W, 4.5, _plain(stripped))

    return bytes(pdf.output())


def send_delivery_email(
    customer_email: str,
    customer_name: str,
    rewritten_resume: str,
    linkedin_copy: str | None = None,
) -> None:
    """Send the completed deliverables via Resend API."""
    subject = "Your ResumeRocket rewrite is ready 🚀"

    linkedin_note = (
        "<p>Your LinkedIn optimization is included as a separate attachment.</p>"
        if linkedin_copy else ""
    )
    body_html = f"""
    <p>Hi {customer_name},</p>
    <p>Your rewritten resume is attached as a PDF. Copy the content directly into your job applications.</p>
    {linkedin_note}
    <p>If you land interviews, we'd love to hear about it. If you're not happy for any reason, reply to this email for a full refund - no questions asked.</p>
    <p>Good luck,<br/>ResumeRocket</p>
    """

    try:
        resume_pdf = markdown_to_pdf(rewritten_resume)
        attach_name = "resume_rewritten.pdf"
        attach_data = base64.b64encode(resume_pdf).decode()
    except Exception as pdf_err:
        print(f"PDF generation failed ({type(pdf_err).__name__}: {pdf_err}), sending as text")
        attach_name = "resume_rewritten.txt"
        attach_data = base64.b64encode(rewritten_resume.encode("utf-8")).decode()

    attachments = [{"filename": attach_name, "content": attach_data}]

    if linkedin_copy:
        attachments.append({
            "filename": "linkedin_optimization.txt",
            "content": base64.b64encode(linkedin_copy.encode()).decode(),
        })

    payload = {
        "from": FROM_EMAIL,
        "to": [customer_email],
        "subject": subject,
        "html": body_html,
        "attachments": attachments,
    }

    resp = httpx.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()


def process_order(order: dict) -> None:
    """
    Main handler. Called by webhook from n8n/Zapier after Stripe payment confirmed.

    Expected order dict:
    {
        "customer_email": str,
        "customer_name": str,
        "target_role": str,
        "career_level": str,  # "IC" | "Manager" | "Director" | "VP" | "C-suite"
        "key_achievement": str,
        "resume_file_content": bytes,  # raw file bytes
        "resume_filename": str,
        "linkedin_url": str | None,     # None for resume-only orders
        "bundle": bool,                 # True if $149 bundle
    }
    """
    print(f"Processing order for {order['customer_email']} — role: {order['target_role']}")

    resume_text = extract_text_from_resume(
        order["resume_file_content"],
        order["resume_filename"],
    )

    rewritten = rewrite_resume(
        resume_text=resume_text,
        target_role=order["target_role"],
        career_level=order["career_level"],
        key_achievement=order["key_achievement"],
    )

    linkedin_copy = None
    if order.get("bundle") and order.get("linkedin_url"):
        linkedin_copy = optimize_linkedin(
            linkedin_url=order["linkedin_url"],
            target_role=order["target_role"],
            resume_text=resume_text,
        )

    send_delivery_email(
        customer_email=order["customer_email"],
        customer_name=order["customer_name"],
        rewritten_resume=rewritten,
        linkedin_copy=linkedin_copy,
    )

    print(f"Delivered to {order['customer_email']}")


# --- Webhook handler (Flask) ---
# Drop this behind a Vercel serverless function or Railway app

import threading
from flask import Flask, request, jsonify

app = Flask(__name__)


def get_field(raw_fields, label_fragment: str):
    """Find a field value by partial label match (case-insensitive)."""
    label_fragment = label_fragment.lower()
    for f in raw_fields:
        if label_fragment in f.get("label", "").lower():
            return f.get("value")
    return None


def process_tally_payload(payload: dict) -> None:
    """Parse Tally payload and run the order pipeline. Runs in a background thread."""
    try:
        print("Tally payload keys:", list(payload.keys()))

        # Tally wraps fields under payload["data"]["fields"]
        raw_fields = (
            payload.get("data", {}).get("fields", [])
            or payload.get("fields", [])
        )
        print(f"Found {len(raw_fields)} fields:")
        for f in raw_fields:
            print(f"  label={f.get('label')!r}  type={f.get('type')}  value_type={type(f.get('value')).__name__}")

        # File upload
        resume_files = get_field(raw_fields, "resume") or get_field(raw_fields, "upload")
        if not isinstance(resume_files, list) or not resume_files:
            print("ERROR: No resume file found in fields")
            return

        file_obj = resume_files[0]
        file_url = file_obj.get("url", "")
        if not file_url:
            print("ERROR: File object has no URL:", file_obj)
            return

        print(f"Downloading resume from {file_url}")
        file_resp = httpx.get(file_url, timeout=60)
        file_resp.raise_for_status()

        customer_name = get_field(raw_fields, "name") or get_field(raw_fields, "full") or "Customer"
        if isinstance(customer_name, list):
            customer_name = customer_name[0] if customer_name else "Customer"

        order = {
            "customer_email": get_field(raw_fields, "email") or "",
            "customer_name": str(customer_name),
            "target_role": get_field(raw_fields, "role") or get_field(raw_fields, "targeting") or "",
            "career_level": get_field(raw_fields, "career level") or get_field(raw_fields, "level") or "IC",
            "key_achievement": get_field(raw_fields, "achievement") or "",
            "resume_file_content": file_resp.content,
            "resume_filename": file_obj.get("name", "resume.pdf"),
            "linkedin_url": get_field(raw_fields, "linkedin"),
            "bundle": False,
        }

        print(f"Order built: email={order['customer_email']} role={order['target_role']}")
        process_order(order)

    except Exception as e:
        print(f"ERROR in process_tally_payload: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


@app.route("/webhook/tally", methods=["POST"])
def tally_webhook():
    """Receives Tally form submission and immediately returns 200.
    Processing happens in a background thread so Tally never times out."""
    payload = request.get_json(force=True)
    # Acknowledge immediately -- Claude rewrite takes 30-60s, Tally would time out
    thread = threading.Thread(target=process_tally_payload, args=(payload,))
    thread.daemon = True
    thread.start()
    return jsonify({"status": "accepted"}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)
