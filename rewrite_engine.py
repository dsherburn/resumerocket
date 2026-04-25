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

import html
import re

import anthropic
import httpx
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

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


def _md_inline(text: str) -> str:
    """Escape HTML then convert **bold** and *italic* to reportlab XML tags."""
    text = html.escape(text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    return text


def markdown_to_pdf(md: str) -> bytes:
    """Convert markdown resume text to a formatted PDF."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        rightMargin=0.75 * inch, leftMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    )

    name_s = ParagraphStyle("name", fontSize=20, fontName="Helvetica-Bold",
                            alignment=TA_CENTER, spaceAfter=2)
    contact_s = ParagraphStyle("contact", fontSize=9, fontName="Helvetica",
                               alignment=TA_CENTER, spaceAfter=6,
                               textColor=colors.HexColor("#555555"))
    h2_s = ParagraphStyle("h2", fontSize=10.5, fontName="Helvetica-Bold",
                          spaceBefore=10, spaceAfter=2)
    h3_s = ParagraphStyle("h3", fontSize=10, fontName="Helvetica-Bold",
                          spaceBefore=5, spaceAfter=1)
    bullet_s = ParagraphStyle("bullet", fontSize=9.5, leftIndent=14,
                              spaceAfter=1, spaceBefore=1)
    body_s = ParagraphStyle("body", fontSize=9.5, spaceAfter=3)

    story = []
    in_name_block = True

    for line in md.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("# "):
            story.append(Paragraph(_md_inline(stripped[2:].strip()), name_s))
            in_name_block = True
        elif stripped.startswith("## "):
            in_name_block = False
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=colors.HexColor("#cccccc"), spaceAfter=2))
            story.append(Paragraph(_md_inline(stripped[3:].strip().upper()), h2_s))
        elif stripped.startswith("### "):
            in_name_block = False
            story.append(Paragraph(_md_inline(stripped[4:].strip()), h3_s))
        elif stripped.startswith(("- ", "* ")):
            in_name_block = False
            story.append(Paragraph(f"• {_md_inline(stripped[2:].strip())}", bullet_s))
        elif in_name_block:
            story.append(Paragraph(_md_inline(stripped), contact_s))
        else:
            story.append(Paragraph(_md_inline(stripped), body_s))

    doc.build(story)
    return buf.getvalue()


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

    resume_pdf = markdown_to_pdf(rewritten_resume)
    attachments = [
        {
            "filename": "resume_rewritten.pdf",
            "content": base64.b64encode(resume_pdf).decode(),
        }
    ]

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
