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

import html as html_lib
import re

import anthropic
import httpx
try:
    from fpdf import FPDF
    _FPDF_AVAILABLE = True
except ImportError:
    _FPDF_AVAILABLE = False

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
FROM_EMAIL = os.environ.get("FROM_EMAIL", "onboarding@resend.dev")
FALLBACK_FROM_EMAIL = "onboarding@resend.dev"

_last_error: dict = {}

RESUME_SYSTEM_PROMPT = """You are an expert resume writer with 15 years of experience helping professionals land roles at top companies. You write clear, concise, achievement-focused resumes that pass ATS screening and impress hiring managers.

Your task: Rewrite the provided resume to maximize interview callbacks for the stated target role.

FORMAT & STRUCTURE:
- Single-column layout only - no tables, text boxes, columns, headers/footers, or graphics
- Standard section headings: Summary, Experience, Education, Skills, Certifications
- Standard fonts only (Arial, Helvetica, Calibri, Georgia) - do not specify fonts in markdown output
- 1 page for under 10 years experience; 2 pages max for senior roles
- Deliver as clean markdown that can be converted to both .docx and .pdf

CONTENT RULES:
- Keep all facts accurate - never invent achievements or numbers
- Lead with a 2-4 line professional summary tailored to the target role (not an objective statement)
- Every bullet: strong action verb + what you did + measurable result/impact. Quantify wherever possible (%, $, time saved, team size, scale)
- Prioritize accomplishments over responsibilities - cut generic duties
- Remove filler phrases: "responsible for", "worked on", "helped with", "assisted in"
- Mirror keywords and phrases from the target role naturally throughout (especially in Skills and Experience)
- Reverse-chronological order for experience
- Remove: photos, full mailing address, references, "References available upon request"
- Location: city + state or "Remote" only - no street address
- Include LinkedIn URL and GitHub/portfolio if mentioned in original
- Skills section: clean list of relevant hard skills, tools, technologies only - no skill bars or ratings
- Match tone to industry (creative roles can be slightly more expressive; finance/legal/corporate stay conservative)

BEFORE FINALIZING:
1. Resume is tailored to the specific target role (not generic)
2. Every bullet has a measurable outcome where possible
3. No typos, inconsistent verb tenses, or formatting inconsistencies
4. Would parse cleanly in ATS (no images, tables, fancy formatting)
5. Most relevant experience and keywords appear in top third of page one

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


def _fmt_html(text: str) -> str:
    """Escape HTML then apply markdown bold/italic inline."""
    text = html_lib.escape(text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    return text


def markdown_to_html(md: str) -> str:
    """Convert markdown resume to inline-styled HTML suitable for email body."""
    parts = [
        '<div style="font-family:Arial,Helvetica,sans-serif;max-width:680px;'
        'margin:0 auto;color:#1a1a1a;line-height:1.45;font-size:10pt;">'
    ]
    in_name_block = True

    for line in md.strip().splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("# "):
            parts.append(
                f'<h1 style="text-align:center;font-size:18pt;font-weight:700;'
                f'margin:0 0 3px 0;">{_fmt_html(s[2:].strip())}</h1>'
            )
            in_name_block = True
        elif s.startswith("## "):
            in_name_block = False
            parts.append(
                f'<h2 style="font-size:9.5pt;font-weight:700;letter-spacing:1px;'
                f'text-transform:uppercase;margin:14px 0 2px 0;padding-bottom:2px;'
                f'border-bottom:1px solid #ccc;">{_fmt_html(s[3:].strip())}</h2>'
            )
        elif s.startswith("### "):
            in_name_block = False
            parts.append(
                f'<p style="margin:6px 0 1px 0;font-weight:700;">'
                f'{_fmt_html(s[4:].strip())}</p>'
            )
        elif s.startswith(("- ", "* ")):
            in_name_block = False
            parts.append(
                f'<p style="margin:1px 0 1px 14px;">'
                f'&bull; {_fmt_html(s[2:].strip())}</p>'
            )
        elif in_name_block:
            parts.append(
                f'<p style="text-align:center;margin:0 0 2px 0;'
                f'color:#555;font-size:9pt;">{_fmt_html(s)}</p>'
            )
        else:
            parts.append(f'<p style="margin:3px 0;">{_fmt_html(s)}</p>')

    parts.append("</div>")
    return "\n".join(parts)


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
    """Convert markdown resume text to a formatted PDF using fpdf2. Raises if unavailable."""
    if not _FPDF_AVAILABLE:
        raise RuntimeError("fpdf2 not installed")
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

    # Always embed the formatted resume in the email body
    resume_html = markdown_to_html(rewritten_resume)
    linkedin_note = (
        "<p>Your LinkedIn optimization is included below.</p>" if linkedin_copy else ""
    )
    linkedin_section = ""
    if linkedin_copy:
        linkedin_section = (
            "<hr style='margin:32px 0;border:none;border-top:1px solid #eee;'>"
            "<h2 style='font-size:14pt;'>LinkedIn Optimization</h2>"
            f"<pre style='white-space:pre-wrap;font-family:inherit;'>"
            f"{html_lib.escape(linkedin_copy)}</pre>"
        )

    body_html = f"""
    <p>Hi {html_lib.escape(customer_name)},</p>
    <p>Your rewritten resume is below. You can also print this email to PDF (File &gt; Print &gt; Save as PDF) for a clean one-page document.</p>
    {linkedin_note}
    <p>If you land interviews, reply and let us know. Not happy for any reason - reply for a full refund, no questions asked.</p>
    <p>Good luck,<br/>ResumeRocket</p>
    <hr style="margin:24px 0;border:none;border-top:2px solid #eee;">
    {resume_html}
    {linkedin_section}
    """

    # Try to also attach a PDF; fall back gracefully if fpdf2 isn't available
    attachments = []
    try:
        resume_pdf = markdown_to_pdf(rewritten_resume)
        attachments.append({
            "filename": "resume_rewritten.pdf",
            "content": base64.b64encode(resume_pdf).decode(),
        })
        print("PDF attachment generated successfully")
    except Exception as pdf_err:
        print(f"PDF generation skipped ({type(pdf_err).__name__}: {pdf_err}) - resume in email body")

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
    if resp.status_code == 403:
        print(f"Customer send failed (403) for {customer_email}. Sending fallback to owner.")
        fallback_payload = {
            "from": FALLBACK_FROM_EMAIL,
            "to": ["dsherburn@gmail.com"],
            "subject": f"[ACTION NEEDED] ResumeRocket order for {customer_email}",
            "html": (
                f"<p><strong>A customer order needs to be forwarded manually.</strong></p>"
                f"<p>Customer email: <strong>{html_lib.escape(customer_email)}</strong></p>"
                f"<p>The domain tryresumerocket.com is not yet fully verified in Resend "
                f"(SPF records missing from DNS). Forward the resume below to the customer.</p>"
                f"<hr>"
                f"{body_html}"
            ),
            "attachments": attachments,
        }
        fallback_resp = httpx.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json=fallback_payload,
            timeout=30,
        )
        fallback_resp.raise_for_status()
        print(f"Fallback notification sent to dsherburn@gmail.com for order: {customer_email}")
        return
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
        import traceback, datetime
        err_msg = f"{type(e).__name__}: {e}"
        print(f"ERROR in process_tally_payload: {err_msg}")
        traceback.print_exc()
        _last_error["message"] = err_msg
        _last_error["traceback"] = traceback.format_exc()
        _last_error["at"] = datetime.datetime.utcnow().isoformat()


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


@app.route("/diagnose", methods=["GET"])
def diagnose():
    """Check all service dependencies are reachable and keys are valid."""
    results = {}
    # Anthropic key
    try:
        import anthropic as _ant
        c = _ant.Anthropic(api_key=ANTHROPIC_API_KEY)
        c.messages.create(model="claude-haiku-4-5-20251001", max_tokens=5, messages=[{"role": "user", "content": "ping"}])
        results["anthropic"] = "ok"
    except Exception as e:
        results["anthropic"] = f"ERROR: {type(e).__name__}: {str(e)[:120]}"
    # Resend key
    try:
        import httpx as _hx
        r = _hx.get("https://api.resend.com/domains", headers={"Authorization": f"Bearer {RESEND_API_KEY}"}, timeout=10)
        results["resend"] = "ok" if r.status_code == 200 else f"HTTP {r.status_code}"
    except Exception as e:
        results["resend"] = f"ERROR: {str(e)[:80]}"
    results["from_email"] = FROM_EMAIL
    results["fallback_from_email"] = FALLBACK_FROM_EMAIL
    results["anthropic_key_prefix"] = ANTHROPIC_API_KEY[:20] + "..." if ANTHROPIC_API_KEY else "NOT SET"
    results["last_error"] = _last_error if _last_error else None
    return jsonify(results), 200


@app.route("/last-error", methods=["GET"])
def last_error():
    return jsonify(_last_error if _last_error else {"message": "no errors recorded"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)
