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

import anthropic
import httpx

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


def send_delivery_email(
    customer_email: str,
    customer_name: str,
    rewritten_resume: str,
    linkedin_copy: str | None = None,
) -> None:
    """Send the completed deliverables via Resend API."""
    subject = "Your ResumeRocket rewrite is ready 🚀"

    body_html = f"""
    <p>Hi {customer_name},</p>
    <p>Your rewritten resume is attached as a PDF and DOCX. Copy it directly into your job applications.</p>
    {"<p>Your LinkedIn optimization is included below the resume in the attached document.</p>" if linkedin_copy else ""}
    <p>If you land interviews, we'd love to hear about it. If you're not happy for any reason, reply to this email for a full refund — no questions asked.</p>
    <p>Good luck,<br/>ResumeRocket</p>
    """

    # Combine deliverables into a single markdown document
    full_doc = f"# Rewritten Resume\n\n{rewritten_resume}"
    if linkedin_copy:
        full_doc += f"\n\n---\n\n# LinkedIn Optimization\n\n{linkedin_copy}"

    payload = {
        "from": FROM_EMAIL,
        "to": [customer_email],
        "subject": subject,
        "html": body_html,
        "attachments": [
            {
                "filename": "resume_rewritten.md",
                "content": base64.b64encode(full_doc.encode()).decode(),
            }
        ],
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

from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/webhook/tally", methods=["POST"])
def tally_webhook():
    """Receives Tally form submission after Stripe payment."""
    data = request.get_json(force=True)

    # Tally webhook payload structure
    fields = {f["key"]: f["value"] for f in data.get("fields", [])}

    # Fetch resume file from Tally file URL
    file_url = fields.get("resume_file", [{}])[0].get("url", "") if isinstance(fields.get("resume_file"), list) else ""
    if not file_url:
        return jsonify({"error": "No resume file"}), 400

    file_resp = httpx.get(file_url, timeout=30)
    file_resp.raise_for_status()

    order = {
        "customer_email": fields.get("email", ""),
        "customer_name": fields.get("full_name", "Customer"),
        "target_role": fields.get("target_role", ""),
        "career_level": fields.get("career_level", "IC"),
        "key_achievement": fields.get("key_achievement", ""),
        "resume_file_content": file_resp.content,
        "resume_filename": fields.get("resume_file", [{}])[0].get("name", "resume.pdf"),
        "linkedin_url": fields.get("linkedin_url"),
        "bundle": fields.get("bundle", False),
    }

    process_order(order)
    return jsonify({"status": "delivered"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)
