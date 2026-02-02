"""
Integrated Functions for Cell Culture Monitoring System
Combines: Sensor Simulation + RAG Knowledge Base Chatbot
"""

import csv
import math
import os
import random
from datetime import datetime, timezone

# ============= EMAIL PROVIDERS =============
# Option 1: Resend API (Currently Active - Easy setup, free tier)
import resend

# Option 2: Gmail SMTP (Commented - Requires Gmail App Password)
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ============= CONFIGURATION =============
LOG_PATH = "culture_log.csv"

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2
)

# Initialize embeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ============= SENSOR SIMULATION FUNCTIONS =============

def compute_status_and_score(temp, ph, o2, imp, prev_imp):
    """
    Compute health status and score based on sensor readings.
    Returns: (score, status_global, statuses_dict)
    """
    statuses = {}

    # Temperature status (optimal: 36-38°C)
    if 36 <= temp <= 38:
        statuses["temperature"] = "OK"
    elif 35 <= temp < 36 or 38 < temp <= 39:
        statuses["temperature"] = "WARNING"
    else:
        statuses["temperature"] = "CRITICAL"

    # pH status (optimal: 7.2-7.4)
    if 7.2 <= ph <= 7.4:
        statuses["pH"] = "OK"
    elif 7.1 <= ph < 7.2 or 7.4 < ph <= 7.6:
        statuses["pH"] = "WARNING"
    else:
        statuses["pH"] = "CRITICAL"

    # O2 status (optimal: 18-22%)
    if 18 <= o2 <= 22:
        statuses["o2"] = "OK"
    elif 10 <= o2 < 18 or 22 < o2 <= 25:
        statuses["o2"] = "WARNING"
    else:
        statuses["o2"] = "CRITICAL"

    # Impedance status (trend-based)
    if prev_imp is None:
        delta = 0.0
    else:
        delta = imp - prev_imp

    if imp < 0.5 or delta <= -0.05:
        statuses["impedance"] = "CRITICAL"
    elif imp < 0.7 or delta < 0.0:
        statuses["impedance"] = "WARNING"
    else:
        statuses["impedance"] = "OK"

    # Calculate global score
    warning_count = sum(1 for v in statuses.values() if v == "WARNING")
    critical_count = sum(1 for v in statuses.values() if v == "CRITICAL")

    score = 100 - (10 * warning_count) - (30 * critical_count)
    score = max(0, min(100, score))

    if score >= 80:
        status_global = "OK"
    elif score >= 50:
        status_global = "WARNING"
    else:
        status_global = "CRITICAL"

    return score, status_global, statuses


def simulate_step(state, params):
    """
    Simulate one step of sensor readings.
    Returns: (values_dict, new_state_dict)
    """
    sim_time_hours = state.get("sim_time_hours", 0.0) + params["sim_step_hours"]

    # Simulate sensor readings with noise and offsets
    temperature = 37.0 + random.gauss(0.0, 0.1) + params["temp_offset"]
    ph = 7.3 + random.gauss(0.0, 0.02) + params["ph_offset"]
    o2 = 20.0 + random.gauss(0.0, 0.5) + params["o2_offset"]

    # Logistic-like growth for impedance
    t = sim_time_hours
    imp_base = 0.2 + 0.7 / (1 + math.exp(-0.25 * (t - 12)))

    prev_imp = state.get("impedance_index", imp_base)
    contamination = params["contamination_level"]

    noise = random.gauss(0.0, 0.01)
    imp = prev_imp + (imp_base - prev_imp) * 0.2 + noise - contamination * 0.03

    # Contamination shocks
    if contamination > 0.7 and random.random() < contamination * 0.2:
        imp -= contamination * 0.05

    imp = max(0.0, min(1.0, imp))

    new_state = {
        "sim_time_hours": sim_time_hours,
        "impedance_index": imp,
        "prev_impedance_index": prev_imp,
    }

    values = {
        "sim_time_hours": sim_time_hours,
        "temperature_degC": temperature,
        "pH": ph,
        "o2_percent": o2,
        "impedance_index": imp,
    }

    return values, new_state


def append_to_log_csv(row_dict, log_path=LOG_PATH):
    """Append a row to the CSV log file."""
    fieldnames = [
        "timestamp_utc",
        "sim_time_hours",
        "temperature_degC",
        "pH",
        "o2_percent",
        "impedance_index",
        "score_global",
        "status_global",
    ]

    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def get_current_timestamp():
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def get_readable_timestamp():
    """Get human-readable timestamp."""
    return datetime.now().strftime("%H:%M:%S")


def generate_forecast(values, score, status_global, statuses):
    """
    Generate future forecast based on current readings.
    Returns forecast text for progress notifications.
    """
    sim_time = values["sim_time_hours"]
    impedance = values["impedance_index"]

    forecasts = []

    # Growth forecast based on impedance trend
    if impedance < 0.4:
        forecasts.append("📈 Early growth phase - expect rapid cell proliferation in next 6-12 hours")
    elif impedance < 0.7:
        forecasts.append("📈 Active growth phase - cells expanding well, continue monitoring")
    elif impedance < 0.85:
        forecasts.append("📈 Approaching optimal density - consider passaging in 4-8 hours")
    else:
        forecasts.append("⚠️ Near confluence - passaging recommended soon to prevent overcrowding")

    # Temperature forecast
    temp = values["temperature_degC"]
    if 36.5 <= temp <= 37.5:
        forecasts.append("🌡️ Temperature stable - maintain current settings")
    elif temp < 36.5:
        forecasts.append("🌡️ Temperature slightly low - growth may slow by 10-15%")
    else:
        forecasts.append("🌡️ Temperature elevated - monitor for heat stress signs")

    # pH forecast
    ph = values["pH"]
    if ph < 7.2:
        forecasts.append("🧪 pH trending acidic - media change recommended in 2-4 hours")
    elif ph > 7.5:
        forecasts.append("🧪 pH trending basic - check CO2 levels")
    else:
        forecasts.append("🧪 pH optimal - next media change in 12-24 hours")

    # Overall timeline
    if score >= 80:
        forecasts.append(f"✅ On track - estimated ready for surgery in {max(1, int(20 - sim_time))} hours")
    elif score >= 50:
        forecasts.append(f"⚠️ Slightly delayed - may need {max(1, int(24 - sim_time))} more hours")
    else:
        forecasts.append("🚨 Timeline at risk - immediate intervention needed")

    return forecasts


def create_progress_notification(values, score, status_global, statuses):
    """
    Create a progress notification dict with details and forecast.
    """
    forecasts = generate_forecast(values, score, status_global, statuses)

    notification = {
        "type": "PROGRESS",
        "timestamp": get_readable_timestamp(),
        "sim_time": f"{values['sim_time_hours']:.1f}h",
        "score": score,
        "status": status_global,
        "readings": {
            "Temperature": f"{values['temperature_degC']:.2f}°C ({statuses['temperature']})",
            "pH": f"{values['pH']:.2f} ({statuses['pH']})",
            "O2": f"{values['o2_percent']:.1f}% ({statuses['o2']})",
            "Impedance": f"{values['impedance_index']:.2f} ({statuses['impedance']})",
        },
        "forecast": forecasts,
    }

    return notification


def get_dummy_notifications():
    """
    Generate 7 dummy notification situations for testing and demo.
    Returns: list of 7 notifications (3 critical + 4 progress)
    """
    dummy_notifications = [
        # ============= CRITICAL SITUATIONS (3) =============
        {
            "type": "CRITICAL",
            "timestamp": "09:15:32",
            "sim_time": "18.5h",
            "score": 35,
            "status": "CRITICAL",
            "problem": "Temperature dropped to 33.2°C - Cell viability at immediate risk",
            "readings": {
                "Temperature": "33.2°C (CRITICAL)",
                "pH": "7.28 (OK)",
                "O2": "19.5% (OK)",
                "Impedance": "0.72 (WARNING)",
            },
            "steps": [
                "1. CHECK incubator door seal immediately",
                "2. VERIFY power supply and heating element",
                "3. DO NOT open incubator door repeatedly",
                "4. TRANSFER cultures to backup incubator if temp doesn't recover in 15 min",
                "5. DOCUMENT time of temperature drop for records",
                "6. NOTIFY supervisor if cultures were exposed >30 minutes",
            ],
            "links": [
                "- Link: https://www.thermofisher.com/us/en/home/references/gibco-cell-culture-basics.html",
                "- Link: https://www.fda.gov/media/73976/download",
            ],
        },
        {
            "type": "CRITICAL",
            "timestamp": "14:42:18",
            "sim_time": "22.3h",
            "score": 25,
            "status": "CRITICAL",
            "problem": "pH dropped to 6.8 - Media turning yellow, severe acidification detected",
            "readings": {
                "Temperature": "37.1°C (OK)",
                "pH": "6.82 (CRITICAL)",
                "O2": "18.2% (WARNING)",
                "Impedance": "0.65 (WARNING)",
            },
            "steps": [
                "1. PREPARE fresh complete media immediately (warm to 37°C)",
                "2. ASPIRATE old yellow media carefully",
                "3. WASH cells gently with PBS",
                "4. ADD fresh media and return to incubator",
                "5. CHECK CO2 levels in incubator (should be 5%)",
                "6. MONITOR every 2 hours for next 12 hours",
            ],
            "links": [
                "- Link: https://www.usp.org/biologics/cell-gene-therapy",
                "- Link: https://www.ema.europa.eu/en/documents/scientific-guideline/guideline-human-cell-based-medicinal-products_en.pdf",
            ],
        },
        {
            "type": "CRITICAL",
            "timestamp": "03:27:45",
            "sim_time": "45.2h",
            "score": 15,
            "status": "CRITICAL",
            "problem": "Impedance dropping rapidly - Possible contamination or mass cell death",
            "readings": {
                "Temperature": "37.0°C (OK)",
                "pH": "7.15 (WARNING)",
                "O2": "21.0% (OK)",
                "Impedance": "0.38 (CRITICAL)",
            },
            "steps": [
                "1. DO NOT open flask near other cultures",
                "2. EXAMINE under microscope for contamination signs",
                "3. CHECK for cloudiness, floating particles, or unusual smell",
                "4. ISOLATE this culture immediately",
                "5. IF contaminated: bleach and autoclave, DO NOT attempt to save",
                "6. ACTIVATE backup cultures from other incubators",
            ],
            "links": [
                "- Link: https://www.ecfr.gov/current/title-21/chapter-I/subchapter-L/part-1271",
                "- Link: https://www.iso.org/obp/ui/#iso:std:iso:10993:-5:en",
            ],
        },

        # ============= PROGRESS SITUATIONS (4) =============
        {
            "type": "PROGRESS",
            "timestamp": "08:00:00",
            "sim_time": "6.0h",
            "score": 95,
            "status": "OK",
            "readings": {
                "Temperature": "37.0°C (OK)",
                "pH": "7.35 (OK)",
                "O2": "20.1% (OK)",
                "Impedance": "0.35 (OK)",
            },
            "forecast": [
                "📈 Early growth phase - cells attached and beginning proliferation",
                "🌡️ Temperature stable at optimal 37°C - maintain current settings",
                "🧪 pH optimal at 7.35 - next media change recommended in 18-24 hours",
                "✅ On track - estimated ready for surgery in 14-16 hours",
            ],
        },
        {
            "type": "PROGRESS",
            "timestamp": "14:00:00",
            "sim_time": "12.0h",
            "score": 90,
            "status": "OK",
            "readings": {
                "Temperature": "37.2°C (OK)",
                "pH": "7.28 (OK)",
                "O2": "19.8% (OK)",
                "Impedance": "0.58 (OK)",
            },
            "forecast": [
                "📈 Active growth phase - cells expanding well, ~40% confluence",
                "🌡️ Temperature slightly elevated but within range - monitor closely",
                "🧪 pH trending slightly acidic - media change in 6-8 hours",
                "✅ On track - estimated ready for surgery in 8-10 hours",
            ],
        },
        {
            "type": "PROGRESS",
            "timestamp": "20:00:00",
            "sim_time": "18.0h",
            "score": 85,
            "status": "OK",
            "readings": {
                "Temperature": "37.0°C (OK)",
                "pH": "7.22 (OK)",
                "O2": "20.5% (OK)",
                "Impedance": "0.78 (OK)",
            },
            "forecast": [
                "📈 Approaching optimal density - ~70% confluence reached",
                "🌡️ Temperature stable - maintain current settings",
                "🧪 pH at lower optimal range - fresh media added, monitoring",
                "✅ Almost ready - consider passaging or preparing for surgery in 4-6 hours",
            ],
        },
        {
            "type": "PROGRESS",
            "timestamp": "02:00:00",
            "sim_time": "24.0h",
            "score": 92,
            "status": "OK",
            "readings": {
                "Temperature": "37.0°C (OK)",
                "pH": "7.32 (OK)",
                "O2": "20.0% (OK)",
                "Impedance": "0.88 (OK)",
            },
            "forecast": [
                "📈 Optimal density reached - ~85% confluence, ready for next step",
                "🌡️ Temperature optimal and stable throughout cultivation",
                "🧪 pH stable - culture healthy and thriving",
                "✅ READY - Culture can be split into multiple incubators or prepared for surgery",
            ],
        },
    ]

    return dummy_notifications


def get_random_dummy_notification():
    """Get a random dummy notification for test emails."""
    import random
    notifications = get_dummy_notifications()
    return random.choice(notifications)


def create_critical_notification(values, score, status_global, statuses, problem, solution_text):
    """
    Create a critical notification dict with steps and resource links.
    """
    # Parse solution to extract steps and links
    steps = []
    links = []

    # Extract numbered steps from solution
    lines = solution_text.split('\n')
    for line in lines:
        line = line.strip()
        # Look for numbered steps
        if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
            if 'http' not in line.lower():
                steps.append(line)
        # Look for links
        if 'Link:' in line or 'http' in line.lower():
            links.append(line)

    # If no steps found, use first few lines
    if not steps:
        steps = [l.strip() for l in lines[:5] if l.strip() and not l.startswith('**')]

    notification = {
        "type": "CRITICAL",
        "timestamp": get_readable_timestamp(),
        "sim_time": f"{values['sim_time_hours']:.1f}h",
        "score": score,
        "status": status_global,
        "problem": problem,
        "readings": {
            "Temperature": f"{values['temperature_degC']:.2f}°C ({statuses['temperature']})",
            "pH": f"{values['pH']:.2f} ({statuses['pH']})",
            "O2": f"{values['o2_percent']:.1f}% ({statuses['o2']})",
            "Impedance": f"{values['impedance_index']:.2f} ({statuses['impedance']})",
        },
        "steps": steps[:6],  # Limit to 6 steps
        "links": links[:4],  # Limit to 4 links
    }

    return notification


def generate_alert_question(statuses, values):
    """
    Generate a question for the RAG based on current sensor alerts.
    Returns: question string or None if no alerts
    """
    critical_params = [k for k, v in statuses.items() if v == "CRITICAL"]
    warning_params = [k for k, v in statuses.items() if v == "WARNING"]

    if critical_params:
        param = critical_params[0]
        if param == "temperature":
            temp = values["temperature_degC"]
            if temp < 36:
                return f"Temperature dropped to {temp:.1f}°C which is below normal. What should I do?"
            else:
                return f"Temperature is too high at {temp:.1f}°C. What should I do?"
        elif param == "pH":
            ph = values["pH"]
            if ph < 7.2:
                return f"pH is too acidic at {ph:.2f}. Media may be turning yellow. What should I do?"
            else:
                return f"pH is too basic at {ph:.2f}. What should I do?"
        elif param == "o2":
            return f"Oxygen level is abnormal at {values['o2_percent']:.1f}%. What should I do?"
        elif param == "impedance":
            return f"Impedance is dropping rapidly, indicating possible cell death or contamination. What should I do?"

    elif warning_params:
        param = warning_params[0]
        return f"The {param} reading is showing a warning. What preventive steps should I take?"

    return None


# ============= RAG KNOWLEDGE BASE FUNCTIONS =============

def load_knowledge_base(knowledge_base_path="knowledge_base"):
    """
    Load all text files from the knowledge base directory.
    Returns: (documents_list, error_message)
    """
    documents = []

    if not os.path.exists(knowledge_base_path):
        return documents, f"Knowledge base directory not found: {knowledge_base_path}"

    for filename in os.listdir(knowledge_base_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(knowledge_base_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                doc = Document(
                    page_content=content,
                    metadata={"source": filename}
                )
                documents.append(doc)

    return documents, None


def create_chunks_with_sources(documents):
    """Split documents into chunks while preserving source information."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n===========================================================================\n", "\n\n", "\n", " "]
    )

    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])

        for chunk in chunks:
            content = chunk.page_content
            # Extract SOURCE metadata
            if "SOURCE:" in content:
                source_lines = [line for line in content.split('\n') if line.startswith("SOURCE:")]
                if source_lines:
                    chunk.metadata["section"] = source_lines[0].replace("SOURCE:", "").strip()

            # Extract LINK metadata
            if "LINK:" in content:
                link_lines = [line for line in content.split('\n') if line.startswith("LINK:")]
                if link_lines:
                    chunk.metadata["url"] = link_lines[0].replace("LINK:", "").strip()

            all_chunks.append(chunk)

    return all_chunks


def create_vector_store(chunks):
    """Create vector store from document chunks."""
    vector_store = Chroma.from_documents(chunks, embedding)
    return vector_store


def rag_answer_with_sources(question, vectorstore):
    """Answer questions using RAG and return sources with links."""
    # Retrieve relevant chunks
    results = vectorstore.similarity_search(question, k=4)

    # Build context with source tracking
    context_parts = []
    sources = []

    for i, result in enumerate(results):
        context_parts.append(f"[Source {i+1}]: {result.page_content}")

        # Extract source information
        section = result.metadata.get("section", "")
        source_file = result.metadata.get("source", "Unknown")
        url = result.metadata.get("url", "")

        display_text = section if section else source_file

        if url:
            sources.append(f"**[{i+1}]** {display_text}\n   - Link: {url}")
        else:
            sources.append(f"**[{i+1}]** {display_text}")

    context_text = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_template("""
You are a CRITICAL RESPONSE PLAYBOOK for a skin cultivation laboratory treating burn patients.

Your job is to give IMMEDIATE, STEP-BY-STEP ACTIONABLE INSTRUCTIONS that an operator can follow right now.

FORMAT YOUR RESPONSE LIKE THIS:

**SITUATION ASSESSMENT:**
[Brief 1-line summary of the problem and its severity: CRITICAL / WARNING / INFO]

**IMMEDIATE ACTIONS:**
1. [First thing to do RIGHT NOW]
2. [Second step]
3. [Third step]
(Continue as needed)

**WHY THIS MATTERS:**
[Brief explanation of consequences if not addressed - 1-2 sentences max]

**NEXT STEPS:**
- [What to monitor after immediate actions]
- [When to escalate to supervisor]

**PREVENTION:**
- [How to prevent this in future - if applicable]

RULES:
- Use ONLY the provided context from the knowledge base
- Be DIRECT and ACTIONABLE - operators need clear steps, not explanations
- Number all action steps
- If this is a CRITICAL situation (risk of losing culture/patient safety), say so clearly
- ALWAYS cite which sources you used (e.g., "Based on [1] and [3]") at the end of your response
- If the answer is not in the context, say: "This situation is not covered in the playbook. IMMEDIATELY contact your lab supervisor."

CONTEXT FROM KNOWLEDGE BASE:
{context}

OPERATOR QUESTION:
{question}

PLAYBOOK RESPONSE:
""")

    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": question})

    # Format the response with sources
    answer = response.content

    # Add sources section with links
    sources_text = "\n\n---\n**📚 SOURCES (Verified References):**\n\n" + "\n\n".join(sources)

    return answer + sources_text


# ============= EMAIL NOTIFICATION FUNCTIONS =============
#
# Two options available:
# 1. RESEND API (Active) - Easy setup, just need API key from resend.com
# 2. GMAIL SMTP (Commented) - Requires Gmail account + App Password
#

def send_email(sender_email, sender_password, recipient_email, subject, body_html):
    """
    Send an email using Resend API.

    Args:
        sender_email: Display name for sender (Resend uses onboarding@resend.dev)
        sender_password: Resend API Key (get from resend.com)
        recipient_email: Recipient's email address
        subject: Email subject line
        body_html: HTML content of the email

    Returns: (success: bool, message: str)
    """
    try:
        # Set API key
        resend.api_key = sender_password  # sender_password field holds the API key

        # Send email via Resend
        params = {
            "from": "CODE4CARE <onboarding@resend.dev>",  # Free tier uses resend.dev domain
            "to": [recipient_email],
            "subject": subject,
            "html": body_html,
        }

        email = resend.Emails.send(params)

        if email and email.get("id"):
            return True, f"Email sent successfully! ID: {email['id']}"
        else:
            return False, "Failed to send email. Check API key."

    except Exception as e:
        return False, f"Failed to send email: {str(e)}"


# ============= GMAIL SMTP (Alternative - Uncomment to use) =============
#
# def send_email(sender_email, sender_password, recipient_email, subject, body_html):
#     """
#     Send an email using Gmail SMTP.
#
#     Requirements:
#     - Gmail account with 2FA enabled
#     - App Password from: https://myaccount.google.com/apppasswords
#
#     Args:
#         sender_email: Gmail address to send from
#         sender_password: Gmail App Password (NOT regular password)
#         recipient_email: Recipient's email address
#         subject: Email subject line
#         body_html: HTML content of the email
#
#     Returns: (success: bool, message: str)
#     """
#     try:
#         # Create message
#         msg = MIMEMultipart("alternative")
#         msg["Subject"] = subject
#         msg["From"] = sender_email
#         msg["To"] = recipient_email
#
#         # Attach HTML body
#         html_part = MIMEText(body_html, "html")
#         msg.attach(html_part)
#
#         # Connect to Gmail SMTP
#         with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
#             server.login(sender_email, sender_password)
#             server.sendmail(sender_email, recipient_email, msg.as_string())
#
#         return True, "Email sent successfully!"
#
#     except smtplib.SMTPAuthenticationError:
#         return False, "Authentication failed. Check email/password. Use Gmail App Password."
#     except Exception as e:
#         return False, f"Failed to send email: {str(e)}"


def generate_progress_report_html(values, score, status_global, statuses, history_summary=None):
    """
    Generate HTML content for progress report email.
    """
    status_colors = {
        "OK": "#2e7d32",
        "WARNING": "#ef6c00",
        "CRITICAL": "#c62828",
    }

    timestamp = get_current_timestamp()

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: linear-gradient(135deg, #1a237e, #4a148c); color: white; padding: 20px; border-radius: 10px; }}
            .status-badge {{ display: inline-block; padding: 8px 16px; border-radius: 20px; color: white; font-weight: bold; }}
            .metrics {{ display: flex; flex-wrap: wrap; gap: 15px; margin: 20px 0; }}
            .metric-card {{ background: #f5f5f5; padding: 15px; border-radius: 8px; min-width: 120px; text-align: center; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #1a237e; }}
            .metric-label {{ font-size: 12px; color: #666; }}
            .param-status {{ margin: 5px 0; padding: 5px 10px; border-radius: 4px; }}
            .footer {{ margin-top: 20px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧬 CODE4CARE - Culture Progress Report</h1>
            <p>Automated monitoring report generated at {timestamp}</p>
        </div>

        <h2>Overall Status:
            <span class="status-badge" style="background-color: {status_colors[status_global]};">
                {status_global}
            </span>
        </h2>

        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{values['temperature_degC']:.2f}°C</div>
                <div class="metric-label">Temperature</div>
                <div class="param-status" style="background-color: {status_colors[statuses['temperature']]}20; color: {status_colors[statuses['temperature']]};">
                    {statuses['temperature']}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{values['pH']:.2f}</div>
                <div class="metric-label">pH Level</div>
                <div class="param-status" style="background-color: {status_colors[statuses['pH']]}20; color: {status_colors[statuses['pH']]};">
                    {statuses['pH']}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{values['o2_percent']:.1f}%</div>
                <div class="metric-label">Oxygen</div>
                <div class="param-status" style="background-color: {status_colors[statuses['o2']]}20; color: {status_colors[statuses['o2']]};">
                    {statuses['o2']}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{values['impedance_index']:.2f}</div>
                <div class="metric-label">Impedance</div>
                <div class="param-status" style="background-color: {status_colors[statuses['impedance']]}20; color: {status_colors[statuses['impedance']]};">
                    {statuses['impedance']}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{score}</div>
                <div class="metric-label">Health Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{values['sim_time_hours']:.1f}h</div>
                <div class="metric-label">Culture Age</div>
            </div>
        </div>

        <div class="footer">
            <p>This is an automated report from CODE4CARE Cell Culture Monitoring System.</p>
            <p>For critical alerts, immediate action is required. Contact lab supervisor if needed.</p>
        </div>
    </body>
    </html>
    """
    return html


def convert_playbook_to_html(solution_text):
    """
    Convert markdown-style playbook response to proper HTML formatting.
    """
    import re

    html_lines = []
    lines = solution_text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            html_lines.append("<br>")
            continue

        # Convert **bold** to <strong>
        line = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', line)

        # Section headers (e.g., **SITUATION ASSESSMENT:**)
        if line.startswith('<strong>') and line.endswith(':</strong>'):
            html_lines.append(f'<h4 style="color: #1a237e; margin: 15px 0 5px 0;">{line}</h4>')
        # Numbered steps (1. 2. 3. etc)
        elif re.match(r'^\d+\.', line):
            html_lines.append(f'<p style="margin: 5px 0 5px 20px;">✓ {line}</p>')
        # Bullet points (- or •)
        elif line.startswith('-') or line.startswith('•'):
            content = line.lstrip('-•').strip()
            html_lines.append(f'<p style="margin: 5px 0 5px 20px;">• {content}</p>')
        # Links
        elif 'http' in line.lower() or 'Link:' in line:
            # Make URLs clickable
            line = re.sub(r'(https?://[^\s<>"]+)', r'<a href="\1" style="color: #1565c0;">\1</a>', line)
            html_lines.append(f'<p style="margin: 5px 0 5px 20px; font-size: 12px;">{line}</p>')
        # Source references
        elif line.startswith('[') or 'Source' in line:
            html_lines.append(f'<p style="margin: 5px 0; font-size: 12px; color: #666;">{line}</p>')
        else:
            html_lines.append(f'<p style="margin: 5px 0;">{line}</p>')

    return '\n'.join(html_lines)


def generate_critical_alert_html(values, score, status_global, statuses, problem_description, solution):
    """
    Generate HTML content for critical alert email with problem and solution.
    """
    timestamp = get_current_timestamp()

    # Find critical parameters
    critical_params = [k for k, v in statuses.items() if v == "CRITICAL"]
    warning_params = [k for k, v in statuses.items() if v == "WARNING"]

    critical_list = ", ".join(critical_params) if critical_params else "None"
    warning_list = ", ".join(warning_params) if warning_params else "None"

    # Convert playbook solution to proper HTML
    solution_html = convert_playbook_to_html(solution)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .alert-header {{ background: #c62828; color: white; padding: 20px; border-radius: 10px; }}
            .problem-box {{ background: #ffebee; border-left: 4px solid #c62828; padding: 15px; margin: 20px 0; }}
            .solution-box {{ background: #e8f5e9; border-left: 4px solid #2e7d32; padding: 15px; margin: 20px 0; }}
            .metrics {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 15px 0; }}
            .metric {{ background: #f5f5f5; padding: 10px; border-radius: 5px; }}
            .footer {{ margin-top: 20px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px; }}
            h3 {{ color: #1a237e; }}
            h4 {{ color: #2e7d32; }}
        </style>
    </head>
    <body>
        <div class="alert-header">
            <h1>🚨 CRITICAL ALERT - Immediate Action Required</h1>
            <p>Alert generated at {timestamp}</p>
        </div>

        <h2>Current Status: CRITICAL (Score: {score}/100)</h2>

        <div class="problem-box">
            <h3>⚠️ Problem Detected</h3>
            <p><strong>Critical Parameters:</strong> {critical_list}</p>
            <p><strong>Warning Parameters:</strong> {warning_list}</p>
            <p><strong>Description:</strong> {problem_description}</p>
        </div>

        <h3>📊 Current Readings</h3>
        <div class="metrics">
            <div class="metric"><strong>Temperature:</strong> {values['temperature_degC']:.2f}°C ({statuses['temperature']})</div>
            <div class="metric"><strong>pH:</strong> {values['pH']:.2f} ({statuses['pH']})</div>
            <div class="metric"><strong>O2:</strong> {values['o2_percent']:.1f}% ({statuses['o2']})</div>
            <div class="metric"><strong>Impedance:</strong> {values['impedance_index']:.2f} ({statuses['impedance']})</div>
            <div class="metric"><strong>Culture Age:</strong> {values['sim_time_hours']:.1f} hours</div>
        </div>

        <div class="solution-box">
            <h3>✅ PLAYBOOK RESPONSE - Follow These Steps Immediately</h3>
            {solution_html}
        </div>

        <div class="footer">
            <p><strong>IMPORTANT:</strong> This alert requires immediate attention. Follow the recommended actions above.</p>
            <p>If you cannot resolve the issue, contact the lab supervisor immediately.</p>
            <p>—CODE4CARE Cell Culture Monitoring System</p>
        </div>
    </body>
    </html>
    """
    return html


def send_progress_report(email_config, values, score, status_global, statuses):
    """
    Send a progress report email.

    Args:
        email_config: dict with sender_email, sender_password, recipient_email
        values, score, status_global, statuses: Current sensor data

    Returns: (success: bool, message: str)
    """
    subject = f"🧬 Culture Progress Report - Status: {status_global} (Score: {score})"
    body_html = generate_progress_report_html(values, score, status_global, statuses)

    return send_email(
        email_config["sender_email"],
        email_config["sender_password"],
        email_config["recipient_email"],
        subject,
        body_html
    )


def send_critical_alert(email_config, values, score, status_global, statuses, problem_description, solution):
    """
    Send a critical alert email with problem and solution.

    Args:
        email_config: dict with sender_email, sender_password, recipient_email
        values, score, status_global, statuses: Current sensor data
        problem_description: What went wrong
        solution: RAG-generated solution from playbook

    Returns: (success: bool, message: str)
    """
    subject = f"🚨 CRITICAL ALERT - Cell Culture Emergency - Score: {score}"
    body_html = generate_critical_alert_html(values, score, status_global, statuses, problem_description, solution)

    return send_email(
        email_config["sender_email"],
        email_config["sender_password"],
        email_config["recipient_email"],
        subject,
        body_html
    )
