"""
CODE4CARE - Integrated Cell Culture Monitoring System
Dashboard + RAG Chatbot Playbook + Email Notifications
"""

import os
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from integrated_functions import (
    # Sensor functions
    compute_status_and_score,
    simulate_step,
    append_to_log_csv,
    get_current_timestamp,
    generate_alert_question,
    # RAG functions
    load_knowledge_base,
    create_chunks_with_sources,
    create_vector_store,
    rag_answer_with_sources,
    # Email functions
    send_email,
    send_progress_report,
    send_critical_alert,
)

# ============= PAGE CONFIG =============
st.set_page_config(
    page_title="CODE4CARE - Cell Culture Monitor",
    page_icon="🧬",
    layout="wide"
)

# ============= INITIALIZE SESSION STATE =============
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "kb_loaded" not in st.session_state:
    st.session_state.kb_loaded = False

if "state" not in st.session_state:
    st.session_state.state = {
        "sim_time_hours": 0.0,
        "impedance_index": 0.3,
        "prev_impedance_index": None,
    }

if "history" not in st.session_state:
    st.session_state.history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_alert" not in st.session_state:
    st.session_state.current_alert = None

# Email session state - Load defaults from .env if available
if "email_config" not in st.session_state:
    env_sender = os.getenv("SENDER_EMAIL", "")
    env_password = os.getenv("SENDER_PASSWORD", "")
    env_recipient = os.getenv("RECIPIENT_EMAIL", "")

    # Auto-enable if all env vars are set
    auto_enabled = bool(env_sender and env_password and env_recipient
                        and env_sender != "your.email@gmail.com")

    st.session_state.email_config = {
        "sender_email": env_sender if env_sender != "your.email@gmail.com" else "",
        "sender_password": env_password if env_password != "xxxx xxxx xxxx xxxx" else "",
        "recipient_email": env_recipient if env_recipient != "supervisor@example.com" else "",
        "enabled": auto_enabled,
        "report_interval_hours": 1.0,
        "critical_alerts_enabled": True,
    }

if "last_report_time" not in st.session_state:
    st.session_state.last_report_time = 0.0

if "last_critical_alert_sent" not in st.session_state:
    st.session_state.last_critical_alert_sent = False

# ============= LOAD KNOWLEDGE BASE (Auto on first run) =============
if not st.session_state.kb_loaded:
    with st.spinner("🔄 Loading knowledge base..."):
        docs, error = load_knowledge_base()
        if error:
            st.error(error)
        elif docs:
            chunks = create_chunks_with_sources(docs)
            vectorstore = create_vector_store(chunks)
            st.session_state.vector_store = vectorstore
            st.session_state.kb_loaded = True

# ============= SIDEBAR =============
with st.sidebar:
    st.title("🧬 CODE4CARE")
    st.markdown("### Cell Culture Monitor")
    st.markdown("---")

    # Tab selector
    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "💬 Playbook Chat", "📧 Email Settings"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    if page == "📊 Dashboard":
        st.markdown("**⚙️ Simulation Controls**")

        temp_offset = st.slider("Temp offset (°C)", -3.0, 3.0, 0.0, 0.1)
        ph_offset = st.slider("pH offset", -0.5, 0.5, 0.0, 0.01)
        o2_offset = st.slider("O2 offset (%)", -10.0, 10.0, 0.0, 0.5)
        contamination_level = st.slider("Contamination", 0.0, 1.0, 0.0, 0.01)

        st.markdown("---")
        st.markdown("**⏱️ Timing**")
        refresh_seconds = st.slider("Refresh (sec)", 0.5, 3.0, 1.0, 0.5)
        sim_step_hours = st.slider("Sim hours/tick", 0.01, 1.0, 0.1, 0.01)
        history_max = st.slider("History points", 50, 500, 200, 10)

        st.markdown("---")
        run_sim = st.toggle("▶️ Run Simulation", value=False)

        # Email status indicator
        if st.session_state.email_config["enabled"]:
            st.markdown("---")
            st.success("📧 Email notifications ON")

    elif page == "💬 Playbook Chat":
        st.markdown("**⚡ Quick Questions:**")

        quick_questions = [
            "What if temperature drops below 35°C?",
            "Media is turning yellow, what to do?",
            "How to prevent contamination?",
            "Incubator failed, what now?",
            "How to split cultures safely?"
        ]

        for q in quick_questions:
            if st.button(q, use_container_width=True, key=f"quick_{q}"):
                st.session_state.pending_question = q

        st.markdown("---")
        if st.button("🔄 Reload Knowledge Base", use_container_width=True):
            with st.spinner("Reloading..."):
                docs, error = load_knowledge_base()
                if docs:
                    chunks = create_chunks_with_sources(docs)
                    st.session_state.vector_store = create_vector_store(chunks)
                    st.success(f"✅ Loaded {len(docs)} docs")

    else:  # Email Settings
        st.markdown("**📧 Email Configuration**")
        st.caption("Configure in main panel →")

# ============= MAIN CONTENT =============

if page == "📊 Dashboard":
    st.title("📊 Live Culture Dashboard")
    st.markdown("*Real-time monitoring of skin culture parameters*")

    if not run_sim:
        st.info("👈 Toggle '▶️ Run Simulation' in sidebar to start monitoring")

        # Show last known values if available
        if st.session_state.history:
            st.markdown("### Last Known Values")
            last = st.session_state.history[-1]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Temperature", f"{last['temperature_degC']:.2f}°C")
            col2.metric("pH", f"{last['pH']:.2f}")
            col3.metric("O2", f"{last['o2_percent']:.1f}%")
            col4.metric("Score", f"{last['score_global']}")
    else:
        # Placeholders for live updates
        alert_placeholder = st.empty()
        email_status_placeholder = st.empty()
        metrics_placeholder = st.empty()
        charts_placeholder = st.empty()

        # Live simulation loop
        while run_sim:
            params = {
                "temp_offset": temp_offset,
                "ph_offset": ph_offset,
                "o2_offset": o2_offset,
                "contamination_level": contamination_level,
                "sim_step_hours": sim_step_hours,
            }

            # Simulate one step
            values, new_state = simulate_step(st.session_state.state, params)
            st.session_state.state = new_state

            # Calculate status
            score, status_global, statuses = compute_status_and_score(
                values["temperature_degC"],
                values["pH"],
                values["o2_percent"],
                values["impedance_index"],
                new_state.get("prev_impedance_index"),
            )

            # Log to CSV
            timestamp_utc = get_current_timestamp()
            row = {
                "timestamp_utc": timestamp_utc,
                "sim_time_hours": round(values["sim_time_hours"], 3),
                "temperature_degC": round(values["temperature_degC"], 3),
                "pH": round(values["pH"], 3),
                "o2_percent": round(values["o2_percent"], 3),
                "impedance_index": round(values["impedance_index"], 3),
                "score_global": int(score),
                "status_global": status_global,
            }
            append_to_log_csv(row)

            # Update history
            st.session_state.history.append(row)
            if len(st.session_state.history) > history_max:
                st.session_state.history = st.session_state.history[-history_max:]

            # Generate alert for chatbot
            alert_q = generate_alert_question(statuses, values)
            if alert_q:
                st.session_state.current_alert = alert_q

            # ============= EMAIL NOTIFICATIONS =============
            email_cfg = st.session_state.email_config

            if email_cfg["enabled"] and email_cfg["sender_password"] and email_cfg["recipient_email"]:
                current_sim_time = values["sim_time_hours"]

                # Send scheduled progress report
                report_interval = email_cfg["report_interval_hours"]
                time_since_last_report = current_sim_time - st.session_state.last_report_time

                if time_since_last_report >= report_interval:
                    success, msg = send_progress_report(
                        email_cfg, values, score, status_global, statuses
                    )
                    if success:
                        st.session_state.last_report_time = current_sim_time
                        with email_status_placeholder.container():
                            st.success(f"📧 Progress report sent to {email_cfg['recipient_email']}")

                # Send critical alert (only once per critical episode)
                if email_cfg["critical_alerts_enabled"]:
                    if status_global == "CRITICAL" and not st.session_state.last_critical_alert_sent:
                        # Get solution from RAG
                        if st.session_state.vector_store and alert_q:
                            solution = rag_answer_with_sources(alert_q, st.session_state.vector_store)
                        else:
                            solution = "Knowledge base not available. Contact supervisor immediately."

                        success, msg = send_critical_alert(
                            email_cfg, values, score, status_global, statuses,
                            alert_q or "Critical condition detected",
                            solution
                        )
                        if success:
                            st.session_state.last_critical_alert_sent = True
                            with email_status_placeholder.container():
                                st.error(f"🚨 Critical alert email sent to {email_cfg['recipient_email']}")

                    elif status_global != "CRITICAL":
                        # Reset flag when status recovers
                        st.session_state.last_critical_alert_sent = False

            # Status colors
            status_colors = {
                "OK": "#2e7d32",
                "WARNING": "#ef6c00",
                "CRITICAL": "#c62828",
            }

            # Show alert banner if WARNING or CRITICAL
            with alert_placeholder.container():
                if status_global == "CRITICAL":
                    st.error(f"🚨 **CRITICAL ALERT**: {alert_q or 'Multiple parameters out of range!'}")
                    st.markdown("👉 Go to **Playbook Chat** for immediate guidance")
                elif status_global == "WARNING":
                    st.warning(f"⚠️ **WARNING**: {alert_q or 'Parameters approaching limits'}")

            # Display metrics
            with metrics_placeholder.container():
                st.subheader("Current Readings")
                col1, col2, col3, col4, col5, col6 = st.columns(6)

                col1.metric("🌡️ Temp (°C)", f"{values['temperature_degC']:.2f}",
                           delta=f"{statuses['temperature']}")
                col2.metric("🧪 pH", f"{values['pH']:.2f}",
                           delta=f"{statuses['pH']}")
                col3.metric("💨 O2 (%)", f"{values['o2_percent']:.1f}",
                           delta=f"{statuses['o2']}")
                col4.metric("📈 Impedance", f"{values['impedance_index']:.2f}",
                           delta=f"{statuses['impedance']}")
                col5.metric("📊 Score", f"{score:.0f}")
                col6.markdown(
                    f"**Status:** <span style='color:{status_colors[status_global]}; font-size:24px;'>{status_global}</span>",
                    unsafe_allow_html=True,
                )

                st.caption(f"⏱️ Simulation Time: {values['sim_time_hours']:.2f} hours | Next report at: {st.session_state.last_report_time + email_cfg['report_interval_hours']:.1f}h")

            # Display charts
            with charts_placeholder.container():
                st.subheader("📈 Trend History")

                if st.session_state.history:
                    df = pd.DataFrame(st.session_state.history)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Environmental Parameters**")
                        st.line_chart(
                            df.set_index("sim_time_hours")[["temperature_degC", "pH", "o2_percent"]],
                            use_container_width=True
                        )

                    with col2:
                        st.markdown("**Growth & Health**")
                        st.line_chart(
                            df.set_index("sim_time_hours")[["impedance_index", "score_global"]],
                            use_container_width=True
                        )

            time.sleep(refresh_seconds)

elif page == "💬 Playbook Chat":
    # ============= PLAYBOOK CHAT =============
    st.title("💬 Playbook Chat")
    st.markdown("*AI-powered troubleshooting assistant for cell culture operations*")

    # Show current alert if any
    if st.session_state.current_alert:
        st.info(f"🔔 **Current Alert:** {st.session_state.current_alert}")
        if st.button("Ask about this alert"):
            st.session_state.pending_question = st.session_state.current_alert

    st.divider()

    # Check if knowledge base is loaded
    if not st.session_state.kb_loaded or st.session_state.vector_store is None:
        st.warning("⚠️ Knowledge base not loaded. Please reload from sidebar.")
    else:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle pending question from sidebar or alert
        if "pending_question" in st.session_state:
            prompt = st.session_state.pending_question
            del st.session_state.pending_question

            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching playbook..."):
                    response = rag_answer_with_sources(prompt, st.session_state.vector_store)
                st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

        # Chat input
        prompt = st.chat_input("Describe your situation or ask a question...")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching playbook..."):
                    response = rag_answer_with_sources(prompt, st.session_state.vector_store)
                st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

else:
    # ============= EMAIL SETTINGS =============
    st.title("📧 Email Notification Settings")
    st.markdown("*Configure email alerts for supervisors*")

    st.divider()

    # Instructions
    with st.expander("📖 How to set up Resend for sending emails", expanded=True):
        st.markdown("""
        **Easy 2-minute setup with Resend (Free: 100 emails/day):**

        1. Go to [resend.com](https://resend.com) and sign up (free)
        2. Go to **API Keys** in your dashboard
        3. Click **Create API Key**
        4. Copy the key (starts with `re_...`)
        5. Paste it below as "API Key"

        **That's it!** No password hassles, works with any recipient email.
        """)

    st.markdown("### Resend API Configuration")
    col1, col2 = st.columns(2)

    with col1:
        sender_email = st.text_input(
            "Sender Display Name (optional)",
            value=st.session_state.email_config["sender_email"],
            placeholder="Lab Operator",
            help="Just for display - emails come from CODE4CARE"
        )

    with col2:
        sender_password = st.text_input(
            "Resend API Key",
            value=st.session_state.email_config["sender_password"],
            type="password",
            placeholder="re_xxxxxxxxx..."
        )

    st.markdown("### Recipient Configuration (Supervisor)")
    recipient_email = st.text_input(
        "Supervisor Email Address",
        value=st.session_state.email_config["recipient_email"],
        placeholder="supervisor@hospital.com"
    )

    st.markdown("### Notification Settings")
    col1, col2 = st.columns(2)

    with col1:
        report_interval = st.selectbox(
            "Send Progress Reports Every",
            options=[0.5, 1.0, 2.0, 4.0, 6.0, 12.0, 24.0],
            index=1,
            format_func=lambda x: f"{x} hour(s) (sim time)"
        )

    with col2:
        critical_alerts = st.checkbox(
            "Send Critical Alerts Immediately",
            value=st.session_state.email_config["critical_alerts_enabled"],
            help="When enabled, sends email immediately when status becomes CRITICAL"
        )

    st.divider()

    # Save and Test buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("💾 Save Settings", use_container_width=True, type="primary"):
            st.session_state.email_config = {
                "sender_email": sender_email,
                "sender_password": sender_password,
                "recipient_email": recipient_email,
                "enabled": True,
                "report_interval_hours": report_interval,
                "critical_alerts_enabled": critical_alerts,
            }
            st.success("✅ Settings saved!")

    with col2:
        if st.button("🔌 Disable Emails", use_container_width=True):
            st.session_state.email_config["enabled"] = False
            st.info("📧 Email notifications disabled")

    with col3:
        if st.button("📤 Send Test Email", use_container_width=True):
            if sender_password and recipient_email:
                with st.spinner("Sending test email..."):
                    test_html = """
                    <html>
                    <body style="font-family: Arial, sans-serif; padding: 20px;">
                        <div style="background: linear-gradient(135deg, #1a237e, #4a148c); color: white; padding: 20px; border-radius: 10px;">
                            <h1>🧬 CODE4CARE - Test Email</h1>
                        </div>
                        <h2 style="color: #2e7d32;">✅ Email Configuration Successful!</h2>
                        <p>This is a test email from the CODE4CARE Cell Culture Monitoring System.</p>
                        <p>If you received this, your email notifications are properly configured.</p>
                        <hr>
                        <p style="color: #666; font-size: 12px;">
                            You will receive:
                            <ul>
                                <li>Progress reports at your configured interval</li>
                                <li>Critical alerts when parameters go out of range</li>
                            </ul>
                        </p>
                    </body>
                    </html>
                    """
                    success, msg = send_email(
                        sender_email,
                        sender_password,
                        recipient_email,
                        "🧬 CODE4CARE - Test Email",
                        test_html
                    )
                    if success:
                        st.success(f"✅ Test email sent to {recipient_email}!")
                    else:
                        st.error(f"❌ Failed: {msg}")
            else:
                st.warning("⚠️ Please enter Resend API Key and Recipient Email")

    # Current Status
    st.divider()
    st.markdown("### Current Status")

    if st.session_state.email_config["enabled"]:
        st.success("📧 **Email notifications are ENABLED**")
        st.markdown(f"""
        - **Sender:** {st.session_state.email_config['sender_email']}
        - **Recipient:** {st.session_state.email_config['recipient_email']}
        - **Report Interval:** Every {st.session_state.email_config['report_interval_hours']} hour(s)
        - **Critical Alerts:** {'Enabled' if st.session_state.email_config['critical_alerts_enabled'] else 'Disabled'}
        """)
    else:
        st.info("📧 Email notifications are disabled. Configure and save settings above to enable.")
