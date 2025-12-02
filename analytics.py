import json
import os
import smtplib
import pandas as pd
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

LOG_FILE = "query_logs.jsonl"

def get_monthly_analysis(days_back=30):
    """
    Reads logs, filters for the last X days, and uses Claude 
    to generate an HR Training Report.
    """
    if not os.path.exists(LOG_FILE):
        return "❌ No log file found."

    # 1. Load Logs into Pandas
    data = []
    with open(LOG_FILE, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    
    if not data:
        return "⚠️ Log file is empty."

    df = pd.DataFrame(data)
    
    # Convert timestamp to datetime
    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Filter for last 30 days
    cutoff_date = datetime.now() - timedelta(days=days_back)
    recent_logs = df[df['date'] > cutoff_date]
    
    if recent_logs.empty:
        return "⚠️ No queries found in the selected time range."

    # 2. Extract Questions for Analysis
    # We don't need the answers, just what employees are confused about.
    questions_list = recent_logs['question'].tolist()
    questions_text = "\n".join([f"- {q}" for q in questions_list])

    # 3. Generate Report with Claude
    llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
    
    analysis_prompt = ChatPromptTemplate.from_template("""
    You are an expert HR Training Consultant. 
    Below is a list of questions that our employees asked the AI Knowledge Base in the last month.
    
    Your goal is to identify "Knowledge Gaps" to help the HR team improve training.
    
    List of Employee Questions:
    {questions}
    
    Please provide a structured Training Report containing:
    1. **Top 3 Recurring Topics**: What are employees asking about most?
    2. **Confusion Points**: Are there specific nuances (e.g., specific tire models, specific warranty exclusions) that seem to trip people up?
    3. **Training Recommendations**: Suggest 2-3 specific topics HR should cover in the next workshop.
    
    Format the output as a professional email body.
    """)
    
    chain = analysis_prompt | llm
    report = chain.invoke({"questions": questions_text})
    
    return report.content

def send_hr_email(report_content, recipient_email):
    """
    Sends the generated report via SMTP (Gmail, Outlook, etc.)
    """
    sender_email = os.getenv("EMAIL_SENDER")     # Add to .env
    sender_password = os.getenv("EMAIL_PASSWORD") # Add to .env (App Password)
    
    if not sender_email or not sender_password:
        return "⚠️ Email credentials not set in .env. Report generated but not sent."

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = f"Monthly Employee Training Insights - {datetime.now().strftime('%B %Y')}"

    msg.attach(MIMEText(report_content, 'plain'))

    try:
        # Example using Gmail (smtp.gmail.com, port 587)
        # Update server if using Outlook/Office365
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return "✅ Email sent successfully to HR!"
    except Exception as e:
        return f"❌ Failed to send email: {str(e)}"
    


# Add this to the bottom of your file
if __name__ == "__main__":
    print("Analyze logs...")
    
    # 1. Generate the report
    hr_report = get_monthly_analysis(days_back=30)
    
    # Check if analysis returned an error string (starts with warning/error icon)
    if hr_report.startswith("❌") or hr_report.startswith("⚠️"):
        print(hr_report)
    else:
        print("Report generated! Sending email...")
        
        # 2. Send the email
        # Replace this string with the actual email you want to send TO
        recipient = "nathanjshaw88@gmail.com" 
        
        result = send_hr_email(hr_report, recipient)
        print(result)