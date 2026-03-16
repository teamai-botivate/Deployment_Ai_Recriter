import os
import sys
import uuid
import json
import time
import smtplib
import re # Added for title cleaning
from typing import List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Framework & Utilities
try:
    from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from dotenv import load_dotenv
    import uvicorn
except ImportError as e:
    print(f"CRITICAL: Missing dependency. Run 'pip install fastapi uvicorn pydantic-settings python-dotenv'. Error: {e}")

# Load environment variables (reaching out to root .env)
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, "../../.env"))

# Global sys.path setup for Backend services
app_root = os.path.abspath(os.path.join(basedir, "../../Backend/app"))
if app_root not in sys.path:
    sys.path.append(app_root)

# Local imports
try:
    from agent import generate_aptitude_questions, evaluate_code
except ImportError:
    # Fallback for different execution contexts
    try:
        from .agent import generate_aptitude_questions, evaluate_code
    except ImportError:
        print("WARNING: agent.py not found in current path.")

# Import services that rely on the updated sys.path
try:
    from services.gmail_oauth import gmail_oauth_service
except ImportError as e:
    print(f"WARNING: Could not import gmail_oauth_service. Error: {e}")

# Database file at root level
DB_FILE = os.path.abspath(os.path.join(basedir, "../../assessments_db.json"))
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:5500") # Default for local

def init_db():
    try:
        if not os.path.exists(DB_FILE):
            print(f"DEBUG: Creating new database at {DB_FILE}")
            with open(DB_FILE, "w") as f:
                json.dump({"assessments": [], "submissions": []}, f)
        else:
            print(f"DEBUG: Database found at {DB_FILE}")
    except Exception as e:
        print(f"CRITICAL: Failed to initialize DB: {e}")

def get_db():
    init_db()
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)
    print(f"DEBUG: Database saved successfully to {DB_FILE}")

app = FastAPI(title="Aptitude Generator API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JDRequest(BaseModel):
    jd_text: str
    difficulty_level: Optional[str] = "Medium"
    custom_instructions: Optional[str] = ""
    mcq_count: Optional[int] = 25

class RunCodeRequest(BaseModel):
    code: str
    language: str
    problem_text: str
    test_cases: list

@app.post("/run-code")
async def run_code(request: RunCodeRequest):
    print(f"\n--- 💻 REQUEST: Evaluating Code ({request.language}) ---")
    try:
        result = evaluate_code(
            request.problem_text, 
            request.code, 
            request.language, 
            request.test_cases
        )
        return result
    except Exception as e:
        print(f"Error evaluating code: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Helper Functions ---
def clean_job_title(title: str) -> str:
    """Removes experience tags like (0-1 Year), [Remote], etc."""
    if not title: return ""
    import re
    # Remove everything inside parentheses or brackets
    cleaned = re.sub(r'\(.*?\)|\[.*?\]', '', title)
    # Remove common experience keywords if they remain
    cleaned = re.sub(r'(?i)\b(senior|junior|intern|fresher|lead|staff|principal)\b', '', cleaned)
    # Remove extra spaces
    return cleaned.strip()

class CandidateItem(BaseModel):
    email: str
    name: Optional[str] = "Candidate"
    resume_path: Optional[str] = ""
    ai_analysis: Optional[dict] = {}

class EmailRequest(BaseModel):
    candidates: list[CandidateItem]
    job_title: str
    mcq_count: int
    coding_count: int
    assessment_link: str
    mcqs: list[dict]
    coding_questions: list[dict]
    company_name: Optional[str] = "RecruitAI"

class ScheduleInterviewRequest(BaseModel):
    emails: list[str]
    job_title: str
    date: str
    time: str
    location: str
    company_name: Optional[str] = "RecruitAI"

@app.post("/generate-aptitude")
async def generate_aptitude(request: JDRequest):
    print(f"\n--- 🤖 REQUEST: Generate Aptitude & Coding ---")
    if not request.jd_text.strip():
        raise HTTPException(status_code=400, detail="Job Description text is empty")
    
    try:
        result = generate_aptitude_questions(
            request.jd_text, 
            request.difficulty_level, 
            request.custom_instructions,
            request.mcq_count
        )
        return result # returns {"mcqs": [...], "coding_questions": [...]}
    except Exception as e:
        print(f"Error generating content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def update_db_task(request: EmailRequest):
    try:
        token = request.assessment_link.split("token=")[-1]
        db = get_db()
        db["assessments"].append({
            "id": str(uuid.uuid4()),
            "token": token,
            "job_title": request.job_title,
            "candidates": [c.dict() for c in request.candidates],
            "emails": [c.email for c in request.candidates], # Backward compatibility
            "mcqs": request.mcqs,
            "coding_questions": request.coding_questions,
            "timestamp": time.time(),
            "status": "Sent"
        })
        save_db(db)
        print(f"✅ Assessment {token} saved to DB")
    except Exception as e:
        print(f"❌ DB Update Error: {e}")

@app.post("/send-assessment")
async def send_assessment(request: EmailRequest, background_tasks: BackgroundTasks):
    print(f"\n--- 📧 REQUEST: Send Assessment to {len(request.candidates)} candidates ---")
    
    # Run DB update synchronously to ensure the link is valid before emails are received
    update_db_task(request)

    try:
        # --- Gmail OAuth (API-based, safe for Hugging Face) ---
        is_sent = False
        try:
            # Sync with the main Backend/app context
            from services.gmail_oauth import gmail_oauth_service
            
            company_id = "default_company"
            if gmail_oauth_service.is_connected(company_id):
                print(f"🔗 Using Gmail OAuth API to bypass SMTP network restrictions...")
                
                # --- CLEAN JOB TITLE ---
                clean_title = clean_job_title(request.job_title)
                
                format_info = ""
                if request.mcq_count > 0: format_info += f"<li><strong>Aptitude:</strong> {request.mcq_count} MCQs</li>"
                if request.coding_count > 0: format_info += f"<li><strong>Coding:</strong> {request.coding_count} DSA Questions</li>"

                for cand in request.candidates:
                    email = cand.email
                    subject = f"Career Opportunity | {clean_title} Technical Evaluation"
                    body = f"""
                    <html>
                    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: auto; padding: 20px; border: 1px solid #e2e8f0; border-radius: 10px;">
                        <h2 style="color: #6366f1;">Congratulations!</h2>
                        <p>Dear {cand.name},</p>
                        <p>Your profile for the <strong>{clean_title}</strong> role has been shortlisted. Please complete the following technical assessment.</p>
                        
                        <div style="background: #f4f4f9; padding: 20px; border-radius: 10px; border-left: 5px solid #6366f1; margin: 20px 0;">
                            <p><strong>Assessment Details:</strong></p>
                            <ul>
                                {format_info}
                                <li><strong>Environment:</strong> Online IDE (Multiple Languages Supported)</li>
                                <li><strong>Mandatory:</strong> Working Camera & Face Verification required to start.</li>
                                <li><strong>Estimated Time:</strong> 1 Hour</li>
                            </ul>

                            <p style="text-align: center; margin-top: 25px;">
                                <a href="{request.assessment_link}" style="background: #6366f1; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; font-weight: bold;">Enter Test Environment</a>
                            </p>
                        </div>
                        <p>Best Regards,<br><strong>{request.company_name}</strong><br>RecruitAI</p>
                        <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
                        <p style="font-size: 0.8rem; color: #94a3b8; text-align: center;">This is an automated message. Please <strong>do not reply</strong> to this email.</p>
                    </body>
                    </html>
                    """
                    # Adding No-Reply Header for Gmail API if supported by service, 
                    # else instructions in body are the standard way.
                    gmail_oauth_service.send_email(company_id, email, subject, body)
                is_sent = True
        except Exception as oauth_e:
            print(f"⚠️ Gmail OAuth Send failed or not connected: {oauth_e}. Falling back to SMTP...")

        # --- Standard SMTP Fallback ---
        if not is_sent:
            smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
            smtp_port = int(os.getenv("SMTP_PORT", 587))
            smtp_user = os.getenv("SMTP_USER")
            smtp_password = os.getenv("SMTP_PASSWORD")

            # On Hugging Face (or similar), SMTP usually fails with "Network is unreachable"
            # We catch this and show a friendly message to Use OAuth instead
            if not all([smtp_user, smtp_password]):
                raise HTTPException(status_code=500, detail="Mail service unreachable. Please ensure Gmail is connected via Dashboard with the 'Send Email' permission.")

            try:
                server = smtplib.SMTP(smtp_server, smtp_port, timeout=10)
                server.starttls()
                server.login(str(smtp_user or ""), str(smtp_password or ""))
                
                # --- CLEAN JOB TITLE (SMTP) ---
                clean_title = clean_job_title(request.job_title)

                # ... SMTP Sending Logic ...
                for cand in request.candidates:
                    email = cand.email
                    msg = MIMEMultipart()
                    msg['From'] = str(smtp_user or "")
                    msg['To'] = str(email or "")
                    msg['Subject'] = str(f"Career Opportunity | {clean_title} Technical Evaluation")
                    msg['Reply-To'] = "no-reply@recruitai.com"
                    body = f"Technical assessment link for {cand.name}: {request.assessment_link}\n\nCareer Opportunity for {clean_title}.\n\nThis is an automated email. Please do not reply."
                    msg.attach(MIMEText(body, 'plain'))
                    server.send_message(msg)
                server.quit()
            except Exception as smtp_err:
                print(f"SMTP Error: {smtp_err}")
                if "101" in str(smtp_err) or "unreachable" in str(smtp_err).lower():
                    raise HTTPException(status_code=500, detail="Network blocked (SMTP). Please connect Gmail on Dashboard to bypass this restriction.")
                else:
                    raise smtp_err
            
        return {"status": "success"}
    except Exception as e:
        print(f"❌ Comprehensive Mail Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class RejectionRequest(BaseModel):
    emails: list[str]
    job_title: str

@app.post("/send-rejection")
async def send_rejection(request: RejectionRequest):
    print(f"\n--- 📧 REQUEST: Send Rejection to {len(request.emails)} candidates ---")
    
    try:
        is_sent = False
        # --- Try Gmail OAuth API ---
        try:
            from services.gmail_oauth import gmail_oauth_service
            company_id = "default_company"
            
            if gmail_oauth_service.is_connected(company_id):
                print(f"🔗 Using Gmail OAuth API for rejections...")
                
                clean_title = clean_job_title(request.job_title)
                
                for email in request.emails:
                    subject = f"Update on your application for {clean_title}"
                    body = f"""
                    <html>
                    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                        <p>Dear Candidate,</p>
                        <p>Thank you for giving us the opportunity to consider your application for the <strong>{clean_title}</strong> position.</p>
                        <p>We have reviewed your profile, and while we were impressed with your qualifications, we have decided to proceed with other candidates who more closely align with our current requirements.</p>
                        <p>We will keep your resume in our database and may contact you if a suitable opening arises in the future.</p>
                        <p>We wish you the best in your job search.</p>
                        <br>
                        <p>Best Regards,<br><strong>Talent Acquisition Team</strong><br>RecruitAI</p>
                        <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
                        <p style="font-size: 0.8rem; color: #94a3b8; text-align: center;">This is an automated message. Please <strong>do not reply</strong> to this email.</p>
                    </body>
                    </html>
                    """
                    gmail_oauth_service.send_email(company_id, email, subject, body)
                is_sent = True
        except Exception as oauth_e:
            print(f"⚠️ Gmail OAuth Rejection failed: {oauth_e}")

        # --- SMTP Fallback ---
        if not is_sent:
            smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
            smtp_port = int(os.getenv("SMTP_PORT", 587))
            smtp_user = os.getenv("SMTP_USER")
            smtp_password = os.getenv("SMTP_PASSWORD")

            if not all([smtp_user, smtp_password]):
                raise HTTPException(status_code=500, detail="Mail service unreachable.")

            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(str(smtp_user or ""), str(smtp_password or ""))
            
            clean_title = clean_job_title(request.job_title)
            for email in request.emails:
                msg = MIMEMultipart()
                msg['From'] = str(smtp_user or "")
                msg['To'] = str(email or "")
                msg['Subject'] = str(f"Update on your application for {clean_title}")
                body_text = f"Dear Candidate,\n\nThank you for applying for the {clean_title} position. After reviewing your profile, we have decided to proceed with other candidates.\n\nBest Regards,\nRecruitAI Team"
                msg.attach(MIMEText(body_text, 'plain'))
                server.send_message(msg)
            server.quit()

        return {"status": "success", "message": f"Sent rejection to {len(request.emails)} candidates"}
    except Exception as e:
        print(f"❌ SMTP Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-assessment/{token}")
async def get_assessment(token: str):
    db = get_db()
    assessment = next((a for a in db["assessments"] if a["token"] == token), None)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return {
        "mcqs": assessment.get("mcqs", []), 
        "coding": assessment.get("coding_questions", []), 
        "job_title": assessment["job_title"]
    }

@app.post("/submit-assessment")
async def submit_assessment(data: dict, background_tasks: BackgroundTasks):
    # data: { token, email, mcq_score, mcq_total, coding_score, coding_total, suspicious, mcq_answers, coding_answers }
    print(f"\n--- 📝 REQUEST: Candidate Submission ({data.get('email')}) ---")
    try:
        db = get_db()
        submission = {
            "token": data["token"],
            "email": data["email"],
            "mcq_score": data.get("mcq_score", 0),
            "mcq_total": data.get("mcq_total", 0),
            "coding_score": data.get("coding_score", 0),
            "coding_total": data.get("coding_total", 0),
            "timestamp": time.time(),
            "suspicious": data.get("suspicious", "Normal"),
            "mcq_answers": data.get("mcq_answers", []),
            "coding_answers": data.get("coding_answers", [])
        }
        db["submissions"].append(submission)
        save_db(db)

        # Send notification to Recruiter
        background_tasks.add_task(send_submission_notification, submission)

        return {"status": "success"}
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def send_submission_notification(submission: dict):
    """Sends an email to the recruiter when a candidate submits the test."""
    try:
        recruiter_email = os.getenv("SMTP_USER")
        if not recruiter_email:
            print("⚠️ No SMTP_USER configured, skipping submission email.")
            return

        # Find job title from token
        db = get_db()
        assessment = next((a for a in db["assessments"] if a["token"] == submission["token"]), None)
        job_title = assessment["job_title"] if assessment else "Unknown Role"

        subject = f"Assessment Submitted: {submission['email']} - {job_title}"
        
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: auto; padding: 20px; border: 1px solid #e2e8f0; border-radius: 10px;">
            <h2 style="color: #4f46e5;">New Assessment Submission</h2>
            <p>A candidate has completed their technical evaluation.</p>
            
            <div style="background: #f8fafc; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #4f46e5;">
                <p><strong>Candidate:</strong> {submission['email']}</p>
                <p><strong>Role:</strong> {job_title}</p>
                <p><strong>MCQ Score:</strong> {submission['mcq_score']} / {submission['mcq_total']}</p>
                <p><strong>Coding Score:</strong> {submission['coding_score']} / {submission['coding_total']}</p>
                <p><strong>Proctoring Status:</strong> <span style="color: {'#ef4444' if submission['suspicious'] != 'Normal' else '#10b981'}; font-weight: bold;">{submission['suspicious']}</span></p>
            </div>
            
            <p>You can view full details in the RecruitAI dashboard.</p>
            <p>Best Regards,<br><strong>RecruitAI Agent</strong></p>
        </body>
        </html>
        """

        # Try Gmail OAuth first
        is_sent = False
        try:
            from services.gmail_oauth import gmail_oauth_service
            company_id = "default_company"
            if gmail_oauth_service.is_connected(company_id):
                gmail_oauth_service.send_email(company_id, recruiter_email, subject, body)
                is_sent = True
        except: pass

        # Fallback to SMTP
        if not is_sent:
            smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
            smtp_port = int(os.getenv("SMTP_PORT", 587))
            smtp_password = os.getenv("SMTP_PASSWORD")
            
            if recruiter_email and smtp_password:
                msg = MIMEMultipart()
                msg['From'] = recruiter_email
                msg['To'] = recruiter_email # Send to self
                msg['Subject'] = subject
                msg.attach(MIMEText(body, 'html'))
                
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls()
                    server.login(recruiter_email, smtp_password)
                    server.send_message(msg)
        
        print(f"✅ Submission notification sent for {submission['email']}")
    except Exception as e:
        print(f"❌ Failed to send submission notification: {e}")

@app.get("/get-analytics")
async def get_analytics():
    db = get_db()
    return db

@app.post("/schedule-interview")
async def schedule_interview(request: ScheduleInterviewRequest):
    print(f"\n--- 📅 REQUEST: Schedule Interview for {len(request.emails)} candidates ---")
    try:
        from services.gmail_oauth import gmail_oauth_service
        company_id = "default_company"
        
        is_sent = False
        if gmail_oauth_service.is_connected(company_id):
            clean_title = clean_job_title(request.job_title)
            
            for email in request.emails:
                subject = f"Interview Invitation: {clean_title} Role"
                body = f"""
                <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: auto; padding: 20px; border: 1px solid #e2e8f0; border-radius: 10px;">
                    <h2 style="color: #6366f1;">Interview Scheduled!</h2>
                    <p>Dear Candidate,</p>
                    <p>Congratulations! Based on your assessment results, we would like to invite you for an <strong>offline</strong> interview for the <strong>{clean_title}</strong> position.</p>
                    
                    <div style="background: #f4f4f9; padding: 20px; border-radius: 10px; border-left: 5px solid #6366f1; margin: 20px 0;">
                        <p><strong>Interview Details:</strong></p>
                        <ul>
                            <li><strong>Date:</strong> {request.date}</li>
                            <li><strong>Time:</strong> {request.time}</li>
                            <li><strong>Location:</strong> {request.location}</li>
                        </ul>
                    </div>
                    
                    <p>Best Regards,<br><strong>{request.company_name}</strong><br>RecruitAI</p>
                    <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
                    <p style="font-size: 0.8rem; color: #94a3b8; text-align: center;">This is an automated message. Please <strong>do not reply</strong> to this email.</p>
                </body>
                </html>
                """
                gmail_oauth_service.send_email(company_id, email, subject, body)
            is_sent = True
        
        if not is_sent:
            # Simple SMTP fallback
            smtp_user = os.getenv("SMTP_USER")
            smtp_password = os.getenv("SMTP_PASSWORD")
            if not all([smtp_user, smtp_password]):
                raise HTTPException(status_code=500, detail="Mail service unreachable. Connect Gmail via Dashboard.")
            
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(str(smtp_user or ""), str(smtp_password or ""))
            
            clean_title = clean_job_title(request.job_title)
            
            for email in request.emails:
                msg = MIMEMultipart()
                msg['From'] = str(smtp_user or "")
                msg['To'] = str(email or "")
                msg['Subject'] = str(f"Interview Invitation: {clean_title}")
                msg['Reply-To'] = "no-reply@recruitai.com"
                body_text = f"Your offline interview for {clean_title} is scheduled on {request.date} at {request.time} at {request.location}.\n\nBest Regards,\n{request.company_name}\nRecruitAI\n\n(Auto-generated email - Do not reply)"
                msg.attach(MIMEText(body_text, 'plain'))
                server.send_message(msg)
            server.quit()

        return {"status": "success", "message": "Interview invitations sent successfully"}
    except Exception as e:
        print(f"❌ Interview Scheduling Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete-assessment/{token}")
async def delete_assessment(token: str):
    db = get_db()
    db["assessments"] = [a for a in db["assessments"] if a["token"] != token]
    db["submissions"] = [s for s in db["submissions"] if s["token"] != token]
    save_db(db)
    return {"status": "success"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
