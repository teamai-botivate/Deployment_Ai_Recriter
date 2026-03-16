# 🚀 Easy Setup Guide — Agentic Hiring Suite

## 👋 Introduction

Welcome! This guide is designed to help you set up your AI-powered recruitment system from scratch. Even if you have never touched code before, just follow these steps one by one, and you'll be up and running in about 15-20 minutes.

---

## 📋 Prerequisites (What You Need)

1. A **Gmail Account** (Your company's hiring email).
2. A **Windows PC**.
3. An internet connection.

---

## 🛠️ Step 1: Install Python (2 minutes)

Python is the "engine" that runs this application.

1. Go to [Python.org](https://www.python.org/downloads/windows/).
2. Download the **latest version** (e.g., Python 3.12).
3. **CRITICAL:** When the installer opens, check the box at the bottom that says **"Add Python to PATH"**.
   _If you miss this, you will get "Command not found" errors later!_
4. Click **"Install Now"** and wait for it to finish.

---

## 🔑 Step 2: Get Your AI API Keys (3 minutes)

The system uses "AI Brains" to read and understand resumes.

1. **OpenAI API (Required):**
   - Go to [OpenAI Platform](https://platform.openai.com/).
   - Sign in or create an account.
   - Go to the **API Keys** section.
   - Click **"Create new secret key"**, name it "My Hiring App", and **copy the key immediately**.
   - **Note:** You need a small amount of credit (e.g., $5) in your OpenAI billing account for the key to work.
2. **Hugging Face Token (Optional - for visual scanning):**
   - Go to [Hugging Face Settings](https://huggingface.co/settings/tokens).
   - Create a "Read" token and copy it.

---

## 🌐 Step 3: Google Cloud Setup (The "Secret" File) (8 minutes)

This allows the app to securely connect to your Gmail to fetch resumes and send test links to candidates.

### A. Create a New Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/).
2. Click the **Project Dropdown** at the top top (it might say "Select a project") → Click **"New Project"**.
3. Name it: `Recruit-AI` and click **"Create"**.
4. Important: Wait for the notification, then **Select the project** from the top dropdown.

### B. Enable the Gmail API

1. In the top search bar, type **"Gmail API"**.
2. Click on the result that says "Gmail API".
3. Click the blue **"ENABLE"** button.

### C. Configure the "OAuth Consent Screen"

1. Click the "Three lines" menu (Top Left) → **APIs & Services** → **OAuth consent screen**.
2. Choose **"External"** and click **"Create"**.
3. Fill in these 3 boxes:
   - **App Name:** `Hiring Assistant`
   - **User support email:** (Your own email)
   - **Developer contact info:** (Your own email)
4. Click **"Save and Continue"**.
5. **Scopes (EXTREMELY IMPORTANT):**
   - Click **"Add or Remove Scopes"**.
   - In the search box, type `gmail.readonly` and press Enter. **Check the box** for it.
   - In the search box, type `gmail.send` and press Enter. **Check the box** for it.
   - Click **"Update"** at the bottom, then click **"Save and Continue"**.
6. **Test Users:**
   - Click **"+ ADD USERS"**.
   - Type the email address you plan to use for the system.
   - Click **"Add"**, then **"Save and Continue"**.

### D. Download Your Credentials

1. Click **"Credentials"** in the left sidebar.
2. Click **"+ CREATE CREDENTIALS"** at the top → **"OAuth client ID"**.
3. **Application type:** Select "Web application".
4. **Authorized redirect URIs:**
   - Click **"+ ADD URI"**.
   - If running on your PC, paste: `http://localhost:8000/auth/gmail/callback`
   - If using Hugging Face Spaces, also add: `https://your-space-name.hf.space/auth/gmail/callback`
5. Click **"Create"**.
6. A popup appears. Click **"DOWNLOAD JSON"**.
7. **RENAME** this file to exactly: `client_secret.json`.

---

## 📂 Step 4: Setting Up the Desktop Folder (2 minutes)

1. Go to your `Deployment_Package` folder.
2. **Move** your `client_secret.json` file into the `Backend/` folder.
3. Find the file named **`.env.example`** in the main folder.
4. Right-click it → **Rename** to just **`.env`** (delete the `.example` part).
5. Open the `.env` file with Notepad and replace the placeholder with your key:
   ```text
   GROQ_API_KEY=paste_your_key_here
   ```
6. Save and close.

---

## 🚀 Step 5: Launching the System (2 minutes)

1. Open the folder where all the files are.
2. Click on the **Address Bar** at the top of the folder window (where the folder path is), type `cmd`, and press **Enter**.
3. A black box will open. Type this command first and press Enter:
   ```bash
   pip install -r requirements.txt
   ```
   _(Wait for it to finish installing all the tools)_
4. Now, type this to start the app:
   ```bash
   python -m uvicorn Backend.app.unified_server:app --host 0.0.0.0 --port 8000
   ```
5. Open your web browser (Chrome/Edge) and go to: `http://localhost:8000`

---

## 📧 Step 6: Final Link (Gmail Connection)

1. Once the dashboard opens, click the **"Connect Gmail"** button.
2. A Google login popup will appear. Login with your hiring email.
3. **Important:** You will see a warning "Google hasn't verified this app". Click **"Advanced"** → **"Go to Hiring Assistant (unsafe)"**. (It is safe, you just made it yourself!)
4. **CRITICAL BOX:** On the permissions page, you **MUST check the box** that says **"Send email on your behalf"**. If you don't check this, the "Send Assessment" button won't work!
5. Click **"Allow"**.

✅ **SUCCESS!** Your system is now fully set up. You can now start scanning resumes and sending tests automatically!

---

## 🆘 Common Issues (Troubleshooting)

- **"Python is not recognized":** You didn't check "Add to PATH" in Step 1. Please re-install Python.
- **"Redirect URI Mismatch":** Ensure the URL in Step 3D matches exactly where you are running the app.
- **"Missing Scope Error":** You forgot to check the "Send" box in Step 6. Click "Disconnect" on the dashboard and try Step 6 again.
- **"No OpenAI Key Found":** Ensure your `.env` file is named correctly and has the `OPENAI_API_KEY` inside.
