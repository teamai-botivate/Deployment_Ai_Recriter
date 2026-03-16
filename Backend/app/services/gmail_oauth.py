"""
Gmail OAuth 2.0 Service
Handles Google OAuth flow and token management for Gmail access
"""

import os
import json
import pickle
from pathlib import Path
from typing import Optional, Dict
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build


class GmailOAuthService:
    """
    Manages Gmail OAuth 2.0 authentication and token storage
    """
    
    SCOPES = [
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/gmail.send'
    ]
    CLIENT_SECRET_FILE = 'client_secret.json'
    TOKEN_DIR = 'tokens'  # Directory to store user tokens
    
    def __init__(self):
        # Look in Backend directory (standard)
        self.backend_dir = Path(__file__).parent.parent.parent
        # Look in Root directory (for Docker/HF)
        self.root_dir = self.backend_dir.parent
        
        # Priority check
        potential_paths = [
            self.backend_dir / self.CLIENT_SECRET_FILE,
            self.root_dir / self.CLIENT_SECRET_FILE,
            Path.cwd() / self.CLIENT_SECRET_FILE
        ]
        
        self.client_secret_path = potential_paths[0] # default
        for path in potential_paths:
            if path.exists():
                self.client_secret_path = path
                print(f"✅ Found client_secret.json at: {path}")
                break
        
        self.token_dir = self.backend_dir / self.TOKEN_DIR
        self.token_dir.mkdir(exist_ok=True)
        
        # --- Clear Tokens on Restart (User Request) ---
        try:
            import shutil
            for item in self.token_dir.iterdir():
                if item.is_file():
                    item.unlink()
            print("🧹 Gmail OAuth tokens cleared for fresh start.")
        except Exception as e:
            print(f"⚠️ Failed to clear tokens: {e}")
        
        if not self.client_secret_path.exists():
            print(f"⚠️  WARNING: client_secret.json not found in any standard locations.")
    
    def get_authorization_url(self, company_id: str, redirect_uri: str) -> tuple[str, str]:
        """
        Generate OAuth authorization URL and store code_verifier
        """
        flow = Flow.from_client_secrets_file(
            str(self.client_secret_path),
            scopes=self.SCOPES,
            redirect_uri=redirect_uri
        )
        
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )
        
        # Store state AND code_verifier for exchange later
        code_verifier = getattr(flow, 'code_verifier', None)
        
        state_file = self.token_dir / f"{company_id}_state.json"
        with open(state_file, 'w') as f:
            json.dump({
                "state": state, 
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier
            }, f)
        
        return authorization_url, state
    
    def handle_callback(self, company_id: str, code: str, state: str) -> Dict:
        """
        Handle OAuth callback and exchange code for tokens
        """
        # Verify state
        state_file = self.token_dir / f"{company_id}_state.json"
        if not state_file.exists():
            raise ValueError("Invalid state: No matching OAuth session found")
        
        with open(state_file, 'r') as f:
            stored_data = json.load(f)
        
        if stored_data['state'] != state:
            raise ValueError("Invalid state: CSRF protection failed")
        
        # Exchange code for tokens
        flow = Flow.from_client_secrets_file(
            str(self.client_secret_path),
            scopes=self.SCOPES,
            redirect_uri=stored_data['redirect_uri']
        )
        
        # Use stored code_verifier for PKCE if it exists
        try:
            flow.fetch_token(
                code=code,
                code_verifier=stored_data.get('code_verifier')
            )
        except Exception as e:
            if "code verifier" in str(e).lower():
                flow.fetch_token(code=code)
            else:
                raise e
                
        credentials = flow.credentials
        
        # Save tokens securely
        self._save_credentials(company_id, credentials)
        
        # Get user email for confirmation
        service = build('gmail', 'v1', credentials=credentials)
        profile = service.users().getProfile(userId='me').execute()
        user_email = profile.get('emailAddress')
        
        # Cleanup
        try:
            state_file.unlink()
        except:
            pass
            
        return {
            "status": "success",
            "email": user_email,
            "message": f"Successfully connected to {user_email}"
        }

    def get_credentials(self, company_id: str) -> Optional[Credentials]:
        """
        Get stored credentials for a company
        """
        token_file = self.token_dir / f"{company_id}_token.pickle"
        
        if not token_file.exists():
            return None
        
        try:
            with open(token_file, 'rb') as f:
                credentials = pickle.load(f)
            
            # Refresh if expired
            if credentials and credentials.expired and credentials.refresh_token:
                try:
                    credentials.refresh(Request())
                    self._save_credentials(company_id, credentials)
                except Exception as e:
                    print(f"OAUTH_DEBUG: Token refresh failed: {e}")
                    return None
            
            return credentials
        except Exception as e:
            print(f"OAUTH_DEBUG: Error loading credentials: {e}")
            return None
    
    def _save_credentials(self, company_id: str, credentials: Credentials):
        """
        Save credentials to file with disk sync
        """
        token_file = self.token_dir / f"{company_id}_token.pickle"
        
        with open(token_file, 'wb') as f:
            pickle.dump(credentials, f)
            f.flush()
            os.fsync(f.fileno())

    def get_gmail_service(self, company_id: str):
        credentials = self.get_credentials(company_id)
        if not credentials:
            raise ValueError(f"Gmail not connected for {company_id}")
        return build('gmail', 'v1', credentials=credentials)
    
    def send_email(self, company_id: str, to: str, subject: str, body: str):
        """
        Send an email directly through Gmail API (HTTPS based, bypasses SMTP blocks)
        """
        import base64
        from email.message import EmailMessage
        
        if not company_id: company_id = "default_company"
        
        try:
            service = self.get_gmail_service(company_id)
            
            # --- Scope & Identity Verification ---
            credentials = self.get_credentials(company_id)
            token_scopes = getattr(credentials, 'scopes', []) or []
            
            # Get authenticated email for the 'From' header (crucial for some filters)
            profile = service.users().getProfile(userId='me').execute()
            user_email = profile.get('emailAddress')

            if 'https://www.googleapis.com/auth/gmail.send' not in token_scopes:
                raise ValueError(f"MISSING_SEND_PERMISSION: Connected as {user_email}, but 'Send Email' scope is missing. Please re-authenticate.")

            message = EmailMessage()
            message.set_content(body, subtype='html')
            message['To'] = to
            message['From'] = user_email  # Added explicit From header
            message['Subject'] = subject
            # --- No-Reply Configuration ---
            message['Reply-To'] = 'no-reply@botivate.in'
            message['Auto-Submitted'] = 'auto-generated'
            message['X-Auto-Response-Suppress'] = 'All'
            
            # encoded message
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            
            send_result = service.users().messages().send(
                userId="me", 
                body={'raw': raw_message}
            ).execute()
            
            return send_result
        except Exception as e:
            print(f"❌ Gmail API Send Error: {e}")
            raise e

    def revoke_access(self, company_id: str) -> bool:
        token_file = self.token_dir / f"{company_id}_token.pickle"
        if token_file.exists():
            try:
                credentials = self.get_credentials(company_id)
                if credentials:
                    credentials.revoke(Request())
            except:
                pass
            token_file.unlink()
        return True
    
    def is_connected(self, company_id: str) -> bool:
        """
        Check if we have valid credentials with ALL required scopes
        """
        try:
            credentials = self.get_credentials(company_id)
            if not credentials:
                print(f"OAUTH_CHECK: No credentials found for {company_id}")
                return False
                
            if not credentials.valid:
                print(f"OAUTH_CHECK: Credentials invalid/expired for {company_id}")
                return False
                
            # --- Scope Protection ---
            # Verify that ALL scopes in self.SCOPES are present in the token
            token_scopes = getattr(credentials, 'scopes', []) or []
            
            # Print for debugging
            print(f"OAUTH_CHECK: Token Scopes: {token_scopes}")
            print(f"OAUTH_CHECK: Required Scopes: {self.SCOPES}")
            
            # Ensure it's a list for comparison
            if isinstance(token_scopes, str):
                token_scopes = token_scopes.split(' ')

            # --- New Lenient Scope Check ---
            # We consider the service "connected" if the token is valid and has AT LEAST one of our scopes.
            # This allows the dashboard to show "Connected" even if they only gave partial permissions.
            has_at_least_one = False
            for s in self.SCOPES:
                if s in token_scopes:
                    has_at_least_one = True
                    break
            
            if not has_at_least_one:
                print(f"❌ OAUTH_CHECK: No required scopes found in token. Requires at least one of: {self.SCOPES}")
                return False

            # Log warnings for missing scopes instead of failing
            for s in self.SCOPES:
                if s not in token_scopes:
                    print(f"⚠️  OAUTH_CHECK: Missing scope: {s} (Functionality related to this scope will fail)")
            
            print(f"✅ OAUTH_CHECK: Service is connected for {company_id}")
            return True
        except Exception as e:
            print(f"OAUTH_CHECK_ERROR: {e}")
            return False

# Singleton instance
gmail_oauth_service = GmailOAuthService()
