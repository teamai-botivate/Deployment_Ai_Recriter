"""
Role Matching Service - Zero-Shot Classification
Matches resumes to job descriptions using facebook/bart-large-mnli
More accurate than semantic similarity for role detection.
"""

import re
from typing import Optional, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Initialize Zero-Shot Classification Pipeline (Singleton)
_zero_shot_classifier = None

def get_zero_shot_classifier():
    """Load Zero-Shot Classification model (BART-large-MNLI)"""
    global _zero_shot_classifier
    if _zero_shot_classifier is None:
        try:
            from transformers import pipeline
            import torch
            
            logger.info("⏳ Loading Zero-Shot Classifier (facebook/bart-large-mnli)...")
            
            # Use GPU if available
            device = 0 if torch.cuda.is_available() else -1
            
            _zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=device
            )
            
            logger.info("✅ Zero-Shot Classifier Loaded Successfully.")
        except Exception as e:
            logger.error(f"Failed to load Zero-Shot Classifier: {e}")
            raise
    
    return _zero_shot_classifier


def extract_text_segment(text: str, max_chars: int = 1000) -> str:
    """Helper to safely get start of text"""
    if not text: return ""
    return text[:max_chars].replace('\n', ' ').strip()


def clean_role_name(title: str) -> str:
    """Extracts only the core role name for better AI matching"""
    if not title: return "Candidate"
    # Remove patterns like (0-1 Year), [Senior], etc.
    clean = re.sub(r'[\(\[\{].*?[\)\]\}]', '', title)
    # Remove common filler words
    clean = re.sub(r'(?i)(opening|role|position|vacancy|career|immediate|hiring|full-time|part-time)', '', clean)
    return clean.strip()


def extract_potential_role(text: str) -> Optional[str]:
    """Attempts to extract a role string from text"""
    if not text: return None
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines: return None
    return lines[0][:100]


def detect_and_match_role(
    jd_title: str,
    email_subject: str,
    email_body: str,
    resume_text: str,
    threshold: float = 0.6,  # Zero-shot typically needs higher threshold (0.6-0.7)
    jd_title_embedding: np.ndarray = None  # Kept for backward compatibility
) -> Dict[str, any]:
    """
    Role detection using Zero-Shot Classification (High Accuracy).
    
    Args:
        jd_title: Target role from job description (e.g., "Backend Developer")
        email_subject: Subject line of application email
        email_body: Body of application email
        resume_text: Full resume text
        threshold: Minimum confidence score (0.0-1.0, default 0.6)
    
    Returns:
        Dict with detected_role, is_match, similarity score, etc.
    """
    
    # Construct combined text from email + resume header
    combined_text_parts = []
    
    # Priority 1: Email Subject (Cleaned)
    if email_subject:
        clean_subj = re.sub(
            r'(?i)(application|applying|resume|for|regarding|re:|ref:)', 
            '', 
            email_subject
        ).strip()
        if clean_subj:
            combined_text_parts.append(clean_subj)
    
    # Priority 2: Email Body Preview
    if email_body:
        body_preview = extract_text_segment(email_body, max_chars=300)
        if body_preview:
            combined_text_parts.append(body_preview)
    
    # Priority 3: Resume Header (Top 500 chars - contains role, skills, summary)
    if resume_text:
        resume_header = extract_text_segment(resume_text, max_chars=500)
        if resume_header:
            combined_text_parts.append(resume_header)
    
    # Combine all parts
    combined_text = ". ".join(combined_text_parts)
    
    if not combined_text:
        logger.warning(f"No text available for role matching")
        return {
            "detected_role": "Unknown",
            "source": None,
            "is_match": True,  # Give benefit of doubt
            "similarity": 0.0,
            "jd_title": jd_title
        }
    
    # Clean JD Title for better labeling
    core_role = clean_role_name(jd_title)
    
    # We use a set of labels to compare. If it matches the JD role better than 
    # being a "Meeting Minutes" or "Generic Document", then it's a match.
    labels = [core_role, "Other/Irrelevant Document", "Meeting Minutes (MOM)"]
    
    # Run Zero-Shot Classification
    try:
        classifier = get_zero_shot_classifier()
        
        # Classification: Comparative mode (multi_label=False)
        # This tells us WHICH label is the best fit.
        result = classifier(
            combined_text,
            candidate_labels=labels,
            multi_label=False
        )
        
        # Extract results for our target role
        scores_map = dict(zip(result["labels"], result["scores"]))
        relevance_score = scores_map.get(core_role, 0.0)
        
        logger.info(f"DEBUG: Comparative Scores for '{core_role}': {scores_map}")
        
        # Logic Change: If core_role is the TOP prediction, or its score is > threshold
        is_top_choice = result["labels"][0] == core_role
        is_match = is_top_choice or (relevance_score >= threshold)
        
        # Extract detected role from text for display
        detected_role_text = clean_subj if email_subject else extract_potential_role(resume_text)
        if not detected_role_text or len(detected_role_text) < 3:
            detected_role_text = core_role
        
        return {
            "detected_role": detected_role_text,
            "source": "comparative_classification",
            "is_match": is_match,
            "similarity": round(relevance_score, 2),
            "jd_title": core_role
        }
        
    except Exception as e:
        logger.error(f"Zero-Shot Classification Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: Give benefit of doubt
        return {
            "detected_role": "Error", 
            "source": "error", 
            "is_match": True, 
            "similarity": 0.0, 
            "jd_title": jd_title
        }


# Legacy function kept for backward compatibility (not used anymore)
def get_text_embedding(text: str) -> Optional[np.ndarray]:
    """Deprecated: Kept for backward compatibility"""
    return None


def calculate_semantic_similarity(
    role1_text: str = None, 
    role2_text: str = None,
    role1_embedding: np.ndarray = None,
    role2_embedding: np.ndarray = None
) -> float:
    """Deprecated: Kept for backward compatibility"""
    return 0.0
