document.addEventListener('DOMContentLoaded', () => {
    initializeElements();
    lucide.createIcons();
    setupEventListeners();
});

// --- Constants ---
const NEON_INNER_HTML = `
    <div class="neon-checkbox__frame">
        <div class="neon-checkbox__box">
            <div class="neon-checkbox__check-container">
                <svg viewBox="0 0 24 24" class="neon-checkbox__check">
                    <path d="M3,12.5l7,7L21,5"></path>
                </svg>
            </div>
            <div class="neon-checkbox__glow"></div>
            <div class="neon-checkbox__borders">
                <span></span><span></span><span></span><span></span>
            </div>
        </div>
        <div class="neon-checkbox__effects">
            <div class="neon-checkbox__particles">
                <span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span>
            </div>
            <div class="neon-checkbox__rings">
                <div class="ring"></div><div class="ring"></div><div class="ring"></div>
            </div>
            <div class="neon-checkbox__sparks">
                <span></span><span></span><span></span><span></span>
            </div>
        </div>
    </div>
`;

// --- State Management ---
let allMcqs = [];
let allCodingQuestions = [];
let selectedMcqs = new Set();
let selectedCoding = new Set();

// --- Elements ---
let generateBtn, jdInput, fileInput, fileNameDisplay, loader;
let selectionSection, questionsList, selectedCount, selectAllCheckbox, doneBtn;
let finalResultSection, finalAptitudeList, finalCodingList, codingQuestionsList, selectAllCoding;
let copyBtn, downloadPdfBtn, emailBtn;
let emailModal, closeEmailModal, cancelEmailBtn, confirmSendEmailBtn, receiverEmailsInput;
let viewAnalysisBtn, analysisDashboard, mainGeneratorCard, jobRolesView, candidateDetailsView, detailJobTitleText, candidatesTbody, mcqTabBtn, codingTabBtn, confirmGmailBtn;
let confirmModal, confirmTitle, confirmMsg, confirmOkBtn, confirmCancelBtn, closeConfirmBtn, confirmIconDynamic;
let currentAptitudeCandidates = []; // Persists full candidate objects from screening

// --- Utilities ---
function escapeHTML(str) {
    if (!str) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// --- Setup ---
function initializeElements() {
    generateBtn = document.getElementById('generate-aptitude-btn');
    jdInput = document.getElementById('jd-input');
    fileInput = document.getElementById('file-upload');
    fileNameDisplay = document.getElementById('file-name');
    loader = document.getElementById('loader');
    difficultyLevelInput = document.getElementById('difficulty-level');
    customInstructionsInput = document.getElementById('custom-instructions');
    mcqCountInput = document.getElementById('mcq-count');

    selectionSection = document.getElementById('selection-section');
    questionsList = document.getElementById('questions-list');
    selectedCount = document.getElementById('selected-count');
    selectAllCheckbox = document.getElementById('select-all-checkbox');
    doneBtn = document.getElementById('done-btn');

    finalResultSection = document.getElementById('final-result-section');
    finalAptitudeList = document.getElementById('final-aptitude-list');
    finalCodingList = document.getElementById('final-coding-list');
    codingQuestionsList = document.getElementById('coding-questions-list');
    selectAllCoding = document.getElementById('select-all-coding');
    copyBtn = document.getElementById('copy-btn');
    downloadPdfBtn = document.getElementById('download-pdf-btn');
    emailBtn = document.getElementById('email-btn');

    emailModal = document.getElementById('email-modal');
    closeEmailModal = document.getElementById('close-email-modal');
    cancelEmailBtn = document.getElementById('cancel-email');
    confirmSendEmailBtn = document.getElementById('confirm-send-email');
    receiverEmailsInput = document.getElementById('receiver-emails');

    viewAnalysisBtn = document.getElementById('view-analysis-btn');
    analysisDashboard = document.getElementById('analysis-dashboard-section');
    mainGeneratorCard = document.querySelector('.generator-layout .main-card:first-child');
    jobRolesView = document.getElementById('job-roles-view');
    candidateDetailsView = document.getElementById('candidate-details-view');
    detailJobTitleText = document.getElementById('detail-job-title');
    candidatesTbody = document.getElementById('candidate-submissions-list');
    
    // New Elements
    selectAllCandidates = document.getElementById('select-all-candidates');
    scheduleBtn = document.getElementById('schedule-btn');
    scheduleModal = document.getElementById('schedule-modal');
    closeSchedule = document.getElementById('close-schedule');
    scheduleCancelBtn = document.getElementById('schedule-cancel-btn');
    scheduleSendBtn = document.getElementById('schedule-send-btn');
    selectedCandidatesCount = document.getElementById('selected-candidates-count');
    interviewDateInput = document.getElementById('interview-date');
    interviewTimeInput = document.getElementById('interview-time');
    interviewLocationInput = document.getElementById('interview-location');
    
    analyticsModal = document.getElementById('analytics-modal');
    analyticsContent = document.getElementById('analytics-content');
    answerKeyModal = document.getElementById('answer-key-modal');
    answerKeyContent = document.getElementById('answer-key-content');

    mcqTabBtn = document.getElementById('mcq-tab-btn');
    codingTabBtn = document.getElementById('coding-tab-btn');
    confirmGmailBtn = document.getElementById('confirm-gmail-btn');

    confirmModal = document.getElementById('confirm-modal');
    confirmTitle = document.getElementById('confirm-title');
    confirmMsg = document.getElementById('confirm-message');
    confirmOkBtn = document.getElementById('confirm-ok-btn');
    confirmCancelBtn = document.getElementById('confirm-cancel-btn');
    closeConfirmBtn = document.getElementById('close-confirm');
    confirmIconDynamic = document.getElementById('confirm-icon-dynamic');
}

function setupEventListeners() {
    // Pre-fill JD from Generator
    const savedJD = localStorage.getItem('recruiter_generated_jd');
    if (savedJD) {
        jdInput.value = savedJD;
    }

    // File Upload handling
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            fileNameDisplay.textContent = file.name;
            const reader = new FileReader();
            reader.onload = (re) => {
                jdInput.value = re.target.result;
            };
            reader.readAsText(file);
        }
    });

    // Step 1: Generate Real Questions from Backend
    generateBtn.addEventListener('click', async () => {
        const jdText = jdInput.value.trim();
        if (!jdText) {
            alert("Please paste a Job Description or upload a file first.");
            return;
        }

        showLoader(true);

        try {
            const difficultyLevel = difficultyLevelInput.value;
            const customInstructions = customInstructionsInput.value.trim();
            const mcqCount = parseInt(mcqCountInput.value) || 25;

            const response = await fetch('/aptitude-api/generate-aptitude', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    jd_text: jdText,
                    difficulty_level: difficultyLevel,
                    custom_instructions: customInstructions,
                    mcq_count: mcqCount
                })
            });

            if (!response.ok) throw new Error("Failed to generate questions");

            const data = await response.json();
            console.log("DEBUG: Received Data:", data);
            allMcqs = data.mcqs || [];
            allCodingQuestions = data.coding_questions || [];

            renderMcqsToSelect();
            renderCodingToSelect();

            // Update tab labels with actual counts
            if (mcqTabBtn) mcqTabBtn.textContent = `MCQs (${allMcqs.length})`;
            if (codingTabBtn) codingTabBtn.textContent = `Coding Questions (${allCodingQuestions.length})`;

            showLoader(false);
            showSection(selectionSection);
            selectionSection.scrollIntoView({ behavior: 'smooth' });
        } catch (error) {
            console.error("GENERATION ERROR:", error);
            alert(`Error: ${error.message}\n\nPlease ensure the backend is running and your network allows the connection.`);
            showLoader(false);
        }
    });

    // Selection Tabs Logic
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
            document.getElementById(tab === 'mcqs' ? 'mcq-selection-view' : 'coding-selection-view').classList.remove('hidden');
        });
    });

    // Step 2: Confirmation
    doneBtn.addEventListener('click', () => {
        if (selectedMcqs.size === 0 && selectedCoding.size === 0) {
            alert("Please select at least one question (MCQ or Coding).");
            return;
        }
        renderFinalLists();
        showSection(finalResultSection);
        finalResultSection.scrollIntoView({ behavior: 'smooth' });
    });

    // Output Tabs Logic
    document.querySelectorAll('.out-tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const t = btn.dataset.outTab;
            document.querySelectorAll('.out-tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            document.querySelectorAll('.out-tab-content').forEach(c => c.classList.add('hidden'));
            document.getElementById(t).classList.remove('hidden');
        });
    });

    // Final: Actions
    copyBtn.addEventListener('click', copyToClipboard);
    downloadPdfBtn.addEventListener('click', simulatePdfDownload);

    // Modal Events
    emailBtn.addEventListener('click', async () => {
        emailModal.classList.remove('hidden');

        // Auto-fill from Resume Screening
        const candidatesUrl = localStorage.getItem('aptitude_candidates_url');
        if (candidatesUrl && !receiverEmailsInput.value.trim()) {
            try {
                receiverEmailsInput.placeholder = "Loading candidates...";
                const resp = await fetch(candidatesUrl);
                if (resp.ok) {
                    currentAptitudeCandidates = await resp.json();
                    const emails = currentAptitudeCandidates.map(c => c.email).filter(e => e).join(', ');
                    receiverEmailsInput.value = emails;
                }
            } catch (e) {
                console.error("Auto-fill failed", e);
                receiverEmailsInput.placeholder = "Failed to load candidates. Please paste manually.";
            }
        }
    });

    closeEmailModal.addEventListener('click', () => emailModal.classList.add('hidden'));
    cancelEmailBtn.addEventListener('click', () => emailModal.classList.add('hidden'));

    confirmSendEmailBtn.addEventListener('click', async () => {
        const emailsString = receiverEmailsInput.value.trim();
        if (!emailsString) {
            alert("Please enter at least one email address.");
            return;
        }

        // Dynamically get Job Title from JD text (looking for 'JOB TITLE: ...')
        const jdValue = jdInput.value;
        const jobTitleMatch = jdValue.match(/JOB TITLE:\s*(.*)/i);
        let jobTitle = "Technical Assessment";

        if (jobTitleMatch && jobTitleMatch[1]) {
            jobTitle = jobTitleMatch[1].trim();
        } else {
            // Fallback to header if JD parsing fails
            const headerTitle = document.querySelector('.main-generator h1')?.textContent.replace('Generated Questions for: ', '').trim();
            if (headerTitle) jobTitle = headerTitle;
        }

        const emailsArray = emailsString.split(',').map(e => e.trim()).filter(e => e);

        // Use current window location to generate link (helps in mobile/network testing)
        const baseUrl = window.location.origin + "/aptitude/test.html";
        const assessmentLink = `${baseUrl}?role=${encodeURIComponent(jobTitle)}&token=${Math.random().toString(36).substr(2, 9)}`;

        // Show loading state on button
        confirmSendEmailBtn.innerHTML = '<span>Sending...</span><i class="ai-spinner-small"></i>';
        confirmSendEmailBtn.disabled = true;

        try {
            const response = await fetch('/aptitude-api/send-assessment', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    candidates: emailsArray.map(email => {
                        const original = currentAptitudeCandidates.find(c => c.email.toLowerCase() === email.toLowerCase());
                        return { 
                            email: email, 
                            name: original ? (original.name || original.candidate_name || "Candidate") : "Candidate",
                            resume_path: original ? original.resume_path : "",
                            ai_analysis: original ? (original.ai_analysis || original.analysis || {}) : {}
                        };
                    }),
                    job_title: jobTitle,
                    mcq_count: selectedMcqs.size,
                    coding_count: selectedCoding.size,
                    assessment_link: assessmentLink,
                    mcqs: Array.from(selectedMcqs),
                    coding_questions: Array.from(selectedCoding),
                    company_name: (() => {
                        const companyDataStr = localStorage.getItem('recruitAI_companyData');
                        if (companyDataStr) {
                            try {
                                return JSON.parse(companyDataStr)['company-name'] || "RecruitAI";
                            } catch(e) { return "RecruitAI"; }
                        }
                        return "RecruitAI";
                    })()
                })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Failed to send emails");
            }

            await showCustomAlert("✅ Success!", `Assessments have been delivered to ${emailsArray.length} candidates successfully.`);
            emailModal.classList.add('hidden');
            receiverEmailsInput.value = '';
        } catch (error) {
            console.error(error);
            const msg = error.message || "";
            if (msg.includes("blocked (SMTP)") || msg.includes("connect Gmail")) {
                showGmailConnectAlert("📧 Mail Sync Required", "To send assessments from Hugging Face, you must connect your Gmail account via OAuth.");
            } else {
                showCustomAlert("❌ Error", "Failed to complete request: " + msg);
            }
        } finally {
            confirmSendEmailBtn.innerHTML = '<span>Send Assessment</span><i data-lucide="send"></i>';
            confirmSendEmailBtn.disabled = false;
            lucide.createIcons();
        }
    });

    // Analysis Events
    viewAnalysisBtn.addEventListener('click', showAnalysisDashboard);

    // Select All Toggle (MCQs)
    selectAllCheckbox.addEventListener('change', () => {
        const isChecked = selectAllCheckbox.checked;
        const items = questionsList.querySelectorAll('.question-item');
        selectedMcqs.clear();
        items.forEach((item, index) => {
            const qObj = allMcqs[index];
            const checkbox = item.querySelector('.q-real-checkbox');
            if (isChecked) {
                item.classList.add('selected');
                checkbox.checked = true;
                selectedMcqs.add(qObj);
            } else {
                item.classList.remove('selected');
                checkbox.checked = false;
            }
        });
        updateCount();
    });

    // Select All Toggle (Coding)
    if (selectAllCoding) {
        selectAllCoding.addEventListener('change', () => {
            const isChecked = selectAllCoding.checked;
            const items = codingQuestionsList.querySelectorAll('.question-item');
            selectedCoding.clear();
            items.forEach((item, index) => {
                const cObj = allCodingQuestions[index];
                const checkbox = item.querySelector('.c-real-checkbox');
                if (isChecked) {
                    item.classList.add('selected');
                    checkbox.checked = true;
                    selectedCoding.add(cObj);
                } else {
                    item.classList.remove('selected');
                    checkbox.checked = false;
                }
            });
            updateCount();
        });
    }

    // Batch Interview Scheduling
    if (selectAllCandidates) {
        selectAllCandidates.addEventListener('change', () => {
            const isChecked = selectAllCandidates.checked;
            const checkboxes = candidatesTbody.querySelectorAll('.cand-checkbox');
            checkboxes.forEach(cb => {
                cb.checked = isChecked;
                const row = cb.closest('tr');
                if (isChecked) row.classList.add('selected');
                else row.classList.remove('selected');
            });
            updateBatchActionState();
        });
    }

    if (scheduleBtn) {
        scheduleBtn.addEventListener('click', () => {
            const selectedCount = candidatesTbody.querySelectorAll('.cand-checkbox:checked').length;
            selectedCandidatesCount.textContent = selectedCount;
            scheduleModal.classList.remove('hidden');
        });
    }

    if (closeSchedule) closeSchedule.addEventListener('click', () => scheduleModal.classList.add('hidden'));
    if (scheduleCancelBtn) scheduleCancelBtn.addEventListener('click', () => scheduleModal.classList.add('hidden'));
    
    if (scheduleSendBtn) {
        scheduleSendBtn.addEventListener('click', async () => {
            const date = interviewDateInput.value;
            const time = interviewTimeInput.value;
            const location = interviewLocationInput.value.trim();
            
            if (!date || !time || !location) {
                alert("Please select Date, Time, and provide a Location for the interview.");
                return;
            }
            
            const selectedEmails = Array.from(candidatesTbody.querySelectorAll('.cand-checkbox:checked'))
                .map(cb => cb.dataset.email);
            
            scheduleSendBtn.disabled = true;
            scheduleSendBtn.innerHTML = '<span>Sending...</span><i class="ai-spinner-small"></i>';
            
            try {
                // Get Company Name from JD Generator persistent data
                const companyDataStr = localStorage.getItem('recruitAI_companyData');
                let companyName = "RecruitAI";
                if (companyDataStr) {
                    const companyData = JSON.parse(companyDataStr);
                    companyName = companyData['company-name'] || "RecruitAI";
                }

                const response = await fetch('/aptitude-api/schedule-interview', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        emails: selectedEmails,
                        job_title: detailJobTitleText.textContent,
                        date: date,
                        time: time,
                        location: location,
                        company_name: companyName
                    })
                });
                
                if (!response.ok) throw new Error("Failed to send invitations");
                
                await showCustomAlert("Success", `Interview invitations sent to ${selectedEmails.length} candidates.`);
                scheduleModal.classList.add('hidden');
            } catch (error) {
                alert("Error scheduling interview: " + error.message);
            } finally {
                scheduleSendBtn.disabled = false;
                scheduleSendBtn.innerHTML = 'Send Invites';
            }
        });
    }
}

function updateBatchActionState() {
    const selectedCount = candidatesTbody.querySelectorAll('.cand-checkbox:checked').length;
    const batchBar = document.getElementById('batch-action-bar');
    const selectionText = batchBar.querySelector('.selection-count');
    
    if (batchBar) {
        if (selectedCount > 0) {
            batchBar.classList.remove('hidden');
            selectionText.textContent = `${selectedCount} candidate${selectedCount > 1 ? 's' : ''} selected`;
            scheduleBtn.disabled = false;
            scheduleBtn.classList.remove('disabled');
        } else {
            batchBar.classList.add('hidden');
            scheduleBtn.disabled = true;
            scheduleBtn.classList.add('disabled');
            if (selectAllCandidates) selectAllCandidates.checked = false;
        }
    }
}

// --- Logic Functions ---

function renderMcqsToSelect() {
    questionsList.innerHTML = '';
    selectedMcqs.clear();
    selectAllCheckbox.checked = false;
    updateCount();

    allMcqs.forEach((qObj, index) => {
        const item = document.createElement('div');
        item.className = 'question-item';

        const questionText = qObj.question;
        const options = qObj.options || [];
        const qId = qObj.id || `Q${index + 1}`;

        let optionsHtml = `
            <div class="options-grid">
                ${options.map((opt, i) => `<div class="option-box"><span>${String.fromCharCode(65 + i)})</span> ${escapeHTML(opt)}</div>`).join('')}
            </div>
        `;

        item.innerHTML = `
            <div class="q-checkbox-wrapper">
                <label class="neon-checkbox">
                    <input type="checkbox" class="q-real-checkbox">
                    ${NEON_INNER_HTML}
                </label>
            </div>
            <div class="q-content">
                <div class="q-id-text">${qId}: ${questionText}</div>
                ${optionsHtml}
            </div>
        `;

        const checkbox = item.querySelector('.q-real-checkbox');
        item.addEventListener('click', (e) => {
            // If clicking the checkbox area itself, let the label/input handle it
            if (e.target.closest('.neon-checkbox')) return;

            checkbox.checked = !checkbox.checked;
            toggleMcqSelection();
        });
        checkbox.addEventListener('change', toggleMcqSelection);

        function toggleMcqSelection() {
            if (checkbox.checked) {
                item.classList.add('selected');
                selectedMcqs.add(qObj);
            } else {
                item.classList.remove('selected');
                selectedMcqs.delete(qObj);
            }
            selectAllCheckbox.checked = (selectedMcqs.size === allMcqs.length);
            updateCount();
        }

        questionsList.appendChild(item);
    });
    lucide.createIcons();
}

function renderCodingToSelect() {
    codingQuestionsList.innerHTML = '';
    selectedCoding.clear();

    allCodingQuestions.forEach((cObj, index) => {
        const item = document.createElement('div');
        item.className = 'question-item';

        item.innerHTML = `
            <div class="q-checkbox-wrapper">
                <label class="neon-checkbox">
                    <input type="checkbox" class="c-real-checkbox">
                    ${NEON_INNER_HTML}
                </label>
            </div>
            <div class="q-content">
                <div class="q-id-text">${cObj.id || "C" + (index + 1)}: ${String(cObj.title || cObj.name || cObj.problem_name || "Untitled Question")}</div>
                <div class="q-desc">${String(cObj.description || cObj.problem || cObj.desc || cObj.problem_statement || "No description provided.")}</div>
                ${(cObj.constraints || cObj.constraint) ? `<div class="q-constraints" style="font-size: 0.8rem; color: #ef4444; font-weight: 700; margin-top: 8px;">Constraints: ${String(cObj.constraints || cObj.constraint)}</div>` : ''}
                <div class="code-example">
                    <span class="example-label">Example Input</span>
                    <pre style="white-space: pre-wrap;">${typeof (cObj.example_input || cObj.input) === 'object' ? JSON.stringify(cObj.example_input || cObj.input, null, 2) : String(cObj.example_input || cObj.input || "N/A")}</pre>
                    <span class="example-label" style="margin-top:10px;">Example Output</span>
                    <pre style="white-space: pre-wrap;">${typeof (cObj.example_output || cObj.output) === 'object' ? JSON.stringify(cObj.example_output || cObj.output, null, 2) : String(cObj.example_output || cObj.output || "N/A")}</pre>
                </div>
            </div>
        `;

        const checkbox = item.querySelector('.c-real-checkbox');
        item.addEventListener('click', (e) => {
            // If clicking the checkbox area itself, let the label/input handle it
            if (e.target.closest('.neon-checkbox')) return;

            checkbox.checked = !checkbox.checked;
            toggleCodingSelection();
        });
        checkbox.addEventListener('change', toggleCodingSelection);

        function toggleCodingSelection() {
            if (checkbox.checked) {
                item.classList.add('selected');
                selectedCoding.add(cObj);
            } else {
                item.classList.remove('selected');
                selectedCoding.delete(cObj);
            }
            // Update Select All checkbox state
            selectAllCoding.checked = (selectedCoding.size === allCodingQuestions.length);
            updateCount();
        }

        codingQuestionsList.appendChild(item);
    });
    lucide.createIcons();
}

function updateCount() {
    selectedCount.textContent = selectedMcqs.size + selectedCoding.size;
}

function renderFinalLists() {
    finalAptitudeList.innerHTML = '';
    finalCodingList.innerHTML = '';

    // Render MCQs
    Array.from(selectedMcqs).forEach((qObj, i) => {
        const container = document.createElement('div');
        container.className = 'final-q-card';
        const optionsHtml = `
            <div class="final-options-grid">
                ${(qObj.options || []).map((opt, i) => `<div class="final-opt"><span>${String.fromCharCode(65 + i)})</span> ${escapeHTML(opt)}</div>`).join('')}
            </div>
        `;
        container.innerHTML = `
            <div class="final-q-text"><strong>Q${i + 1}:</strong> ${qObj.question}</div>
            ${optionsHtml}
            <div class="final-answer">Correct Answer: ${qObj.answer}</div>
        `;
        finalAptitudeList.appendChild(container);
    });

    // Render Coding
    Array.from(selectedCoding).forEach((cObj, i) => {
        const container = document.createElement('div');
        container.className = 'final-q-card';
        const title = String(cObj.title || cObj.name || cObj.problem_name || "Untitled Question");
        const desc = String(cObj.description || cObj.problem || cObj.desc || cObj.problem_statement || "No description provided.");
        const constraints = String(cObj.constraints || cObj.constraint || "");
        const input = typeof (cObj.example_input || cObj.input) === 'object' ? JSON.stringify(cObj.example_input || cObj.input, null, 2) : String(cObj.example_input || cObj.input || "N/A");
        const output = typeof (cObj.example_output || cObj.output) === 'object' ? JSON.stringify(cObj.example_output || cObj.output, null, 2) : String(cObj.example_output || cObj.output || "N/A");

        container.innerHTML = `
            <div class="final-q-text"><strong>C${i + 1}:</strong> ${title}</div>
            <div class="q-desc">${desc}</div>
            ${constraints ? `<div class="q-constraints" style="font-size: 0.85rem; color: #ef4444; font-weight: 700; margin-bottom: 10px;">Constraints: ${constraints}</div>` : ''}
            <div class="code-example">
                <span class="example-label">Example Input</span> <pre>${input}</pre>
                <span class="example-label" style="margin-top:10px;">Example Output</span> <pre>${output}</pre>
            </div>
        `;
        finalCodingList.appendChild(container);
    });
}

function showLoader(show) {
    if (show) loader.classList.remove('hidden');
    else loader.classList.add('hidden');
}

function showSection(section) {
    section.classList.remove('hidden');
}

function copyToClipboard() {
    let text = "--- MCQs ---\n\n";
    Array.from(selectedMcqs).forEach((q, i) => {
        text += `Q${i + 1}: ${q.question}\nOptions: ${q.options.join(', ')}\nAnswer: ${q.answer}\n\n`;
    });

    if (selectedCoding.size > 0) {
        text += "\n--- Coding Questions ---\n\n";
        Array.from(selectedCoding).forEach((c, i) => {
            text += `C${i + 1}: ${c.title}\nDescription: ${c.description}\nExample Input: ${c.example_input}\nExample Output: ${c.example_output}\n\n`;
        });
    }

    navigator.clipboard.writeText(text).then(() => {
        const originalText = copyBtn.innerHTML;
        copyBtn.innerHTML = '<i data-lucide="check"></i> Copied!';
        lucide.createIcons();
        setTimeout(() => {
            copyBtn.innerHTML = originalText;
            lucide.createIcons();
        }, 2000);
    });
}

function simulatePdfDownload() {
    alert("In the full implementation, this will generate a formatted PDF of your selected questions.");
}

// --- Custom Modal Helper ---

function showCustomAlert(title, message) {
    confirmTitle.textContent = title;
    confirmMsg.textContent = message;
    
    // Set Info/Alert icon
    if (confirmIconDynamic) {
        confirmIconDynamic.setAttribute('data-lucide', 'info');
        lucide.createIcons({ root: confirmModal });
    }

    confirmCancelBtn.classList.add('hidden');
    confirmModal.classList.remove('hidden');
    return new Promise((resolve) => {
        const handleOk = () => {
            confirmModal.classList.add('hidden');
            confirmOkBtn.removeEventListener('click', handleOk);
            // Reset icon
            if (confirmIconDynamic) confirmIconDynamic.setAttribute('data-lucide', 'help-circle');
            resolve(true);
        };
        confirmOkBtn.addEventListener('click', handleOk);
    });
}

function showCustomConfirm(title, message) {
    confirmTitle.textContent = title;
    confirmMsg.textContent = message;
    confirmCancelBtn.classList.remove('hidden');
    confirmModal.classList.remove('hidden');
    return new Promise((resolve) => {
        const handleOk = () => {
            confirmModal.classList.add('hidden');
            cleanup();
            resolve(true);
        };
        const handleCancel = () => {
            confirmModal.classList.add('hidden');
            cleanup();
            resolve(false);
        };
        const cleanup = () => {
            confirmOkBtn.removeEventListener('click', handleOk);
            confirmCancelBtn.removeEventListener('click', handleCancel);
            closeConfirmBtn.removeEventListener('click', handleCancel);
        };
        confirmOkBtn.addEventListener('click', handleOk);
        confirmCancelBtn.addEventListener('click', handleCancel);
        closeConfirmBtn.addEventListener('click', handleCancel);
    });
}

window.showCustomAlert = showCustomAlert;
window.showCustomConfirm = showCustomConfirm;

// --- Gmail OAuth Logic (Direct from Popup) ---
const API_BASE = window.location.origin;
const COMPANY_ID = 'default_company';

async function showGmailConnectAlert(title, message) {
    confirmTitle.textContent = title;
    confirmMsg.textContent = message;
    
    // Set Mail Icon
    if (confirmIconDynamic) {
        confirmIconDynamic.setAttribute('data-lucide', 'mail');
        lucide.createIcons({ root: confirmModal });
    }
    
    confirmCancelBtn.classList.remove('hidden'); // Show cancel
    confirmOkBtn.classList.add('hidden');       // Hide normal OK
    confirmGmailBtn.classList.remove('hidden'); // Show Connect button
    confirmModal.classList.remove('hidden');

    return new Promise((resolve) => {
        const handleConnect = () => {
            connectGmail();
            cleanup();
            resolve(true);
        };
        const handleCancel = () => {
            confirmModal.classList.add('hidden');
            cleanup();
            resolve(false);
        };
        const cleanup = () => {
            confirmGmailBtn.removeEventListener('click', handleConnect);
            confirmCancelBtn.removeEventListener('click', handleCancel);
            closeConfirmBtn.removeEventListener('click', handleCancel);
            // Restore buttons and icon for next alert
            confirmOkBtn.classList.remove('hidden');
            confirmGmailBtn.classList.add('hidden');
            if (confirmIconDynamic) {
                confirmIconDynamic.setAttribute('data-lucide', 'help-circle');
            }
        };

        confirmGmailBtn.addEventListener('click', handleConnect);
        confirmCancelBtn.addEventListener('click', handleCancel);
        closeConfirmBtn.addEventListener('click', handleCancel);
    });
}

function connectGmail() {
    confirmModal.classList.add('hidden');
    const width = 600;
    const height = 700;
    const left = (screen.width - width) / 2;
    const top = (screen.height - height) / 2;

    const authUrl = `${API_BASE}/auth/gmail/start?company_id=${COMPANY_ID}`;
    const popup = window.open(authUrl, 'Gmail OAuth', `width=${width},height=${height},left=${left},top=${top}`);

    // Poll for success
    let attempts = 0;
    const poller = setInterval(async () => {
        attempts++;
        const resp = await fetch(`${API_BASE}/auth/gmail/status?company_id=${COMPANY_ID}`);
        const data = await resp.json();
        
        if (data.connected || attempts > 80 || (popup && popup.closed)) {
            clearInterval(poller);
            if (data.connected) {
                showCustomAlert("✅ Connected", "Gmail synced successfully! You can now send assessments.");
            }
        }
    }, 2000);
}

// --- Helper: Format Date as DD/MM/YYYY : HH/MM/SS ---
function formatProctoringDate(timestamp) {
    if (!timestamp) return '-';
    const d = new Date(timestamp * 1000);
    const pad = (n) => n.toString().padStart(2, '0');

    const date = `${pad(d.getDate())}/${pad(d.getMonth() + 1)}/${d.getFullYear()}`;
    const time = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;

    return `${date} : ${time}`;
}

async function showAnalysisDashboard() {
    mainGeneratorCard.classList.add('hidden');
    selectionSection.classList.add('hidden');
    finalResultSection.classList.add('hidden');
    analysisDashboard.classList.remove('hidden');

    const rolesTbody = document.getElementById('job-roles-tbody');
    const emptyState = document.getElementById('empty-state');
    const table = document.querySelector('.modern-table');
    const totalSentStat = document.getElementById('total-tests-stat');
    const completionRateStat = document.getElementById('completion-rate-stat');

    // Reset view
    rolesTbody.innerHTML = '';
    totalSentStat.textContent = '0';
    completionRateStat.textContent = '0%';
    emptyState.classList.add('hidden');
    table.classList.remove('hidden');

    try {
        const response = await fetch('/aptitude-api/get-analytics');
        if (!response.ok) throw new Error("Backend not reachable");

        const db = await response.json();

        if (!db.assessments || db.assessments.length === 0) {
            table.classList.add('hidden');
            emptyState.classList.remove('hidden');
            return;
        }

        // Update Stats
        const totalSent = db.assessments.reduce((acc, curr) => acc + curr.emails.length, 0);
        totalSentStat.textContent = totalSent;

        const totalAttempted = db.submissions.length;
        const rate = totalSent > 0 ? Math.round((totalAttempted / totalSent) * 100) : 0;
        completionRateStat.textContent = rate + '%';

        // Update Header for Date Column
        const thead = document.querySelector('#roles-table-header');
        if (thead && !thead.innerHTML.includes('SENT DATE')) {
            const actionsTh = thead.lastElementChild;
            const dateTh = document.createElement('th');
            dateTh.textContent = 'SENT DATE';
            thead.insertBefore(dateTh, actionsTh);
        }

        // Render Roles Table
        rolesTbody.innerHTML = db.assessments.map(a => {
            const attempted = db.submissions.filter(s => s.token === a.token).length;
            const pending = a.emails.length - attempted;
            const sentDate = formatProctoringDate(a.timestamp);

            const mcqCount = a.mcqs ? a.mcqs.length : (a.questions ? a.questions.length : 0);
            const codeCount = a.coding_questions ? a.coding_questions.length : 0;

            return `
                <tr>
                    <td onclick="viewCandidateDetails('${a.job_title}', '${a.token}')">
                        <div style="font-weight: 700;">${a.job_title}</div>
                        <div style="font-size: 0.7rem; color: #94a3b8;">${mcqCount} MCQ | ${codeCount} Code</div>
                    </td>
                    <td><span class="status-badge status-sent">Sent</span></td>
                    <td>${a.emails.length}</td>
                    <td>${attempted}</td>
                    <td>${new Date(a.timestamp * 1000).toLocaleDateString()}</td>
                    <td class="actions-cell">
                        <button class="glass-btn sm" onclick="viewCandidateDetails('${a.job_title}', '${a.token}')">View Details</button>
                        <button class="delete-btn" onclick="event.stopPropagation(); deleteAssessment('${a.token}')" title="Delete Assessment">
                            <i data-lucide="trash-2"></i>
                        </button>
                    </td>
                </tr>
            `;
        }).join('');
        lucide.createIcons();
    } catch (error) {
        console.error("Failed to fetch analytics:", error);
        table.classList.add('hidden');
        emptyState.querySelector('p').textContent = "Unable to connect to service. Please ensure the backend server is running.";
        emptyState.classList.remove('hidden');
    }
}

window.deleteAssessment = async function (token) {
    const confirmed = await showCustomConfirm("Delete Assessment", "Are you sure you want to delete this assessment and all its submission data? This cannot be undone.");
    if (!confirmed) return;

    try {
        const response = await fetch(`/aptitude-api/delete-assessment/${token}`, { method: 'DELETE' });
        if (response.ok) {
            showAnalysisDashboard(); // Refresh
        } else {
            showCustomAlert("Error", "Failed to delete assessment.");
        }
    } catch (error) {
        showCustomAlert("Error", "Connection error. Could not delete.");
    }
}

window.hideAnalysis = function () {
    analysisDashboard.classList.add('hidden');
    mainGeneratorCard.classList.remove('hidden');
}

const candidateList = document.getElementById('candidate-submissions-list');

window.viewCandidateDetails = async function (jobTitle, token) {
    detailJobTitleText.textContent = jobTitle;
    jobRolesView.classList.add('hidden');
    candidateDetailsView.classList.remove('hidden');
    
    // Reset selection state
    if (selectAllCandidates) selectAllCandidates.checked = false;
    updateBatchActionState();

    candidatesTbody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding:20px;">Loading candidates...</td></tr>';

    try {
        const response = await fetch('/aptitude-api/get-analytics');
        const db = await response.json();

        const assessment = db.assessments.find(a => a.token === token);
        const submissions = db.submissions.filter(s => s.token === token);

        if (!assessment) return;

        // Use 'candidates' if available for full meta, fallback to 'emails'
        const candidateItems = assessment.candidates || assessment.emails.map(e => ({ email: e, name: 'Candidate' }));

        candidatesTbody.innerHTML = candidateItems.map(cand => {
            const email = typeof cand === 'string' ? cand : cand.email;
            const name = typeof cand === 'string' ? 'Candidate' : cand.name;
            const sub = submissions.find(s => s.email === email);

            let status = "Not Started";
            let statusClass = "not-started";

            if (sub) {
                if (sub.suspicious === "Suspicious activity") {
                    status = "Major Violation";
                    statusClass = "suspicious";
                } else if (sub.suspicious !== "Normal") {
                    statusClass = "pending";
                    status = sub.suspicious || "Pending";
                } else {
                    status = "Completed";
                    statusClass = "attempted";
                }
            }

            const mcqScore = sub ? (sub.mcq_score !== undefined ? `${sub.mcq_score}/${sub.mcq_total}` : `${sub.score || 0}/${sub.total || 0}`) : '-';
            const codingScore = sub ? (sub.coding_score !== undefined ? `${sub.coding_score}/${sub.coding_total}` : '-') : '-';

            return `
                <tr data-email="${email}">
                    <td>
                        <label class="neon-checkbox sm">
                            <input type="checkbox" class="cand-checkbox" data-email="${email}">
                            ${NEON_INNER_HTML}
                        </label>
                    </td>
                    <td>
                        <div class="candidate-info">
                            <strong>${name}</strong>
                            <small>${email}</small>
                        </div>
                    </td>
                    <td style="font-weight:700; color:var(--primary);">${mcqScore}</td>
                    <td style="font-weight:700; color:#10b981;">${codingScore}</td>
                    <td><span class="status-badge ${statusClass}">${status}</span></td>
                    <td class="candidate-actions">
                        <button class="action-btn-icon btn-ai" onclick="showAIAnalytics('${email}', '${token}')" title="AI Analytics">
                            <i data-lucide="brain"></i>
                        </button>
                        <button class="action-btn-icon btn-resume" onclick="viewResume('${cand.resume_path || ''}')" title="View Resume" ${!cand.resume_path ? 'disabled style="opacity:0.3;"' : ''}>
                            <i data-lucide="file-text"></i>
                        </button>
                        <button class="action-btn-icon btn-key" onclick="showAnswerKey('${email}', '${token}')" title="Answer Key" ${!sub ? 'disabled style="opacity:0.3;"' : ''}>
                            <i data-lucide="key"></i>
                        </button>
                    </td>
                </tr>
            `;
        }).join('');

        // Re-attach listeners for individual checkboxes
        candidatesTbody.querySelectorAll('.cand-checkbox').forEach(cb => {
            cb.addEventListener('change', () => {
                const row = cb.closest('tr');
                if (cb.checked) row.classList.add('selected');
                else row.classList.remove('selected');
                updateBatchActionState();
            });
        });

    } catch (error) {
        console.error("Failed to fetch candidate details:", error);
        candidatesTbody.innerHTML = '<tr><td colspan="6" style="text-align:center; color:red;">Error loading details.</td></tr>';
    }
    lucide.createIcons();
}

window.showAIAnalytics = async function(email, token) {
    try {
        const response = await fetch('/aptitude-api/get-analytics');
        const db = await response.json();
        const assessment = db.assessments.find(a => a.token === token);
        const cand = assessment.candidates ? assessment.candidates.find(c => c.email === email) : null;
        
        if (!cand || !cand.ai_analysis || Object.keys(cand.ai_analysis).length === 0) {
            await showCustomAlert("⚠️ No Analytics", "The AI resume analysis data is missing for this candidate. This usually happens if they were manually added.");
            return;
        }
        
        const analysis = cand.ai_analysis;
        analyticsContent.innerHTML = `
            <div class="ai-insights-grid">
                <div class="score-card">
                    <div class="score-main">
                        <span class="score-label">Fitment Score</span>
                        <span class="score-value">${analysis.score || analysis.total_score || 'N/A'}%</span>
                    </div>
                </div>
                
                <div class="insight-card full-width">
                    <h5><i data-lucide="info" style="width:16px;"></i> Executive Summary</h5>
                    <p style="font-size: 0.95rem; line-height: 1.5; color: var(--text-muted);">${analysis.summary || analysis.reasoning || "No executive summary available."}</p>
                </div>

                <div class="insight-card">
                    <h5 style="color: var(--success);"><i data-lucide="trending-up" style="width:16px;"></i> Key Strengths</h5>
                    <ul>
                        ${(analysis.strengths || []).length ? analysis.strengths.map(s => `<li>${s}</li>`).join('') : "<li>No specific strengths recorded.</li>"}
                    </ul>
                </div>

                <div class="insight-card">
                    <h5 style="color: var(--danger);"><i data-lucide="alert-triangle" style="width:16px;"></i> Potential Risks</h5>
                    <ul>
                        ${(analysis.weaknesses || analysis.risks || []).length ? (analysis.weaknesses || analysis.risks).map(w => `<li>${w}</li>`).join('') : "<li>No significant risks detected.</li>"}
                    </ul>
                </div>
            </div>
        `;
        analyticsModal.classList.remove('hidden');
        lucide.createIcons();
    } catch (e) {
        console.error(e);
        showCustomAlert("❌ Error", "Failed to load analytics: " + e.message);
    }
}

window.viewResume = function(resumePath) {
    if (!resumePath || resumePath === '-' || resumePath === 'undefined') {
        showCustomAlert("ℹ️ Info", "Resume not available for this candidate. They might have been added manually.");
        return;
    }
    window.open(resumePath, '_blank');
}

window.showAnswerKey = async function(email, token) {
    try {
        const response = await fetch('/aptitude-api/get-analytics');
        const db = await response.json();
        const assessment = db.assessments.find(a => a.token === token);
        const sub = db.submissions.find(s => s.token === token && s.email === email);
        
        if (!sub) {
            await showCustomAlert("🕒 No Submission", "This candidate has not yet submitted their assessment responses.");
            return;
        }

        const mcqs = assessment.mcqs || [];
        const codingQs = assessment.coding_questions || [];

        let html = `
            <div class="answer-key-view">
                <h5 style="margin-bottom: 20px; color: var(--primary); display: flex; align-items: center; gap: 8px; border-bottom: 1px solid #eef2f6; padding-bottom: 15px;">
                    <span style="background: var(--primary); color: white; width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; margin-right: 5px;">
                        <i data-lucide="list-checks" style="width: 18px; height: 18px;"></i>
                    </span>
                    MCQ EVALUATION (${mcqs.length} Questions)
                </h5>
        `;

        if (mcqs.length > 0) {
            mcqs.forEach((q, idx) => {
                const candAns = sub.mcq_answers ? sub.mcq_answers.find(a => a.id === q.id || (a.question === q.question)) : null;
                const isCorrect = candAns && candAns.answer === q.answer;
                const statusLabel = candAns ? (isCorrect ? 'Correct' : 'Incorrect') : 'Not Attempted';
                const statusClass = candAns ? (isCorrect ? 'bg-green' : 'bg-red') : 'bg-red';

                html += `
                    <div class="answer-item ${candAns ? (isCorrect ? 'correct' : 'incorrect') : 'not-attempted'}" style="margin-bottom:15px; border-radius:12px; padding:20px; border-left: 6px solid ${candAns ? (isCorrect ? 'var(--success)' : 'var(--danger)') : '#cbd5e1'}; background: ${candAns ? (isCorrect ? '#f0fdf4' : '#fef2f2') : '#f8fafc'};">
                        <div style="font-weight: 800; color: #1e293b; margin-bottom: 10px; font-size: 1rem;">
                            Q${idx + 1}: ${q.question}
                        </div>
                        <div style="display: flex; align-items: center; gap: 15px; margin-top: 12px; font-size: 0.9rem;">
                            <div style="color: var(--text-muted);">
                                <strong>Candidate Response:</strong> 
                                <span class="badge ${statusClass}" style="margin-left: 5px; padding: 4px 12px;">${candAns ? candAns.answer : 'No Answer'}</span>
                            </div>
                            <div style="color: var(--success); font-weight: 700;">
                                <strong>Correct Answer:</strong> ${q.answer}
                            </div>
                        </div>
                    </div>
                `;
            });
        } else {
            html += `<p class="text-muted" style="padding: 20px; text-align: center;">No MCQ questions found in this assessment.</p>`;
        }

        html += `
            <h5 style="margin-top: 50px; margin-bottom: 20px; color: var(--primary); display: flex; align-items: center; gap: 8px; border-bottom: 1px solid #eef2f6; padding-bottom: 15px;">
                <span style="background: var(--success); color: white; width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; margin-right: 5px;">
                    <i data-lucide="code" style="width: 18px; height: 18px;"></i>
                </span>
                CODING ASSESSMENT (${codingQs.length} Problems)
            </h5>
        `;

        if (codingQs.length > 0) {
            codingQs.forEach((q, idx) => {
                const candCode = sub.coding_answers ? sub.coding_answers.find(a => (a.id === q.id || a.title === q.title)) : null;
                const language = candCode ? candCode.language : "N/A";
                const isSuccess = candCode ? candCode.passed : false;
                
                html += `
                    <div class="code-answer-item" style="margin-bottom: 30px; background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 25px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                            <p style="font-weight: 800; font-size: 1.1rem; color: #1e293b; margin: 0;">${idx + 1}. ${q.title || "Coding Challenge"}</p>
                            <div style="display: flex; gap: 10px;">
                                <span class="badge" style="background: #f1f5f9; color: #64748b; padding: 4px 12px;">${language}</span>
                                <span class="badge ${candCode ? (isSuccess ? 'bg-green' : 'bg-red') : 'bg-red'}" style="padding: 4px 12px;">
                                    ${candCode ? (isSuccess ? 'PASSED' : 'FAILED') : 'NOT ATTEMPTED'}
                                </span>
                            </div>
                        </div>
                        <p style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 20px; font-style: italic;">${q.description ? (q.description.substring(0, 150) + "...") : "No description available."}</p>
                        <div style="position: relative;">
                            <div style="position: absolute; top: 0; right: 0; padding: 5px 12px; background: rgba(255,255,255,0.1); color: #94a3b8; font-size: 0.7rem; border-radius: 0 8px 0 8px;">Candidate's Code</div>
                            <pre class="code-block" style="margin: 0; font-weight: 500; border: none; background: #0f172a; border-radius: 12px; padding: 25px;"><code>${candCode ? escapeHTML(candCode.code) : '// No code submitted'}</code></pre>
                        </div>
                    </div>
                `;
            });
        } else {
            html += `<p class="text-muted" style="padding: 20px; text-align: center;">No coding questions found in this assessment.</p>`;
        }

        html += `</div>`;
        answerKeyContent.innerHTML = html;
        answerKeyModal.classList.remove('hidden');
        lucide.createIcons();
    } catch (e) {
        console.error(e);
        showCustomAlert("❌ Error", "Failed to load answer key: " + e.message);
    }
}

window.backToRoles = function () {
    candidateDetailsView.classList.add('hidden');
    jobRolesView.classList.remove('hidden');
}
