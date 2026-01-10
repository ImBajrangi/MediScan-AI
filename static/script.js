// Tab Switching
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        tabBtns.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));

        btn.classList.add('active');
        const tabId = btn.getAttribute('data-tab');
        document.getElementById(tabId + 'Section').classList.add('active');

        // Reset results on tab change
        resultsSection.style.display = 'none';
    });
});

// Image Upload Logic
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const analyzeBtn = document.getElementById('analyzeBtn');
const cancelUpload = document.getElementById('cancelUpload');

// Results Logic
const resultsSection = document.getElementById('resultsSection');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const confidenceValue = document.getElementById('confidenceValue');
const progressFill = document.getElementById('progressFill');
const conditionName = document.getElementById('conditionName');
const conditionBadge = document.getElementById('conditionBadge');
const modelUsedLabel = document.getElementById('modelUsed');
const warningBox = document.getElementById('warningBox');
const resultTimestamp = document.getElementById('resultTimestamp');
const downloadPDFBtn = document.getElementById('downloadPDFBtn');

// Symptom Logic
const symptomInput = document.getElementById('symptomInput');
const symptomSearchBtn = document.getElementById('symptomSearchBtn');
const symptomTags = document.getElementById('symptomTags');
let currentSymptoms = [];

// Drag and drop for image
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--primary)';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = 'var(--glass-border)';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleFile(file);
});

uploadArea.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
});

function handleFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        previewSection.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

cancelUpload.addEventListener('click', () => {
    fileInput.value = '';
    uploadArea.style.display = 'block';
    previewSection.style.display = 'none';
});

// Symptom Input Handling
symptomInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        const val = symptomInput.value.trim();
        if (val) {
            const syms = val.split(',').map(s => s.trim()).filter(s => s);
            syms.forEach(addSymptom);
            symptomInput.value = '';
        }
    }
});

function addSymptom(s) {
    if (!currentSymptoms.includes(s)) {
        currentSymptoms.push(s);
        const tag = document.createElement('div');
        tag.className = 'mini-tag';
        tag.textContent = s + ' ';

        const closeBtn = document.createElement('span');
        closeBtn.textContent = '×';
        closeBtn.style.cursor = 'pointer';
        closeBtn.onclick = () => removeSymptom(s, tag);

        tag.appendChild(closeBtn);
        symptomTags.appendChild(tag);
    }
}

function removeSymptom(s, tagElement) {
    currentSymptoms = currentSymptoms.filter(item => item !== s);
    tagElement.remove();
}

// Analysis Calls
symptomSearchBtn.addEventListener('click', async () => {
    const val = symptomInput.value.trim();
    if (val) {
        val.split(',').map(s => s.trim()).filter(s => s).forEach(addSymptom);
        symptomInput.value = '';
    }

    if (currentSymptoms.length === 0) return alert('Please enter at least one symptom');

    showLoading(symptomSearchBtn);
    try {
        const res = await fetch('/predict_symptoms', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symptoms: currentSymptoms })
        });
        const data = await res.json();
        if (res.ok) showResults(data, 'Symptom-based ML (Random Forest)');
        else alert(data.error);
    } catch (e) { alert(e.message); }
    finally { hideLoading(symptomSearchBtn, 'Analyze Symptoms'); }
});

analyzeBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) return;

    showLoading(analyzeBtn);
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (res.ok) showResults(data, 'Pictorial AI (DermaMNIST CNN)');
        else alert(data.error);
    } catch (e) { alert(e.message); }
    finally { hideLoading(analyzeBtn, 'Start Visual Analysis'); }
});

function showLoading(btn) {
    const span = btn.querySelector('span') || btn;
    if (btn.querySelector('.loader')) {
        btn.querySelector('.loader').style.display = 'block';
        span.style.display = 'none';
    }
    btn.disabled = true;
    resultsSection.style.display = 'none';
}

function hideLoading(btn, originalText) {
    const span = btn.querySelector('span') || btn;
    if (btn.querySelector('.loader')) {
        btn.querySelector('.loader').style.display = 'none';
        span.style.display = 'inline';
    }
    btn.disabled = false;
}

function showResults(data, modelInfo) {
    resultsSection.style.display = 'block';

    // Set Timestamp
    const now = new Date();
    resultTimestamp.textContent = now.toLocaleString();

    // Generic Result Mapping (handles both API types)
    const name = data.disease || data.condition;
    const isSerious = data.is_serious || false;

    conditionName.textContent = name;
    conditionBadge.textContent = isSerious ? 'Attention Required' : 'Diagnostic Lead';
    conditionBadge.className = 'condition-badge ' + (isSerious ? 'serious' : 'benign');

    confidenceValue.textContent = data.confidence + '%';
    progressFill.style.width = data.confidence + '%';
    modelUsedLabel.textContent = modelInfo;

    // Warnings
    warningBox.style.display = isSerious ? 'flex' : 'none';

    // Multi-modal content
    const mBox = document.getElementById('matchedSymptomsBox');
    const mList = document.getElementById('matchedSymptomsList');
    if (data.matched) {
        mBox.style.display = 'block';
        mList.innerHTML = data.matched.map(s => `<span class="mini-tag">${s.replace(/_/g, ' ')}</span>`).join('');
    } else {
        mBox.style.display = 'none';
    }

    const pBox = document.getElementById('precautionBox');
    const pList = document.getElementById('precautionList');
    if (data.precautions && data.precautions.length > 0) {
        pBox.style.display = 'block';
        pList.innerHTML = data.precautions.map(p => `<li>${p}</li>`).join('');
    } else {
        pBox.style.display = 'none';
    }

    if (resultsSection) {
        resultsSection.style.display = 'block';
        populatePDFTemplate(data, modelInfo); // Populate the hidden template
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }
}

function populatePDFTemplate(data, modelInfo) {
    const name = data.disease || data.condition;
    document.getElementById('pdfCondition').textContent = name;
    document.getElementById('pdfConfidence').textContent = data.confidence + '%';
    document.getElementById('pdfModel').textContent = modelInfo;
    document.getElementById('pdfDate').textContent = new Date().toLocaleString();

    const symSection = document.getElementById('pdfSymptomsSection');
    const symList = document.getElementById('pdfSymptoms');
    if (data.matched) {
        symSection.style.display = 'block';
        symList.textContent = "Reported indicators: " + data.matched.map(s => s.replace(/_/g, ' ')).join(', ');
    } else {
        symSection.style.display = 'none';
    }

    const pList = document.getElementById('pdfPrecautions');
    if (data.precautions && data.precautions.length > 0) {
        pList.innerHTML = data.precautions.map(p => `<li>${p}</li>`).join('');
    } else {
        pList.innerHTML = '<li>Maintained clinical observation advised.</li>';
    }

    document.getElementById('pdfWarning').style.display = data.is_serious ? 'block' : 'none';
}

async function generatePDF() {
    const condition = document.getElementById('pdfCondition').textContent || 'Unknown Condition';
    const confidence = document.getElementById('pdfConfidence').textContent || 'N/A';
    const model = document.getElementById('pdfModel').textContent || 'N/A';
    const symptoms = document.getElementById('pdfSymptoms').textContent || '';
    const precautionItems = document.getElementById('pdfPrecautions').querySelectorAll('li');
    const isWarning = document.getElementById('pdfWarning').style.display !== 'none';

    // Get the uploaded image if available
    const previewImg = document.getElementById('previewImage');
    const hasImage = previewImg && previewImg.src && previewImg.src.startsWith('data:');

    // Use jsPDF directly (bundled with html2pdf)
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF('p', 'mm', 'a4');

    const pageWidth = doc.internal.pageSize.getWidth();
    let y = 20;

    // Header bar
    doc.setFillColor(99, 102, 241);
    doc.rect(0, 0, pageWidth, 35, 'F');

    doc.setTextColor(255, 255, 255);
    doc.setFontSize(24);
    doc.setFont('helvetica', 'bold');
    doc.text('MediScan AI', 15, 18);

    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.text('Clinical Intelligence Assessment Report', 15, 26);

    doc.setFontSize(9);
    doc.text(new Date().toLocaleString(), pageWidth - 15, 18, { align: 'right' });
    doc.text('Ref: MS-AI-2026', pageWidth - 15, 26, { align: 'right' });

    y = 50;

    // Section 1: Diagnosis
    doc.setTextColor(15, 23, 42);
    doc.setFontSize(14);
    doc.setFont('helvetica', 'bold');
    doc.text('1. Diagnostic Result', 15, y);

    y += 10;
    doc.setFillColor(241, 245, 249);
    doc.roundedRect(15, y, pageWidth - 30, 40, 3, 3, 'F');

    y += 10;
    doc.setFontSize(9);
    doc.setTextColor(100, 116, 139);
    doc.setFont('helvetica', 'normal');
    doc.text('CLINICAL LEAD POSITION:', 20, y);

    y += 8;
    doc.setTextColor(30, 27, 75);
    doc.setFontSize(18);
    doc.setFont('helvetica', 'bold');
    doc.text(condition, 20, y);

    y += 12;
    doc.setFontSize(9);
    doc.setTextColor(100, 116, 139);
    doc.setFont('helvetica', 'normal');
    doc.text('AI Confidence:', 20, y);
    doc.text('Methodology:', 90, y);

    y += 6;
    doc.setFontSize(14);
    doc.setTextColor(99, 102, 241);
    doc.setFont('helvetica', 'bold');
    doc.text(confidence, 20, y);
    doc.setTextColor(168, 85, 247);
    doc.text(model, 90, y);

    y += 20;

    // Uploaded Image Section (for pictorial diagnosis)
    if (hasImage) {
        doc.setTextColor(15, 23, 42);
        doc.setFontSize(14);
        doc.setFont('helvetica', 'bold');
        doc.text('Analyzed Image', 15, y);

        y += 5;

        // Add border around image
        doc.setDrawColor(226, 232, 240);
        doc.setLineWidth(0.5);
        doc.roundedRect(15, y, 50, 50, 2, 2, 'S');

        // Add the image
        try {
            doc.addImage(previewImg.src, 'JPEG', 16, y + 1, 48, 48);
        } catch (e) {
            console.log('Could not add image:', e);
        }

        // Caption
        doc.setFontSize(8);
        doc.setTextColor(100, 116, 139);
        doc.setFont('helvetica', 'normal');
        doc.text('Submitted for AI Analysis', 15, y + 56);

        y += 65;
    }

    // Section 2: Symptoms
    if (symptoms && symptoms.trim()) {
        doc.setTextColor(15, 23, 42);
        doc.setFontSize(14);
        doc.setFont('helvetica', 'bold');
        doc.text('2. Observed Indicators', 15, y);

        y += 8;
        doc.setFontSize(10);
        doc.setTextColor(51, 65, 85);
        doc.setFont('helvetica', 'normal');
        const symptomLines = doc.splitTextToSize(symptoms, pageWidth - 40);
        doc.text(symptomLines, 20, y);
        y += symptomLines.length * 5 + 10;
    }

    // Section 3: Precautions
    doc.setTextColor(15, 23, 42);
    doc.setFontSize(14);
    doc.setFont('helvetica', 'bold');
    doc.text('3. Recommended Precautions', 15, y);

    y += 8;
    doc.setFontSize(10);
    doc.setTextColor(51, 65, 85);
    doc.setFont('helvetica', 'normal');

    precautionItems.forEach((item) => {
        doc.text('• ' + item.textContent, 20, y);
        y += 6;
    });

    if (precautionItems.length === 0) {
        doc.text('• Maintain regular clinical observation.', 20, y);
        y += 6;
    }

    y += 10;

    // Warning box
    if (isWarning) {
        doc.setFillColor(254, 226, 226);
        doc.setDrawColor(239, 68, 68);
        doc.roundedRect(15, y, pageWidth - 30, 25, 3, 3, 'FD');

        y += 10;
        doc.setTextColor(153, 27, 27);
        doc.setFontSize(12);
        doc.setFont('helvetica', 'bold');
        doc.text('! URGENT MEDICAL EVALUATION ADVISED', 20, y);

        y += 8;
        doc.setFontSize(9);
        doc.setFont('helvetica', 'normal');
        doc.text('This condition requires professional dermatological consultation immediately.', 20, y);
        y += 15;
    }

    // Footer
    y = doc.internal.pageSize.getHeight() - 30;
    doc.setDrawColor(99, 102, 241);
    doc.line(15, y, pageWidth - 15, y);

    y += 5;
    doc.setFontSize(7);
    doc.setTextColor(100, 116, 139);
    doc.setFont('helvetica', 'normal');
    const disclaimer = 'Medical Disclaimer: This report is generated by AI for informational purposes only. It does not constitute medical advice. Always consult a qualified healthcare professional.';
    const disclaimerLines = doc.splitTextToSize(disclaimer, pageWidth - 30);
    doc.text(disclaimerLines, 15, y);

    // Save
    doc.save('MediScan_Report_' + condition.replace(/\s+/g, '_') + '.pdf');
}

downloadPDFBtn.addEventListener('click', generatePDF);

newAnalysisBtn.addEventListener('click', () => {
    resultsSection.style.display = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
    // If image tab, reset image. If symptom tab, reset symptoms.
    if (document.getElementById('imagesSection').classList.contains('active')) {
        cancelUpload.click();
    } else {
        currentSymptoms = [];
        symptomTags.innerHTML = '';
        symptomInput.value = '';
    }
});
