document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const clinicalForm = document.getElementById('clinicalForm');
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const previewContainer = document.querySelector('.preview-container');
    const uploadPlaceholder = document.querySelector('.upload-placeholder');
    const removeBtn = document.querySelector('.remove-btn');
    const btnVision = document.getElementById('btnVision');
    const historyList = document.getElementById('historyList');

    // Local Cache Key
    const CACHE_KEY = 'mediscan_ckd_history';

    // Initialize History
    updateHistoryUI();

    // --- Clinical Assessment ---
    clinicalForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = document.getElementById('btnClinical');
        const loader = btn.querySelector('.loader');
        const btnText = btn.querySelector('.btn-text');
        
        const data = {
            age: parseInt(document.getElementById('age').value),
            gfr: parseFloat(document.getElementById('gfr').value),
            albuminuria: parseFloat(document.getElementById('albuminuria').value)
        };

        // Loading state
        btn.classList.add('disabled');
        loader.style.display = 'block';
        btnText.style.opacity = '0';

        try {
            const response = await fetch('/predict_ckd', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            
            if (result.error) throw new Error(result.error);

            // Update UI
            document.getElementById('clinicalResult').classList.remove('hidden');
            document.getElementById('clinicalStatus').textContent = result.stage;
            document.getElementById('clinicalConf').textContent = `${result.confidence}% Confidence`;
            document.getElementById('clinicalDesc').textContent = result.description;

            // Cache result
            saveToHistory({
                id: Date.now(),
                type: 'Clinical',
                result: result.stage,
                metrics: `GFR: ${data.gfr}, Alb: ${data.albuminuria}`,
                timestamp: new Date().toLocaleTimeString()
            });

        } catch (error) {
            console.error('Error:', error);
            alert('Prediction failed: ' + error.message);
        } finally {
            btn.classList.remove('disabled');
            loader.style.display = 'none';
            btnText.style.opacity = '1';
        }
    });

    // --- Imaging Analysis ---
    
    // Trigger file input
    dropZone.addEventListener('click', () => fileInput.click());

    // File handling
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleFile(file);
    });

    // Drag & Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--emerald)';
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = 'var(--border)';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--border)';
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.classList.remove('hidden');
            uploadPlaceholder.classList.add('hidden');
            btnVision.classList.remove('disabled');
        };
        reader.readAsDataURL(file);
    }

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.value = '';
        previewContainer.classList.add('hidden');
        uploadPlaceholder.classList.remove('hidden');
        btnVision.classList.add('disabled');
        document.getElementById('visionResult').classList.add('hidden');
    });

    btnVision.addEventListener('click', async () => {
        if (btnVision.classList.contains('disabled')) return;

        const loader = btnVision.querySelector('.loader');
        const btnText = btnVision.querySelector('.btn-text');
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        // Loading state
        btnVision.classList.add('disabled');
        loader.style.display = 'block';
        btnText.style.opacity = '0';

        try {
            const response = await fetch('/predict_ckd_vision', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (result.error) throw new Error(result.error);

            // Update UI
            document.getElementById('visionResult').classList.remove('hidden');
            document.getElementById('visionStatus').textContent = result.status;
            document.getElementById('visionConf').textContent = `${result.confidence}% Confidence`;

            // Cache result
            saveToHistory({
                id: Date.now(),
                type: 'Imaging',
                result: result.status,
                metrics: 'Neural Ultrasound Scan',
                timestamp: new Date().toLocaleTimeString()
            });

        } catch (error) {
            console.error('Error:', error);
            alert('Vision analysis failed: ' + error.message);
        } finally {
            btnVision.classList.remove('disabled');
            loader.style.display = 'none';
            btnText.style.opacity = '1';
        }
    });

    // --- Caching System ---
    function saveToHistory(item) {
        let history = JSON.parse(localStorage.getItem(CACHE_KEY) || '[]');
        history.unshift(item); // Add to beginning
        history = history.slice(0, 5); // Keep last 5
        localStorage.setItem(CACHE_KEY, JSON.stringify(history));
        updateHistoryUI();
    }

    function updateHistoryUI() {
        const history = JSON.parse(localStorage.getItem(CACHE_KEY) || '[]');
        
        if (history.length === 0) {
            historyList.innerHTML = '<p class="empty-state">No recent records found.</p>';
            return;
        }

        historyList.innerHTML = history.map(item => `
            <div class="history-item">
                <div class="hist-left">
                    <span class="hist-label">${item.type} Analysis</span>
                    <span class="hist-val">${item.result}</span>
                </div>
                <div class="hist-right" style="text-align: right;">
                    <span class="hist-label">${item.timestamp}</span>
                    <span class="hist-val" style="font-size: 0.8rem; color: var(--text-secondary); display: block;">${item.metrics}</span>
                </div>
            </div>
        `).join('');
    }
});
