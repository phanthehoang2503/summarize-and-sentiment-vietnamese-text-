// Vietnamese Text Analysis - Frontend JavaScript
class VietnameseTextAnalyzer {
    constructor() {
        this.initializeEventListeners();
        this.history = this.loadHistory();
        this.updateHistoryDisplay();
    }

    initializeEventListeners() {
        // Main action buttons
        document.getElementById('summarizeBtn').addEventListener('click', () => this.performAnalysis('summarize'));
        document.getElementById('sentimentBtn').addEventListener('click', () => this.performAnalysis('sentiment'));
        document.getElementById('analyzeBtn').addEventListener('click', () => this.performAnalysis('analyze'));
        document.getElementById('clearBtn').addEventListener('click', () => this.clearAll());
        
        // File upload
        const fileInput = document.getElementById('fileInput');
        const fileUploadArea = document.getElementById('fileUploadArea');
        
        fileUploadArea.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => this.handleFileUpload(e.target.files[0]));
        
        // Drag and drop
        fileUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileUploadArea.classList.add('dragover');
        });
        
        fileUploadArea.addEventListener('dragleave', () => {
            fileUploadArea.classList.remove('dragover');
        });
        
        fileUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            fileUploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file) this.handleFileUpload(file);
        });
    }

    async handleFileUpload(file) {
        if (!file) return;

        const allowedTypes = ['text/plain', 'text/csv', 'application/csv'];
        if (!allowedTypes.includes(file.type) && !file.name.match(/\.(txt|csv)$/i)) {
            this.showError('Vui lòng thả tệp .txt vào đây');
            return;
        }

        this.showLoading(true);

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                document.getElementById('textInput').value = result.text;
                this.showSuccess(`Tệp đã được tải lên thành công! Đã trích xuất ${result.text.length} ký tự.`);
            } else {
                throw new Error(result.error || 'Tải lên thất bại');
            }
        } catch (error) {
            console.error('File upload error:', error);
            this.showError(`Tải tệp thất bại: ${error.message}`);
        } finally {
            this.showLoading(false);
        }
    }

    async performAnalysis(type) {
        const text = document.getElementById('textInput').value.trim();
        
        if (!text) {
            this.showError('Vui lòng nhập văn bản để phân tích');
            return;
        }

        this.showLoading(true, type);
        this.hideResults();

        try {
            let endpoint;
            let requestBody = { text: text };
            
            switch (type) {
                case 'summarize':
                    endpoint = '/api/summarize';
                    break;
                case 'sentiment':
                    endpoint = '/api/sentiment';
                    break;
                case 'analyze':
                    endpoint = '/api/analyze';
                    break;
                default:
                    throw new Error('Invalid analysis type');
            }

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                throw new Error(`Request failed: ${response.statusText}`);
            }

            const result = await response.json();
            
            if (result.success) {
                this.displayResults(result, type);
                this.addToHistory(text, result, type);
            } else {
                throw new Error(result.error || 'Phân tích thất bại');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError(`Phân tích thất bại: ${error.message}`);
        } finally {
            this.showLoading(false, type);
        }
    }

    showLoading(show, type = null) {
        const buttons = document.querySelectorAll('#summarizeBtn, #sentimentBtn, #analyzeBtn');
        
        buttons.forEach(btn => {
            btn.disabled = show;
            if (show) {
                const originalIcon = btn.querySelector('i').className;
                btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Đang xử lý...';
                btn.setAttribute('data-original-icon', originalIcon);
            } else {
                // Reset button text based on button type
                const originalIcon = btn.getAttribute('data-original-icon');
                const labels = {
                    'summarizeBtn': 'Tóm Tắt',
                    'sentimentBtn': 'Phân Tích Cảm Xúc',
                    'analyzeBtn': 'Cả Hai (Tóm Tắt + Cảm Xúc)'
                };
                btn.innerHTML = `<i class="${originalIcon} me-2"></i>${labels[btn.id]}`;
            }
        });

        // Show specific loading indicators for output sections
        if (show && type) {
            if (type === 'summarize' || type === 'analyze') {
                const summaryLoading = document.getElementById('summaryLoading');
                if (summaryLoading) summaryLoading.style.display = 'flex';
            }
            if (type === 'sentiment' || type === 'analyze') {
                const sentimentLoading = document.getElementById('sentimentLoading');
                if (sentimentLoading) sentimentLoading.style.display = 'flex';
            }
        } else {
            // Hide all loading indicators
            const loadingIndicators = document.querySelectorAll('#summaryLoading, #sentimentLoading');
            loadingIndicators.forEach(indicator => {
                if (indicator) indicator.style.display = 'none';
            });
        }
    }

    displayResults(result, type) {
        const resultsSection = document.getElementById('resultsSection');
        const summaryResults = document.getElementById('summaryResults');
        const sentimentResults = document.getElementById('sentimentResults');

        // Show results section
        resultsSection.style.display = 'block';

        // Extract the actual result data
        const data = result.result || result;

        // Display summary if available
        if (data.summary && data.summary.trim() !== '') {
            document.getElementById('summaryText').textContent = data.summary;
            summaryResults.style.display = 'block';
        } else if (type === 'analyze' && data.summary === null) {
            // For combine button when summary is null (short text), show explanation
            document.getElementById('summaryText').textContent = 'Văn bản quá ngắn để tóm tắt (dưới 50 ký tự). Chỉ phân tích cảm xúc.';
            summaryResults.style.display = 'block';
        } else {
            summaryResults.style.display = 'none';
        }

        // Display sentiment if available - handle both direct sentiment and nested sentiment
        let sentimentData = null;
        
        // For analyze endpoint - sentiment is nested under 'sentiment' key
        if (data.sentiment && typeof data.sentiment === 'object') {
            sentimentData = {
                label: data.sentiment.predicted_label || data.sentiment.label,
                confidence: data.sentiment.confidence || 0,
                probabilities: data.sentiment.probabilities || {}
            };
        } 
        // For direct sentiment endpoint - data is at root level
        else if (data.predicted_label || data.label) {
            sentimentData = {
                label: data.predicted_label || data.label,
                confidence: data.confidence || 0,
                probabilities: data.probabilities || {}
            };
        }
        
        if (sentimentData && sentimentData.label && sentimentData.label !== 'UNKNOWN') {
            // Pass additional context for sentiment source
            const sentimentContext = {
                ...sentimentData,
                usedSummarization: data.used_summarization || false,
                analysisMethod: data.analysis_method || 'direct sentiment only'
            };
            this.displaySentimentResults(sentimentContext);
            sentimentResults.style.display = 'block';
        } else {
            sentimentResults.style.display = 'none';
        }

        // Display statistics with proper data mapping
        this.displayStatistics(data);

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    displaySentimentResults(sentiment) {
        const label = document.getElementById('sentimentLabel');
        const confidenceBar = document.getElementById('confidenceBar');
        const confidenceText = document.getElementById('confidenceText');
        const probabilitiesSection = document.getElementById('probabilitiesSection');
        const sentimentSource = document.getElementById('sentimentSource');
        const sentimentSourceText = document.getElementById('sentimentSourceText');

        // Set sentiment label with appropriate color and styling
        const labelText = sentiment.label || 'Unknown';
        label.textContent = labelText;
        
        // Apply color classes based on sentiment
        label.className = 'sentiment-label';
        if (labelText.toLowerCase().includes('pos') || labelText.toLowerCase().includes('tích cực')) {
            label.classList.add('sentiment-positive');
            confidenceBar.className = 'confidence-fill bg-success';
        } else if (labelText.toLowerCase().includes('neg') || labelText.toLowerCase().includes('tiêu cực')) {
            label.classList.add('sentiment-negative');
            confidenceBar.className = 'confidence-fill bg-danger';
        } else {
            label.classList.add('sentiment-neutral');
            confidenceBar.className = 'confidence-fill bg-warning';
        }

        // Show sentiment source information
        if (sentiment.usedSummarization) {
            sentimentSourceText.textContent = 'văn bản đã tóm tắt';
            sentimentSource.style.display = 'block';
        } else if (sentiment.analysisMethod && sentiment.analysisMethod.includes('direct')) {
            sentimentSourceText.textContent = 'văn bản gốc';
            sentimentSource.style.display = 'block';
        } else {
            sentimentSource.style.display = 'none';
        }

        // Set confidence bar
        const confidence = (sentiment.confidence || 0) * 100;
        confidenceBar.style.width = `${confidence}%`;
        confidenceText.textContent = `${confidence.toFixed(1)}%`;

        // Display probabilities
        if (sentiment.probabilities && Object.keys(sentiment.probabilities).length > 0) {
            let probabilitiesHtml = '<h6 class="small mb-3"><i class="fas fa-chart-pie me-1"></i>Phân Bố Xác Suất:</h6>';
            probabilitiesHtml += '<div class="row">';
            
            for (const [label, prob] of Object.entries(sentiment.probabilities)) {
                const percentage = (prob * 100).toFixed(1);
                let colorClass = 'text-secondary';
                if (label.toLowerCase().includes('pos')) colorClass = 'text-success';
                else if (label.toLowerCase().includes('neg')) colorClass = 'text-danger';
                else if (label.toLowerCase().includes('neu')) colorClass = 'text-warning';
                
                probabilitiesHtml += `
                    <div class="col-6 mb-2">
                        <div class="d-flex justify-content-between align-items-center">
                            <span class="small ${colorClass} fw-bold">${label}:</span>
                            <span class="small fw-bold">${percentage}%</span>
                        </div>
                        <div class="progress" style="height: 4px;">
                            <div class="progress-bar ${colorClass.replace('text-', 'bg-')}" style="width: ${percentage}%"></div>
                        </div>
                    </div>
                `;
            }
            probabilitiesHtml += '</div>';
            probabilitiesSection.innerHTML = probabilitiesHtml;
        } else {
            probabilitiesSection.innerHTML = '';
        }
    }

    displayStatistics(data) {
        const statisticsSection = document.getElementById('statisticsSection');
        
        // Map all possible statistics from different API responses
        const stats = [
            {
                label: 'Thời gian xử lý',
                value: data.processing_time ? `${data.processing_time}s` : 'N/A',
                icon: 'fas fa-clock'
            },
            {
                label: 'Tokens văn bản gốc',
                value: data.original_tokens || data.tokens || (data.sentiment && data.sentiment.tokens) || 'N/A',
                icon: 'fas fa-file-text'
            },
            {
                label: 'Tokens tóm tắt',
                value: data.summary_tokens || (data.summary ? data.summary.split(' ').length : 'N/A'),
                icon: 'fas fa-compress-alt'
            }
        ];

        let statsHtml = '';
        stats.forEach(stat => {
            if (stat.value !== 'N/A') {
                statsHtml += `
                    <div class="stat-item">
                        <div class="stat-icon">
                            <i class="${stat.icon}"></i>
                        </div>
                        <div class="stat-content">
                            <div class="stat-label">${stat.label}</div>
                            <div class="stat-value">${stat.value}</div>
                        </div>
                    </div>
                `;
            }
        });

        statisticsSection.innerHTML = statsHtml;
    }

    hideResults() {
        const resultsSection = document.getElementById('resultsSection');
        const summaryResults = document.getElementById('summaryResults');
        const sentimentResults = document.getElementById('sentimentResults');
        
        resultsSection.style.display = 'none';
        summaryResults.style.display = 'none';
        sentimentResults.style.display = 'none';
    }

    clearAll() {
        document.getElementById('textInput').value = '';
        this.hideResults();
        document.getElementById('statisticsSection').innerHTML = '';
    }

    showError(message) {
        // Create or show error message
        let errorDiv = document.getElementById('errorMessage');
        if (!errorDiv) {
            errorDiv = document.createElement('div');
            errorDiv.id = 'errorMessage';
            errorDiv.className = 'alert alert-danger alert-dismissible fade show mt-3';
            errorDiv.innerHTML = `
                <i class="fas fa-exclamation-triangle me-2"></i>
                <span id="errorText"></span>
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            document.querySelector('.input-section .card-body').appendChild(errorDiv);
        }
        
        document.getElementById('errorText').textContent = message;
        errorDiv.style.display = 'block';
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            if (errorDiv) {
                errorDiv.style.display = 'none';
            }
        }, 5000);
    }

    showSuccess(message) {
        // Create or show success message
        let successDiv = document.getElementById('successMessage');
        if (!successDiv) {
            successDiv = document.createElement('div');
            successDiv.id = 'successMessage';
            successDiv.className = 'alert alert-success alert-dismissible fade show mt-3';
            successDiv.innerHTML = `
                <i class="fas fa-check-circle me-2"></i>
                <span id="successText"></span>
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            document.querySelector('.input-section .card-body').appendChild(successDiv);
        }
        
        document.getElementById('successText').textContent = message;
        successDiv.style.display = 'block';
        
        // Auto-hide after 3 seconds
        setTimeout(() => {
            if (successDiv) {
                successDiv.style.display = 'none';
            }
        }, 3000);
    }

    // History management
    loadHistory() {
        try {
            return JSON.parse(localStorage.getItem('vietnameseTextAnalysisHistory') || '[]');
        } catch {
            return [];
        }
    }

    saveHistory() {
        localStorage.setItem('vietnameseTextAnalysisHistory', JSON.stringify(this.history));
    }

    addToHistory(text, result, type) {
        const entry = {
            id: Date.now(),
            timestamp: new Date().toLocaleString('vi-VN'),
            text: text.substring(0, 100) + (text.length > 100 ? '...' : ''),
            type: type,
            result: result,
            date: new Date().toISOString()
        };
        
        this.history.unshift(entry);
        if (this.history.length > 10) {
            this.history = this.history.slice(0, 10);
        }
        
        this.saveHistory();
        this.updateHistoryDisplay();
    }

    updateHistoryDisplay() {
        const historyPanel = document.getElementById('historyPanel');
        
        if (this.history.length === 0) {
            historyPanel.innerHTML = '<p class="text-muted small text-center">Chưa có phân tích nào được thực hiện</p>';
            return;
        }

        let historyHtml = '';
        this.history.forEach(entry => {
            const typeLabels = {
                'summarize': 'Tóm tắt',
                'sentiment': 'Cảm xúc',
                'analyze': 'Cả hai'
            };
            
            const typeIcons = {
                'summarize': 'fas fa-compress-alt',
                'sentiment': 'fas fa-smile',
                'analyze': 'fas fa-cogs'
            };

            historyHtml += `
                <div class="history-item mb-2 p-2 border rounded">
                    <div class="d-flex justify-content-between align-items-start">
                        <div class="flex-grow-1">
                            <small class="text-muted">${entry.timestamp}</small>
                            <div class="small">
                                <i class="${typeIcons[entry.type]} me-1"></i>
                                <strong>${typeLabels[entry.type]}</strong>
                            </div>
                            <div class="small text-muted mt-1" style="max-height: 40px; overflow: hidden;">
                                ${entry.text}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });

        historyPanel.innerHTML = historyHtml;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new VietnameseTextAnalyzer();
});