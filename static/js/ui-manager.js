/**
 * UI Manager Module  
 * Handles all UI updates and visual feedback
 */
class UIManager {
    constructor() {
        this.elements = this.initializeElements();
    }

    initializeElements() {
        return {
            resultsSection: document.getElementById('resultsSection'),
            summaryResults: document.getElementById('summaryResults'),
            sentimentResults: document.getElementById('sentimentResults'),
            summaryText: document.getElementById('summaryText'),
            sentimentLabel: document.getElementById('sentimentLabel'),
            confidenceBar: document.getElementById('confidenceBar'),
            confidenceText: document.getElementById('confidenceText'),
            probabilitiesSection: document.getElementById('probabilitiesSection'),
            sentimentSource: document.getElementById('sentimentSource'),
            sentimentSourceText: document.getElementById('sentimentSourceText'),
            statisticsSection: document.getElementById('statisticsSection'),
            textInput: document.getElementById('textInput')
        };
    }

    /**
     * Show/hide loading state
     */
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
                    'summarizeBtn': '<i class="fas fa-compress-alt me-2"></i>Tóm tắt',
                    'sentimentBtn': '<i class="fas fa-smile me-2"></i>Cảm xúc', 
                    'analyzeBtn': '<i class="fas fa-cogs me-2"></i>Cả hai'
                };
                btn.innerHTML = labels[btn.id] || btn.innerHTML;
            }
        });

        // Show specific loading indicators
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
            ['summaryLoading', 'sentimentLoading'].forEach(id => {
                const indicator = document.getElementById(id);
                if (indicator) indicator.style.display = 'none';
            });
        }
    }

    /**
     * Display analysis results
     */
    displayResults(result, type) {
        // Show results section
        this.elements.resultsSection.style.display = 'block';

        // Extract the actual result data
        const data = result.result || result;

        // Display summary if available
        this.displaySummary(data, type);

        // Display sentiment if available
        this.displaySentiment(data);

        // Display statistics
        this.displayStatistics(data);

        // Scroll to results
        this.elements.resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    /**
     * Display summary results
     */
    displaySummary(data, type) {
        if (data.summary && data.summary.trim() !== '') {
            this.elements.summaryText.textContent = data.summary;
            this.elements.summaryResults.style.display = 'block';
        } else if (type === 'analyze' && data.summary === null) {
            // For combine button when summary is null (short text), show explanation
            this.elements.summaryText.textContent = 'Văn bản quá ngắn để tóm tắt (dưới 50 ký tự). Chỉ phân tích cảm xúc.';
            this.elements.summaryResults.style.display = 'block';
        } else {
            this.elements.summaryResults.style.display = 'none';
        }
    }

    /**
     * Display sentiment results
     */
    displaySentiment(data) {
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
            this.elements.sentimentResults.style.display = 'block';
        } else {
            this.elements.sentimentResults.style.display = 'none';
        }
    }

    /**
     * Display detailed sentiment results
     */
    displaySentimentResults(sentiment) {
        // Set sentiment label with appropriate color and styling
        const labelText = sentiment.label || 'Unknown';
        this.elements.sentimentLabel.textContent = labelText;
        
        // Apply color classes based on sentiment
        this.elements.sentimentLabel.className = 'sentiment-label';
        if (labelText.toLowerCase().includes('pos') || labelText.toLowerCase().includes('tích cực')) {
            this.elements.sentimentLabel.classList.add('sentiment-positive');
            this.elements.confidenceBar.className = 'confidence-fill bg-success';
        } else if (labelText.toLowerCase().includes('neg') || labelText.toLowerCase().includes('tiêu cực')) {
            this.elements.sentimentLabel.classList.add('sentiment-negative');
            this.elements.confidenceBar.className = 'confidence-fill bg-danger';
        } else {
            this.elements.sentimentLabel.classList.add('sentiment-neutral');
            this.elements.confidenceBar.className = 'confidence-fill bg-warning';
        }

        // Show sentiment source information
        if (sentiment.usedSummarization) {
            this.elements.sentimentSourceText.textContent = 'văn bản đã tóm tắt';
            this.elements.sentimentSource.style.display = 'block';
        } else if (sentiment.analysisMethod && sentiment.analysisMethod.includes('direct')) {
            this.elements.sentimentSourceText.textContent = 'văn bản gốc';
            this.elements.sentimentSource.style.display = 'block';
        } else {
            this.elements.sentimentSource.style.display = 'none';
        }

        // Set confidence bar
        const confidence = (sentiment.confidence || 0) * 100;
        this.elements.confidenceBar.style.width = `${confidence}%`;
        this.elements.confidenceText.textContent = `${confidence.toFixed(1)}%`;

        // Display probabilities
        this.displayProbabilities(sentiment.probabilities);
    }

    /**
     * Display probability distribution
     */
    displayProbabilities(probabilities) {
        if (probabilities && Object.keys(probabilities).length > 0) {
            let probabilitiesHtml = '<h6 class="small mb-3"><i class="fas fa-chart-pie me-1"></i>Phân Bố Xác Suất:</h6>';
            probabilitiesHtml += '<div class="row">';
            
            for (const [label, prob] of Object.entries(probabilities)) {
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
            this.elements.probabilitiesSection.innerHTML = probabilitiesHtml;
        } else {
            this.elements.probabilitiesSection.innerHTML = '';
        }
    }

    /**
     * Display processing statistics
     */
    displayStatistics(data) {
        let statsHtml = '';
        
        // Processing time
        if (data.processing_time !== undefined) {
            statsHtml += `
                <div class="stat-item">
                    <div class="stat-label">Thời gian xử lý</div>
                    <div class="stat-value">${data.processing_time}s</div>
                </div>
            `;
        }
        
        // Token counts
        if (data.original_tokens !== undefined) {
            statsHtml += `
                <div class="stat-item">
                    <div class="stat-label">Tokens văn bản gốc</div>
                    <div class="stat-value">${data.original_tokens}</div>
                </div>
            `;
        }
        
        if (data.summary_tokens !== undefined && data.summary_tokens > 0) {
            statsHtml += `
                <div class="stat-item">
                    <div class="stat-label">Tokens tóm tắt</div>
                    <div class="stat-value">${data.summary_tokens}</div>
                </div>
            `;
        }
        
        // Compression ratio
        if (data.compression_ratio !== undefined && data.compression_ratio !== null) {
            const ratio = (data.compression_ratio * 100).toFixed(1);
            statsHtml += `
                <div class="stat-item">
                    <div class="stat-label">Tỷ lệ nén</div>
                    <div class="stat-value">${ratio}%</div>
                </div>
            `;
        }

        this.elements.statisticsSection.innerHTML = statsHtml;
    }

    /**
     * Hide results section
     */
    hideResults() {
        this.elements.resultsSection.style.display = 'none';
        this.elements.summaryResults.style.display = 'none';
        this.elements.sentimentResults.style.display = 'none';
    }

    /**
     * Show error message
     */
    showError(message) {
        // Create error alert if it doesn't exist
        let errorAlert = document.getElementById('errorAlert');
        if (!errorAlert) {
            errorAlert = document.createElement('div');
            errorAlert.id = 'errorAlert';
            errorAlert.className = 'alert alert-danger alert-dismissible fade show mt-3';
            errorAlert.innerHTML = `
                <span id="errorMessage"></span>
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            this.elements.textInput.parentNode.appendChild(errorAlert);
        }
        
        document.getElementById('errorMessage').textContent = message;
        errorAlert.style.display = 'block';
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            if (errorAlert) errorAlert.style.display = 'none';
        }, 5000);
    }

    /**
     * Show success message
     */
    showSuccess(message) {
        // Create success alert if it doesn't exist
        let successAlert = document.getElementById('successAlert');
        if (!successAlert) {
            successAlert = document.createElement('div');
            successAlert.id = 'successAlert';
            successAlert.className = 'alert alert-success alert-dismissible fade show mt-3';
            successAlert.innerHTML = `
                <span id="successMessage"></span>
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            this.elements.textInput.parentNode.appendChild(successAlert);
        }
        
        document.getElementById('successMessage').textContent = message;
        successAlert.style.display = 'block';
        
        // Auto-hide after 3 seconds
        setTimeout(() => {
            if (successAlert) successAlert.style.display = 'none';
        }, 3000);
    }

    /**
     * Clear all content and results
     */
    clearAll() {
        this.elements.textInput.value = '';
        this.hideResults();
        this.elements.statisticsSection.innerHTML = '';
    }
}

// Export for module use
window.UIManager = UIManager;