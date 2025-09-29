// UI Manager for Vietnamese Text Analysis
class UIManager {
    constructor() {
        this.elements = {
            textInput: document.getElementById('textInput'),
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
            statisticsSection: document.getElementById('statisticsSection')
        };
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
                    'summarizeBtn': `<i class="${originalIcon}"></i> Tóm tắt`,
                    'sentimentBtn': `<i class="${originalIcon}"></i> Cảm xúc`,
                    'analyzeBtn': `<i class="${originalIcon}"></i> Cả hai`
                };
                btn.innerHTML = labels[btn.id] || btn.innerHTML;
            }
        });

        // Show/hide specific loading indicators
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

    hideResults() {
        this.elements.resultsSection.style.display = 'none';
        this.elements.summaryResults.style.display = 'none';
        this.elements.sentimentResults.style.display = 'none';
    }

    clearAll() {
        this.elements.textInput.value = '';
        this.hideResults();
        this.elements.statisticsSection.innerHTML = '';
    }

    showError(message) {
        // Remove any existing alerts
        const existingAlert = document.querySelector('.alert');
        if (existingAlert) {
            existingAlert.remove();
        }

        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-danger alert-dismissible fade show mt-3';
        alertDiv.innerHTML = `
            <i class="fas fa-exclamation-triangle me-2"></i>
            <strong>Lỗi:</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        const container = document.querySelector('.container');
        const firstCard = container.querySelector('.card');
        container.insertBefore(alertDiv, firstCard);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (alertDiv && alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }

    showSuccess(message) {
        // Remove any existing alerts
        const existingAlert = document.querySelector('.alert');
        if (existingAlert) {
            existingAlert.remove();
        }

        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-success alert-dismissible fade show mt-3';
        alertDiv.innerHTML = `
            <i class="fas fa-check-circle me-2"></i>
            <strong>Thành công:</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        const container = document.querySelector('.container');
        const firstCard = container.querySelector('.card');
        container.insertBefore(alertDiv, firstCard);

        // Auto remove after 3 seconds
        setTimeout(() => {
            if (alertDiv && alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 3000);
    }

    getText() {
        return this.elements.textInput.value.trim();
    }

    setText(text) {
        this.elements.textInput.value = text;
    }
}