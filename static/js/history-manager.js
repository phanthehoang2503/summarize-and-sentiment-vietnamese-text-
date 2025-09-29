// History Manager for Vietnamese Text Analysis
class HistoryManager {
    constructor() {
        this.history = this.loadHistory();
        this.maxHistoryItems = 10;
    }

    loadHistory() {
        try {
            return JSON.parse(localStorage.getItem('vietnamese_text_analysis_history') || '[]');
        } catch (e) {
            console.warn('Failed to load history:', e);
            return [];
        }
    }

    saveHistory() {
        try {
            localStorage.setItem('vietnamese_text_analysis_history', JSON.stringify(this.history));
        } catch (e) {
            console.warn('Failed to save history:', e);
        }
    }

    addToHistory(text, result, type) {
        // Prevent duplicate entries
        const isDuplicate = this.history.some(item => 
            item.text === text && item.type === type
        );
        
        if (isDuplicate) return;

        const historyItem = {
            timestamp: new Date().toISOString(),
            text: text.length > 100 ? text.substring(0, 100) + '...' : text,
            fullText: text,
            result: result,
            type: type,
            id: Date.now()
        };
        
        this.history.unshift(historyItem);
        
        // Keep only recent items
        if (this.history.length > this.maxHistoryItems) {
            this.history = this.history.slice(0, this.maxHistoryItems);
        }
        
        this.saveHistory();
        this.updateHistoryDisplay();
    }

    updateHistoryDisplay() {
        const historyContainer = document.querySelector('.history-list');
        if (!historyContainer) return;
        
        if (this.history.length === 0) {
            historyContainer.innerHTML = `
                <div class="text-center text-muted py-3">
                    <i class="fas fa-history fa-2x mb-2"></i>
                    <p>Chưa có lịch sử phân tích</p>
                </div>`;
            return;
        }
        
        let historyHtml = '';
        this.history.forEach((item, index) => {
            const date = new Date(item.timestamp).toLocaleDateString('vi-VN');
            const time = new Date(item.timestamp).toLocaleTimeString('vi-VN', { 
                hour: '2-digit', 
                minute: '2-digit' 
            });
            
            const typeLabels = {
                'summarize': 'Tóm tắt',
                'sentiment': 'Cảm xúc',
                'analyze': 'Cả hai'
            };
            
            historyHtml += `
                <div class="history-item" data-id="${item.id}">
                    <div class="history-header">
                        <span class="history-type badge bg-primary">${typeLabels[item.type] || item.type}</span>
                        <span class="history-time text-muted small">${date} ${time}</span>
                    </div>
                    <div class="history-text">${item.text}</div>
                    <div class="history-actions">
                        <button class="btn btn-sm btn-outline-primary reuse-btn" data-text="${this.escapeHtml(item.fullText)}">
                            <i class="fas fa-redo"></i> Sử dụng lại
                        </button>
                        <button class="btn btn-sm btn-outline-danger delete-btn" data-id="${item.id}">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>`;
        });
        
        historyContainer.innerHTML = historyHtml;
        
        // Add event listeners for history actions
        this.attachHistoryEventListeners();
    }

    attachHistoryEventListeners() {
        // Reuse button listeners
        document.querySelectorAll('.reuse-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const text = e.target.closest('.reuse-btn').dataset.text;
                const textInput = document.getElementById('textInput');
                if (textInput) {
                    textInput.value = text;
                    textInput.focus();
                }
            });
        });
        
        // Delete button listeners
        document.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const id = parseInt(e.target.closest('.delete-btn').dataset.id);
                this.removeFromHistory(id);
            });
        });
    }

    removeFromHistory(id) {
        this.history = this.history.filter(item => item.id !== id);
        this.saveHistory();
        this.updateHistoryDisplay();
    }

    clearHistory() {
        this.history = [];
        this.saveHistory();
        this.updateHistoryDisplay();
    }

    escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
}