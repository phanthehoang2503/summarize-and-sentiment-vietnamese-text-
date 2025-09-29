/**
 * History Manager Module  
 * Handles analysis history storage and display
 */
class HistoryManager {
    constructor() {
        this.maxHistoryItems = 10;
        this.storageKey = 'vietnamese_text_analysis_history';
        this.history = this.loadHistory();
    }

    /**
     * Load history from localStorage
     */
    loadHistory() {
        try {
            const saved = localStorage.getItem(this.storageKey);
            return saved ? JSON.parse(saved) : [];
        } catch (error) {
            console.error('Failed to load history:', error);
            return [];
        }
    }

    /**
     * Save history to localStorage
     */
    saveHistory() {
        try {
            localStorage.setItem(this.storageKey, JSON.stringify(this.history));
        } catch (error) {
            console.error('Failed to save history:', error);
        }
    }

    /**
     * Add new analysis to history
     */
    addToHistory(text, result, type) {
        const entry = {
            id: Date.now(),
            timestamp: new Date().toLocaleString('vi-VN'),
            text: text.substring(0, 100) + (text.length > 100 ? '...' : ''),
            type: type,
            result: this.sanitizeResult(result)
        };

        // Add to beginning of array
        this.history.unshift(entry);

        // Keep only the last N items
        if (this.history.length > this.maxHistoryItems) {
            this.history = this.history.slice(0, this.maxHistoryItems);
        }

        this.saveHistory();
        this.updateHistoryDisplay();
    }

    /**
     * Sanitize result data for storage
     */
    sanitizeResult(result) {
        // Only store essential data to avoid localStorage size limits
        const data = result.result || result;
        return {
            success: result.success,
            summary: data.summary ? data.summary.substring(0, 200) : null,
            sentiment: data.sentiment || data.predicted_label ? {
                label: data.sentiment?.predicted_label || data.predicted_label,
                confidence: data.sentiment?.confidence || data.confidence
            } : null,
            processing_time: data.processing_time
        };
    }

    /**
     * Update history display in UI
     */
    updateHistoryDisplay() {
        const historyPanel = document.getElementById('historyPanel');
        if (!historyPanel) return;
        
        if (this.history.length === 0) {
            historyPanel.innerHTML = '<p class="text-muted small text-center">Chưa có phân tích nào được thực hiện</p>';
            return;
        }

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

        let historyHtml = '';
        this.history.forEach((entry, index) => {
            const typeLabel = typeLabels[entry.type] || entry.type;
            const typeIcon = typeIcons[entry.type] || 'fas fa-file-text';

            historyHtml += `
                <div class="history-item mb-2 p-2 border rounded" data-index="${index}">
                    <div class="d-flex justify-content-between align-items-start">
                        <div class="flex-grow-1">
                            <small class="text-muted">${entry.timestamp}</small>
                            <div class="small">
                                <i class="${typeIcon} me-1"></i>
                                <strong>${typeLabel}</strong>
                            </div>
                            <div class="small text-muted mt-1" style="max-height: 40px; overflow: hidden;">
                                ${entry.text}
                            </div>
                            ${this.renderHistoryResult(entry.result)}
                        </div>
                        <button class="btn btn-sm btn-outline-danger ms-2" onclick="historyManager.removeHistoryItem(${index})" title="Xóa">
                            <i class="fas fa-trash-alt"></i>
                        </button>
                    </div>
                </div>
            `;
        });

        historyPanel.innerHTML = historyHtml;
    }

    /**
     * Render history result summary
     */
    renderHistoryResult(result) {
        if (!result) return '';
        
        let resultHtml = '<div class="small text-info mt-1">';
        
        if (result.sentiment) {
            resultHtml += `<i class="fas fa-smile me-1"></i>${result.sentiment.label} `;
        }
        
        if (result.summary) {
            resultHtml += `<i class="fas fa-compress-alt me-1"></i>Đã tóm tắt `;
        }
        
        if (result.processing_time) {
            resultHtml += `<i class="fas fa-clock me-1"></i>${result.processing_time}s`;
        }
        
        resultHtml += '</div>';
        
        return resultHtml;
    }

    /**
     * Remove history item
     */
    removeHistoryItem(index) {
        if (index >= 0 && index < this.history.length) {
            this.history.splice(index, 1);
            this.saveHistory();
            this.updateHistoryDisplay();
        }
    }

    /**
     * Clear all history
     */
    clearHistory() {
        this.history = [];
        this.saveHistory();
        this.updateHistoryDisplay();
    }

    /**
     * Get history statistics
     */
    getHistoryStats() {
        const stats = {
            total: this.history.length,
            byType: {},
            averageProcessingTime: 0
        };

        let totalTime = 0;
        let timeCount = 0;

        this.history.forEach(entry => {
            // Count by type
            stats.byType[entry.type] = (stats.byType[entry.type] || 0) + 1;
            
            // Calculate average processing time
            if (entry.result?.processing_time) {
                totalTime += entry.result.processing_time;
                timeCount++;
            }
        });

        if (timeCount > 0) {
            stats.averageProcessingTime = (totalTime / timeCount).toFixed(2);
        }

        return stats;
    }
}

// Export for module use
window.HistoryManager = HistoryManager;