// Main Application - Vietnamese Text Analysis
class VietnameseTextAnalyzer {
    constructor() {
        // Initialize components
        this.apiService = new ApiService();
        this.uiManager = new UIManager();
        this.resultsDisplay = new ResultsDisplay(this.uiManager);
        this.fileUploadHandler = new FileUploadHandler(this.apiService, this.uiManager);
        this.historyManager = new HistoryManager();
        
        // Initialize the application
        this.initializeEventListeners();
        this.historyManager.updateHistoryDisplay();
    }

    initializeEventListeners() {
        // Main action buttons
        document.getElementById('summarizeBtn')?.addEventListener('click', () => this.performAnalysis('summarize'));
        document.getElementById('sentimentBtn')?.addEventListener('click', () => this.performAnalysis('sentiment'));
        document.getElementById('analyzeBtn')?.addEventListener('click', () => this.performAnalysis('analyze'));
        document.getElementById('clearBtn')?.addEventListener('click', () => this.clearAll());
    }

    async performAnalysis(type) {
        const text = this.uiManager.getText();
        
        if (!text) {
            this.uiManager.showError('Vui lòng nhập văn bản để phân tích');
            return;
        }

        this.uiManager.showLoading(true, type);
        this.uiManager.hideResults();

        try {
            const result = await this.apiService.performAnalysis(type, text);
            
            this.resultsDisplay.displayResults(result, type);
            this.historyManager.addToHistory(text, result, type);
            
        } catch (error) {
            console.error('Analysis error:', error);
            this.uiManager.showError(`Phân tích thất bại: ${error.message}`);
        } finally {
            this.uiManager.showLoading(false, type);
        }
    }

    clearAll() {
        this.uiManager.clearAll();
    }
}

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.textAnalyzer = new VietnameseTextAnalyzer();
});