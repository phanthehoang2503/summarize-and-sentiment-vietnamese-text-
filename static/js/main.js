/**
 * Vietnamese Text Analyzer - Main Application
 * Orchestrates all modules and handles user interactions
 */
class VietnameseTextAnalyzer {
    constructor() {
        // Initialize modules
        this.apiClient = new ApiClient();
        this.uiManager = new UIManager();
        this.historyManager = new HistoryManager();
        this.fileHandler = new FileHandler(this.apiClient, this.uiManager);
        
        // Initialize the application
        this.initializeEventListeners();
        this.historyManager.updateHistoryDisplay();
    }

    /**
     * Initialize main event listeners
     */
    initializeEventListeners() {
        // Main action buttons
        const buttons = {
            summarizeBtn: () => this.performAnalysis('summarize'),
            sentimentBtn: () => this.performAnalysis('sentiment'),
            analyzeBtn: () => this.performAnalysis('analyze'),
            clearBtn: () => this.clearAll()
        };

        Object.entries(buttons).forEach(([id, handler]) => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('click', handler);
            }
        });

        // History panel clear button (if exists)
        const clearHistoryBtn = document.getElementById('clearHistoryBtn');
        if (clearHistoryBtn) {
            clearHistoryBtn.addEventListener('click', () => this.clearHistory());
        }
    }

    /**
     * Perform text analysis
     */
    async performAnalysis(type) {
        const textInput = document.getElementById('textInput');
        const text = textInput.value.trim();
        
        if (!text) {
            this.uiManager.showError('Vui lòng nhập văn bản để phân tích');
            return;
        }

        this.uiManager.showLoading(true, type);
        this.uiManager.hideResults();

        try {
            const result = await this.apiClient.performAnalysis(text, type);
            
            // Display results
            this.uiManager.displayResults(result, type);
            
            // Add to history
            this.historyManager.addToHistory(text, result, type);
            
        } catch (error) {
            console.error('Analysis error:', error);
            this.uiManager.showError(`Phân tích thất bại: ${error.message}`);
        } finally {
            this.uiManager.showLoading(false, type);
        }
    }

    /**
     * Clear all content and results
     */
    clearAll() {
        this.uiManager.clearAll();
    }

    /**
     * Clear analysis history
     */
    clearHistory() {
        if (confirm('Bạn có chắc muốn xóa toàn bộ lịch sử phân tích?')) {
            this.historyManager.clearHistory();
        }
    }

    /**
     * Get application status
     */
    getStatus() {
        return {
            modules: {
                apiClient: !!this.apiClient,
                uiManager: !!this.uiManager,
                historyManager: !!this.historyManager,
                fileHandler: !!this.fileHandler
            },
            history: this.historyManager.getHistoryStats()
        };
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Make historyManager globally accessible for button clicks
    window.historyManager = null;
    
    try {
        const app = new VietnameseTextAnalyzer();
        window.historyManager = app.historyManager;
        window.textAnalyzer = app;
        
        console.log('Vietnamese Text Analyzer initialized successfully');
        console.log('App status:', app.getStatus());
        
    } catch (error) {
        console.error('Failed to initialize Vietnamese Text Analyzer:', error);
    }
});