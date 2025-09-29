/**
 * File Handler Module
 * Handles file upload and processing
 */
class FileHandler {
    constructor(apiClient, uiManager) {
        this.apiClient = apiClient;
        this.uiManager = uiManager;
        this.allowedTypes = ['text/plain', 'text/csv', 'application/csv'];
        this.initializeEventListeners();
    }

    /**
     * Initialize file upload event listeners
     */
    initializeEventListeners() {
        const fileInput = document.getElementById('fileInput');
        const fileUploadArea = document.getElementById('fileUploadArea');
        
        if (!fileInput || !fileUploadArea) return;

        // Click to select file
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

    /**
     * Handle file upload
     */
    async handleFileUpload(file) {
        if (!file) return;

        // Validate file type
        if (!this.isValidFileType(file)) {
            this.uiManager.showError('Vui lòng chọn tệp .txt hoặc .csv');
            return;
        }

        this.uiManager.showLoading(true);

        try {
            const result = await this.apiClient.uploadFile(file);
            
            // Set the uploaded content to textarea
            document.getElementById('textInput').value = result.text || result.content;
            
            this.uiManager.showSuccess(
                `Tệp đã được tải lên thành công! Đã trích xuất ${(result.text || result.content).length} ký tự.`
            );
            
        } catch (error) {
            console.error('File upload error:', error);
            this.uiManager.showError(`Tải tệp thất bại: ${error.message}`);
        } finally {
            this.uiManager.showLoading(false);
        }
    }

    /**
     * Validate file type
     */
    isValidFileType(file) {
        return this.allowedTypes.includes(file.type) || 
               file.name.match(/\.(txt|csv)$/i);
    }
}

// Export for module use
window.FileHandler = FileHandler;