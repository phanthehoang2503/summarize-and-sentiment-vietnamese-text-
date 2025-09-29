// File Upload Handler for Vietnamese Text Analysis
class FileUploadHandler {
    constructor(apiService, uiManager) {
        this.apiService = apiService;
        this.uiManager = uiManager;
        this.allowedTypes = ['text/plain', 'text/csv', 'application/csv'];
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        const fileInput = document.getElementById('fileInput');
        const fileUploadArea = document.getElementById('fileUploadArea');
        
        if (!fileInput || !fileUploadArea) return;

        fileUploadArea.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => this.handleFileUpload(e.target.files[0]));
        
        // Drag and drop events
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

        // Validate file type
        if (!this.isValidFileType(file)) {
            this.uiManager.showError('Vui lòng thả tệp .txt hoặc .csv vào đây');
            return;
        }

        this.uiManager.showLoading(true);

        try {
            const result = await this.apiService.uploadFile(file);
            
            if (result.success) {
                this.uiManager.setText(result.content || result.text);
                this.uiManager.showSuccess(
                    `Tệp đã được tải lên thành công! Đã trích xuất ${(result.content || result.text).length} ký tự.`
                );
            }
        } catch (error) {
            console.error('File upload error:', error);
            this.uiManager.showError(`Tải tệp thất bại: ${error.message}`);
        } finally {
            this.uiManager.showLoading(false);
        }
    }

    isValidFileType(file) {
        return this.allowedTypes.includes(file.type) || file.name.match(/\.(txt|csv)$/i);
    }
}