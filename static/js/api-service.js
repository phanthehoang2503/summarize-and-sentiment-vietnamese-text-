// API Service for Vietnamese Text Analysis
class ApiService {
    constructor() {
        this.baseUrl = '/api';
    }

    async performAnalysis(type, text) {
        let endpoint;
        const requestBody = { text: text };
        
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
                throw new Error(`Unknown analysis type: ${type}`);
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
        
        if (!result.success) {
            throw new Error(result.error || 'Phân tích thất bại');
        }

        return result;
    }

    async uploadFile(file) {
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
        
        if (!result.success) {
            throw new Error(result.error || 'Tải lên thất bại');
        }

        return result;
    }
}