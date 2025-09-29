/**
 * API Client Module
 * Handles all communication with backend APIs
 */
class ApiClient {
    constructor() {
        this.baseUrl = '';
    }

    /**
     * Upload file to server
     */
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
            throw new Error(result.error || 'Upload failed');
        }

        return result;
    }

    /**
     * Perform text analysis
     */
    async performAnalysis(text, type) {
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
            throw new Error(result.error || 'Analysis failed');
        }

        return result;
    }
}

// Export for module use
window.ApiClient = ApiClient;