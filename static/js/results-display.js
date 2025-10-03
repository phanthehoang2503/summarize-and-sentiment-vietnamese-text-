// Results Display Component for Vietnamese Text Analysis
class ResultsDisplay {
    constructor(uiManager) {
        this.ui = uiManager;
    }

    displayResults(result, type) {
        const { resultsSection, summaryResults, sentimentResults } = this.ui.elements;

        // Show results section
        resultsSection.style.display = 'block';

        // Extract the actual result data
        const data = result.result || result;

        // Display summary if available
        this._displaySummary(data, type);
        
        // Display sentiment if available
        this._displaySentiment(data);

        // Display statistics
        this._displayStatistics(data);

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    _displaySummary(data, type) {
        const { summaryResults, summaryText } = this.ui.elements;

        if (data.summary && data.summary.trim() !== '') {
            summaryText.textContent = data.summary;
            summaryResults.style.display = 'block';
        } else if (type === 'analyze' && data.summary === null) {
            // For combine button when summary is null (short text), show explanation
            summaryText.textContent = 'Văn bản quá ngắn để tóm tắt (dưới 50 ký tự). Chỉ phân tích cảm xúc.';
            summaryResults.style.display = 'block';
        } else {
            summaryResults.style.display = 'none';
        }
    }

    _displaySentiment(data) {
        const { sentimentResults } = this.ui.elements;
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
            this._displaySentimentResults(sentimentContext);
            sentimentResults.style.display = 'block';
        } else {
            sentimentResults.style.display = 'none';
        }
    }

    _displaySentimentResults(sentiment) {
        const { sentimentLabel, confidenceBar, confidenceText, probabilitiesSection,
                sentimentSource, sentimentSourceText } = this.ui.elements;

        // Set sentiment label with appropriate color and styling
        const labelText = sentiment.label || 'Unknown';
        sentimentLabel.textContent = labelText;
        
        // Apply color classes based on sentiment
        sentimentLabel.className = 'sentiment-label';
        if (labelText.toLowerCase().includes('pos') || labelText.toLowerCase().includes('tích cực')) {
            sentimentLabel.classList.add('sentiment-positive');
            confidenceBar.className = 'confidence-fill bg-success';
        } else if (labelText.toLowerCase().includes('neg') || labelText.toLowerCase().includes('tiêu cực')) {
            sentimentLabel.classList.add('sentiment-negative');
            confidenceBar.className = 'confidence-fill bg-danger';
        } else {
            sentimentLabel.classList.add('sentiment-neutral');
            confidenceBar.className = 'confidence-fill bg-warning';
        }

        // Show sentiment source information
        if (sentiment.usedSummarization) {
            sentimentSourceText.textContent = 'văn bản đã tóm tắt';
            sentimentSource.style.display = 'block';
        } else if (sentiment.analysisMethod && sentiment.analysisMethod.includes('direct')) {
            sentimentSourceText.textContent = 'văn bản gốc';
            sentimentSource.style.display = 'block';
        } else {
            sentimentSource.style.display = 'none';
        }

        // Set confidence bar
        const confidence = (sentiment.confidence || 0) * 100;
        confidenceBar.style.width = `${confidence}%`;
        confidenceText.textContent = `${confidence.toFixed(1)}%`;

        // Display probabilities
        this._displayProbabilities(sentiment.probabilities);
    }

    _displayProbabilities(probabilities) {
        const { probabilitiesSection } = this.ui.elements;

        if (probabilities && Object.keys(probabilities).length > 0) {
            let probabilitiesHtml = '<h6 class="small mb-3">Phân Bố Xác Suất:</h6>';
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
                        <div class="progress progress-sm">
                            <div class="progress-bar ${colorClass.replace('text-', 'bg-')}" 
                                 style="width: ${percentage}%"></div>
                        </div>
                    </div>`;
            }
            
            probabilitiesHtml += '</div>';
            probabilitiesSection.innerHTML = probabilitiesHtml;
        } else {
            probabilitiesSection.innerHTML = '';
        }
    }

    _displayStatistics(data) {
        const { statisticsSection } = this.ui.elements;

        // Map all possible statistics from different API responses (matching original)
        const stats = [
            {
                label: 'Thời gian xử lý',
                value: data.processing_time ? `${data.processing_time}s` : 'N/A'
            },
            {
                label: 'Tokens văn bản gốc',
                value: data.original_tokens || data.tokens || (data.sentiment && data.sentiment.tokens) || 'N/A'
            },
            {
                label: 'Tokens tóm tắt',
                value: data.summary_tokens || (data.summary ? data.summary.split(' ').length : 'N/A')
            }
        ];

        let statsHtml = '';
        stats.forEach(stat => {
            if (stat.value !== 'N/A') {
                statsHtml += `
                    <div class="stat-item">
                        <div class="stat-content">
                            <div class="stat-label">${stat.label}</div>
                            <div class="stat-value">${stat.value}</div>
                        </div>
                    </div>
                `;
            }
        });

        statisticsSection.innerHTML = statsHtml;
    }
}