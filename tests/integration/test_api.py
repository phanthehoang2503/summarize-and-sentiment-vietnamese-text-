"""
Integration tests for the Flask API
"""
import pytest
import json
from app.main import create_app


@pytest.fixture
def client():
    """Create test client"""
    app = create_app({'TESTING': True})
    with app.test_client() as client:
        yield client


class TestAPIEndpoints:
    """Test suite for API endpoints"""
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get('/api/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'version' in data
    
    def test_summarize_endpoint_no_data(self, client):
        """Test summarize endpoint with no data"""
        response = client.post('/api/summarize')
        assert response.status_code == 400
    
    def test_summarize_endpoint_empty_text(self, client):
        """Test summarize endpoint with empty text"""
        response = client.post('/api/summarize', 
                              json={'text': ''},
                              content_type='application/json')
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'error' in data
    
    def test_sentiment_endpoint_empty_text(self, client):
        """Test sentiment endpoint with empty text"""
        response = client.post('/api/sentiment',
                              json={'text': ''},
                              content_type='application/json')
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'error' in data
    
    def test_analyze_endpoint_empty_text(self, client):
        """Test analyze endpoint with empty text"""
        response = client.post('/api/analyze',
                              json={'text': ''},
                              content_type='application/json')
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'error' in data
    
    def test_main_page(self, client):
        """Test main page loads"""
        response = client.get('/')
        assert response.status_code == 200