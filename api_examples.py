#!/usr/bin/env python3
"""
Example usage of the Text Summarization FastAPI server
"""

import requests
import json
from pathlib import Path

# Server configuration
BASE_URL = "http://localhost:9000"


def test_health_check():
    """Test the health check endpoint."""
    print("🔍 Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)


def test_text_summarization():
    """Test basic text summarization."""
    print("📝 Testing text summarization...")
    
    sample_text = """
    Artificial Intelligence (AI) has emerged as one of the most transformative technologies of the 21st century, 
    fundamentally altering how we work, communicate, and solve complex problems. From its humble beginnings in 
    academic research laboratories to its current widespread adoption across industries, AI has evolved from a 
    theoretical concept to a practical tool that impacts millions of lives daily. The current AI revolution is 
    largely driven by three key factors: the exponential growth in computing power, the availability of vast 
    amounts of data, and significant advances in machine learning algorithms, particularly deep learning.
    """
    
    # Basic summarization
    payload = {
        "text": sample_text,
        "summary_type": "basic",
        "chain_type": "map_reduce"
    }
    
    response = requests.post(f"{BASE_URL}/summarize/text", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Summary: {result['summary']}")
        print(f"Stats: {result['stats']}")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)


def test_advanced_summarization():
    """Test advanced LangGraph summarization."""
    print("🚀 Testing advanced summarization...")
    
    sample_text = """
    Machine learning, a subset of AI, has become particularly prominent in recent years. Unlike traditional 
    programming where developers write explicit instructions for computers to follow, machine learning systems 
    learn patterns from data and make predictions or decisions based on those patterns. Deep learning, which 
    uses artificial neural networks inspired by the human brain, has achieved remarkable success in areas such 
    as image recognition, natural language processing, and game playing. The applications of AI in modern 
    society are vast and growing rapidly. In healthcare, AI systems are being used to analyze medical images, 
    assist in drug discovery, and even predict patient outcomes.
    """
    
    payload = {
        "text": sample_text,
        "summary_type": "advanced"
    }
    
    response = requests.post(f"{BASE_URL}/summarize/text", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Final Summary: {result['final_summary']}")
        print(f"Key Points: {result['key_points']}")
        print(f"Metadata: {result['metadata']}")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)


def test_file_upload():
    """Test file upload and summarization."""
    print("📁 Testing file upload...")
    
    # Create a sample file
    sample_file_path = "temp_sample.txt"
    with open(sample_file_path, "w", encoding="utf-8") as f:
        f.write("""
        The Rise of Artificial Intelligence in Modern Society
        
        Artificial Intelligence (AI) has emerged as one of the most transformative technologies of the 21st century, 
        fundamentally altering how we work, communicate, and solve complex problems. From its humble beginnings in 
        academic research laboratories to its current widespread adoption across industries, AI has evolved from a 
        theoretical concept to a practical tool that impacts millions of lives daily.
        
        The current AI revolution is largely driven by three key factors: the exponential growth in computing power, 
        the availability of vast amounts of data, and significant advances in machine learning algorithms, particularly 
        deep learning. Modern AI systems can process and analyze data at scales that were unimaginable just a few decades ago.
        """)
    
    try:
        with open(sample_file_path, "rb") as f:
            files = {"file": (sample_file_path, f, "text/plain")}
            data = {
                "summary_type": "advanced",
                "model_name": "gpt-3.5-turbo"
            }
            
            response = requests.post(f"{BASE_URL}/summarize/file", files=files, data=data)
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Final Summary: {result['final_summary']}")
                if result.get('key_points'):
                    print(f"Key Points: {result['key_points']}")
            else:
                print(f"Error: {response.text}")
    
    finally:
        # Clean up
        Path(sample_file_path).unlink(missing_ok=True)
    
    print("-" * 50)


def test_get_config():
    """Test configuration endpoint."""
    print("⚙️ Testing configuration...")
    response = requests.get(f"{BASE_URL}/config")
    print(f"Status: {response.status_code}")
    print(f"Config: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)


def test_get_models():
    """Test available models endpoint."""
    print("🤖 Testing available models...")
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status: {response.status_code}")
    print(f"Models: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)


def main():
    """Run all API tests."""
    print("🚀 Text Summarization API Examples")
    print("=" * 60)
    
    try:
        test_health_check()
        test_get_config()
        test_get_models()
        test_text_summarization()
        test_advanced_summarization()
        test_file_upload()
        
        print("✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API server.")
        print("Make sure the server is running with: python app.py")
        print("Or: uvicorn app:app --reload")
        print("Also ensure PERPLEXITY_API_KEY is set in your .env file")
    
    except Exception as e:
        print(f"❌ Error during testing: {e}")


if __name__ == "__main__":
    main()
