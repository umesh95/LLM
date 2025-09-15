#!/usr/bin/env python3
"""
API Demo script showing how to use the Text Summarization FastAPI server
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import json
from pathlib import Path

# Server configuration
BASE_URL = "http://localhost:9000"


def check_server_status():
    """Check if the server is running."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def demo_basic_text_summarization():
    """Demonstrate basic text summarization via API."""
    print("=" * 60)
    print("BASIC TEXT SUMMARIZATION DEMO")
    print("=" * 60)
    
    # Read sample text
    sample_file = Path(__file__).parent / "sample_text.txt"
    with open(sample_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"Original text length: {len(text)} characters")
    print(f"Original word count: {len(text.split())} words")
    print("\n" + "-" * 40)
    
    # API request for basic summarization
    payload = {
        "text": text,
        "summary_type": "basic",
        "chain_type": "map_reduce"
    }
    
    print("Sending request to API...")
    response = requests.post(f"{BASE_URL}/summarize/text", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("\nSUMMARY:")
        print("-" * 20)
        print(result["summary"])
        
        if result.get("stats"):
            stats = result["stats"]
            print(f"\nSTATISTICS:")
            print(f"- Original words: {stats['original_words']}")
            print(f"- Summary words: {stats['summary_words']}")
            print(f"- Compression: {stats['compression_percentage']}")
            print(f"- Model used: {result['model_used']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def demo_advanced_summarization():
    """Demonstrate advanced LangGraph summarization via API."""
    print("\n" + "=" * 60)
    print("ADVANCED LANGGRAPH SUMMARIZATION DEMO")
    print("=" * 60)
    
    # Read sample text
    sample_file = Path(__file__).parent / "sample_text.txt"
    with open(sample_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    # API request for advanced summarization
    payload = {
        "text": text,
        "summary_type": "advanced",
        "model_name": "gpt-3.5-turbo"
    }
    
    print("Running advanced multi-step summarization workflow...")
    response = requests.post(f"{BASE_URL}/summarize/text", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        
        print("\nINITIAL SUMMARY:")
        print("-" * 30)
        print(result.get("initial_summary", "Not available"))
        
        print("\nKEY POINTS:")
        print("-" * 20)
        for i, point in enumerate(result.get("key_points", []), 1):
            print(f"{i}. {point}")
        
        print("\nREFINED SUMMARY:")
        print("-" * 25)
        print(result.get("refined_summary", "Not available"))
        
        print("\nFINAL POLISHED SUMMARY:")
        print("-" * 35)
        print(result.get("final_summary", "Not available"))
        
        print(f"\nMETADATA:")
        metadata = result.get("metadata", {})
        print(f"- Document count: {metadata.get('document_count', 'N/A')}")
        print(f"- Workflow completed: {metadata.get('completed', 'N/A')}")
        print(f"- Model used: {result.get('model_used', 'N/A')}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def demo_file_upload():
    """Demonstrate file upload and summarization."""
    print("\n" + "=" * 60)
    print("FILE UPLOAD SUMMARIZATION DEMO")
    print("=" * 60)
    
    sample_file = Path(__file__).parent / "sample_text.txt"
    
    print(f"Uploading file: {sample_file.name}")
    
    with open(sample_file, "rb") as f:
        files = {"file": (sample_file.name, f, "text/plain")}
        data = {
            "summary_type": "advanced",
            "model_name": "gpt-3.5-turbo"
        }
        
        response = requests.post(f"{BASE_URL}/summarize/file", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print("\nFILE SUMMARY:")
        print("-" * 25)
        print(result["final_summary"])
        
        if result.get("key_points"):
            print("\nKEY POINTS:")
            print("-" * 20)
            for i, point in enumerate(result["key_points"], 1):
                print(f"{i}. {point}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def demo_custom_prompt():
    """Demonstrate custom prompt functionality via API."""
    print("\n" + "=" * 60)
    print("CUSTOM PROMPT DEMO")
    print("=" * 60)
    
    # Read sample text
    sample_file = Path(__file__).parent / "sample_text.txt"
    with open(sample_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    custom_prompt = """
    Create a bullet-point summary focusing on:
    1. Key technological developments
    2. Current applications
    3. Future challenges
    
    Format as clear bullet points with brief explanations.
    """
    
    payload = {
        "text": text,
        "summary_type": "basic",
        "custom_prompt": custom_prompt
    }
    
    print("Using custom prompt for bullet-point summary...")
    response = requests.post(f"{BASE_URL}/summarize/text", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("\nCUSTOM SUMMARY (Bullet Points):")
        print("-" * 40)
        print(result["summary"])
    else:
        print(f"Error: {response.status_code} - {response.text}")


def demo_api_info():
    """Demonstrate API information endpoints."""
    print("\n" + "=" * 60)
    print("API INFORMATION DEMO")
    print("=" * 60)
    
    # Health check
    print("1. HEALTH CHECK:")
    print("-" * 20)
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    
    # Configuration
    print("\n2. CONFIGURATION:")
    print("-" * 20)
    response = requests.get(f"{BASE_URL}/config")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    
    # Available models
    print("\n3. AVAILABLE MODELS:")
    print("-" * 20)
    response = requests.get(f"{BASE_URL}/models")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))


def main():
    """Run all API demonstrations."""
    print("🚀 TEXT SUMMARIZATION API DEMO")
    print("Using LangChain and LangGraph via FastAPI")
    print("=" * 60)
    
    # Check if server is running
    if not check_server_status():
        print("❌ API server is not running!")
        print("Please start the server first:")
        print("  python app.py")
        print("  or")
        print("  uvicorn app:app --reload")
        print("Also ensure PERPLEXITY_API_KEY is set in your .env file")
        return
    
    print("✅ API server is running!")
    
    try:
        # Run all demos
        demo_api_info()
        demo_basic_text_summarization()
        demo_advanced_summarization()
        demo_file_upload()
        demo_custom_prompt()
        
        print("\n" + "=" * 60)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("- Visit http://localhost:9000/docs for interactive API documentation")
        print("- Visit http://localhost:9000/redoc for alternative documentation")
        print("- Integrate the API into your applications")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
    except Exception as e:
        print(f"❌ Error during demo: {e}")


if __name__ == "__main__":
    main()
