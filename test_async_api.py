"""Test script for async LangGraph API endpoints."""

import requests
import time
import json
from typing import Dict, Any

BASE_URL = "http://localhost:9000"

def test_async_text_summarization():
    """Test async text summarization workflow."""
    print("🚀 Testing Async Text Summarization")
    print("=" * 50)
    
    # Step 1: Start async summarization
    print("1. Starting async text summarization...")
    
    text = """
    Artificial Intelligence (AI) has revolutionized numerous industries and continues to shape our daily lives. 
    From machine learning algorithms that power recommendation systems to natural language processing that 
    enables chatbots and virtual assistants, AI technologies are becoming increasingly sophisticated.
    
    The healthcare industry has particularly benefited from AI advancements. Medical imaging analysis, 
    drug discovery, and personalized treatment plans are just a few areas where AI is making significant 
    contributions. AI-powered diagnostic tools can detect diseases earlier and more accurately than 
    traditional methods, potentially saving countless lives.
    
    In the business world, AI is transforming operations through automation, predictive analytics, 
    and intelligent decision-making systems. Companies are using AI to optimize supply chains, 
    improve customer service, and enhance product development processes.
    
    However, the rapid advancement of AI also brings challenges. Ethical considerations around 
    privacy, bias, and job displacement are critical issues that need to be addressed. 
    As AI systems become more powerful, ensuring they are used responsibly and fairly becomes 
    increasingly important.
    
    The future of AI holds immense promise, but it requires careful navigation of both 
    opportunities and risks to ensure beneficial outcomes for society as a whole.
    """
    
    request_data = {
        "text": text,
        "summary_type": "advanced",
        "model_name": "llama-3.1-sonet-large-128k-online"
    }
    
    response = requests.post(f"{BASE_URL}/summarize/async/text", json=request_data)
    
    if response.status_code == 200:
        result = response.json()
        thread_id = result["thread_id"]
        print(f"✅ Task started successfully!")
        print(f"   Thread ID: {thread_id}")
        print(f"   Status: {result['status']}")
        print(f"   Message: {result['message']}")
        print(f"   Created: {result['created_at']}")
        print(f"   Estimated Duration: {result['estimated_duration']}")
    else:
        print(f"❌ Failed to start task: {response.text}")
        return
    
    # Step 2: Monitor task status
    print(f"\n2. Monitoring task status...")
    print("-" * 30)
    
    max_attempts = 30  # 5 minutes max
    attempt = 0
    
    while attempt < max_attempts:
        status_response = requests.get(f"{BASE_URL}/task/status/{thread_id}")
        
        if status_response.status_code == 200:
            status_data = status_response.json()
            status = status_data["status"]
            updated_at = status_data["updated_at"]
            
            print(f"   Attempt {attempt + 1}: Status = {status} (Updated: {updated_at})")
            
            if status == "completed":
                print(f"\n🎉 Task completed successfully!")
                print(f"   Final Summary: {status_data['result']['final_summary'][:200]}...")
                print(f"   Key Points: {len(status_data['result']['key_points'])} points extracted")
                print(f"   Completed at: {status_data['completed_at']}")
                break
            elif status == "failed":
                print(f"\n❌ Task failed: {status_data.get('error', 'Unknown error')}")
                break
            elif status == "running":
                # Show workflow state if available
                if status_data.get('workflow_state'):
                    workflow_state = status_data['workflow_state']
                    print(f"   Workflow State: {workflow_state}")
        
        else:
            print(f"   Failed to get status: {status_response.text}")
        
        time.sleep(10)  # Wait 10 seconds
        attempt += 1
    
    if attempt >= max_attempts:
        print(f"\n⏰ Timeout: Task did not complete within 5 minutes")
    
    return thread_id

def test_get_all_tasks():
    """Test getting all tasks."""
    print(f"\n📋 Getting all tasks...")
    print("-" * 30)
    
    response = requests.get(f"{BASE_URL}/tasks/all")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Total tasks: {result['total_tasks']}")
        
        for thread_id, task in result['tasks'].items():
            print(f"   {thread_id}: {task['status']} ({task['task_type']})")
    else:
        print(f"❌ Failed to get tasks: {response.text}")

def test_workflow_state_detailed(thread_id: str):
    """Test getting detailed workflow state."""
    print(f"\n🔍 Getting detailed workflow state for {thread_id}...")
    print("-" * 50)
    
    response = requests.get(f"{BASE_URL}/task/status/{thread_id}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Task Status: {result['status']}")
        print(f"   Task Type: {result['task_type']}")
        print(f"   Created: {result['created_at']}")
        print(f"   Updated: {result['updated_at']}")
        
        if result.get('completed_at'):
            print(f"   Completed: {result['completed_at']}")
        
        if result.get('result'):
            result_data = result['result']
            print(f"\n📊 Results:")
            print(f"   Initial Summary: {result_data.get('initial_summary', 'N/A')[:100]}...")
            print(f"   Refined Summary: {result_data.get('refined_summary', 'N/A')[:100]}...")
            print(f"   Final Summary: {result_data.get('final_summary', 'N/A')[:100]}...")
            print(f"   Key Points: {len(result_data.get('key_points', []))} points")
            
            if result_data.get('metadata'):
                metadata = result_data['metadata']
                print(f"   Metadata: {metadata}")
        
        if result.get('workflow_state'):
            print(f"\n🔄 Workflow State: {result['workflow_state']}")
        
        if result.get('error'):
            print(f"\n❌ Error: {result['error']}")
    
    else:
        print(f"❌ Failed to get detailed status: {response.text}")

def main():
    """Main test function."""
    print("🧪 Async LangGraph API Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Async text summarization
        thread_id = test_async_text_summarization()
        
        # Test 2: Get all tasks
        test_get_all_tasks()
        
        # Test 3: Get detailed workflow state
        if thread_id:
            test_workflow_state_detailed(thread_id)
        
        print(f"\n✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to server. Make sure the server is running on {BASE_URL}")
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")

if __name__ == "__main__":
    main()
