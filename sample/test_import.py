#!/usr/bin/env python3
"""
Test importing dab-evaluation from TestPyPI
"""

print("🧪 Testing DAB Evaluation Import from TestPyPI")
print("=" * 50)

try:
    # Test basic import
    print("1. Testing basic import...")
    import dab_eval
    print(f"   ✅ Successfully imported dab_eval v{dab_eval.__version__}")
    
    # Test importing main classes
    print("2. Testing class imports...")
    from dab_eval import DABEvaluator, AgentMetadata, TaskCategory, EvaluationMethod
    print("   ✅ Successfully imported main classes")
    
    # Test importing evaluation modules
    print("3. Testing evaluation module imports...")
    from dab_eval import BaseEvaluator, LLMEvaluator, HybridEvaluator
    print("   ✅ Successfully imported evaluation modules")
    
    # Test creating evaluator instance
    print("4. Testing evaluator instantiation...")
    llm_config = {
        "model": "gpt-4",
        "base_url": "https://api.openai.com/v1",
        "api_key": "test-key",
        "temperature": 0.3,
        "max_tokens": 2000
    }
    
    evaluator = DABEvaluator(llm_config, "test_output")
    print("   ✅ Successfully created DABEvaluator instance")
    
    # Test creating agent metadata
    print("5. Testing agent metadata creation...")
    agent = AgentMetadata(
        url="http://localhost:8000",
        capabilities=[TaskCategory.WEB_RETRIEVAL],
        timeout=30
    )
    print("   ✅ Successfully created AgentMetadata")
    
    # Test enum values
    print("6. Testing enum values...")
    print(f"   TaskCategory.WEB_RETRIEVAL: {TaskCategory.WEB_RETRIEVAL.value}")
    print(f"   EvaluationMethod.HYBRID: {EvaluationMethod.HYBRID.value}")
    print("   ✅ Enums working correctly")
    
    print("\n🎉 All tests passed! DAB Evaluation is working correctly!")
    print(f"📦 Package version: {dab_eval.__version__}")
    print(f"👥 Author: {dab_eval.__author__}")
    print(f"📧 Email: {dab_eval.__email__}")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("💡 Make sure to install from TestPyPI:")
    print("   pip install -i https://test.pypi.org/simple/ dab-evaluation")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
