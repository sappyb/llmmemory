#!/usr/bin/env python3
"""
Test script for Evelyn AI - Student Simulation System
"""
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test if all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from core import (
            load_config, setup_logger, get_logger,
            RAGProcessor, EvelynAIServer,
            load_student_config, StudentPersonaManager,
            StudentDataProcessor, ResponseGenerator,
            StudentSimulatorServer
        )
        print("✅ Core modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import core modules: {e}")
        return False
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError:
        print("⚠️  Streamlit not available (web interface won't work)")
    
    try:
        import openai
        print("✅ OpenAI imported successfully")
    except ImportError:
        print("⚠️  OpenAI not available (LLM features won't work)")
    
    try:
        import faiss
        print("✅ FAISS imported successfully")
    except ImportError:
        print("⚠️  FAISS not available (vector search won't work)")
    
    return True

def test_configuration():
    """Test configuration loading"""
    print("\n🔧 Testing configuration...")
    
    try:
        from core import load_config, load_student_config
        
        # Test general config
        config = load_config()
        print("✅ General configuration loaded")
        
        # Test student config
        student_config = load_student_config()
        print("✅ Student simulator configuration loaded")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_persona_manager():
    """Test persona manager"""
    print("\n🎭 Testing persona manager...")
    
    try:
        from core import StudentPersonaManager
        
        manager = StudentPersonaManager()
        personas = manager.list_personas()
        
        print(f"✅ Found {len(personas)} personas")
        
        # Test specific persona
        persona = manager.get_persona("HED")
        if persona:
            print(f"✅ Persona HED: {persona.description}")
        else:
            print("❌ Failed to get persona HED")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Persona manager test failed: {e}")
        return False

def test_data_processor():
    """Test data processor"""
    print("\n📊 Testing data processor...")
    
    try:
        from core import StudentDataProcessor, load_student_config
        
        config = load_student_config()
        processor = StudentDataProcessor(config)
        
        # Check if data file exists
        data_path = Path(config.data.data_path)
        if data_path.exists():
            print(f"✅ Data file found: {data_path}")
        else:
            print(f"⚠️  Data file not found: {data_path}")
            print("   Some features may not work without training data")
        
        return True
    except Exception as e:
        print(f"❌ Data processor test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Evelyn AI - Test Suite")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_configuration,
        test_persona_manager,
        test_data_processor
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Evelyn AI is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
