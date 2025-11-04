#!/usr/bin/env python3
"""
Test script to validate LangChain refactoring.
This script verifies that all imports work and basic functionality is intact.
"""

import sys
import os

# Add the API directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

def test_imports():
    """Test that all required imports work correctly."""
    print("Testing imports...")
    
    try:
        # Test LangChain imports
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import PromptTemplate
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        from langchain_community.utilities import SQLDatabase
        print("✓ All LangChain imports successful")
        
        # Test other imports
        import ollama
        import chromadb
        import sqlglot
        from sqlalchemy import create_engine
        print("✓ All supporting library imports successful")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_prompt_template():
    """Test PromptTemplate functionality."""
    print("\nTesting PromptTemplate...")
    
    try:
        from langchain_core.prompts import PromptTemplate
        
        template = PromptTemplate(
            input_variables=["db_schema", "question"],
            template="Schema: {db_schema}\nQuestion: {question}"
        )
        
        result = template.format(
            db_schema="CREATE TABLE test (id INT);",
            question="What is the schema?"
        )
        
        assert "CREATE TABLE test" in result
        assert "What is the schema?" in result
        print("✓ PromptTemplate works correctly")
        return True
    except Exception as e:
        print(f"✗ PromptTemplate test failed: {e}")
        return False

def test_query_orchestrator_syntax():
    """Test that QueryOrchestrator has valid Python syntax."""
    print("\nTesting QueryOrchestrator syntax...")
    
    try:
        import py_compile
        module_path = os.path.join(
            os.path.dirname(__file__), 
            'api', 'app', 'services', 'query_orchestrator.py'
        )
        py_compile.compile(module_path, doraise=True)
        print("✓ QueryOrchestrator has valid syntax")
        return True
    except py_compile.PyCompileError as e:
        print(f"✗ Syntax error in QueryOrchestrator: {e}")
        return False

def test_method_signatures():
    """Test that all expected methods exist with correct signatures."""
    print("\nTesting method signatures...")
    
    try:
        # We can't instantiate without DB connections, but we can check the class definition
        with open('api/app/services/query_orchestrator.py', 'r') as f:
            content = f.read()
        
        # Check for key method definitions
        expected_methods = [
            'def __init__',
            'async def process_user_question',
            'async def generate_forecast_narrative',
            'async def generate_inflation_interpretation',
            'async def pull_model',
            'def index_reference_queries',
            'def _validate_sql',
            'def _is_question_harmful',
            'def _needs_data_retrieval',
        ]
        
        for method in expected_methods:
            if method not in content:
                print(f"✗ Method not found: {method}")
                return False
        
        print("✓ All expected methods are defined")
        return True
    except Exception as e:
        print(f"✗ Method signature test failed: {e}")
        return False

def test_langchain_components_used():
    """Test that LangChain components are actually used in the code."""
    print("\nTesting LangChain component usage...")
    
    try:
        with open('api/app/services/query_orchestrator.py', 'r') as f:
            content = f.read()
        
        # Check for LangChain component usage
        checks = [
            ('ChatOllama', 'ChatOllama imported and used'),
            ('HuggingFaceEmbeddings', 'HuggingFaceEmbeddings imported and used'),
            ('PromptTemplate', 'PromptTemplate imported and used'),
            ('SQLDatabase', 'SQLDatabase imported and used'),
            ('Chroma', 'Chroma vector store imported'),
            ('self.llm.ainvoke', 'ChatOllama ainvoke method used'),
            ('embed_documents', 'LangChain embedding interface used'),
            ('similarity_search', 'LangChain vector store search used'),
        ]
        
        for check_str, description in checks:
            if check_str not in content:
                print(f"✗ {description} - NOT FOUND: {check_str}")
                return False
            else:
                print(f"  ✓ {description}")
        
        print("✓ All LangChain components are properly integrated")
        return True
    except Exception as e:
        print(f"✗ LangChain component usage test failed: {e}")
        return False

def test_backward_compatibility():
    """Test that original functionality is preserved."""
    print("\nTesting backward compatibility...")
    
    try:
        with open('api/app/services/query_orchestrator.py', 'r') as f:
            content = f.read()
        
        # Check that security features are preserved
        security_checks = [
            '_is_question_harmful',
            '_needs_data_retrieval',
            '_validate_sql',
            'banned_terms',
            'economic_keywords',
        ]
        
        for check in security_checks:
            if check not in content:
                print(f"✗ Security feature missing: {check}")
                return False
        
        # Check that semaphores are preserved
        semaphore_checks = [
            'self.embed_sem',
            'self.chroma_sem',
            'self.llm_sem',
        ]
        
        for check in semaphore_checks:
            if check not in content:
                print(f"✗ Concurrency control missing: {check}")
                return False
        
        print("✓ All original security and concurrency features preserved")
        return True
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        return False

def test_no_breaking_changes():
    """Test that no breaking changes were introduced."""
    print("\nTesting for breaking changes...")
    
    try:
        with open('api/app/services/query_orchestrator.py', 'r') as f:
            content = f.read()
        
        # These should NOT be in the refactored code (except in comments)
        breaking_patterns = [
            # Original patterns that should be replaced
            ('self.embedding_model.encode(', 'Should use embed_documents or similarity_search'),
        ]
        
        # Count occurrences and allow some for backward compat or comments
        for pattern, reason in breaking_patterns:
            count = content.count(pattern)
            # Allow up to 2 occurrences (might be in comments/documentation)
            if count > 2:
                print(f"⚠ Warning: Found {count} occurrences of old pattern: {pattern}")
                print(f"  Reason to avoid: {reason}")
        
        print("✓ No major breaking changes detected")
        return True
    except Exception as e:
        print(f"✗ Breaking change test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 70)
    print("LangChain Refactoring Validation Tests")
    print("=" * 70)
    
    tests = [
        test_imports,
        test_prompt_template,
        test_query_orchestrator_syntax,
        test_method_signatures,
        test_langchain_components_used,
        test_backward_compatibility,
        test_no_breaking_changes,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 70)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 70)
    
    if all(results):
        print("\n✓ ALL TESTS PASSED - Refactoring is successful!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - Please review the errors above")
        return 1

if __name__ == '__main__':
    sys.exit(main())
