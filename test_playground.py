#!/usr/bin/env python3
"""
Simple test script to validate playground files and structure
"""

import os
import sys
import webbrowser
from pathlib import Path

def test_playground_structure():
    """Test that all playground files exist and have content"""
    playground_dir = Path("playground")
    
    required_files = [
        "index.html",
        "css/playground.css", 
        "js/playground.js",
        "README.md"
    ]
    
    print("🔍 Testing playground structure...")
    
    for file_path in required_files:
        full_path = playground_dir / file_path
        if not full_path.exists():
            print(f"❌ Missing file: {file_path}")
            return False
        
        if full_path.stat().st_size == 0:
            print(f"❌ Empty file: {file_path}")
            return False
            
        print(f"✅ Found: {file_path}")
    
    return True

def test_html_structure():
    """Test HTML file for basic structure"""
    html_file = Path("playground/index.html")
    
    print("\n🔍 Testing HTML structure...")
    
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    required_elements = [
        "<!DOCTYPE html>",
        '<title>ReservoirChat Playground',
        'class="playground-header"',
        'class="playground-nav"',
        'data-section="chat"',
        'data-section="features"', 
        'data-section="explore"',
        'data-section="future"',
        'iframe',
        'https://chat.reservoirpy.inria.fr',
        'playground.css',
        'playground.js'
    ]
    
    for element in required_elements:
        if element not in content:
            print(f"❌ Missing HTML element: {element}")
            return False
        print(f"✅ Found: {element}")
    
    return True

def test_css_structure():
    """Test CSS file for basic styling"""
    css_file = Path("playground/css/playground.css")
    
    print("\n🔍 Testing CSS structure...")
    
    with open(css_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    required_styles = [
        ":root",
        "--primary-color",
        ".playground-header",
        ".playground-nav",
        ".section",
        ".chat-iframe",
        ".feature-card",
        "@media",
        "transition"
    ]
    
    for style in required_styles:
        if style not in content:
            print(f"❌ Missing CSS element: {style}")
            return False
        print(f"✅ Found: {style}")
    
    return True

def test_js_structure():
    """Test JavaScript file for basic functionality"""
    js_file = Path("playground/js/playground.js")
    
    print("\n🔍 Testing JavaScript structure...")
    
    with open(js_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    required_functions = [
        "class PlaygroundApp",
        "constructor()",
        "setupTheme()",
        "setupNavigation()",
        "setupInteractiveElements()",
        "setupKnowledgeGraph()", 
        "renderKnowledgeGraph",
        "analyzeQuery",
        "addEventListener"
    ]
    
    for func in required_functions:
        if func not in content:
            print(f"❌ Missing JS element: {func}")
            return False
        print(f"✅ Found: {func}")
    
    return True

def open_playground():
    """Open playground in default browser"""
    playground_file = Path("playground/index.html").absolute()
    if playground_file.exists():
        print(f"\n🚀 Opening playground in browser...")
        print(f"File: {playground_file}")
        webbrowser.open(f"file://{playground_file}")
        return True
    return False

def main():
    """Run all tests"""
    print("🎉 ReservoirChat Playground Test Suite")
    print("=" * 50)
    
    # Change to repository root
    os.chdir(Path(__file__).parent)
    
    tests = [
        test_playground_structure,
        test_html_structure,
        test_css_structure,
        test_js_structure
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test failed: {test.__name__}")
        except Exception as e:
            print(f"❌ Test error in {test.__name__}: {e}")
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Playground is ready.")
        
        # Ask if user wants to open playground
        try:
            response = input("\n🌐 Open playground in browser? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                open_playground()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())