#!/bin/bash

# ReservoirCogs Project Setup Script
# This script helps set up the GitHub Project configuration for ReservoirCogs

echo "🚀 ReservoirCogs Project Setup"
echo "=============================="
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: This script must be run from within the ReservoirCogs git repository"
    exit 1
fi

# Check if GitHub CLI is available
if ! command -v gh &> /dev/null; then
    echo "⚠️  GitHub CLI (gh) is not installed. Some features will be limited."
    echo "   Install it from: https://cli.github.com/"
    echo ""
fi

echo "📋 Available Project Setup Options:"
echo "1. View project configuration"
echo "2. Create project labels (requires GitHub CLI)"
echo "3. Create project milestones (requires GitHub CLI)"
echo "4. Show project roadmap"
echo "5. Validate project structure"
echo ""

read -p "Select an option (1-5): " choice

case $choice in
    1)
        echo ""
        echo "📄 Project Configuration Overview:"
        echo "=================================="
        if [ -f ".github/PROJECT_CONFIG.yml" ]; then
            echo "✅ Project configuration file found"
            echo "📁 Location: .github/PROJECT_CONFIG.yml"
            echo ""
            echo "🏷️  Configured Labels:"
            grep -A 1 "name:" .github/PROJECT_CONFIG.yml | grep "name:" | head -10
            echo ""
            echo "📊 Configured Views:"
            grep -A 2 "name:" .github/PROJECT_CONFIG.yml | grep "name:" | grep -E "(All Items|Technical Features|Future Roadmap|Priority Dashboard|Research Track)"
        else
            echo "❌ Project configuration file not found"
        fi
        ;;
    2)
        if command -v gh &> /dev/null; then
            echo ""
            echo "🏷️  Creating project labels..."
            gh workflow run setup-project.yml
            echo "✅ Project setup workflow triggered"
            echo "   Check the Actions tab in GitHub to see the progress"
        else
            echo "❌ GitHub CLI is required for this operation"
        fi
        ;;
    3)
        if command -v gh &> /dev/null; then
            echo ""
            echo "📊 Creating project milestones..."
            gh workflow run setup-project.yml
            echo "✅ Project setup workflow triggered"
            echo "   Milestones will be created automatically"
        else
            echo "❌ GitHub CLI is required for this operation"
        fi
        ;;
    4)
        echo ""
        echo "🗺️  Project Roadmap:"
        echo "=================="
        if [ -f ".github/PROJECT_ROADMAP.md" ]; then
            echo "📍 Development roadmap available at: .github/PROJECT_ROADMAP.md"
            echo ""
            echo "📋 Current Technical Features:"
            grep -A 1 "###.*GraphRAG\|###.*Codestral\|###.*AtomSpace\|###.*Hybrid" .github/PROJECT_ROADMAP.md | head -8
            echo ""
            echo "🚀 Future Development Features:"
            grep -A 1 "###.*P-Systems\|###.*B-Series\|###.*J-Surface\|###.*Differential" .github/PROJECT_ROADMAP.md | head -8
        else
            echo "❌ Project roadmap not found"
        fi
        ;;
    5)
        echo ""
        echo "🔍 Validating Project Structure:"
        echo "==============================="
        
        # Check for required files
        files=(
            ".github/PROJECT_ROADMAP.md"
            ".github/PROJECT_CONFIG.yml"
            ".github/ISSUE_TEMPLATE/config.yml"
            ".github/ISSUE_TEMPLATE/feature-technical.yml"
            ".github/ISSUE_TEMPLATE/feature-future.yml"
            ".github/ISSUE_TEMPLATE/bug-report.yml"
            ".github/ISSUE_TEMPLATE/documentation.yml"
            ".github/workflows/project-automation.yml"
            ".github/workflows/setup-project.yml"
        )
        
        echo "📁 Required Files:"
        for file in "${files[@]}"; do
            if [ -f "$file" ]; then
                echo "✅ $file"
            else
                echo "❌ $file (missing)"
            fi
        done
        
        echo ""
        echo "🔗 Integration Points:"
        if grep -q "GitHub Project" README.md; then
            echo "✅ README.md contains project links"
        else
            echo "❌ README.md missing project links"
        fi
        
        if [ -f "playground/README.md" ] && grep -q "Future Development Roadmap" playground/README.md; then
            echo "✅ Playground documentation includes roadmap"
        else
            echo "⚠️  Playground documentation could include more roadmap details"
        fi
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo ""
echo "📚 Additional Resources:"
echo "========================"
echo "🔗 GitHub Project: https://github.com/orgs/HyperCogWizard/projects/1"
echo "📖 Project Roadmap: .github/PROJECT_ROADMAP.md"
echo "🎮 Interactive Playground: playground/index.html"
echo "💬 ReservoirChat: https://chat.reservoirpy.inria.fr"
echo ""
echo "✨ Project setup completed!"