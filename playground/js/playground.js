// ReservoirChat Playground JavaScript

class PlaygroundApp {
    constructor() {
        this.currentSection = 'chat';
        this.theme = localStorage.getItem('playground-theme') || 'light';
        this.knowledgeGraph = null;
        
        this.init();
    }
    
    init() {
        this.setupTheme();
        this.setupNavigation();
        this.setupInteractiveElements();
        this.setupKnowledgeGraph();
        this.setupConfigSliders();
        this.setupQueryAnalyzer();
    }
    
    setupTheme() {
        document.documentElement.setAttribute('data-theme', this.theme);
        
        const themeToggle = document.querySelector('.theme-toggle');
        themeToggle.addEventListener('click', () => {
            this.theme = this.theme === 'light' ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', this.theme);
            localStorage.setItem('playground-theme', this.theme);
        });
    }
    
    setupNavigation() {
        const navButtons = document.querySelectorAll('.nav-btn');
        const sections = document.querySelectorAll('.section');
        
        navButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetSection = button.getAttribute('data-section');
                
                // Update active button
                navButtons.forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
                
                // Update active section
                sections.forEach(section => section.classList.remove('active'));
                document.getElementById(`${targetSection}-section`).classList.add('active');
                
                this.currentSection = targetSection;
                
                // Trigger section-specific initialization
                this.onSectionChange(targetSection);
            });
        });
    }
    
    setupInteractiveElements() {
        // Add loading animation to chat iframe
        const chatIframe = document.querySelector('.chat-iframe');
        if (chatIframe) {
            chatIframe.addEventListener('load', () => {
                console.log('ReservoirChat interface loaded successfully');
            });
            
            chatIframe.addEventListener('error', () => {
                console.warn('Failed to load ReservoirChat interface');
                this.showChatFallback();
            });
        }
        
        // Setup feature card animations
        this.setupFeatureCards();
        
        // Setup timeline animations
        this.setupTimelineAnimations();
    }
    
    setupFeatureCards() {
        const featureCards = document.querySelectorAll('.feature-card');
        
        featureCards.forEach(card => {
            card.addEventListener('mouseenter', () => {
                card.style.transform = 'translateY(-8px) scale(1.02)';
            });
            
            card.addEventListener('mouseleave', () => {
                card.style.transform = 'translateY(0) scale(1)';
            });
        });
    }
    
    setupTimelineAnimations() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateX(0)';
                }
            });
        }, { threshold: 0.1 });
        
        document.querySelectorAll('.timeline-item').forEach(item => {
            item.style.opacity = '0';
            item.style.transform = 'translateX(-50px)';
            item.style.transition = 'all 0.6s ease';
            observer.observe(item);
        });
    }
    
    setupKnowledgeGraph() {
        const canvas = document.getElementById('knowledge-graph');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        this.knowledgeGraph = { canvas, ctx, nodes: [], connections: [] };
        
        // Setup graph buttons
        const graphButtons = document.querySelectorAll('.graph-btn');
        graphButtons.forEach(button => {
            button.addEventListener('click', () => {
                const topic = button.getAttribute('data-topic');
                this.renderKnowledgeGraph(topic);
                
                // Update active button
                graphButtons.forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
            });
        });
        
        // Initial graph render
        this.renderKnowledgeGraph('reservoir');
    }
    
    renderKnowledgeGraph(topic) {
        const { ctx, canvas } = this.knowledgeGraph;
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Define graph data based on topic
        const graphData = this.getGraphData(topic);
        
        // Draw connections first
        ctx.strokeStyle = this.theme === 'dark' ? '#404040' : '#dee2e6';
        ctx.lineWidth = 2;
        
        graphData.connections.forEach(connection => {
            const startNode = graphData.nodes[connection.from];
            const endNode = graphData.nodes[connection.to];
            
            ctx.beginPath();
            ctx.moveTo(startNode.x, startNode.y);
            ctx.lineTo(endNode.x, endNode.y);
            ctx.stroke();
        });
        
        // Draw nodes
        graphData.nodes.forEach(node => {
            // Node circle
            ctx.fillStyle = node.color || '#459db9';
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius || 20, 0, Math.PI * 2);
            ctx.fill();
            
            // Node border
            ctx.strokeStyle = this.theme === 'dark' ? '#ffffff' : '#333333';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // Node label
            ctx.fillStyle = this.theme === 'dark' ? '#ffffff' : '#333333';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(node.label, node.x, node.y + node.radius + 15);
        });
    }
    
    getGraphData(topic) {
        const graphs = {
            reservoir: {
                nodes: [
                    { x: 100, y: 100, label: 'Input Layer', color: '#28a745' },
                    { x: 300, y: 100, label: 'Reservoir', color: '#459db9' },
                    { x: 500, y: 100, label: 'Output Layer', color: '#dc3545' },
                    { x: 300, y: 200, label: 'Recurrent\nConnections', color: '#ffc107' },
                    { x: 200, y: 250, label: 'Echo State\nProperty', color: '#6f42c1' }
                ],
                connections: [
                    { from: 0, to: 1 },
                    { from: 1, to: 2 },
                    { from: 1, to: 3 },
                    { from: 3, to: 1 },
                    { from: 3, to: 4 }
                ]
            },
            atomspace: {
                nodes: [
                    { x: 150, y: 80, label: 'Atoms', color: '#459db9' },
                    { x: 300, y: 80, label: 'Links', color: '#28a745' },
                    { x: 450, y: 80, label: 'Types', color: '#dc3545' },
                    { x: 200, y: 180, label: 'Pattern\nMatcher', color: '#ffc107' },
                    { x: 400, y: 180, label: 'Inference\nEngine', color: '#6f42c1' },
                    { x: 300, y: 280, label: 'Knowledge\nBase', color: '#fd7e14' }
                ],
                connections: [
                    { from: 0, to: 1 },
                    { from: 1, to: 2 },
                    { from: 0, to: 3 },
                    { from: 2, to: 4 },
                    { from: 3, to: 5 },
                    { from: 4, to: 5 }
                ]
            },
            esn: {
                nodes: [
                    { x: 100, y: 120, label: 'Input\nWeights', color: '#28a745' },
                    { x: 300, y: 80, label: 'Reservoir\nMatrix', color: '#459db9' },
                    { x: 300, y: 200, label: 'Internal\nStates', color: '#ffc107' },
                    { x: 500, y: 120, label: 'Output\nWeights', color: '#dc3545' },
                    { x: 200, y: 300, label: 'Spectral\nRadius', color: '#6f42c1' }
                ],
                connections: [
                    { from: 0, to: 1 },
                    { from: 1, to: 2 },
                    { from: 2, to: 3 },
                    { from: 1, to: 4 },
                    { from: 4, to: 2 }
                ]
            }
        };
        
        return graphs[topic] || graphs.reservoir;
    }
    
    setupConfigSliders() {
        const sliders = document.querySelectorAll('.config-slider');
        
        sliders.forEach(slider => {
            const valueSpan = slider.parentElement.querySelector('.slider-value');
            
            slider.addEventListener('input', (e) => {
                valueSpan.textContent = e.target.value;
                this.onConfigChange(slider.name || 'unknown', e.target.value);
            });
        });
    }
    
    setupQueryAnalyzer() {
        const analyzeBtn = document.querySelector('.analyze-btn');
        const queryInput = document.querySelector('.query-input');
        const analysisOutput = document.querySelector('.analysis-output');
        
        if (analyzeBtn && queryInput && analysisOutput) {
            analyzeBtn.addEventListener('click', () => {
                const query = queryInput.value.trim();
                if (!query) {
                    analysisOutput.innerHTML = '<p><em>Please enter a query to analyze.</em></p>';
                    return;
                }
                
                this.analyzeQuery(query, analysisOutput);
            });
            
            // Allow Enter key to trigger analysis
            queryInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && e.ctrlKey) {
                    analyzeBtn.click();
                }
            });
        }
    }
    
    analyzeQuery(query, outputElement) {
        // Simulate query analysis
        outputElement.innerHTML = '<p><em>Analyzing query...</em></p>';
        
        setTimeout(() => {
            const analysis = this.performQueryAnalysis(query);
            outputElement.innerHTML = analysis;
        }, 1000);
    }
    
    performQueryAnalysis(query) {
        const keywords = {
            'reservoir': '🌊 Reservoir Computing',
            'echo state': '🔄 Echo State Networks',
            'atomspace': '🧠 OpenCog AtomSpace',
            'neural': '🧠 Neural Networks',
            'training': '📚 Learning/Training',
            'python': '🐍 Python Programming',
            'c++': '⚡ C++ Programming',
            'matrix': '📊 Matrix Operations',
            'temporal': '⏰ Temporal Processing'
        };
        
        const foundKeywords = [];
        const lowercaseQuery = query.toLowerCase();
        
        Object.entries(keywords).forEach(([key, value]) => {
            if (lowercaseQuery.includes(key)) {
                foundKeywords.push(value);
            }
        });
        
        const complexity = query.split(' ').length > 10 ? 'High' : 
                          query.split(' ').length > 5 ? 'Medium' : 'Low';
        
        const hasQuestionWords = /\b(what|how|why|when|where|which|who)\b/i.test(query);
        const queryType = hasQuestionWords ? 'Question' : 'Statement';
        
        return `
            <div class="analysis-result">
                <h4>📊 Query Analysis Results</h4>
                <div class="analysis-grid">
                    <div class="analysis-item">
                        <strong>Type:</strong> ${queryType}
                    </div>
                    <div class="analysis-item">
                        <strong>Complexity:</strong> ${complexity}
                    </div>
                    <div class="analysis-item">
                        <strong>Word Count:</strong> ${query.split(' ').length}
                    </div>
                    <div class="analysis-item">
                        <strong>Detected Topics:</strong><br>
                        ${foundKeywords.length > 0 ? foundKeywords.join('<br>') : 'General inquiry'}
                    </div>
                </div>
                <div class="analysis-recommendation">
                    <strong>💡 Recommendation:</strong> 
                    ${this.getQueryRecommendation(foundKeywords, complexity, hasQuestionWords)}
                </div>
            </div>
        `;
    }
    
    getQueryRecommendation(keywords, complexity, isQuestion) {
        if (keywords.length === 0) {
            return "Try including specific reservoir computing terms like 'reservoir', 'echo state', or 'AtomSpace' for better results.";
        }
        
        if (complexity === 'High') {
            return "Complex query detected. Consider breaking it into smaller, focused questions for clearer responses.";
        }
        
        if (!isQuestion) {
            return "Consider rephrasing as a question (What, How, Why) to get more detailed explanations.";
        }
        
        return "Well-structured query! This should produce comprehensive results from ReservoirChat.";
    }
    
    onSectionChange(section) {
        switch (section) {
            case 'chat':
                console.log('Switched to Chat section');
                break;
            case 'features':
                console.log('Switched to Features section');
                this.animateFeatureCards();
                break;
            case 'explore':
                console.log('Switched to Explore section');
                if (this.knowledgeGraph) {
                    this.renderKnowledgeGraph('reservoir');
                }
                break;
            case 'future':
                console.log('Switched to Future section');
                this.animateTimeline();
                break;
        }
    }
    
    animateFeatureCards() {
        const cards = document.querySelectorAll('.feature-card');
        cards.forEach((card, index) => {
            setTimeout(() => {
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, index * 100);
        });
    }
    
    animateTimeline() {
        const items = document.querySelectorAll('.timeline-item');
        items.forEach((item, index) => {
            setTimeout(() => {
                item.style.opacity = '1';
                item.style.transform = 'translateX(0)';
            }, index * 200);
        });
    }
    
    onConfigChange(parameter, value) {
        console.log(`Configuration changed: ${parameter} = ${value}`);
        
        // Here you could implement real-time configuration updates
        // For now, we'll just log the changes
        
        // Example: Update GraphRAG context window
        if (parameter === 'context-window') {
            console.log(`GraphRAG context window updated to ${value} tokens`);
        }
        
        // Example: Update Codestral temperature
        if (parameter === 'temperature') {
            console.log(`Codestral temperature updated to ${value}`);
        }
    }
    
    showChatFallback() {
        const chatContainer = document.querySelector('.chat-iframe-container');
        if (chatContainer) {
            chatContainer.innerHTML = `
                <div class="chat-fallback">
                    <div class="fallback-content">
                        <h3>🔗 ReservoirChat Access</h3>
                        <p>The embedded chat interface is temporarily unavailable.</p>
                        <a href="https://chat.reservoirpy.inria.fr/" target="_blank" class="chat-link-btn">
                            🚀 Open ReservoirChat in New Tab
                        </a>
                        <div class="fallback-info">
                            <h4>In the meantime, you can explore:</h4>
                            <ul>
                                <li>Technical features documentation</li>
                                <li>Interactive exploration tools</li>
                                <li>Future development roadmap</li>
                            </ul>
                        </div>
                    </div>
                </div>
            `;
        }
    }
}

// CSS for analysis results (injected dynamically)
const analysisCSS = `
    .analysis-result {
        background: var(--bg-color);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid var(--border-color);
    }
    
    .analysis-result h4 {
        color: var(--secondary-color);
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    .analysis-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .analysis-item {
        background: var(--bg-secondary);
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: 1px solid var(--border-color);
    }
    
    .analysis-item strong {
        color: var(--primary-color);
    }
    
    .analysis-recommendation {
        background: var(--bg-secondary);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid var(--primary-color);
        margin-top: 1rem;
    }
    
    .chat-fallback {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        padding: 2rem;
        text-align: center;
    }
    
    .fallback-content h3 {
        color: var(--secondary-color);
        margin-bottom: 1rem;
    }
    
    .chat-link-btn {
        display: inline-block;
        background: var(--primary-color);
        color: white;
        padding: 1rem 2rem;
        border-radius: 0.5rem;
        text-decoration: none;
        font-weight: 600;
        margin: 1rem 0;
        transition: var(--transition);
    }
    
    .chat-link-btn:hover {
        background: var(--primary-hover);
        transform: translateY(-2px);
    }
    
    .fallback-info {
        margin-top: 2rem;
        text-align: left;
        max-width: 300px;
    }
    
    .fallback-info ul {
        list-style: none;
        padding: 0;
    }
    
    .fallback-info li {
        padding: 0.5rem 0;
        color: var(--text-light);
    }
    
    .fallback-info li::before {
        content: '▶';
        color: var(--primary-color);
        margin-right: 0.5rem;
    }
`;

// Inject the CSS
const style = document.createElement('style');
style.textContent = analysisCSS;
document.head.appendChild(style);

// Initialize the playground when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new PlaygroundApp();
});

// Export for testing/debugging
window.PlaygroundApp = PlaygroundApp;