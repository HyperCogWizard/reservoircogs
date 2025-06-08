# ReservoirChat Playground Guide

## Overview

The ReservoirChat Playground is an interactive web-based platform that provides a comprehensive exploration environment for ReservoirChat's capabilities. It integrates the chat interface with technical documentation, interactive tools, and future development insights.

## Accessing the Playground

The playground can be accessed through several methods:

1. **Direct Access**: Open `playground/index.html` in any modern web browser
2. **Local Server**: Use a local HTTP server for development
3. **Documentation Link**: Available from the main README

## Features

### Chat Interface

The playground provides direct access to ReservoirChat through an embedded iframe:

- **Real-time interaction** with the AI model
- **Contextual suggestions** for effective queries
- **Fallback options** if the embedded interface is unavailable

### Technical Features Documentation

Comprehensive documentation of the underlying technologies:

#### GraphRAG Integration
- **Knowledge Graph Construction**: How reservoir computing concepts are represented
- **Retrieval Mechanisms**: Query processing and context retrieval
- **Response Generation**: How retrieved knowledge enhances AI responses

#### Codestral AI Engine
- **Model Architecture**: Specialized language model for technical domains
- **Code Generation**: Python and C++ code generation capabilities
- **Technical Explanations**: Mathematical formulation and algorithm explanations

#### AtomSpace Intelligence
- **Symbolic Representation**: How reservoir states map to symbolic concepts
- **Inference Engine**: Logical reasoning over temporal patterns
- **Knowledge Extraction**: Pattern learning and concept formation

### Interactive Exploration Tools

#### Query Analysis Tool
```html
<!-- Example usage in playground -->
<textarea class="query-input" placeholder="Enter a question..."></textarea>
<button class="analyze-btn">Analyze Query Structure</button>
```

Features:
- **Complexity Analysis**: Word count, structure assessment
- **Topic Detection**: Identification of reservoir computing concepts
- **Query Optimization**: Suggestions for better formulation

#### Knowledge Graph Visualization

Interactive canvas-based visualization:

```javascript
// Example topic rendering
renderKnowledgeGraph('reservoir');  // Reservoir computing concepts
renderKnowledgeGraph('atomspace');  // AtomSpace components
renderKnowledgeGraph('esn');        // Echo State Network structure
```

#### Configuration Explorer

Real-time parameter adjustment interface:

- **GraphRAG Settings**: Context window, graph depth
- **Codestral Parameters**: Temperature, max tokens
- **Visual Feedback**: Immediate parameter impact visualization

## Future Development Integration

### Planned Features

#### P-Systems Membrane Computing
Implementation roadmap for bio-inspired computational models:

```python
# Future API example
from reservoirpy.psystems import MembraneComputing

membrane_reservoir = MembraneComputing(
    membranes=3,
    p_lingua_rules="rules.pl",
    hierarchical=True
)
```

#### B-Series Mathematical Framework
Advanced optimization using mathematical series:

```python
# Future mathematical framework
from reservoirpy.optimization import BSeries

optimizer = BSeries(
    method="runge_kutta",
    order=4,
    ridge_regression=True
)
```

#### Julia Integration
Differential equation solving with Julia backend:

```python
# Julia Differential ESN - NOW AVAILABLE!
from reservoirpy.compat import JuliaDifferentialESN

# Create differential ESN with Julia backend
model = JuliaDifferentialESN(
    n_reservoir=100,
    spectral_radius=0.95,
    solver="Tsit5",  # Uses DifferentialEquations.jl
    dt=0.01
)

# Initialize and train
model.initialize(dim_in=3, dim_out=1) 
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

# Automatic fallback to Python when Julia unavailable
print(f"Using Julia: {model.julia_available}")
```

## Usage Examples

### Basic Navigation

```javascript
// Programmatic section switching
playground.switchSection('features');
playground.switchSection('explore');
```

### Theme Management

```javascript
// Theme toggle implementation
playground.toggleTheme();  // Switches between light/dark
playground.setTheme('dark');  // Explicit theme setting
```

### Interactive Elements

```javascript
// Query analysis
playground.analyzeQuery("How do echo state networks work?");

// Graph visualization
playground.renderGraph('reservoir', {
    animation: true,
    highlight: ['input', 'reservoir', 'output']
});
```

## Technical Implementation

### Architecture

```
playground/
├── index.html              # Main interface structure
├── css/
│   └── playground.css      # Responsive styling system
├── js/
│   └── playground.js       # Interactive functionality
└── README.md              # Comprehensive documentation
```

### Key Components

#### PlaygroundApp Class
```javascript
class PlaygroundApp {
    constructor() {
        this.currentSection = 'chat';
        this.theme = localStorage.getItem('theme') || 'light';
        this.knowledgeGraph = null;
    }
    
    init() {
        this.setupTheme();
        this.setupNavigation();
        this.setupInteractiveElements();
    }
}
```

#### Theme System
```css
:root {
    --primary-color: #459db9;
    --secondary-color: #5b1e1e;
    --text-color: #333;
    --bg-color: #ffffff;
}

[data-theme="dark"] {
    --text-color: #e9ecef;
    --bg-color: #1a1a1a;
}
```

### Responsive Design

The playground implements mobile-first responsive design:

```css
@media (max-width: 768px) {
    .chat-container {
        grid-template-columns: 1fr;
    }
    
    .features-grid {
        grid-template-columns: 1fr;
    }
}
```

## Integration with ReservoirCogs

### Documentation Links
- Connects to existing user guide sections
- References AtomSpace integration documentation
- Links to API documentation and examples

### Code Examples
The playground demonstrates real ReservoirCogs usage:

```python
# Example from playground features section
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge

# Create reservoir with AtomSpace integration
reservoir = Reservoir(100, lr=0.3, sr=0.9, atomspace=True)
readout = Ridge(ridge=1e-6)

# Build and train model
model = reservoir >> readout
model.fit(X_train, y_train, symbolic_learning=True)
```

## Best Practices

### Performance Optimization
- Lazy loading of interactive components
- Efficient canvas rendering for knowledge graphs
- Debounced configuration updates

### Accessibility
- Semantic HTML structure
- ARIA labels for interactive elements
- Keyboard navigation support
- Screen reader compatibility

### Browser Compatibility
- Modern browser features with graceful degradation
- Progressive enhancement for advanced features
- Cross-platform testing recommendations

## Troubleshooting

### Common Issues

#### Chat Interface Not Loading
```javascript
// Fallback implementation
if (!chatIframe.contentWindow) {
    playground.showChatFallback();
}
```

#### Theme Persistence Issues
```javascript
// Clear and reset theme
localStorage.removeItem('playground-theme');
playground.setTheme('light');
```

#### Graph Rendering Problems
```javascript
// Canvas context reset
const canvas = document.getElementById('knowledge-graph');
const ctx = canvas.getContext('2d');
ctx.clearRect(0, 0, canvas.width, canvas.height);
```

## Contributing

### Development Setup
1. Clone the repository
2. Navigate to `playground/` directory
3. Start local server: `python -m http.server 8000`
4. Open `http://localhost:8000` in browser

### Adding New Features
1. Follow existing code structure and patterns
2. Update CSS custom properties for theming
3. Add appropriate documentation
4. Test across different browsers and devices

### Code Style Guidelines
- Use semantic HTML5 elements
- Follow BEM methodology for CSS classes
- Write modular, documented JavaScript
- Maintain accessibility standards

## Support

For technical support and feature requests:
- GitHub Issues: Report bugs and suggest features
- Documentation: Reference existing guides and API docs
- Email Contact: xavier.hinaut@inria.fr for research inquiries