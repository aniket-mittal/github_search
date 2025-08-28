# GitHub Repository Search - AST-Based Code Chunking System

This project provides a comprehensive system for extracting and chunking GitHub repository source code using Abstract Syntax Trees (AST) for intelligent code analysis and search.

## 🚀 Features

### Repository Extraction
- **BigQuery Integration**: Extract repository data from Google BigQuery's public GitHub dataset
- **Smart Filtering**: Focus on popular repositories (>10 stars, no forks)
- **Batch Processing**: Handle large-scale data extraction efficiently
- **Incremental Storage**: Save code files to disk during processing

### AST-Based Code Chunking
- **Multi-language Support**: Python, JavaScript, TypeScript, React, HTML, CSS, and more
- **Intelligent Parsing**: Use AST for Python, regex patterns for other languages
- **Rich Metadata**: Extract classes, functions, imports, docstrings, and signatures
- **Batch Processing**: Process entire repositories or directories efficiently
- **Statistics & Analysis**: Get detailed insights about code structure

## 📁 Project Structure

```
github_search/
├── src/
│   ├── extract_repos.py      # Repository extraction from BigQuery
│   └── ast_chunker.py        # AST-based code chunking system
├── data/
│   ├── code_files/           # Extracted repository source code
│   ├── repositories.csv      # Repository metadata
│   └── sample_repos.json    # Sample repository data
├── test_chunker.py           # Comprehensive test suite
├── demo_chunker.py           # Simple demonstration script
├── batch_chunker.py          # Batch processing for large repositories
├── chunk_summary.py          # System capabilities overview
├── requirements.txt          # Python dependencies
└── README_CHUNKER.md        # Detailed chunking system documentation
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd github_search
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google Cloud credentials** (for repository extraction):
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   ```

## 🎯 Quick Start

### 1. Extract GitHub Repositories

```bash
# Extract popular repositories with source code
python src/extract_repos.py

# This will:
# - Query BigQuery for popular repositories
# - Download source code files
# - Save to data/code_files/
# - Generate metadata CSV and JSON files
```

### 2. Chunk Code Files

```bash
# Chunk a single file
python src/ast_chunker.py path/to/file.py -o chunks.json -s

# Chunk an entire directory
python src/ast_chunker.py path/to/directory -o chunks.json -s

# Chunk extracted repositories
python src/ast_chunker.py data/code_files/ -o all_chunks.json -s
```

### 3. Test the System

```bash
# Run comprehensive tests
python test_chunker.py

# Quick demonstration
python demo_chunker.py src/ast_chunker.py -s

# System overview
python chunk_summary.py
```

## 🔧 Usage Examples

### Python API

```python
from src.ast_chunker import ASTChunker

# Initialize chunker
chunker = ASTChunker()

# Chunk a single file
chunks = chunker.chunk_file('path/to/file.py')

# Chunk a directory
chunks = chunker.chunk_directory('path/to/project')

# Get statistics
stats = chunker.get_chunk_statistics(chunks)
print(f"Generated {stats['total_chunks']} chunks")
```

### Command Line

```bash
# Basic chunking
python src/ast_chunker.py file.py

# With output and statistics
python src/ast_chunker.py project/ -o chunks.json -s

# Batch processing for large repositories
python batch_chunker.py large_repo/ -o output/ -b 500

# Process multiple repositories
python batch_chunker.py repos_directory/ -m -o all_chunks/
```

## 📊 What Gets Chunked

The system identifies and creates chunks for:

- **Classes**: Class definitions with methods and properties
- **Functions**: Function definitions (standalone or within classes)
- **Modules**: Entire file content
- **Imports**: Import statements and dependencies
- **Variables**: Variable declarations and assignments
- **Comments**: Documentation and inline comments
- **Strings**: String literals and docstrings
- **Misc**: Other code elements

## 🌐 Supported Languages

### Primary Support (AST-based)
- **Python**: Full AST parsing with class, function, and import detection

### Secondary Support (Regex-based)
- **JavaScript/TypeScript**: Class, function, and import detection
- **React**: JSX/TSX support
- **HTML**: Structure-based chunking
- **CSS**: Rule-based chunking
- **Markdown**: Section-based chunking

### Generic Support
- **Other languages**: Fallback to content-based chunking

## 📈 Performance

- **Small files (< 1KB)**: ~1-5ms per file
- **Medium files (1-10KB)**: ~5-20ms per file
- **Large files (10-100KB)**: ~20-100ms per file
- **Very large files (>100KB)**: ~100ms+ per file

Memory usage scales linearly with file size and number of chunks.

## 🎯 Use Cases

### Code Search & Discovery
- Break down large codebases into searchable units
- Find specific functions or classes across repositories
- Analyze code structure and organization

### Documentation Generation
- Extract function signatures and docstrings
- Generate API documentation automatically
- Identify undocumented code sections

### Code Analysis
- Measure code complexity and structure
- Analyze dependencies between components
- Identify code patterns and anti-patterns

### Machine Learning
- Prepare code for embedding models
- Create training datasets for code understanding
- Support code generation and completion

## 🔍 Example Output

Each chunk includes rich metadata:

```json
{
  "chunk_id": "unique_identifier",
  "chunk_type": "class|function|module|import|misc",
  "name": "chunk_name",
  "content": "actual_code_content",
  "start_line": 10,
  "end_line": 25,
  "file_path": "path/to/file.py",
  "language": "python",
  "parent_context": "class:ParentClass",
  "docstring": "Documentation string if available",
  "signature": "def function_name(arg1, arg2)",
  "dependencies": ["dependency1", "dependency2"]
}
```

## 🧪 Testing

The system includes comprehensive testing:

```bash
# Run all tests
python test_chunker.py

# Test specific components
python demo_chunker.py src/ast_chunker.py -s
python demo_chunker.py data/code_files/Shopify_js-buy-sdk -s
```

## 📚 Documentation

- **README_CHUNKER.md**: Detailed chunking system documentation
- **Code comments**: Comprehensive inline documentation
- **Type hints**: Full type annotations for all functions
- **Examples**: Working examples in test files

## 🚨 Limitations

- **Python**: Requires valid syntax (handles errors gracefully)
- **JavaScript**: Regex-based parsing may miss complex patterns
- **Large files**: Very large files may be processed as single chunks
- **Binary files**: Only text-based files are supported

## 🤝 Contributing

To extend the system:

1. **Add new language support**: Create a new chunker class
2. **Improve parsing**: Enhance existing chunkers
3. **Add chunk types**: Extend the ChunkType enum
4. **Optimize performance**: Improve parsing algorithms

## 📄 License

This project follows standard open-source licensing terms.

## 🆘 Support

For issues and questions:

1. Check the test output for error details
2. Review the logging output
3. Ensure file paths and permissions are correct
4. Verify Python version compatibility (3.7+)
5. Check Google Cloud credentials (for extraction)

## 🎉 Getting Started

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run tests**: `python test_chunker.py`
3. **Try demo**: `python demo_chunker.py src/ast_chunker.py -s`
4. **Extract repos**: `python src/extract_repos.py`
5. **Chunk code**: `python src/ast_chunker.py data/code_files/ -o chunks.json`

The system is designed to be both powerful for production use and easy to understand for learning and experimentation.
