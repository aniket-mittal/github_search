#!/usr/bin/env python3
"""
AST-Based Code Chunking System

This script provides intelligent chunking of source code files using Abstract Syntax Trees (AST)
to break down code into meaningful, searchable chunks based on classes, functions, and other
structural elements.
"""

import os
import ast
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import tokenize
from io import StringIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChunkType(Enum):
    """Types of code chunks."""
    CLASS = "class"
    FUNCTION = "function"
    MODULE = "module"
    IMPORT = "import"
    VARIABLE = "variable"
    COMMENT = "comment"
    STRING = "string"
    MISC = "misc"

@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata."""
    chunk_id: str
    chunk_type: ChunkType
    name: str
    content: str
    start_line: int
    end_line: int
    file_path: str
    language: str
    parent_context: Optional[str] = None
    docstring: Optional[str] = None
    signature: Optional[str] = None
    complexity: Optional[int] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class LanguageDetector:
    """Detects programming language based on file extension and content."""
    
    LANGUAGE_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'react',
        '.tsx': 'react-ts',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'csharp',
        '.php': 'php',
        '.rb': 'ruby',
        '.go': 'go',
        '.rs': 'rust',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.clj': 'clojure',
        '.hs': 'haskell',
        '.ml': 'ocaml',
        '.fs': 'fsharp',
        '.r': 'r',
        '.m': 'matlab',
        '.sh': 'bash',
        '.sql': 'sql',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.sass': 'sass',
        '.xml': 'xml',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.json': 'json',
        '.toml': 'toml',
        '.ini': 'ini',
        '.cfg': 'config',
        '.conf': 'config',
        '.md': 'markdown',
        '.txt': 'text'
    }
    
    @classmethod
    def detect_language(cls, file_path: str, content: str = None) -> str:
        """Detect programming language from file path and optionally content."""
        ext = Path(file_path).suffix.lower()
        
        # Check extension first
        if ext in cls.LANGUAGE_EXTENSIONS:
            return cls.LANGUAGE_EXTENSIONS[ext]
        
        # Fallback to content-based detection
        if content:
            return cls._detect_from_content(content)
        
        return 'unknown'
    
    @classmethod
    def _detect_from_content(cls, content: str) -> str:
        """Detect language from file content using heuristics."""
        content_lower = content.lower()
        
        # Python
        if any(keyword in content_lower for keyword in ['def ', 'class ', 'import ', 'from ']):
            if 'def ' in content_lower or 'class ' in content_lower:
                return 'python'
        
        # JavaScript/TypeScript
        if any(keyword in content_lower for keyword in ['function ', 'const ', 'let ', 'var ', '=>']):
            if 'interface ' in content_lower or 'type ' in content_lower:
                return 'typescript'
            return 'javascript'
        
        # Java
        if any(keyword in content_lower for keyword in ['public class', 'private ', 'protected ', 'import java']):
            return 'java'
        
        # C/C++
        if any(keyword in content_lower for keyword in ['#include', 'int main', 'class ', 'namespace ']):
            if 'std::' in content_lower or 'template<' in content_lower:
                return 'cpp'
            return 'c'
        
        # PHP
        if '<?php' in content_lower or 'function ' in content_lower:
            return 'php'
        
        # Ruby
        if any(keyword in content_lower for keyword in ['def ', 'class ', 'module ', 'require ']):
            return 'ruby'
        
        # Go
        if 'package ' in content_lower or 'func ' in content_lower:
            return 'go'
        
        # Rust
        if 'fn ' in content_lower or 'struct ' in content_lower or 'impl ' in content_lower:
            return 'rust'
        
        return 'unknown'

class PythonChunker:
    """Chunks Python code using AST."""
    
    @classmethod
    def chunk_code(cls, content: str, file_path: str) -> List[CodeChunk]:
        """Chunk Python code into meaningful pieces."""
        chunks = []
        
        try:
            tree = ast.parse(content)
            chunk_id_counter = 0
            
            # Add module-level chunk
            module_chunk = CodeChunk(
                chunk_id=f"{Path(file_path).stem}_module_{chunk_id_counter}",
                chunk_type=ChunkType.MODULE,
                name=Path(file_path).stem,
                content=content,
                start_line=1,
                end_line=len(content.splitlines()),
                file_path=file_path,
                language='python'
            )
            chunks.append(module_chunk)
            chunk_id_counter += 1
            
            # Process all nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    chunk = cls._process_class(node, content, file_path, chunk_id_counter)
                    chunks.append(chunk)
                    chunk_id_counter += 1
                elif isinstance(node, ast.FunctionDef):
                    chunk = cls._process_function(node, content, file_path, chunk_id_counter)
                    chunks.append(chunk)
                    chunk_id_counter += 1
                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    chunk = cls._process_import(node, content, file_path, chunk_id_counter)
                    chunks.append(chunk)
                    chunk_id_counter += 1
            
            # Add misc chunks for any remaining code
            cls._add_misc_chunks(tree, content, file_path, chunks, chunk_id_counter)
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            # Fallback: create a single chunk for the entire file
            chunks.append(CodeChunk(
                chunk_id=f"{Path(file_path).stem}_error_{0}",
                chunk_type=ChunkType.MISC,
                name="syntax_error",
                content=content,
                start_line=1,
                end_line=len(content.splitlines()),
                file_path=file_path,
                language='python'
            ))
        
        return chunks
    
    @classmethod
    def _process_class(cls, node: ast.ClassDef, content: str, file_path: str, chunk_id: int) -> CodeChunk:
        """Process a class definition."""
        lines = content.splitlines()
        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
        
        # Get class content
        class_content = '\n'.join(lines[start_line-1:end_line])
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Get class signature
        bases = [base.id for base in node.bases if hasattr(base, 'id')]
        signature = f"class {node.name}"
        if bases:
            signature += f"({', '.join(bases)})"
        
        return CodeChunk(
            chunk_id=f"{Path(file_path).stem}_class_{chunk_id}",
            chunk_type=ChunkType.CLASS,
            name=node.name,
            content=class_content,
            start_line=start_line,
            end_line=end_line,
            file_path=file_path,
            language='python',
            docstring=docstring,
            signature=signature
        )
    
    @classmethod
    def _process_function(cls, node: ast.FunctionDef, content: str, file_path: str, chunk_id: int) -> CodeChunk:
        """Process a function definition."""
        lines = content.splitlines()
        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
        
        # Get function content
        func_content = '\n'.join(lines[start_line-1:end_line])
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Get function signature
        args = [arg.arg for arg in node.args.args]
        signature = f"def {node.name}({', '.join(args)})"
        
        # Determine parent context
        parent_context = None
        if hasattr(node, 'parent') and node.parent:
            if isinstance(node.parent, ast.ClassDef):
                parent_context = f"class:{node.parent.name}"
        
        return CodeChunk(
            chunk_id=f"{Path(file_path).stem}_func_{chunk_id}",
            chunk_type=ChunkType.FUNCTION,
            name=node.name,
            content=func_content,
            start_line=start_line,
            end_line=end_line,
            file_path=file_path,
            language='python',
            parent_context=parent_context,
            docstring=docstring,
            signature=signature
        )
    
    @classmethod
    def _process_import(cls, node: ast.AST, content: str, file_path: str, chunk_id: int) -> CodeChunk:
        """Process import statements."""
        lines = content.splitlines()
        start_line = node.lineno
        end_line = node.lineno
        
        import_content = lines[start_line-1]
        
        return CodeChunk(
            chunk_id=f"{Path(file_path).stem}_import_{chunk_id}",
            chunk_type=ChunkType.IMPORT,
            name="import",
            content=import_content,
            start_line=start_line,
            end_line=end_line,
            file_path=file_path,
            language='python'
        )
    
    @classmethod
    def _add_misc_chunks(cls, tree: ast.AST, content: str, file_path: str, chunks: List[CodeChunk], start_id: int):
        """Add miscellaneous chunks for remaining code."""
        # This is a simplified approach - in practice, you might want more sophisticated
        # logic to identify other code patterns
        pass

class JavaScriptChunker:
    """Chunks JavaScript/TypeScript code using regex patterns."""
    
    @classmethod
    def chunk_code(cls, content: str, file_path: str) -> List[CodeChunk]:
        """Chunk JavaScript/TypeScript code into meaningful pieces."""
        chunks = []
        chunk_id_counter = 0
        
        # Add module-level chunk
        module_chunk = CodeChunk(
            chunk_id=f"{Path(file_path).stem}_module_{chunk_id_counter}",
            chunk_type=ChunkType.MODULE,
            name=Path(file_path).stem,
            content=content,
            start_line=1,
            end_line=len(content.splitlines()),
            file_path=file_path,
            language='javascript'
        )
        chunks.append(module_chunk)
        chunk_id_counter += 1
        
        # Find classes
        class_pattern = r'class\s+(\w+)(?:\s+extends\s+(\w+))?\s*\{'
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            chunk = cls._process_class(match, content, file_path, chunk_id_counter)
            chunks.append(chunk)
            chunk_id_counter += 1
        
        # Find functions
        function_patterns = [
            r'function\s+(\w+)\s*\([^)]*\)\s*\{',  # function declaration
            r'(\w+)\s*[:=]\s*function\s*\([^)]*\)\s*\{',  # function expression
            r'(\w+)\s*[:=]\s*\([^)]*\)\s*=>\s*\{',  # arrow function
            r'(\w+)\s*[:=]\s*\([^)]*\)\s*=>',  # arrow function without braces
        ]
        
        for pattern in function_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                chunk = cls._process_function(match, content, file_path, chunk_id_counter)
                if chunk:
                    chunks.append(chunk)
                    chunk_id_counter += 1
        
        # Find imports
        import_patterns = [
            r'import\s+.*?from\s+[\'"][^\'"]+[\'"];?',
            r'require\s*\([\'"][^\'"]+[\'"]\)',
        ]
        
        for pattern in import_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                chunk = cls._process_import(match, content, file_path, chunk_id_counter)
                chunks.append(chunk)
                chunk_id_counter += 1
        
        return chunks
    
    @classmethod
    def _process_class(cls, match: re.Match, content: str, file_path: str, chunk_id: int) -> CodeChunk:
        """Process a class definition."""
        lines = content.splitlines()
        start_line = content[:match.start()].count('\n') + 1
        
        # Find the end of the class (simplified)
        brace_count = 0
        start_pos = match.start()
        for i, char in enumerate(content[start_pos:], start_pos):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break
        else:
            end_pos = len(content)
        
        end_line = content[:end_pos].count('\n') + 1
        class_content = content[start_pos:end_pos]
        
        return CodeChunk(
            chunk_id=f"{Path(file_path).stem}_class_{chunk_id}",
            chunk_type=ChunkType.CLASS,
            name=match.group(1),
            content=class_content,
            start_line=start_line,
            end_line=end_line,
            file_path=file_path,
            language='javascript'
        )
    
    @classmethod
    def _process_function(cls, match: re.Match, content: str, file_path: str, chunk_id: int) -> Optional[CodeChunk]:
        """Process a function definition."""
        lines = content.splitlines()
        start_line = content[:match.start()].count('\n') + 1
        
        # Find the end of the function (simplified)
        brace_count = 0
        start_pos = match.start()
        for i, char in enumerate(content[start_pos:], start_pos):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break
        else:
            end_pos = len(content)
        
        end_line = content[:end_pos].count('\n') + 1
        func_content = content[start_pos:end_pos]
        
        # Extract function name
        func_name = match.group(1) if match.group(1) else f"anonymous_{chunk_id}"
        
        return CodeChunk(
            chunk_id=f"{Path(file_path).stem}_func_{chunk_id}",
            chunk_type=ChunkType.FUNCTION,
            name=func_name,
            content=func_content,
            start_line=start_line,
            end_line=end_line,
            file_path=file_path,
            language='javascript'
        )
    
    @classmethod
    def _process_import(cls, match: re.Match, content: str, file_path: str, chunk_id: int) -> CodeChunk:
        """Process import statements."""
        lines = content.splitlines()
        start_line = content[:match.start()].count('\n') + 1
        end_line = start_line
        
        return CodeChunk(
            chunk_id=f"{Path(file_path).stem}_import_{chunk_id}",
            chunk_type=ChunkType.IMPORT,
            name="import",
            content=match.group(0),
            start_line=start_line,
            end_line=end_line,
            file_path=file_path,
            language='javascript'
        )

class GenericChunker:
    """Generic chunker for other file types."""
    
    @classmethod
    def chunk_code(cls, content: str, file_path: str, language: str) -> List[CodeChunk]:
        """Create generic chunks for non-code files."""
        chunks = []
        
        # Create a single chunk for the entire file
        chunk = CodeChunk(
            chunk_id=f"{Path(file_path).stem}_content_{0}",
            chunk_type=ChunkType.MISC,
            name=Path(file_path).stem,
            content=content,
            start_line=1,
            end_line=len(content.splitlines()),
            file_path=file_path,
            language=language
        )
        chunks.append(chunk)
        
        return chunks

class ASTChunker:
    """Main chunking orchestrator."""
    
    def __init__(self):
        """Initialize the chunker."""
        self.chunkers = {
            'python': PythonChunker,
            'javascript': JavaScriptChunker,
            'typescript': JavaScriptChunker,
            'react': JavaScriptChunker,
            'react-ts': JavaScriptChunker,
        }
    
    def chunk_file(self, file_path: str, content: str = None) -> List[CodeChunk]:
        """Chunk a single file."""
        if content is None:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                except Exception as e:
                    logger.error(f"Could not read file {file_path}: {e}")
                    return []
            except Exception as e:
                logger.error(f"Could not read file {file_path}: {e}")
                return []
        
        # Detect language
        language = LanguageDetector.detect_language(file_path, content)
        
        # Get appropriate chunker
        chunker_class = self.chunkers.get(language, GenericChunker)
        
        try:
            if chunker_class == GenericChunker:
                chunks = chunker_class.chunk_code(content, file_path, language)
            else:
                chunks = chunker_class.chunk_code(content, file_path)
            logger.info(f"Chunked {file_path} ({language}) into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking {file_path}: {e}")
            # Fallback to generic chunker
            return GenericChunker.chunk_code(content, file_path, language)
    
    def chunk_directory(self, directory_path: str, output_file: str = None) -> List[CodeChunk]:
        """Chunk all files in a directory recursively."""
        all_chunks = []
        directory_path = Path(directory_path)
        
        # Get all files
        file_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp', 
                          '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.clj', 
                          '.hs', '.ml', '.fs', '.r', '.m', '.sh', '.sql', '.html', '.css', '.scss', 
                          '.sass', '.xml', '.yaml', '.yml', '.json', '.toml', '.ini', '.cfg', '.conf', 
                          '.md', '.txt'}
        
        files = []
        for ext in file_extensions:
            files.extend(directory_path.rglob(f"*{ext}"))
        
        logger.info(f"Found {len(files)} files to chunk in {directory_path}")
        
        # Process each file
        for file_path in files:
            try:
                chunks = self.chunk_file(str(file_path))
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to chunk {file_path}: {e}")
        
        # Save chunks if output file specified
        if output_file:
            self.save_chunks(all_chunks, output_file)
        
        return all_chunks
    
    def save_chunks(self, chunks: List[CodeChunk], output_file: str):
        """Save chunks to a JSON file."""
        try:
            # Convert chunks to dictionaries
            chunk_data = []
            for chunk in chunks:
                chunk_dict = asdict(chunk)
                chunk_dict['chunk_type'] = chunk.chunk_type.value
                chunk_data.append(chunk_dict)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(chunks)} chunks to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save chunks: {e}")
    
    def get_chunk_statistics(self, chunks: List[CodeChunk]) -> Dict[str, Any]:
        """Get statistics about the chunks."""
        stats = {
            'total_chunks': len(chunks),
            'by_type': {},
            'by_language': {},
            'by_file': {},
            'total_lines': 0,
            'avg_chunk_size': 0
        }
        
        for chunk in chunks:
            # Count by type
            chunk_type = chunk.chunk_type.value
            stats['by_type'][chunk_type] = stats['by_type'].get(chunk_type, 0) + 1
            
            # Count by language
            stats['by_language'][chunk.language] = stats['by_language'].get(chunk.language, 0) + 1
            
            # Count by file
            stats['by_file'][chunk.file_path] = stats['by_file'].get(chunk.file_path, 0) + 1
            
            # Count lines
            stats['total_lines'] += chunk.end_line - chunk.start_line + 1
        
        if chunks:
            stats['avg_chunk_size'] = stats['total_lines'] / len(chunks)
        
        return stats

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AST-based code chunking system')
    parser.add_argument('input', help='Input file or directory path')
    parser.add_argument('-o', '--output', help='Output JSON file for chunks')
    parser.add_argument('-s', '--stats', action='store_true', help='Show chunk statistics')
    
    args = parser.parse_args()
    
    chunker = ASTChunker()
    
    if os.path.isfile(args.input):
        # Chunk single file
        chunks = chunker.chunk_file(args.input)
        print(f"Chunked {args.input} into {len(chunks)} chunks")
        
        if args.output:
            chunker.save_chunks(chunks, args.output)
        
        if args.stats:
            stats = chunker.get_chunk_statistics(chunks)
            print("\nChunk Statistics:")
            print(json.dumps(stats, indent=2))
            
    elif os.path.isdir(args.input):
        # Chunk directory
        chunks = chunker.chunk_directory(args.input, args.output)
        print(f"Chunked {args.input} into {len(chunks)} total chunks")
        
        if args.stats:
            stats = chunker.get_chunk_statistics(chunks)
            print("\nChunk Statistics:")
            print(json.dumps(stats, indent=2))
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
