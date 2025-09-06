#!/usr/bin/env python3
"""
Generic CPG Parser

This module implements a generic Code Property Graph parser for languages
that don't have specialized parsers. It uses pattern matching and heuristics
to extract basic structural information.
"""

import re
import logging
from typing import Dict, List, Set, Optional, Any, Tuple
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpg_core import CPGParser, CodePropertyGraph, CPGNode, CPGEdge, NodeType, EdgeType

logger = logging.getLogger(__name__)


class GenericCPGParser(CPGParser):
    """Generic CPG parser using common programming language patterns."""
    
    def __init__(self):
        """Initialize the generic parser."""
        self.current_cpg = None
        self.current_file_path = ""
        self.variable_definitions = {}
        self.variable_uses = {}
        
        # Generic patterns that work across many languages
        self.patterns = {
            # Common function/method patterns
            'function_patterns': [
                r'def\s+(\w+)\s*\(',  # Python, Ruby
                r'function\s+(\w+)\s*\(',  # JavaScript, PHP
                r'(\w+)\s*\([^)]*\)\s*\{',  # C-style languages
                r'fn\s+(\w+)\s*\(',  # Rust, Go (partial)
                r'func\s+(\w+)\s*\(',  # Go, Swift
                r'sub\s+(\w+)\s*\(',  # Perl
                r'procedure\s+(\w+)\s*\(',  # Pascal
                r'fun\s+(\w+)\s*\(',  # Kotlin, ML
            ],
            
            # Common class patterns
            'class_patterns': [
                r'class\s+(\w+)(?:\s*\([^)]*\))?(?:\s*:\s*[\w\s,]+)?\s*[:{]',  # Python, C++, C#, etc.
                r'struct\s+(\w+)\s*[{]',  # C, C++, Go, Rust
                r'interface\s+(\w+)\s*[{]',  # Java, C#, TypeScript
                r'type\s+(\w+)\s*=\s*class',  # Pascal
                r'object\s+(\w+)',  # Some OOP languages
            ],
            
            # Common import/include patterns
            'import_patterns': [
                r'import\s+[\w\.\*]+',  # Java, Python, etc.
                r'from\s+[\w\.]+\s+import',  # Python
                r'#include\s*[<"][^>"]+[>"]',  # C, C++
                r'require\s*\([\'"][^\'"]+[\'"]\)',  # Node.js
                r'use\s+[\w:]+',  # Rust, Perl
                r'with\s+[\w\.]+',  # Ada
            ],
            
            # Common variable/assignment patterns
            'variable_patterns': [
                r'(\w+)\s*=\s*[^=]',  # General assignment
                r'let\s+(\w+)',  # JavaScript, Rust, etc.
                r'var\s+(\w+)',  # JavaScript, C#, etc.
                r'const\s+(\w+)',  # JavaScript, C++, etc.
                r'(\w+)\s+(\w+)\s*[;=]',  # Typed variables
            ],
            
            # Common control flow patterns
            'control_flow_patterns': [
                r'if\s*\(',  # Most C-style languages
                r'if\s+',  # Python, Ruby, etc.
                r'for\s*\(',  # C-style for loops
                r'for\s+\w+\s+in',  # Python, Ruby for-in loops
                r'while\s*\(',  # Most languages
                r'switch\s*\(',  # C-style switch
                r'match\s+',  # Rust, Scala pattern matching
                r'case\s+',  # Various languages
            ],
            
            # Common comment patterns
            'comment_patterns': [
                r'//.*$',  # Single line comments
                r'/\*.*?\*/',  # Multi-line comments
                r'#.*$',  # Shell, Python, Ruby comments
                r'--.*$',  # SQL, Ada comments
                r';.*$',  # Assembly, Lisp comments
            ],
            
            # Common string patterns
            'string_patterns': [
                r'"[^"]*"',  # Double quoted strings
                r"'[^']*'",  # Single quoted strings
                r'`[^`]*`',  # Backtick strings
                r'""".*?"""',  # Triple quoted strings
                r"'''.*?'''",  # Triple quoted strings
            ],
        }
    
    def parse(self, content: str, file_path: str) -> CodePropertyGraph:
        """Parse source code using generic patterns and generate CPG."""
        self.current_file_path = file_path
        self.current_cpg = CodePropertyGraph()
        
        # Reset state
        self.variable_definitions.clear()
        self.variable_uses.clear()
        
        # Build AST
        self.build_ast(content, file_path)
        
        # Build CFG
        self.build_cfg(self.current_cpg)
        
        # Build DFG
        self.build_dfg(self.current_cpg)
        
        return self.current_cpg
    
    def build_ast(self, content: str, file_path: str) -> CodePropertyGraph:
        """Build Abstract Syntax Tree using generic patterns."""
        lines = content.splitlines()
        
        # Detect likely language based on file extension and content
        language = self._detect_language_features(content, file_path)
        
        # Create module node
        module_node = self.create_node(
            NodeType.MODULE,
            Path(file_path).stem,
            content,
            file_path,
            1,
            len(lines),
            0,
            len(lines[-1]) if lines else 0,
            language
        )
        module_id = self.current_cpg.add_node(module_node)
        
        # Process different constructs
        self._process_imports(content, lines, module_id, language)
        self._process_functions(content, lines, module_id, language)
        self._process_classes(content, lines, module_id, language)
        self._process_variables(content, lines, module_id, language)
        self._process_control_flow(content, lines, module_id, language)
        
        return self.current_cpg
    
    def _detect_language_features(self, content: str, file_path: str) -> str:
        """Detect language based on file extension and content patterns."""
        ext = Path(file_path).suffix.lower()
        
        # Common language indicators
        language_indicators = {
            'python': [r'def\s+\w+', r'import\s+\w+', r'if\s+__name__', r'print\s*\('],
            'javascript': [r'function\s+\w+', r'var\s+\w+', r'=>', r'require\s*\('],
            'java': [r'public\s+class', r'import\s+java\.', r'System\.out'],
            'cpp': [r'#include', r'std::', r'namespace\s+\w+', r'cout\s*<<'],
            'c': [r'#include\s*<.*\.h>', r'int\s+main', r'printf\s*\('],
            'csharp': [r'using\s+System', r'public\s+class', r'Console\.Write'],
            'php': [r'<\?php', r'\$\w+', r'echo\s+'],
            'ruby': [r'def\s+\w+', r'class\s+\w+', r'puts\s+', r'require\s+'],
            'go': [r'package\s+\w+', r'func\s+\w+', r'import\s*\(', r'fmt\.Print'],
            'rust': [r'fn\s+\w+', r'let\s+\w+', r'use\s+\w+', r'println!'],
            'swift': [r'func\s+\w+', r'var\s+\w+', r'let\s+\w+', r'print\s*\('],
            'kotlin': [r'fun\s+\w+', r'class\s+\w+', r'val\s+\w+', r'var\s+\w+'],
            'scala': [r'def\s+\w+', r'class\s+\w+', r'object\s+\w+', r'val\s+\w+'],
            'perl': [r'sub\s+\w+', r'my\s+\$\w+', r'use\s+\w+', r'print\s+'],
            'lua': [r'function\s+\w+', r'local\s+\w+', r'require\s*\(', r'print\s*\('],
            'bash': [r'#!/bin/bash', r'function\s+\w+', r'if\s*\[', r'echo\s+'],
        }
        
        # Score each language
        scores = {}
        for lang, patterns in language_indicators.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content, re.MULTILINE | re.IGNORECASE))
                score += matches
            scores[lang] = score
        
        # Return the highest scoring language, or 'unknown' if no clear winner
        if scores:
            best_lang = max(scores, key=scores.get)
            if scores[best_lang] > 0:
                return best_lang
        
        return 'unknown'
    
    def _process_imports(self, content: str, lines: List[str], parent_id: str, language: str):
        """Process import/include statements."""
        for pattern in self.patterns['import_patterns']:
            for match in re.finditer(pattern, content, re.MULTILINE):
                start_line = content[:match.start()].count('\n') + 1
                
                import_node = self.create_node(
                    NodeType.IMPORT,
                    "import",
                    match.group(0),
                    self.current_file_path,
                    start_line,
                    start_line,
                    match.start() - content.rfind('\n', 0, match.start()) - 1,
                    match.end() - content.rfind('\n', 0, match.end()) - 1,
                    language
                )
                import_id = self.current_cpg.add_node(import_node)
                
                # Link to parent
                edge = self.create_edge(parent_id, import_id, EdgeType.AST_CHILD)
                self.current_cpg.add_edge(edge)
    
    def _process_functions(self, content: str, lines: List[str], parent_id: str, language: str):
        """Process function definitions."""
        for pattern in self.patterns['function_patterns']:
            for match in re.finditer(pattern, content, re.MULTILINE):
                # Extract function name
                func_name = match.group(1) if match.groups() else "unknown_function"
                start_line = content[:match.start()].count('\n') + 1
                
                # Try to find function end (simplified)
                end_pos = self._find_function_end(content, match.end(), language)
                end_line = content[:end_pos].count('\n') + 1 if end_pos else start_line
                
                # Extract function content
                func_content = content[match.start():end_pos] if end_pos else match.group(0)
                
                func_node = self.create_node(
                    NodeType.FUNCTION,
                    func_name,
                    func_content,
                    self.current_file_path,
                    start_line,
                    end_line,
                    match.start() - content.rfind('\n', 0, match.start()) - 1,
                    0,
                    language
                )
                func_id = self.current_cpg.add_node(func_node)
                
                # Link to parent
                edge = self.create_edge(parent_id, func_id, EdgeType.AST_CHILD)
                self.current_cpg.add_edge(edge)
                
                # Process function body
                self._process_function_body(func_content, func_id, start_line, language)
    
    def _process_classes(self, content: str, lines: List[str], parent_id: str, language: str):
        """Process class definitions."""
        for pattern in self.patterns['class_patterns']:
            for match in re.finditer(pattern, content, re.MULTILINE):
                # Extract class name
                class_name = match.group(1) if match.groups() else "unknown_class"
                start_line = content[:match.start()].count('\n') + 1
                
                # Try to find class end
                end_pos = self._find_block_end(content, match.end(), language)
                end_line = content[:end_pos].count('\n') + 1 if end_pos else start_line
                
                # Extract class content
                class_content = content[match.start():end_pos] if end_pos else match.group(0)
                
                class_node = self.create_node(
                    NodeType.CLASS,
                    class_name,
                    class_content,
                    self.current_file_path,
                    start_line,
                    end_line,
                    match.start() - content.rfind('\n', 0, match.start()) - 1,
                    0,
                    language
                )
                class_id = self.current_cpg.add_node(class_node)
                
                # Link to parent
                edge = self.create_edge(parent_id, class_id, EdgeType.AST_CHILD)
                self.current_cpg.add_edge(edge)
                
                # Process class members
                self._process_class_members(class_content, class_id, start_line, language)
    
    def _process_variables(self, content: str, lines: List[str], parent_id: str, language: str):
        """Process variable declarations and assignments."""
        for pattern in self.patterns['variable_patterns']:
            for match in re.finditer(pattern, content, re.MULTILINE):
                start_line = content[:match.start()].count('\n') + 1
                
                # Extract variable name (heuristic)
                var_name = "unknown_var"
                if match.groups():
                    # Try to get the variable name from capture groups
                    for group in match.groups():
                        if group and re.match(r'^\w+$', group):
                            var_name = group
                            break
                
                var_node = self.create_node(
                    NodeType.VARIABLE,
                    var_name,
                    match.group(0),
                    self.current_file_path,
                    start_line,
                    start_line,
                    match.start() - content.rfind('\n', 0, match.start()) - 1,
                    match.end() - content.rfind('\n', 0, match.end()) - 1,
                    language
                )
                var_id = self.current_cpg.add_node(var_node)
                
                # Track for DFG
                if var_name not in self.variable_definitions:
                    self.variable_definitions[var_name] = []
                self.variable_definitions[var_name].append(var_id)
                
                # Track variable usage in the expression
                used_vars = self._extract_generic_variables_from_code(match.group(0))
                for used_var in used_vars:
                    if used_var != var_name:  # Don't track self-reference
                        if used_var not in self.variable_uses:
                            self.variable_uses[used_var] = []
                        self.variable_uses[used_var].append(var_id)
                
                # Link to parent
                edge = self.create_edge(parent_id, var_id, EdgeType.AST_CHILD)
                self.current_cpg.add_edge(edge)
    
    def _extract_generic_variables_from_code(self, code: str) -> List[str]:
        """Extract variable names from generic code."""
        variables = []
        
        # Skip common keywords across languages
        generic_keywords = {
            'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default',
            'break', 'continue', 'return', 'function', 'class', 'interface',
            'public', 'private', 'protected', 'static', 'const', 'let', 'var',
            'int', 'string', 'bool', 'char', 'float', 'double', 'void',
            'true', 'false', 'null', 'undefined', 'this', 'super', 'new',
            'import', 'export', 'from', 'as', 'def', 'class', 'import',
            'include', 'using', 'namespace', 'struct', 'enum', 'typedef'
        }
        
        # Find variable patterns - simple identifiers
        for match in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', code):
            var_name = match.group(1)
            if var_name not in generic_keywords and var_name not in variables:
                variables.append(var_name)
        
        return variables
    
    def _process_control_flow(self, content: str, lines: List[str], parent_id: str, language: str):
        """Process control flow statements."""
        for pattern in self.patterns['control_flow_patterns']:
            for match in re.finditer(pattern, content, re.MULTILINE):
                start_line = content[:match.start()].count('\n') + 1
                
                # Determine control flow type
                control_type = "unknown"
                if 'if' in match.group(0).lower():
                    control_type = "if"
                    node_type = NodeType.CONDITION
                elif 'for' in match.group(0).lower() or 'while' in match.group(0).lower():
                    control_type = "loop"
                    node_type = NodeType.LOOP
                elif 'switch' in match.group(0).lower() or 'match' in match.group(0).lower():
                    control_type = "switch"
                    node_type = NodeType.CONDITION
                else:
                    node_type = NodeType.CONDITION
                
                # Try to find block end
                end_pos = self._find_block_end(content, match.end(), language)
                end_line = content[:end_pos].count('\n') + 1 if end_pos else start_line
                
                # Extract control flow content
                control_content = content[match.start():end_pos] if end_pos else match.group(0)
                
                control_node = self.create_node(
                    node_type,
                    control_type,
                    control_content,
                    self.current_file_path,
                    start_line,
                    end_line,
                    match.start() - content.rfind('\n', 0, match.start()) - 1,
                    0,
                    language,
                    control_type=control_type
                )
                control_id = self.current_cpg.add_node(control_node)
                
                # Link to parent
                edge = self.create_edge(parent_id, control_id, EdgeType.AST_CHILD)
                self.current_cpg.add_edge(edge)
    
    def _process_function_body(self, func_content: str, func_id: str, 
                              func_start_line: int, language: str):
        """Process statements within a function body."""
        # Process variables within function
        for pattern in self.patterns['variable_patterns']:
            for match in re.finditer(pattern, func_content, re.MULTILINE):
                start_line = func_start_line + func_content[:match.start()].count('\n')
                
                # Extract variable name
                var_name = "local_var"
                if match.groups():
                    for group in match.groups():
                        if group and re.match(r'^\w+$', group):
                            var_name = group
                            break
                
                var_node = self.create_node(
                    NodeType.VARIABLE,
                    var_name,
                    match.group(0),
                    self.current_file_path,
                    start_line,
                    start_line,
                    0,
                    0,
                    language,
                    scope="local"
                )
                var_id = self.current_cpg.add_node(var_node)
                
                # Track for DFG
                if var_name not in self.variable_definitions:
                    self.variable_definitions[var_name] = []
                self.variable_definitions[var_name].append(var_id)
                
                # Link to function
                edge = self.create_edge(func_id, var_id, EdgeType.AST_CHILD)
                self.current_cpg.add_edge(edge)
    
    def _process_class_members(self, class_content: str, class_id: str, 
                              class_start_line: int, language: str):
        """Process methods and fields within a class."""
        # Process methods within class
        for pattern in self.patterns['function_patterns']:
            for match in re.finditer(pattern, class_content, re.MULTILINE):
                method_name = match.group(1) if match.groups() else "unknown_method"
                start_line = class_start_line + class_content[:match.start()].count('\n')
                
                # Try to find method end
                end_pos = self._find_function_end(class_content, match.end(), language)
                end_line = class_start_line + class_content[:end_pos].count('\n') if end_pos else start_line
                
                # Extract method content
                method_content = class_content[match.start():end_pos] if end_pos else match.group(0)
                
                method_node = self.create_node(
                    NodeType.METHOD,
                    method_name,
                    method_content,
                    self.current_file_path,
                    start_line,
                    end_line,
                    0,
                    0,
                    language
                )
                method_id = self.current_cpg.add_node(method_node)
                
                # Link to class
                edge = self.create_edge(class_id, method_id, EdgeType.AST_CHILD)
                self.current_cpg.add_edge(edge)
    
    def _find_function_end(self, content: str, start_pos: int, language: str) -> Optional[int]:
        """Find the end of a function based on language-specific rules."""
        if language in ['python', 'ruby']:
            # Indentation-based languages
            return self._find_indentation_block_end(content, start_pos)
        elif language in ['javascript', 'java', 'cpp', 'c', 'csharp', 'go', 'rust']:
            # Brace-based languages
            return self._find_brace_block_end(content, start_pos)
        else:
            # Generic approach - try both
            brace_end = self._find_brace_block_end(content, start_pos)
            if brace_end:
                return brace_end
            return self._find_indentation_block_end(content, start_pos)
    
    def _find_block_end(self, content: str, start_pos: int, language: str) -> Optional[int]:
        """Find the end of a code block."""
        return self._find_function_end(content, start_pos, language)
    
    def _find_brace_block_end(self, content: str, start_pos: int) -> Optional[int]:
        """Find the end of a brace-delimited block."""
        # Find the opening brace
        brace_pos = content.find('{', start_pos)
        if brace_pos == -1:
            return None
        
        brace_count = 1
        pos = brace_pos + 1
        
        while pos < len(content) and brace_count > 0:
            if content[pos] == '{':
                brace_count += 1
            elif content[pos] == '}':
                brace_count -= 1
            pos += 1
        
        return pos if brace_count == 0 else None
    
    def _find_indentation_block_end(self, content: str, start_pos: int) -> Optional[int]:
        """Find the end of an indentation-based block."""
        lines = content[start_pos:].split('\n')
        if not lines:
            return None
        
        # Find the base indentation level
        base_indent = 0
        for line in lines:
            stripped = line.lstrip()
            if stripped and not stripped.startswith('#'):
                base_indent = len(line) - len(stripped)
                break
        
        # Find where indentation returns to base level or less
        current_pos = start_pos
        for i, line in enumerate(lines[1:], 1):
            stripped = line.lstrip()
            if stripped and not stripped.startswith('#'):
                current_indent = len(line) - len(stripped)
                if current_indent <= base_indent:
                    # Found end of block
                    return current_pos + sum(len(lines[j]) + 1 for j in range(i))
            current_pos += len(line) + 1
        
        return len(content)  # End of file
    
    def build_cfg(self, cpg: CodePropertyGraph) -> CodePropertyGraph:
        """Build Control Flow Graph from AST."""
        # Find all function and method nodes
        function_nodes = []
        for node in list(cpg.nodes.values()):
            if node.node_type in [NodeType.FUNCTION, NodeType.METHOD]:
                function_nodes.append(node)
        
        # Build CFG for each function (simplified)
        for func_node in function_nodes:
            self._build_simple_cfg(func_node, cpg)
        
        return cpg
    
    def _build_simple_cfg(self, func_node: CPGNode, cpg: CodePropertyGraph):
        """Build a simplified CFG for a function."""
        # Create entry and exit nodes
        entry_node = self.create_node(
            NodeType.ENTRY,
            f"{func_node.name}_entry",
            "",
            func_node.file_path,
            func_node.start_line,
            func_node.start_line,
            language=func_node.language
        )
        entry_id = cpg.add_node(entry_node)
        
        exit_node = self.create_node(
            NodeType.EXIT,
            f"{func_node.name}_exit",
            "",
            func_node.file_path,
            func_node.end_line,
            func_node.end_line,
            language=func_node.language
        )
        exit_id = cpg.add_node(exit_node)
        
        # Get all child nodes of the function
        child_nodes = cpg.get_children(func_node.id)
        
        # Create simple sequential flow
        if child_nodes:
            # Connect entry to first statement
            first_stmt = child_nodes[0]
            edge = self.create_edge(entry_id, first_stmt.id, EdgeType.CONTROL_FLOW)
            cpg.add_edge(edge)
            
            # Connect statements sequentially
            for i in range(len(child_nodes) - 1):
                current_node = child_nodes[i]
                next_node = child_nodes[i + 1]
                
                edge = self.create_edge(current_node.id, next_node.id, EdgeType.CONTROL_FLOW)
                cpg.add_edge(edge)
            
            # Connect last statement to exit
            last_stmt = child_nodes[-1]
            edge = self.create_edge(last_stmt.id, exit_id, EdgeType.CONTROL_FLOW)
            cpg.add_edge(edge)
    
    def build_dfg(self, cpg: CodePropertyGraph) -> CodePropertyGraph:
        """Build Data Flow Graph from AST and CFG - DISABLED for performance."""
        # DFG generation disabled for faster processing
        # Only AST and CFG are generated
        return cpg
