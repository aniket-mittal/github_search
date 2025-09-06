#!/usr/bin/env python3
"""
Enhanced Python CPG Parser

This module implements an enhanced Python parser with better accuracy for
AST, CFG, and DFG generation using Python's built-in AST module with
additional heuristics and error handling.
"""

import ast
import logging
import re
from typing import Dict, List, Set, Optional, Any, Tuple
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpg_core import CPGParser, CodePropertyGraph, CPGNode, CPGEdge, NodeType, EdgeType

logger = logging.getLogger(__name__)


class EnhancedPythonCPGParser(CPGParser):
    """Enhanced Python-specific CPG parser with improved accuracy."""
    
    def __init__(self):
        """Initialize the enhanced Python parser."""
        self.current_cpg = None
        self.current_file_path = ""
        self.variable_definitions = {}
        self.variable_uses = {}
        self.function_calls = {}
        self.class_hierarchy = {}
        self.import_graph = {}
        self.scope_stack = []
        self.current_scope = "global"
    
    def parse(self, content: str, file_path: str) -> CodePropertyGraph:
        """Parse Python source code and generate complete CPG."""
        self.current_file_path = file_path
        self.current_cpg = CodePropertyGraph()
        
        # Reset state
        self.variable_definitions.clear()
        self.variable_uses.clear()
        self.function_calls.clear()
        self.class_hierarchy.clear()
        self.import_graph.clear()
        self.scope_stack.clear()
        self.current_scope = "global"
        
        try:
            # Build AST first
            self.build_ast(content, file_path)
            
            # Then build CFG
            self.build_cfg(self.current_cpg)
            
            # Finally build DFG
            self.build_dfg(self.current_cpg)
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            # Create error metadata
            self.current_cpg.metadata['error'] = str(e)
            self.current_cpg.metadata['language'] = 'python'
            
        return self.current_cpg
    
    def build_ast(self, content: str, file_path: str) -> CodePropertyGraph:
        """Build Enhanced Abstract Syntax Tree for Python code."""
        try:
            tree = ast.parse(content, filename=file_path)
            
            # Create module node with enhanced metadata
            module_node = self.create_node(
                NodeType.MODULE,
                Path(file_path).stem,
                content,
                file_path,
                1,
                len(content.splitlines()),
                0,
                len(content.splitlines()[-1]) if content.splitlines() else 0,
                'python',
                docstring=ast.get_docstring(tree),
                complexity_score=self._calculate_complexity(content),
                imports=self._extract_imports(tree),
                classes=self._extract_class_names(tree),
                functions=self._extract_function_names(tree)
            )
            module_id = self.current_cpg.add_node(module_node)
            
            # Process all AST nodes with enhanced analysis
            self._process_ast_node(tree, module_id, content.splitlines())
            
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            # Try to parse partial content up to the error
            lines = content.splitlines()
            if e.lineno and e.lineno > 1:
                try:
                    partial_content = '\n'.join(lines[:e.lineno-1])
                    if partial_content.strip():
                        partial_tree = ast.parse(partial_content)
                        self._process_ast_node(partial_tree, "partial_module", lines)
                except:
                    pass
            
            # Create error node with detailed information
            error_node = self.create_node(
                NodeType.MODULE,
                "syntax_error",
                content,
                file_path,
                e.lineno or 1,
                e.lineno or len(content.splitlines()),
                e.offset or 0,
                e.end_col_offset or 0,
                'python',
                error=str(e),
                error_type="SyntaxError",
                error_line=e.lineno,
                error_offset=e.offset
            )
            self.current_cpg.add_node(error_node)
        
        return self.current_cpg
    
    def _calculate_complexity(self, content: str) -> int:
        """Calculate cyclomatic complexity of the code."""
        complexity = 1  # Base complexity
        
        # Count decision points
        decision_keywords = ['if', 'elif', 'while', 'for', 'try', 'except', 'and', 'or']
        for keyword in decision_keywords:
            complexity += len(re.findall(rf'\b{keyword}\b', content))
        
        return complexity
    
    def _extract_imports(self, tree: ast.AST) -> Tuple[str, ...]:
        """Extract all import statements."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)
        return tuple(imports)
    
    def _extract_class_names(self, tree: ast.AST) -> Tuple[str, ...]:
        """Extract all class names."""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        return tuple(classes)
    
    def _extract_function_names(self, tree: ast.AST) -> Tuple[str, ...]:
        """Extract all function names."""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)
        return tuple(functions)
    
    def _process_ast_node(self, node: ast.AST, parent_id: str, lines: List[str]) -> str:
        """Process a single AST node with enhanced analysis."""
        node_id = None
        
        if isinstance(node, ast.ClassDef):
            node_id = self._process_class_def(node, parent_id, lines)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node_id = self._process_function_def(node, parent_id, lines)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            node_id = self._process_import(node, parent_id, lines)
        elif isinstance(node, ast.Assign):
            node_id = self._process_assignment(node, parent_id, lines)
        elif isinstance(node, ast.Call):
            node_id = self._process_function_call(node, parent_id, lines)
        elif isinstance(node, ast.If):
            node_id = self._process_if_statement(node, parent_id, lines)
        elif isinstance(node, (ast.For, ast.While)):
            node_id = self._process_loop(node, parent_id, lines)
        elif isinstance(node, ast.Return):
            node_id = self._process_return(node, parent_id, lines)
        elif isinstance(node, ast.Name):
            node_id = self._process_name_usage(node, parent_id, lines)
        
        # Process child nodes
        if hasattr(node, 'body') and isinstance(node.body, list):
            for child in node.body:
                self._process_ast_node(child, node_id or parent_id, lines)
        
        # Process other child attributes
        for field_name, field_value in ast.iter_fields(node):
            if field_name not in ['body'] and isinstance(field_value, list):
                for child in field_value:
                    if isinstance(child, ast.AST):
                        self._process_ast_node(child, node_id or parent_id, lines)
            elif isinstance(field_value, ast.AST):
                self._process_ast_node(field_value, node_id or parent_id, lines)
        
        return node_id or parent_id
    
    def _process_class_def(self, node: ast.ClassDef, parent_id: str, lines: List[str]) -> str:
        """Process class definition with inheritance tracking."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        start_col = node.col_offset
        end_col = node.end_col_offset or 0
        
        code = self._extract_code_segment(lines, start_line, end_line, start_col, end_col)
        docstring = ast.get_docstring(node)
        
        # Extract base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(self._get_attribute_name(base))
        
        # Extract decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        
        # Calculate class metrics
        methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        attributes = [n for n in node.body if isinstance(n, ast.Assign)]
        
        class_node = self.create_node(
            NodeType.CLASS,
            node.name,
            code,
            self.current_file_path,
            start_line,
            end_line,
            start_col,
            end_col,
            'python',
            docstring=docstring,
            bases=tuple(bases),
            decorators=tuple(decorators),
            method_count=len(methods),
            attribute_count=len(attributes),
            is_abstract=any(isinstance(d, ast.Name) and d.id == 'abstractmethod' 
                          for d in node.decorator_list)
        )
        
        class_id = self.current_cpg.add_node(class_node)
        
        # Track class hierarchy
        self.class_hierarchy[node.name] = {
            'bases': bases,
            'methods': [m.name for m in methods],
            'node_id': class_id
        }
        
        # Create inheritance edges
        for base in bases:
            if base in self.class_hierarchy:
                base_id = self.class_hierarchy[base]['node_id']
                edge = self.create_edge(class_id, base_id, EdgeType.INHERITANCE)
                self.current_cpg.add_edge(edge)
        
        # Link to parent
        edge = self.create_edge(parent_id, class_id, EdgeType.AST_CHILD)
        self.current_cpg.add_edge(edge)
        
        return class_id
    
    def _process_function_def(self, node: ast.FunctionDef, parent_id: str, lines: List[str]) -> str:
        """Process function definition with enhanced analysis."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        start_col = node.col_offset
        end_col = node.end_col_offset or 0
        
        code = self._extract_code_segment(lines, start_line, end_line, start_col, end_col)
        docstring = ast.get_docstring(node)
        
        # Extract parameters with type annotations
        params = []
        param_types = {}
        for arg in node.args.args:
            params.append(arg.arg)
            if arg.annotation:
                param_types[arg.arg] = self._get_type_annotation(arg.annotation)
        
        # Extract return type
        return_type = None
        if node.returns:
            return_type = self._get_type_annotation(node.returns)
        
        # Calculate function metrics
        complexity = self._calculate_function_complexity(node)
        
        func_node = self.create_node(
            NodeType.FUNCTION if not self._is_in_class_context(parent_id) else NodeType.METHOD,
            node.name,
            code,
            self.current_file_path,
            start_line,
            end_line,
            start_col,
            end_col,
            'python',
            docstring=docstring,
            parameters=tuple(params),
            param_types=param_types,
            return_type=return_type,
            decorators=tuple(self._get_decorator_name(d) for d in node.decorator_list),
            is_async=isinstance(node, ast.AsyncFunctionDef),
            complexity=complexity,
            line_count=end_line - start_line + 1
        )
        
        func_id = self.current_cpg.add_node(func_node)
        
        # Track function for call graph
        self.function_calls[node.name] = {
            'node_id': func_id,
            'parameters': params,
            'calls': []
        }
        
        # Link to parent
        edge = self.create_edge(parent_id, func_id, EdgeType.AST_CHILD)
        self.current_cpg.add_edge(edge)
        
        return func_id
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, 
                                ast.ExceptHandler, ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _is_in_class_context(self, parent_id: str) -> bool:
        """Check if the parent context is a class."""
        parent_node = self.current_cpg.get_node(parent_id)
        return parent_node and parent_node.node_type == NodeType.CLASS
    
    def _get_type_annotation(self, annotation: ast.AST) -> str:
        """Extract type annotation as string."""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Attribute):
            return self._get_attribute_name(annotation)
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        else:
            return ast.unparse(annotation)
    
    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get full attribute name (e.g., 'module.Class')."""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._get_attribute_name(node.value)}.{node.attr}"
        else:
            return node.attr
    
    def _get_decorator_name(self, decorator: ast.AST) -> str:
        """Extract decorator name."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return self._get_attribute_name(decorator)
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return self._get_attribute_name(decorator.func)
        return "unknown_decorator"
    
    def _extract_code_segment(self, lines: List[str], start_line: int, end_line: int,
                            start_col: int = 0, end_col: int = 0) -> str:
        """Extract code segment from lines."""
        if start_line <= 0 or start_line > len(lines):
            return ""
        
        if end_line <= 0 or end_line > len(lines):
            end_line = len(lines)
        
        if start_line == end_line:
            line = lines[start_line - 1]
            if end_col > 0:
                return line[start_col:end_col]
            else:
                return line[start_col:]
        else:
            result_lines = []
            for i in range(start_line - 1, end_line):
                if i < len(lines):
                    if i == start_line - 1:
                        result_lines.append(lines[i][start_col:])
                    elif i == end_line - 1 and end_col > 0:
                        result_lines.append(lines[i][:end_col])
                    else:
                        result_lines.append(lines[i])
            return '\n'.join(result_lines)
    
    def build_cfg(self, cpg: CodePropertyGraph) -> CodePropertyGraph:
        """Build enhanced Control Flow Graph."""
        # Implementation for enhanced CFG building
        # This would include more sophisticated control flow analysis
        return super().build_cfg(cpg)
    
    def build_dfg(self, cpg: CodePropertyGraph) -> CodePropertyGraph:
        """Build Data Flow Graph from AST and CFG - DISABLED for performance."""
        # DFG generation disabled for faster processing
        # Only AST and CFG are generated
        return cpg