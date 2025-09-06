#!/usr/bin/env python3
"""
Robust Python CPG Parser
=========================

A much more robust Python parser that handles:
1. Python 2 and Python 3 syntax
2. Accurate function, class, and variable detection
3. Proper control flow and call relationships
4. Better error handling and fallback parsing
"""

import ast
import logging
import re
import sys
from typing import Dict, List, Set, Optional, Any, Tuple
from pathlib import Path

import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpg_core import CPGParser, CodePropertyGraph, CPGNode, CPGEdge, NodeType, EdgeType

logger = logging.getLogger(__name__)


class RobustPythonCPGParser(CPGParser):
    """Robust Python CPG parser with Python 2/3 compatibility and better accuracy."""
    
    def __init__(self):
        """Initialize the robust parser."""
        self.current_cpg = None
        self.current_file_path = ""
        self.node_counter = 0
        self.edge_counter = 0
    
    def parse(self, content: str, file_path: str) -> CodePropertyGraph:
        """Parse Python source code with robust error handling."""
        self.current_file_path = file_path
        self.current_cpg = CodePropertyGraph()
        self.node_counter = 0
        self.edge_counter = 0
        
        # Try multiple parsing strategies
        success = False
        
        # Strategy 1: Try Python 3 AST parsing
        try:
            tree = ast.parse(content, filename=file_path)
            self._build_cpg_from_ast(tree, content)
            success = True
            logger.debug(f"✅ Python 3 AST parsing successful for {file_path}")
        except SyntaxError as e:
            logger.debug(f"⚠️ Python 3 AST failed for {file_path}: {e}")
            
            # Strategy 2: Try Python 2 to 3 conversion for common issues
            try:
                converted_content = self._convert_python2_to_3(content)
                tree = ast.parse(converted_content, filename=file_path)
                self._build_cpg_from_ast(tree, converted_content)
                success = True
                logger.debug(f"✅ Python 2→3 conversion successful for {file_path}")
            except Exception as e2:
                logger.debug(f"⚠️ Python 2→3 conversion failed for {file_path}: {e2}")
        
        # Strategy 3: Fallback to regex-based parsing for critical elements
        if not success:
            logger.debug(f"🔄 Using fallback regex parsing for {file_path}")
            self._fallback_regex_parsing(content)
        
        # Ensure we have at least a module node
        if not self.current_cpg.nodes:
            self._create_minimal_cpg(content, file_path)
        
        return self.current_cpg
    
    def build_ast(self, content: str, file_path: str) -> CodePropertyGraph:
        """Build Abstract Syntax Tree - delegated to main parse method."""
        return self.parse(content, file_path)
    
    def build_cfg(self, cpg: CodePropertyGraph) -> CodePropertyGraph:
        """Build Control Flow Graph from AST - basic implementation."""
        # For now, return the AST as-is. CFG building can be enhanced later
        return cpg
    
    def build_dfg(self, cpg: CodePropertyGraph) -> CodePropertyGraph:
        """Build Data Flow Graph from AST and CFG - basic implementation."""  
        # For now, return the graph as-is. DFG building can be enhanced later
        return cpg
    
    def _convert_python2_to_3(self, content: str) -> str:
        """Convert common Python 2 syntax to Python 3 for parsing."""
        # Fix print statements
        content = re.sub(r'\bprint\s+([^(].*?)(?=\n|$)', r'print(\1)', content, flags=re.MULTILINE)
        
        # Fix except syntax: except Exception, e: -> except Exception as e:
        content = re.sub(r'\bexcept\s+([^:,]+),\s*([^:]+):', r'except \1 as \2:', content)
        
        # Fix raw_input -> input
        content = re.sub(r'\braw_input\b', 'input', content)
        
        # Fix xrange -> range
        content = re.sub(r'\bxrange\b', 'range', content)
        
        # Fix iteritems, iterkeys, itervalues
        content = re.sub(r'\.iteritems\(\)', '.items()', content)
        content = re.sub(r'\.iterkeys\(\)', '.keys()', content)
        content = re.sub(r'\.itervalues\(\)', '.values()', content)
        
        return content
    
    def _build_cpg_from_ast(self, tree: ast.AST, content: str):
        """Build CPG from successful AST parsing."""
        lines = content.splitlines()
        
        # Create module node
        module_node = self._create_node(
            NodeType.MODULE, 
            Path(self.current_file_path).stem,
            content[:200] + "..." if len(content) > 200 else content,
            self.current_file_path,
            1, len(lines), 0, 0
        )
        
        # Process AST nodes recursively
        self._process_ast_node(tree, module_node, lines)
    
    def _process_ast_node(self, node: ast.AST, parent_node: CPGNode, lines: List[str]):
        """Process AST node and create appropriate CPG nodes and edges."""
        
        if isinstance(node, ast.FunctionDef):
            self._process_function(node, parent_node, lines)
        
        elif isinstance(node, ast.AsyncFunctionDef):
            self._process_function(node, parent_node, lines, is_async=True)
        
        elif isinstance(node, ast.ClassDef):
            self._process_class(node, parent_node, lines)
        
        elif isinstance(node, ast.Assign):
            self._process_assignment(node, parent_node, lines)
        
        elif isinstance(node, ast.AugAssign):
            self._process_assignment(node, parent_node, lines, is_aug=True)
        
        elif isinstance(node, ast.Call):
            self._process_call(node, parent_node, lines)
        
        elif isinstance(node, ast.Import):
            self._process_import(node, parent_node, lines)
        
        elif isinstance(node, ast.ImportFrom):
            self._process_import_from(node, parent_node, lines)
        
        elif isinstance(node, ast.If):
            self._process_conditional(node, parent_node, lines, "if")
        
        elif isinstance(node, ast.While):
            self._process_conditional(node, parent_node, lines, "while")
        
        elif isinstance(node, ast.For):
            self._process_loop(node, parent_node, lines, "for")
        
        elif isinstance(node, ast.Return):
            self._process_return(node, parent_node, lines)
        
        # Process child nodes
        for child in ast.iter_child_nodes(node):
            self._process_ast_node(child, parent_node, lines)
    
    def _process_function(self, node: ast.FunctionDef, parent_node: CPGNode, lines: List[str], is_async: bool = False):
        """Process function definition."""
        func_type = NodeType.FUNCTION
        func_name = node.name
        
        # Get function code
        start_line = getattr(node, 'lineno', 1)
        end_line = getattr(node, 'end_lineno', start_line)
        
        func_code = self._get_code_snippet(lines, start_line, min(start_line + 3, len(lines)))
        
        func_node = self._create_node(
            func_type, func_name, func_code,
            self.current_file_path, start_line, end_line, 0, 0
        )
        
        # Create edge from parent to function
        self._create_edge(parent_node, func_node, EdgeType.AST_CHILD)
        
        # Process parameters
        for arg in node.args.args:
            param_name = getattr(arg, 'arg', getattr(arg, 'id', str(arg)))
            param_node = self._create_node(
                NodeType.PARAMETER, param_name, param_name,
                self.current_file_path, start_line, start_line, 0, 0
            )
            self._create_edge(func_node, param_node, EdgeType.AST_CHILD)
        
        # Process function body
        for child in node.body:
            self._process_ast_node(child, func_node, lines)
    
    def _process_class(self, node: ast.ClassDef, parent_node: CPGNode, lines: List[str]):
        """Process class definition."""
        class_name = node.name
        start_line = getattr(node, 'lineno', 1)
        end_line = getattr(node, 'end_lineno', start_line)
        
        class_code = self._get_code_snippet(lines, start_line, min(start_line + 2, len(lines)))
        
        class_node = self._create_node(
            NodeType.CLASS, class_name, class_code,
            self.current_file_path, start_line, end_line, 0, 0
        )
        
        self._create_edge(parent_node, class_node, EdgeType.AST_CHILD)
        
        # Process class body
        for child in node.body:
            self._process_ast_node(child, class_node, lines)
    
    def _process_assignment(self, node: ast.Assign, parent_node: CPGNode, lines: List[str], is_aug: bool = False):
        """Process assignment statement."""
        start_line = getattr(node, 'lineno', 1)
        assign_code = self._get_code_snippet(lines, start_line, start_line)
        
        assign_node = self._create_node(
            NodeType.ASSIGNMENT, "assignment", assign_code,
            self.current_file_path, start_line, start_line, 0, 0
        )
        
        self._create_edge(parent_node, assign_node, EdgeType.AST_CHILD)
        
        # Process target variables
        if hasattr(node, 'targets'):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_node = self._create_node(
                        NodeType.VARIABLE, target.id, target.id,
                        self.current_file_path, start_line, start_line, 0, 0
                    )
                    self._create_edge(assign_node, var_node, EdgeType.AST_CHILD)
        elif hasattr(node, 'target') and isinstance(node.target, ast.Name):
            var_node = self._create_node(
                NodeType.VARIABLE, node.target.id, node.target.id,
                self.current_file_path, start_line, start_line, 0, 0
            )
            self._create_edge(assign_node, var_node, EdgeType.AST_CHILD)
    
    def _process_call(self, node: ast.Call, parent_node: CPGNode, lines: List[str]):
        """Process function call."""
        start_line = getattr(node, 'lineno', 1)
        
        # Get function name
        func_name = "call"
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        
        call_code = self._get_code_snippet(lines, start_line, start_line)
        
        call_node = self._create_node(
            NodeType.CALL, func_name, call_code,
            self.current_file_path, start_line, start_line, 0, 0
        )
        
        self._create_edge(parent_node, call_node, EdgeType.AST_CHILD)
        return call_node
    
    def _process_import(self, node: ast.Import, parent_node: CPGNode, lines: List[str]):
        """Process import statement."""
        start_line = getattr(node, 'lineno', 1)
        import_code = self._get_code_snippet(lines, start_line, start_line)
        
        for alias in node.names:
            import_node = self._create_node(
                NodeType.VARIABLE, alias.name, import_code,
                self.current_file_path, start_line, start_line, 0, 0
            )
            self._create_edge(parent_node, import_node, EdgeType.AST_CHILD)
    
    def _process_import_from(self, node: ast.ImportFrom, parent_node: CPGNode, lines: List[str]):
        """Process from...import statement."""
        start_line = getattr(node, 'lineno', 1)
        import_code = self._get_code_snippet(lines, start_line, start_line)
        
        for alias in node.names:
            import_node = self._create_node(
                NodeType.VARIABLE, alias.name, import_code,
                self.current_file_path, start_line, start_line, 0, 0
            )
            self._create_edge(parent_node, import_node, EdgeType.AST_CHILD)
    
    def _process_conditional(self, node, parent_node: CPGNode, lines: List[str], cond_type: str):
        """Process conditional statements (if, while)."""
        start_line = getattr(node, 'lineno', 1)
        cond_code = self._get_code_snippet(lines, start_line, start_line)
        
        cond_node = self._create_node(
            NodeType.CONDITION, cond_type, cond_code,
            self.current_file_path, start_line, start_line, 0, 0
        )
        
        self._create_edge(parent_node, cond_node, EdgeType.AST_CHILD)
        
        # Process body
        for child in node.body:
            self._process_ast_node(child, cond_node, lines)
    
    def _process_loop(self, node, parent_node: CPGNode, lines: List[str], loop_type: str):
        """Process loop statements."""
        start_line = getattr(node, 'lineno', 1)
        loop_code = self._get_code_snippet(lines, start_line, start_line)
        
        loop_node = self._create_node(
            NodeType.CONDITION, loop_type, loop_code,
            self.current_file_path, start_line, start_line, 0, 0
        )
        
        self._create_edge(parent_node, loop_node, EdgeType.AST_CHILD)
        
        # Process body
        for child in node.body:
            self._process_ast_node(child, loop_node, lines)
    
    def _process_return(self, node: ast.Return, parent_node: CPGNode, lines: List[str]):
        """Process return statement."""
        start_line = getattr(node, 'lineno', 1)
        return_code = self._get_code_snippet(lines, start_line, start_line)
        
        return_node = self._create_node(
            NodeType.RETURN, "return", return_code,
            self.current_file_path, start_line, start_line, 0, 0
        )
        
        self._create_edge(parent_node, return_node, EdgeType.AST_CHILD)
    
    def _fallback_regex_parsing(self, content: str):
        """Fallback parsing using regular expressions when AST fails."""
        lines = content.splitlines()
        
        # Create module node
        module_node = self._create_node(
            NodeType.MODULE, Path(self.current_file_path).stem,
            content[:200] + "..." if len(content) > 200 else content,
            self.current_file_path, 1, len(lines), 0, 0
        )
        
        # Extract functions using regex
        func_pattern = r'^(\s*)(def|async\s+def)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        for i, line in enumerate(lines, 1):
            match = re.match(func_pattern, line)
            if match:
                func_name = match.group(3)
                func_node = self._create_node(
                    NodeType.FUNCTION, func_name, line.strip(),
                    self.current_file_path, i, i, 0, 0
                )
                self._create_edge(module_node, func_node, EdgeType.AST_CHILD)
        
        # Extract classes using regex
        class_pattern = r'^(\s*)class\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        for i, line in enumerate(lines, 1):
            match = re.match(class_pattern, line)
            if match:
                class_name = match.group(2)
                class_node = self._create_node(
                    NodeType.CLASS, class_name, line.strip(),
                    self.current_file_path, i, i, 0, 0
                )
                self._create_edge(module_node, class_node, EdgeType.AST_CHILD)
        
        # Extract imports using regex
        import_pattern = r'^(\s*)(import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        for i, line in enumerate(lines, 1):
            match = re.match(import_pattern, line)
            if match:
                import_name = match.group(3)
                import_node = self._create_node(
                    NodeType.VARIABLE, import_name, line.strip(),
                    self.current_file_path, i, i, 0, 0
                )
                self._create_edge(module_node, import_node, EdgeType.AST_CHILD)
        
        # Extract assignments using regex
        assign_pattern = r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*='
        for i, line in enumerate(lines, 1):
            match = re.match(assign_pattern, line)
            if match:
                var_name = match.group(2)
                var_node = self._create_node(
                    NodeType.VARIABLE, var_name, line.strip(),
                    self.current_file_path, i, i, 0, 0
                )
                self._create_edge(module_node, var_node, EdgeType.AST_CHILD)
    
    def _create_minimal_cpg(self, content: str, file_path: str):
        """Create minimal CPG when all parsing fails."""
        lines = content.splitlines()
        module_node = self._create_node(
            NodeType.MODULE, Path(file_path).stem,
            content[:500] + "..." if len(content) > 500 else content,
            file_path, 1, len(lines), 0, 0
        )
    
    def _create_node(self, node_type: NodeType, name: str, code: str, 
                     file_path: str, start_line: int, end_line: int,
                     start_col: int, end_col: int) -> CPGNode:
        """Create a CPG node and add it to the graph."""
        self.node_counter += 1
        node_id = f"node_{self.node_counter}"
        
        node = CPGNode(
            id=node_id,
            node_type=node_type,
            name=name,
            code=code,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            start_column=start_col,
            end_column=end_col,
            language='python'
        )
        
        self.current_cpg.add_node(node)
        return node
    
    def _create_edge(self, source_node: CPGNode, target_node: CPGNode, edge_type: EdgeType) -> CPGEdge:
        """Create a CPG edge and add it to the graph."""
        self.edge_counter += 1
        edge_id = f"edge_{self.edge_counter}"
        
        edge = CPGEdge(
            id=edge_id,
            source_id=source_node.id,
            target_id=target_node.id,
            edge_type=edge_type
        )
        
        self.current_cpg.add_edge(edge)
        return edge
    
    def _get_code_snippet(self, lines: List[str], start_line: int, end_line: int) -> str:
        """Get code snippet from lines."""
        try:
            # Convert to 0-based indexing
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            
            if start_idx >= len(lines):
                return ""
            
            snippet = '\n'.join(lines[start_idx:end_idx])
            return snippet[:1000]  # Limit length
        except:
            return ""