#!/usr/bin/env python3
"""
Python CPG Parser

This module implements a comprehensive Code Property Graph parser for Python,
generating accurate AST, CFG, and DFG representations.
"""

import ast
import logging
from typing import Dict, List, Set, Optional, Any
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpg_core import CPGParser, CodePropertyGraph, CPGNode, CPGEdge, NodeType, EdgeType

logger = logging.getLogger(__name__)


class PythonCPGParser(CPGParser):
    """Python-specific CPG parser using Python's built-in AST module."""
    
    def __init__(self):
        """Initialize the Python parser."""
        self.current_cpg = None
        self.current_file_path = ""
        self.variable_definitions = {}  # Track variable definitions for DFG
        self.variable_uses = {}  # Track variable uses for DFG
        self.basic_blocks = []  # Track basic blocks for CFG
        self.current_block = None
        self.block_counter = 0
    
    def parse(self, content: str, file_path: str) -> CodePropertyGraph:
        """Parse Python source code and generate complete CPG."""
        self.current_file_path = file_path
        self.current_cpg = CodePropertyGraph()
        
        # Reset state
        self.variable_definitions.clear()
        self.variable_uses.clear()
        self.basic_blocks.clear()
        self.current_block = None
        self.block_counter = 0
        
        # Build AST first
        self.build_ast(content, file_path)
        
        # Then build CFG
        self.build_cfg(self.current_cpg)
        
        # Finally build DFG
        self.build_dfg(self.current_cpg)
        
        return self.current_cpg
    
    def build_ast(self, content: str, file_path: str) -> CodePropertyGraph:
        """Build Abstract Syntax Tree for Python code."""
        try:
            tree = ast.parse(content, filename=file_path)
            
            # Create module node
            module_node = self.create_node(
                NodeType.MODULE,
                Path(file_path).stem,
                content,
                file_path,
                1,
                len(content.splitlines()),
                0,
                len(content.splitlines()[-1]) if content.splitlines() else 0,
                'python'
            )
            module_id = self.current_cpg.add_node(module_node)
            
            # Process all AST nodes
            self._process_ast_node(tree, module_id, content.splitlines())
            
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            # Create error node
            error_node = self.create_node(
                NodeType.MODULE,
                "syntax_error",
                content,
                file_path,
                1,
                len(content.splitlines()),
                language='python',
                error=str(e)
            )
            self.current_cpg.add_node(error_node)
        
        return self.current_cpg
    
    def _process_ast_node(self, node: ast.AST, parent_id: str, lines: List[str]) -> str:
        """Process a single AST node and create corresponding CPG node."""
        node_id = None
        
        if isinstance(node, ast.ClassDef):
            node_id = self._process_class(node, parent_id, lines)
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            node_id = self._process_function(node, parent_id, lines)
        elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            node_id = self._process_import(node, parent_id, lines)
        elif isinstance(node, ast.Assign):
            node_id = self._process_assignment(node, parent_id, lines)
        elif isinstance(node, ast.Call):
            node_id = self._process_call(node, parent_id, lines)
        elif isinstance(node, ast.If):
            node_id = self._process_if(node, parent_id, lines)
        elif isinstance(node, ast.For) or isinstance(node, ast.While):
            node_id = self._process_loop(node, parent_id, lines)
        elif isinstance(node, ast.Return):
            node_id = self._process_return(node, parent_id, lines)
        elif isinstance(node, ast.Try):
            node_id = self._process_try(node, parent_id, lines)
        elif isinstance(node, ast.Name):
            node_id = self._process_name(node, parent_id, lines)
        elif isinstance(node, ast.Constant) or isinstance(node, ast.Str) or isinstance(node, ast.Num):
            node_id = self._process_literal(node, parent_id, lines)
        else:
            # Generic node processing
            node_id = self._process_generic_node(node, parent_id, lines)
        
        # Process child nodes
        if node_id:
            for child in ast.iter_child_nodes(node):
                child_id = self._process_ast_node(child, node_id, lines)
                if child_id:
                    # Create AST edge
                    edge = self.create_edge(node_id, child_id, EdgeType.AST_CHILD)
                    self.current_cpg.add_edge(edge)
                    
                    # Create reverse edge
                    parent_edge = self.create_edge(child_id, node_id, EdgeType.AST_PARENT)
                    self.current_cpg.add_edge(parent_edge)
        
        return node_id
    
    def _process_class(self, node: ast.ClassDef, parent_id: str, lines: List[str]) -> str:
        """Process a class definition."""
        start_line = getattr(node, 'lineno', 1)
        end_line = getattr(node, 'end_lineno', start_line)
        start_col = getattr(node, 'col_offset', 0)
        end_col = getattr(node, 'end_col_offset', 0)
        
        # Extract class code
        code_lines = lines[start_line-1:end_line] if end_line <= len(lines) else lines[start_line-1:]
        code = '\n'.join(code_lines)
        
        # Get docstring
        docstring = ast.get_docstring(node)
        
        # Get base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(self._get_attribute_name(base))
        
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
            decorators=tuple(self._get_decorator_name(d) for d in node.decorator_list)
        )
        
        class_id = self.current_cpg.add_node(class_node)
        
        # Create inheritance edges
        for base in bases:
            # This would need to be resolved later with full program analysis
            pass
        
        return class_id
    
    def _process_function(self, node: ast.FunctionDef, parent_id: str, lines: List[str]) -> str:
        """Process a function definition."""
        start_line = getattr(node, 'lineno', 1)
        end_line = getattr(node, 'end_lineno', start_line)
        start_col = getattr(node, 'col_offset', 0)
        end_col = getattr(node, 'end_col_offset', 0)
        
        # Extract function code
        code_lines = lines[start_line-1:end_line] if end_line <= len(lines) else lines[start_line-1:]
        code = '\n'.join(code_lines)
        
        # Get docstring
        docstring = ast.get_docstring(node)
        
        # Get parameters
        params = []
        for arg in node.args.args:
            params.append(arg.arg)
        
        # Determine if it's a method (inside a class)
        node_type = NodeType.METHOD if self._is_inside_class(parent_id) else NodeType.FUNCTION
        
        func_node = self.create_node(
            node_type,
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
            decorators=tuple(self._get_decorator_name(d) for d in node.decorator_list),
            is_async=isinstance(node, ast.AsyncFunctionDef)
        )
        
        func_id = self.current_cpg.add_node(func_node)
        
        # Process parameters as nodes
        for arg in node.args.args:
            param_node = self.create_node(
                NodeType.PARAMETER,
                arg.arg,
                arg.arg,
                self.current_file_path,
                getattr(arg, 'lineno', start_line),
                getattr(arg, 'end_lineno', start_line),
                getattr(arg, 'col_offset', 0),
                getattr(arg, 'end_col_offset', 0),
                'python'
            )
            param_id = self.current_cpg.add_node(param_node)
            
            # Link parameter to function
            edge = self.create_edge(func_id, param_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Track parameter definition for DFG
            self.variable_definitions[arg.arg] = param_id
        
        return func_id
    
    def _process_import(self, node: ast.AST, parent_id: str, lines: List[str]) -> str:
        """Process import statements."""
        start_line = getattr(node, 'lineno', 1)
        end_line = getattr(node, 'end_lineno', start_line)
        start_col = getattr(node, 'col_offset', 0)
        end_col = getattr(node, 'end_col_offset', 0)
        
        # Extract import code
        code_lines = lines[start_line-1:end_line] if end_line <= len(lines) else lines[start_line-1:]
        code = '\n'.join(code_lines)
        
        # Get imported names
        imported_names = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imported_names.append(f"{module}.{alias.name}" if module else alias.name)
        
        import_node = self.create_node(
            NodeType.IMPORT,
            "import",
            code,
            self.current_file_path,
            start_line,
            end_line,
            start_col,
            end_col,
            'python',
            imported_names=tuple(imported_names)
        )
        
        return self.current_cpg.add_node(import_node)
    
    def _process_assignment(self, node: ast.Assign, parent_id: str, lines: List[str]) -> str:
        """Process assignment statements."""
        start_line = getattr(node, 'lineno', 1)
        end_line = getattr(node, 'end_lineno', start_line)
        start_col = getattr(node, 'col_offset', 0)
        end_col = getattr(node, 'end_col_offset', 0)
        
        # Extract assignment code
        code_lines = lines[start_line-1:end_line] if end_line <= len(lines) else lines[start_line-1:]
        code = '\n'.join(code_lines)
        
        # Get target names
        targets = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                targets.append(target.id)
            elif isinstance(target, ast.Attribute):
                targets.append(self._get_attribute_name(target))
        
        assign_node = self.create_node(
            NodeType.ASSIGNMENT,
            "assignment",
            code,
            self.current_file_path,
            start_line,
            end_line,
            start_col,
            end_col,
            'python',
            targets=tuple(targets)
        )
        
        assign_id = self.current_cpg.add_node(assign_node)
        
        # Track variable definitions for DFG
        for target in targets:
            if isinstance(target, str):  # Simple variable name
                self.variable_definitions[target] = assign_id
        
        return assign_id
    
    def _process_call(self, node: ast.Call, parent_id: str, lines: List[str]) -> str:
        """Process function calls."""
        start_line = getattr(node, 'lineno', 1)
        end_line = getattr(node, 'end_lineno', start_line)
        start_col = getattr(node, 'col_offset', 0)
        end_col = getattr(node, 'end_col_offset', 0)
        
        # Extract call code
        code_lines = lines[start_line-1:end_line] if end_line <= len(lines) else lines[start_line-1:]
        code = '\n'.join(code_lines)
        
        # Get function name
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = self._get_attribute_name(node.func)
        
        call_node = self.create_node(
            NodeType.CALL,
            func_name,
            code,
            self.current_file_path,
            start_line,
            end_line,
            start_col,
            end_col,
            'python',
            function_name=func_name,
            num_args=len(node.args)
        )
        
        return self.current_cpg.add_node(call_node)
    
    def _process_if(self, node: ast.If, parent_id: str, lines: List[str]) -> str:
        """Process if statements."""
        start_line = getattr(node, 'lineno', 1)
        end_line = getattr(node, 'end_lineno', start_line)
        start_col = getattr(node, 'col_offset', 0)
        end_col = getattr(node, 'end_col_offset', 0)
        
        # Extract if code
        code_lines = lines[start_line-1:end_line] if end_line <= len(lines) else lines[start_line-1:]
        code = '\n'.join(code_lines)
        
        if_node = self.create_node(
            NodeType.CONDITION,
            "if",
            code,
            self.current_file_path,
            start_line,
            end_line,
            start_col,
            end_col,
            'python',
            has_else=len(node.orelse) > 0
        )
        
        return self.current_cpg.add_node(if_node)
    
    def _process_loop(self, node: ast.AST, parent_id: str, lines: List[str]) -> str:
        """Process loop statements."""
        start_line = getattr(node, 'lineno', 1)
        end_line = getattr(node, 'end_lineno', start_line)
        start_col = getattr(node, 'col_offset', 0)
        end_col = getattr(node, 'end_col_offset', 0)
        
        # Extract loop code
        code_lines = lines[start_line-1:end_line] if end_line <= len(lines) else lines[start_line-1:]
        code = '\n'.join(code_lines)
        
        loop_type = "for" if isinstance(node, ast.For) else "while"
        
        loop_node = self.create_node(
            NodeType.LOOP,
            loop_type,
            code,
            self.current_file_path,
            start_line,
            end_line,
            start_col,
            end_col,
            'python',
            loop_type=loop_type
        )
        
        return self.current_cpg.add_node(loop_node)
    
    def _process_return(self, node: ast.Return, parent_id: str, lines: List[str]) -> str:
        """Process return statements."""
        start_line = getattr(node, 'lineno', 1)
        end_line = getattr(node, 'end_lineno', start_line)
        start_col = getattr(node, 'col_offset', 0)
        end_col = getattr(node, 'end_col_offset', 0)
        
        # Extract return code
        code_lines = lines[start_line-1:end_line] if end_line <= len(lines) else lines[start_line-1:]
        code = '\n'.join(code_lines)
        
        return_node = self.create_node(
            NodeType.RETURN,
            "return",
            code,
            self.current_file_path,
            start_line,
            end_line,
            start_col,
            end_col,
            'python'
        )
        
        return self.current_cpg.add_node(return_node)
    
    def _process_try(self, node: ast.Try, parent_id: str, lines: List[str]) -> str:
        """Process try-except statements."""
        start_line = getattr(node, 'lineno', 1)
        end_line = getattr(node, 'end_lineno', start_line)
        start_col = getattr(node, 'col_offset', 0)
        end_col = getattr(node, 'end_col_offset', 0)
        
        # Extract try code
        code_lines = lines[start_line-1:end_line] if end_line <= len(lines) else lines[start_line-1:]
        code = '\n'.join(code_lines)
        
        try_node = self.create_node(
            NodeType.EXCEPTION,
            "try",
            code,
            self.current_file_path,
            start_line,
            end_line,
            start_col,
            end_col,
            'python',
            num_handlers=len(node.handlers),
            has_finally=len(node.finalbody) > 0
        )
        
        return self.current_cpg.add_node(try_node)
    
    def _process_name(self, node: ast.Name, parent_id: str, lines: List[str]) -> str:
        """Process variable names."""
        start_line = getattr(node, 'lineno', 1)
        end_line = getattr(node, 'end_lineno', start_line)
        start_col = getattr(node, 'col_offset', 0)
        end_col = getattr(node, 'end_col_offset', 0)
        
        name_node = self.create_node(
            NodeType.VARIABLE,
            node.id,
            node.id,
            self.current_file_path,
            start_line,
            end_line,
            start_col,
            end_col,
            'python',
            context=type(node.ctx).__name__
        )
        
        name_id = self.current_cpg.add_node(name_node)
        
        # Track variable use for DFG
        if isinstance(node.ctx, ast.Load):
            if node.id not in self.variable_uses:
                self.variable_uses[node.id] = []
            self.variable_uses[node.id].append(name_id)
        
        return name_id
    
    def _process_literal(self, node: ast.AST, parent_id: str, lines: List[str]) -> str:
        """Process literal values."""
        start_line = getattr(node, 'lineno', 1)
        end_line = getattr(node, 'end_lineno', start_line)
        start_col = getattr(node, 'col_offset', 0)
        end_col = getattr(node, 'end_col_offset', 0)
        
        # Get literal value
        value = ""
        if isinstance(node, ast.Constant):
            value = str(node.value)
        elif hasattr(node, 'n'):  # ast.Num
            value = str(node.n)
        elif hasattr(node, 's'):  # ast.Str
            value = str(node.s)
        
        literal_node = self.create_node(
            NodeType.LITERAL,
            value,
            value,
            self.current_file_path,
            start_line,
            end_line,
            start_col,
            end_col,
            'python',
            literal_type=type(node).__name__
        )
        
        return self.current_cpg.add_node(literal_node)
    
    def _process_generic_node(self, node: ast.AST, parent_id: str, lines: List[str]) -> str:
        """Process generic AST nodes."""
        start_line = getattr(node, 'lineno', 1)
        end_line = getattr(node, 'end_lineno', start_line)
        start_col = getattr(node, 'col_offset', 0)
        end_col = getattr(node, 'end_col_offset', 0)
        
        # Extract code for this node
        if start_line <= len(lines):
            code = lines[start_line-1]
        else:
            code = ""
        
        generic_node = self.create_node(
            NodeType.OPERATOR if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare)) else NodeType.LITERAL,
            type(node).__name__,
            code,
            self.current_file_path,
            start_line,
            end_line,
            start_col,
            end_col,
            'python',
            ast_type=type(node).__name__
        )
        
        return self.current_cpg.add_node(generic_node)
    
    def build_cfg(self, cpg: CodePropertyGraph) -> CodePropertyGraph:
        """Build Control Flow Graph from AST."""
        # Find all function and method nodes
        function_nodes = []
        for node in list(cpg.nodes.values()):
            if node.node_type in [NodeType.FUNCTION, NodeType.METHOD]:
                function_nodes.append(node)
        
        # Build CFG for each function
        for func_node in function_nodes:
            self._build_function_cfg(func_node, cpg)
        
        return cpg
    
    def _build_function_cfg(self, func_node: CPGNode, cpg: CodePropertyGraph):
        """Build CFG for a single function."""
        # Create entry and exit nodes
        entry_node = self.create_node(
            NodeType.ENTRY,
            f"{func_node.name}_entry",
            "",
            func_node.file_path,
            func_node.start_line,
            func_node.start_line,
            language='python'
        )
        entry_id = cpg.add_node(entry_node)
        
        exit_node = self.create_node(
            NodeType.EXIT,
            f"{func_node.name}_exit",
            "",
            func_node.file_path,
            func_node.end_line,
            func_node.end_line,
            language='python'
        )
        exit_id = cpg.add_node(exit_node)
        
        # Get all child nodes of the function (excluding entry/exit)
        child_nodes = [node for node in cpg.get_children(func_node.id) 
                      if node.node_type not in [NodeType.ENTRY, NodeType.EXIT]]
        
        # Create basic blocks and control flow edges
        if child_nodes:
            # Connect entry to first statement
            first_stmt = child_nodes[0]
            edge = self.create_edge(entry_id, first_stmt.id, EdgeType.CONTROL_FLOW)
            cpg.add_edge(edge)
            
            # Connect statements with proper control flow logic
            for i in range(len(child_nodes) - 1):
                current_node = child_nodes[i]
                next_node = child_nodes[i + 1]
                
                # Handle control flow based on node type
                if current_node.node_type == NodeType.CONDITION:
                    # If statement - create conditional edges
                    true_edge = self.create_edge(current_node.id, next_node.id, EdgeType.CONDITIONAL_TRUE)
                    cpg.add_edge(true_edge)
                    
                    # Find the next statement after the conditional block
                    if i + 2 < len(child_nodes):
                        false_target = child_nodes[i + 2]
                        false_edge = self.create_edge(current_node.id, false_target.id, EdgeType.CONDITIONAL_FALSE)
                        cpg.add_edge(false_edge)
                    else:
                        false_edge = self.create_edge(current_node.id, exit_id, EdgeType.CONDITIONAL_FALSE)
                        cpg.add_edge(false_edge)
                
                elif current_node.node_type == NodeType.LOOP:
                    # Loop - create forward edge and back edge
                    loop_edge = self.create_edge(current_node.id, next_node.id, EdgeType.CONTROL_FLOW)
                    cpg.add_edge(loop_edge)
                    
                    # Back edge for loop
                    back_edge = self.create_edge(next_node.id, current_node.id, EdgeType.CONTROL_FLOW)
                    cpg.add_edge(back_edge)
                
                elif current_node.node_type == NodeType.RETURN:
                    # Return statement - connect to exit
                    return_edge = self.create_edge(current_node.id, exit_id, EdgeType.CONTROL_FLOW)
                    cpg.add_edge(return_edge)
                
                else:
                    # Regular statement - sequential flow
                    flow_edge = self.create_edge(current_node.id, next_node.id, EdgeType.CONTROL_FLOW)
                    cpg.add_edge(flow_edge)
            
            # Connect last statement to exit
            if child_nodes:
                last_stmt = child_nodes[-1]
                if last_stmt.node_type not in [NodeType.RETURN, NodeType.ENTRY, NodeType.EXIT]:
                    edge = self.create_edge(last_stmt.id, exit_id, EdgeType.CONTROL_FLOW)
                    cpg.add_edge(edge)
    
    def build_dfg(self, cpg: CodePropertyGraph) -> CodePropertyGraph:
        """Build Data Flow Graph from AST and CFG - DISABLED for performance."""
        # DFG generation disabled for faster processing
        # Only AST and CFG are generated
        return cpg
    
    def _extract_variables_from_code(self, code: str) -> List[str]:
        """Extract variable names from code snippet."""
        import re
        # Simple regex to find variable names (can be improved)
        var_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        variables = []
        
        for match in re.finditer(var_pattern, code):
            var_name = match.group(0)
            # Skip keywords and common tokens
            if var_name not in ['def', 'class', 'if', 'else', 'for', 'while', 'return', 'self', 'import', 'from', 'as', 'try', 'except', 'finally', 'with', 'lambda', 'and', 'or', 'not', 'in', 'is', 'None', 'True', 'False']:
                variables.append(var_name)
        
        return list(set(variables))
    
    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get full attribute name from AST node."""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._get_attribute_name(node.value)}.{node.attr}"
        else:
            return node.attr
    
    def _get_decorator_name(self, node: ast.AST) -> str:
        """Get decorator name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        else:
            return str(node)
    
    def _is_inside_class(self, parent_id: str) -> bool:
        """Check if a node is inside a class definition."""
        if not self.current_cpg or not parent_id:
            return False
        
        parent = self.current_cpg.get_node(parent_id)
        while parent:
            if parent.node_type == NodeType.CLASS:
                return True
            # Get parent's parent
            parent_edges = self.current_cpg.get_incoming_edges(parent.id, EdgeType.AST_PARENT)
            if parent_edges:
                parent = self.current_cpg.get_node(parent_edges[0].source_id)
            else:
                break
        
        return False
