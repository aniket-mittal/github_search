#!/usr/bin/env python3
"""
PHP CPG Parser

This module implements a Code Property Graph parser for PHP,
using regex-based parsing for AST, CFG, and DFG generation.
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


class PhpCPGParser(CPGParser):
    """PHP CPG parser using regex patterns."""
    
    def __init__(self):
        """Initialize the PHP parser."""
        self.current_cpg = None
        self.current_file_path = ""
        self.variable_definitions = {}
        self.variable_uses = {}
        
        # PHP patterns
        self.patterns = {
            'php_open': r'<\?php',
            'namespace': r'namespace\s+([\w\\]+);',
            'use': r'use\s+([\w\\]+)(?:\s+as\s+(\w+))?;',
            'class': r'(?:abstract\s+|final\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w\s,]+))?\s*\{',
            'interface': r'interface\s+(\w+)(?:\s+extends\s+([\w\s,]+))?\s*\{',
            'trait': r'trait\s+(\w+)\s*\{',
            'function': r'function\s+(\w+)\s*\([^)]*\)\s*(?::\s*\w+)?\s*\{',
            'method': r'(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:abstract\s+)?function\s+(\w+)\s*\([^)]*\)\s*(?::\s*\w+)?\s*\{',
            'property': r'(?:public\s+|private\s+|protected\s+)?(?:static\s+)?\$(\w+)(?:\s*=\s*[^;]+)?;',
            'variable': r'\$(\w+)\s*=\s*([^;]+);',
            'constant': r'(?:const\s+(\w+)\s*=\s*[^;]+;|define\s*\(\s*[\'"](\w+)[\'"]\s*,\s*[^)]+\))',
            'function_call': r'(\w+(?:->|::|\w)*)\s*\([^)]*\)',
            'if_statement': r'if\s*\([^)]+\)\s*\{',
            'for_loop': r'for\s*\([^)]*\)\s*\{',
            'foreach_loop': r'foreach\s*\([^)]+\)\s*\{',
            'while_loop': r'while\s*\([^)]+\)\s*\{',
            'switch_statement': r'switch\s*\([^)]+\)\s*\{',
            'try_catch': r'try\s*\{',
            'return': r'return\s+[^;]*;',
            'echo': r'echo\s+[^;]*;',
            'include': r'(?:include|require)(?:_once)?\s*\([^)]+\);?',
        }
    
    def parse(self, content: str, file_path: str) -> CodePropertyGraph:
        """Parse PHP source code and generate complete CPG."""
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
        """Build Abstract Syntax Tree for PHP code."""
        lines = content.splitlines()
        
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
            'php'
        )
        module_id = self.current_cpg.add_node(module_node)
        
        # Process different constructs
        self._process_namespace(content, lines, module_id)
        self._process_includes(content, lines, module_id)
        self._process_classes(content, lines, module_id)
        self._process_functions(content, lines, module_id)
        self._process_variables(content, lines, module_id)
        self._process_function_calls(content, lines, module_id)
        self._process_control_flow(content, lines, module_id)
        
        return self.current_cpg
    
    def _process_namespace(self, content: str, lines: List[str], parent_id: str):
        """Process namespace declarations."""
        for match in re.finditer(self.patterns['namespace'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            namespace_name = match.group(1)
            
            namespace_node = self.create_node(
                NodeType.MODULE,
                namespace_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'php',
                namespace=namespace_name
            )
            namespace_id = self.current_cpg.add_node(namespace_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, namespace_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_includes(self, content: str, lines: List[str], parent_id: str):
        """Process include/require statements."""
        for match in re.finditer(self.patterns['include'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            
            include_node = self.create_node(
                NodeType.IMPORT,
                "include",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'php'
            )
            include_id = self.current_cpg.add_node(include_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, include_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Use statements
        for match in re.finditer(self.patterns['use'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            use_name = match.group(1)
            
            use_node = self.create_node(
                NodeType.IMPORT,
                use_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'php',
                use_name=use_name
            )
            use_id = self.current_cpg.add_node(use_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, use_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_classes(self, content: str, lines: List[str], parent_id: str):
        """Process class definitions."""
        for match in re.finditer(self.patterns['class'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            class_name = match.group(1)
            
            # Find the end of the class
            brace_pos = content.find('{', match.start())
            if brace_pos != -1:
                end_pos = self._find_matching_brace(content, brace_pos)
                if end_pos:
                    end_line = content[:end_pos].count('\n') + 1
                    class_code = content[match.start():end_pos]
                else:
                    end_line = start_line
                    class_code = match.group(0)
            else:
                end_line = start_line
                class_code = match.group(0)
            
            class_node = self.create_node(
                NodeType.CLASS,
                class_name,
                class_code,
                self.current_file_path,
                start_line,
                end_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                (end_pos if end_pos else match.end()) - content.rfind('\n', 0, end_pos if end_pos else match.end()) - 1,
                'php',
                class_name=class_name
            )
            class_id = self.current_cpg.add_node(class_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, class_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Process methods and properties within class
            if class_code:
                self._process_methods_in_class(class_code, class_id, start_line)
                self._process_properties_in_class(class_code, class_id, start_line)
    
    def _process_methods_in_class(self, class_content: str, class_id: str, class_start_line: int):
        """Process methods within a class."""
        for match in re.finditer(self.patterns['method'], class_content, re.MULTILINE):
            start_line = class_start_line + class_content[:match.start()].count('\n')
            method_name = match.group(1)
            
            # Find method body
            brace_pos = class_content.find('{', match.start())
            if brace_pos != -1:
                end_pos = self._find_matching_brace(class_content, brace_pos)
                if end_pos:
                    end_line = class_start_line + class_content[:end_pos].count('\n')
                    method_code = class_content[match.start():end_pos]
                else:
                    end_line = start_line
                    method_code = match.group(0)
            else:
                end_line = start_line
                method_code = match.group(0)
            
            method_node = self.create_node(
                NodeType.METHOD,
                method_name,
                method_code,
                self.current_file_path,
                start_line,
                end_line,
                0,
                0,
                'php',
                method_name=method_name
            )
            method_id = self.current_cpg.add_node(method_node)
            
            # Link to class
            edge = self.create_edge(class_id, method_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Process statements within method
            if method_code:
                self._process_statements_in_function(method_code, method_id, start_line)
    
    def _process_properties_in_class(self, class_content: str, class_id: str, class_start_line: int):
        """Process properties within a class."""
        for match in re.finditer(self.patterns['property'], class_content, re.MULTILINE):
            start_line = class_start_line + class_content[:match.start()].count('\n')
            prop_name = match.group(1)
            
            prop_node = self.create_node(
                NodeType.VARIABLE,
                prop_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'php',
                property_name=prop_name,
                is_property=True
            )
            prop_id = self.current_cpg.add_node(prop_node)
            
            # Track property definition for DFG
            if prop_name not in self.variable_definitions:
                self.variable_definitions[prop_name] = []
            self.variable_definitions[prop_name].append(prop_id)
            
            # Link to class
            edge = self.create_edge(class_id, prop_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_functions(self, content: str, lines: List[str], parent_id: str):
        """Process function definitions."""
        for match in re.finditer(self.patterns['function'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            func_name = match.group(1)
            
            # Find the end of the function
            brace_pos = content.find('{', match.start())
            if brace_pos != -1:
                end_pos = self._find_matching_brace(content, brace_pos)
                if end_pos:
                    end_line = content[:end_pos].count('\n') + 1
                    func_code = content[match.start():end_pos]
                else:
                    end_line = start_line
                    func_code = match.group(0)
            else:
                end_line = start_line
                func_code = match.group(0)
            
            func_node = self.create_node(
                NodeType.FUNCTION,
                func_name,
                func_code,
                self.current_file_path,
                start_line,
                end_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                (end_pos if end_pos else match.end()) - content.rfind('\n', 0, end_pos if end_pos else match.end()) - 1,
                'php',
                function_name=func_name
            )
            func_id = self.current_cpg.add_node(func_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, func_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Process statements within function
            if func_code:
                self._process_statements_in_function(func_code, func_id, start_line)
    
    def _process_statements_in_function(self, func_content: str, func_id: str, func_start_line: int):
        """Process statements within a function body."""
        # Find variable assignments within function
        for match in re.finditer(self.patterns['variable'], func_content, re.MULTILINE):
            start_line = func_start_line + func_content[:match.start()].count('\n')
            var_name = match.group(1)
            
            var_node = self.create_node(
                NodeType.ASSIGNMENT,
                "assignment",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'php',
                variable_name=var_name
            )
            var_id = self.current_cpg.add_node(var_node)
            
            # Track variable definition for DFG
            if var_name not in self.variable_definitions:
                self.variable_definitions[var_name] = []
            self.variable_definitions[var_name].append(var_id)
            
            # Track variable usage in assignment expression
            used_vars = self._extract_php_variables_from_code(match.group(0))
            for used_var in used_vars:
                if used_var != var_name:  # Don't track self-reference
                    if used_var not in self.variable_uses:
                        self.variable_uses[used_var] = []
                    self.variable_uses[used_var].append(var_id)
            
            # Link to function
            edge = self.create_edge(func_id, var_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Process function calls
        self._process_function_calls_in_function(func_content, func_id, func_start_line)
    
    def _extract_php_variables_from_code(self, code: str) -> List[str]:
        """Extract variable names from PHP code."""
        variables = []
        
        # Skip PHP keywords
        php_keywords = {
            'if', 'else', 'elseif', 'for', 'foreach', 'while', 'do', 'switch', 'case', 'default',
            'break', 'continue', 'return', 'function', 'class', 'interface', 'trait',
            'public', 'private', 'protected', 'static', 'abstract', 'final', 'const',
            'true', 'false', 'null', 'this', 'self', 'parent', 'new', 'clone',
            'include', 'require', 'include_once', 'require_once', 'namespace', 'use',
            'echo', 'print', 'isset', 'empty', 'unset', 'array', 'string', 'int', 'float',
            'bool', 'object', 'mixed', 'void', 'callable', 'iterable'
        }
        
        # Find PHP variable patterns ($variable)
        for match in re.finditer(r'\$([a-zA-Z_][a-zA-Z0-9_]*)', code):
            var_name = match.group(1)
            if var_name not in php_keywords and var_name not in variables:
                variables.append(var_name)
        
        return variables
    
    def _process_function_calls_in_function(self, func_content: str, func_id: str, func_start_line: int):
        """Process function calls within a function body."""
        # Find function calls within function
        for match in re.finditer(self.patterns['function_call'], func_content, re.MULTILINE):
            start_line = func_start_line + func_content[:match.start()].count('\n')
            call_name = match.group(1)
            
            call_node = self.create_node(
                NodeType.CALL,
                call_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'php',
                method_name=call_name
            )
            call_id = self.current_cpg.add_node(call_node)
            
            # Track variable usage for DFG - extract variables from the call expression
            used_vars = self._extract_php_variables_from_code(match.group(0))
            for var_name in used_vars:
                if var_name not in self.variable_uses:
                    self.variable_uses[var_name] = []
                self.variable_uses[var_name].append(call_id)
            
            # Link to function
            edge = self.create_edge(func_id, call_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Find control flow statements within function
        for match in re.finditer(self.patterns['if_statement'], func_content, re.MULTILINE):
            start_line = func_start_line + func_content[:match.start()].count('\n')
            
            if_node = self.create_node(
                NodeType.CONDITION,
                "if",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'php'
            )
            if_id = self.current_cpg.add_node(if_node)
            
            # Link to function
            edge = self.create_edge(func_id, if_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Find loops within function
        for pattern_name in ['for_loop', 'foreach_loop', 'while_loop']:
            for match in re.finditer(self.patterns[pattern_name], func_content, re.MULTILINE):
                start_line = func_start_line + func_content[:match.start()].count('\n')
                
                loop_node = self.create_node(
                    NodeType.LOOP,
                    pattern_name.replace('_loop', ''),
                    match.group(0),
                    self.current_file_path,
                    start_line,
                    start_line,
                    0,
                    0,
                    'php',
                    loop_type=pattern_name.replace('_loop', '')
                )
                loop_id = self.current_cpg.add_node(loop_node)
                
                # Link to function
                edge = self.create_edge(func_id, loop_id, EdgeType.AST_CHILD)
                self.current_cpg.add_edge(edge)
        
        # Find return statements within function
        for match in re.finditer(self.patterns['return'], func_content, re.MULTILINE):
            start_line = func_start_line + func_content[:match.start()].count('\n')
            
            return_node = self.create_node(
                NodeType.RETURN,
                "return",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'php'
            )
            return_id = self.current_cpg.add_node(return_node)
            
            # Link to function
            edge = self.create_edge(func_id, return_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_variables(self, content: str, lines: List[str], parent_id: str):
        """Process variable declarations and assignments."""
        for match in re.finditer(self.patterns['variable'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            var_name = match.group(1)
            
            var_node = self.create_node(
                NodeType.ASSIGNMENT,
                "assignment",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'php',
                variable_name=var_name
            )
            var_id = self.current_cpg.add_node(var_node)
            
            # Track variable definition for DFG
            if var_name not in self.variable_definitions:
                self.variable_definitions[var_name] = []
            self.variable_definitions[var_name].append(var_id)
            
            # Link to parent
            edge = self.create_edge(parent_id, var_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_function_calls(self, content: str, lines: List[str], parent_id: str):
        """Process function calls."""
        for match in re.finditer(self.patterns['function_call'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            func_name = match.group(1)
            
            call_node = self.create_node(
                NodeType.CALL,
                func_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'php',
                function_name=func_name
            )
            call_id = self.current_cpg.add_node(call_node)
            
            # Track variable usage for DFG
            base_name = func_name.split('->')[0].split('::')[0]
            if base_name.startswith('$'):
                base_name = base_name[1:]  # Remove $ prefix
                if base_name not in self.variable_uses:
                    self.variable_uses[base_name] = []
                self.variable_uses[base_name].append(call_id)
            
            # Link to parent
            edge = self.create_edge(parent_id, call_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_control_flow(self, content: str, lines: List[str], parent_id: str):
        """Process control flow statements."""
        # If statements
        for match in re.finditer(self.patterns['if_statement'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            
            if_node = self.create_node(
                NodeType.CONDITION,
                "if",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'php'
            )
            if_id = self.current_cpg.add_node(if_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, if_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Loops
        for pattern_name in ['for_loop', 'foreach_loop', 'while_loop']:
            for match in re.finditer(self.patterns[pattern_name], content, re.MULTILINE):
                start_line = content[:match.start()].count('\n') + 1
                
                loop_node = self.create_node(
                    NodeType.LOOP,
                    pattern_name.replace('_loop', ''),
                    match.group(0),
                    self.current_file_path,
                    start_line,
                    start_line,
                    match.start() - content.rfind('\n', 0, match.start()) - 1,
                    match.end() - content.rfind('\n', 0, match.end()) - 1,
                    'php',
                    loop_type=pattern_name.replace('_loop', '')
                )
                loop_id = self.current_cpg.add_node(loop_node)
                
                # Link to parent
                edge = self.create_edge(parent_id, loop_id, EdgeType.AST_CHILD)
                self.current_cpg.add_edge(edge)
    
    def _find_matching_brace(self, content: str, start_pos: int) -> Optional[int]:
        """Find the matching closing brace for an opening brace."""
        if start_pos >= len(content) or content[start_pos] != '{':
            return None
        
        brace_count = 1
        pos = start_pos + 1
        
        while pos < len(content) and brace_count > 0:
            if content[pos] == '{':
                brace_count += 1
            elif content[pos] == '}':
                brace_count -= 1
            pos += 1
        
        return pos if brace_count == 0 else None
    
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
            language='php'
        )
        entry_id = cpg.add_node(entry_node)
        
        exit_node = self.create_node(
            NodeType.EXIT,
            f"{func_node.name}_exit",
            "",
            func_node.file_path,
            func_node.end_line,
            func_node.end_line,
            language='php'
        )
        exit_id = cpg.add_node(exit_node)
        
        # Link entry and exit to function
        edge = self.create_edge(func_node.id, entry_id, EdgeType.AST_CHILD)
        cpg.add_edge(edge)
        edge = self.create_edge(func_node.id, exit_id, EdgeType.AST_CHILD)
        cpg.add_edge(edge)
        
        # Get all child nodes of the function (excluding entry/exit)
        child_nodes = [node for node in cpg.get_children(func_node.id) 
                      if node.node_type not in [NodeType.ENTRY, NodeType.EXIT]]
        
        # Create control flow edges
        if child_nodes:
            # Connect entry to first statement
            first_stmt = child_nodes[0]
            edge = self.create_edge(entry_id, first_stmt.id, EdgeType.CONTROL_FLOW)
            cpg.add_edge(edge)
            
            # Connect statements with control flow logic
            for i in range(len(child_nodes) - 1):
                current_node = child_nodes[i]
                next_node = child_nodes[i + 1]
                
                if current_node.node_type == NodeType.CONDITION:
                    # Conditional branches
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
                    # Loop edges
                    loop_edge = self.create_edge(current_node.id, next_node.id, EdgeType.CONTROL_FLOW)
                    cpg.add_edge(loop_edge)
                    
                    # Back edge for loop
                    back_edge = self.create_edge(next_node.id, current_node.id, EdgeType.CONTROL_FLOW)
                    cpg.add_edge(back_edge)
                
                else:
                    # Sequential flow
                    flow_edge = self.create_edge(current_node.id, next_node.id, EdgeType.CONTROL_FLOW)
                    cpg.add_edge(flow_edge)
            
            # Connect last statement to exit
            if child_nodes:
                last_stmt = child_nodes[-1]
                if last_stmt.node_type not in [NodeType.ENTRY, NodeType.EXIT]:
                    edge = self.create_edge(last_stmt.id, exit_id, EdgeType.CONTROL_FLOW)
                    cpg.add_edge(edge)
    
    def build_dfg(self, cpg: CodePropertyGraph) -> CodePropertyGraph:
        """Build Data Flow Graph from AST and CFG - DISABLED for performance."""
        # DFG generation disabled for faster processing
        # Only AST and CFG are generated
        return cpg