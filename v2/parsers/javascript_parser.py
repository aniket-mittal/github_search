#!/usr/bin/env python3
"""
JavaScript/TypeScript CPG Parser

This module implements a Code Property Graph parser for JavaScript and TypeScript,
using regex-based parsing and pattern matching for AST, CFG, and DFG generation.
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


class JavaScriptCPGParser(CPGParser):
    """JavaScript/TypeScript CPG parser using regex patterns."""
    
    def __init__(self):
        """Initialize the JavaScript parser."""
        self.current_cpg = None
        self.current_file_path = ""
        self.variable_definitions = {}
        self.variable_uses = {}
        self.scope_stack = []
        self.current_scope = "global"
        
        # JavaScript/TypeScript patterns
        self.patterns = {
            'class': r'class\s+(\w+)(?:\s+extends\s+(\w+))?\s*\{',
            'function_decl': r'function\s+(\w+)\s*\([^)]*\)\s*\{',
            'function_expr': r'(\w+)\s*[:=]\s*function\s*\([^)]*\)\s*\{',
            'arrow_function': r'(\w+)\s*[:=]\s*\([^)]*\)\s*=>\s*\{?',
            'arrow_function_simple': r'(\w+)\s*[:=]\s*\([^)]*\)\s*=>',
            'method': r'(\w+)\s*\([^)]*\)\s*\{',
            'import': r'import\s+.*?from\s+[\'"][^\'"]+[\'"];?',
            'require': r'require\s*\([\'"][^\'"]+[\'"]\)',
            'export': r'export\s+(?:default\s+)?(?:class|function|const|let|var)\s+(\w+)',
            'variable_decl': r'(?:var|let|const)\s+(\w+)(?:\s*[:=].*?)?[;\n]',
            'assignment': r'(\w+(?:\.\w+)*)\s*=\s*([^;\n]+)[;\n]',
            'if_statement': r'if\s*\([^)]+\)\s*\{',
            'for_loop': r'for\s*\([^)]*\)\s*\{',
            'while_loop': r'while\s*\([^)]+\)\s*\{',
            'function_call': r'(\w+(?:\.\w+)*)\s*\([^)]*\)',
            'return': r'return\s+[^;\n]*[;\n]',
            'try_catch': r'try\s*\{',
            'interface': r'interface\s+(\w+)(?:\s+extends\s+\w+)?\s*\{',
            'type_alias': r'type\s+(\w+)\s*=\s*[^;\n]+[;\n]',
        }
    
    def parse(self, content: str, file_path: str) -> CodePropertyGraph:
        """Parse JavaScript/TypeScript source code and generate complete CPG."""
        self.current_file_path = file_path
        self.current_cpg = CodePropertyGraph()
        
        # Reset state
        self.variable_definitions.clear()
        self.variable_uses.clear()
        self.scope_stack.clear()
        self.current_scope = "global"
        
        # Build AST
        self.build_ast(content, file_path)
        
        # Build CFG
        self.build_cfg(self.current_cpg)
        
        # Build DFG
        self.build_dfg(self.current_cpg)
        
        return self.current_cpg
    
    def build_ast(self, content: str, file_path: str) -> CodePropertyGraph:
        """Build Abstract Syntax Tree for JavaScript/TypeScript code."""
        # Work on a sanitized snapshot to avoid regex picking up text in comments
        sanitized = self._strip_comments(content)
        lines = sanitized.splitlines()
        
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
            'javascript'
        )
        module_id = self.current_cpg.add_node(module_node)
        
        # Process different constructs
        self._process_imports(sanitized, lines, module_id)
        self._process_classes(sanitized, lines, module_id)
        self._process_functions(sanitized, lines, module_id)
        self._process_variables(sanitized, lines, module_id)
        self._process_control_flow(sanitized, lines, module_id)
        self._process_function_calls(sanitized, lines, module_id)
        
        return self.current_cpg
    
    def _process_imports(self, content: str, lines: List[str], parent_id: str):
        """Process import and require statements."""
        # ES6 imports
        for match in re.finditer(self.patterns['import'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            end_line = start_line
            
            import_node = self.create_node(
                NodeType.IMPORT,
                "import",
                match.group(0),
                self.current_file_path,
                start_line,
                end_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'javascript'
            )
            import_id = self.current_cpg.add_node(import_node)
            
            # Link to module
            edge = self.create_edge(parent_id, import_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # CommonJS requires
        for match in re.finditer(self.patterns['require'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            end_line = start_line
            
            require_node = self.create_node(
                NodeType.IMPORT,
                "require",
                match.group(0),
                self.current_file_path,
                start_line,
                end_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'javascript'
            )
            require_id = self.current_cpg.add_node(require_node)
            
            # Link to module
            edge = self.create_edge(parent_id, require_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_classes(self, content: str, lines: List[str], parent_id: str):
        """Process class definitions."""
        for match in re.finditer(self.patterns['class'], content, re.MULTILINE):
            class_name = match.group(1)
            base_class = match.group(2) if match.groups() and len(match.groups()) > 1 else None
            
            start_line = content[:match.start()].count('\n') + 1
            
            # Find class end (simplified - looks for matching braces)
            end_pos = self._find_matching_brace(content, match.end() - 1)
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
                'javascript',
                base_class=base_class
            )
            class_id = self.current_cpg.add_node(class_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, class_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Process methods within class
            self._process_methods_in_class(class_content, class_id, start_line)
    
    def _process_functions(self, content: str, lines: List[str], parent_id: str):
        """Process function definitions."""
        # Function declarations
        for match in re.finditer(self.patterns['function_decl'], content, re.MULTILINE):
            func_name = match.group(1)
            self._create_function_node(content, match, func_name, parent_id, NodeType.FUNCTION)
        
        # Function expressions
        for match in re.finditer(self.patterns['function_expr'], content, re.MULTILINE):
            func_name = match.group(1)
            self._create_function_node(content, match, func_name, parent_id, NodeType.FUNCTION)
        
        # Arrow functions
        for match in re.finditer(self.patterns['arrow_function'], content, re.MULTILINE):
            func_name = match.group(1)
            self._create_function_node(content, match, func_name, parent_id, NodeType.FUNCTION)
        
        # Simple arrow functions
        for match in re.finditer(self.patterns['arrow_function_simple'], content, re.MULTILINE):
            func_name = match.group(1)
            self._create_function_node(content, match, func_name, parent_id, NodeType.FUNCTION)
    
    def _process_methods_in_class(self, class_content: str, class_id: str, class_start_line: int):
        """Process methods within a class."""
        for match in re.finditer(self.patterns['method'], class_content, re.MULTILINE):
            method_name = match.group(1)
            
            # Skip constructor and common non-method patterns
            if method_name in ['if', 'for', 'while', 'switch']:
                continue
            
            start_line = class_start_line + class_content[:match.start()].count('\n')
            
            # Find method end
            end_pos = self._find_matching_brace(class_content, match.end() - 1)
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
                'javascript'
            )
            method_id = self.current_cpg.add_node(method_node)
            
            # Link to class
            edge = self.create_edge(class_id, method_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Process statements within method
            self._process_statements_in_function(method_content, method_id, start_line)
    
    def _create_function_node(self, content: str, match: re.Match, func_name: str, 
                             parent_id: str, node_type: NodeType) -> str:
        """Create a function node from a regex match."""
        start_line = content[:match.start()].count('\n') + 1
        
        # Find function end
        end_pos = self._find_matching_brace(content, match.end() - 1)
        end_line = content[:end_pos].count('\n') + 1 if end_pos else start_line
        
        # Extract function content
        func_content = content[match.start():end_pos] if end_pos else match.group(0)
        
        func_node = self.create_node(
            node_type,
            func_name,
            func_content,
            self.current_file_path,
            start_line,
            end_line,
            match.start() - content.rfind('\n', 0, match.start()) - 1,
            0,
            'javascript'
        )
        func_id = self.current_cpg.add_node(func_node)
        
        # Link to parent
        edge = self.create_edge(parent_id, func_id, EdgeType.AST_CHILD)
        self.current_cpg.add_edge(edge)
        
        # Extract and process parameters
        self._extract_function_parameters(match.group(0), func_id, start_line)
        
        # Process statements within function
        if func_content:
            self._process_statements_in_function(func_content, func_id, start_line)
        
        return func_id
    
    def _extract_function_parameters(self, func_signature: str, func_id: str, start_line: int):
        """Extract parameters from function signature."""
        # Match parameters in parentheses
        param_match = re.search(r'\(([^)]*)\)', func_signature)
        if param_match:
            params_str = param_match.group(1).strip()
            if params_str:
                # Split parameters by comma and process each
                params = [p.strip() for p in params_str.split(',')]
                for i, param in enumerate(params):
                    if param:
                        # Handle destructuring and default parameters
                        param_name = param.split('=')[0].strip()
                        param_name = param_name.split(':')[0].strip()  # Remove TypeScript type annotations
                        param_name = re.sub(r'[{}[\]]', '', param_name).split()[0]  # Handle destructuring
                        
                        if param_name and param_name.isidentifier():
                            param_node = self.create_node(
                                NodeType.PARAMETER,
                                param_name,
                                param,
                                self.current_file_path,
                                start_line,
                                start_line,
                                0,
                                0,
                                'javascript'
                            )
                            param_id = self.current_cpg.add_node(param_node)
                            
                            # Link to function
                            edge = self.create_edge(func_id, param_id, EdgeType.AST_CHILD)
                            self.current_cpg.add_edge(edge)
    
    def _process_statements_in_function(self, func_content: str, func_id: str, func_start_line: int):
        """Process statements within a function body."""
        # Find variable declarations within function
        for match in re.finditer(self.patterns['variable_decl'], func_content, re.MULTILINE):
            start_line = func_start_line + func_content[:match.start()].count('\n')
            var_name = match.group(1)
            
            var_node = self.create_node(
                NodeType.VARIABLE,
                var_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'javascript'
            )
            var_id = self.current_cpg.add_node(var_node)
            
            # Track variable definition for DFG
            if var_name not in self.variable_definitions:
                self.variable_definitions[var_name] = []
            self.variable_definitions[var_name].append(var_id)
            
            # Link to function
            edge = self.create_edge(func_id, var_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Find assignments within function
        for match in re.finditer(self.patterns['assignment'], func_content, re.MULTILINE):
            start_line = func_start_line + func_content[:match.start()].count('\n')
            var_name = match.group(1)
            
            assign_node = self.create_node(
                NodeType.ASSIGNMENT,
                "assignment",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'javascript'
            )
            assign_id = self.current_cpg.add_node(assign_node)
            
            # Track variable definition for DFG
            if var_name not in self.variable_definitions:
                self.variable_definitions[var_name] = []
            self.variable_definitions[var_name].append(assign_id)
            
            # Link to function
            edge = self.create_edge(func_id, assign_id, EdgeType.AST_CHILD)
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
                'javascript'
            )
            if_id = self.current_cpg.add_node(if_node)
            
            # Link to function
            edge = self.create_edge(func_id, if_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Find loops within function
        for match in re.finditer(self.patterns['for_loop'], func_content, re.MULTILINE):
            start_line = func_start_line + func_content[:match.start()].count('\n')
            
            loop_node = self.create_node(
                NodeType.LOOP,
                "for",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'javascript'
            )
            loop_id = self.current_cpg.add_node(loop_node)
            
            # Link to function
            edge = self.create_edge(func_id, loop_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        for match in re.finditer(self.patterns['while_loop'], func_content, re.MULTILINE):
            start_line = func_start_line + func_content[:match.start()].count('\n')
            
            loop_node = self.create_node(
                NodeType.LOOP,
                "while",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'javascript'
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
                'javascript'
            )
            return_id = self.current_cpg.add_node(return_node)
            
            # Track variable usage in return statement for DFG
            used_vars = self._extract_variables_from_code(match.group(0))
            for var_name in used_vars:
                                if var_name not in self.variable_uses:
                                    self.variable_uses[var_name] = []
                                self.variable_uses[var_name].append(return_id)
            
            # Link to function
            edge = self.create_edge(func_id, return_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Find function calls within function
        for match in re.finditer(self.patterns['function_call'], func_content, re.MULTILINE):
            start_line = func_start_line + func_content[:match.start()].count('\n')
            func_name = match.group(1)
            
            call_node = self.create_node(
                NodeType.CALL,
                func_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'javascript',
                function_name=func_name
            )
            call_id = self.current_cpg.add_node(call_node)
            
            # Track variable usage for DFG
            if '.' not in func_name:  # Simple function name
                if func_name not in self.variable_uses:
                    self.variable_uses[func_name] = []
                self.variable_uses[func_name].append(call_id)
            
            # Link to function
            edge = self.create_edge(func_id, call_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_variables(self, content: str, lines: List[str], parent_id: str):
        """Process variable declarations and assignments."""
        # Variable declarations
        for match in re.finditer(self.patterns['variable_decl'], content, re.MULTILINE):
            var_name = match.group(1)
            start_line = content[:match.start()].count('\n') + 1
            
            var_node = self.create_node(
                NodeType.VARIABLE,
                var_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'javascript',
                declaration_type=match.group(0).split()[0]  # var, let, or const
            )
            var_id = self.current_cpg.add_node(var_node)
            
            # Track for DFG
            # Track variable definition for DFG
            if var_name not in self.variable_definitions:
                self.variable_definitions[var_name] = []
            self.variable_definitions[var_name].append(var_id)
            
            # Link to parent
            edge = self.create_edge(parent_id, var_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Assignments
        for match in re.finditer(self.patterns['assignment'], content, re.MULTILINE):
            var_name = match.group(1)
            start_line = content[:match.start()].count('\n') + 1
            
            assign_node = self.create_node(
                NodeType.ASSIGNMENT,
                "assignment",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'javascript',
                target=var_name
            )
            assign_id = self.current_cpg.add_node(assign_node)
            
            # Track for DFG
            # Track variable definition for DFG
            if var_name not in self.variable_definitions:
                self.variable_definitions[var_name] = []
            self.variable_definitions[var_name].append(assign_id)
            
            # Link to parent
            edge = self.create_edge(parent_id, assign_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_control_flow(self, content: str, lines: List[str], parent_id: str):
        """Process control flow statements."""
        # If statements
        for match in re.finditer(self.patterns['if_statement'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            
            # Find end of if block
            end_pos = self._find_matching_brace(content, match.end() - 1)
            end_line = content[:end_pos].count('\n') + 1 if end_pos else start_line
            
            if_content = content[match.start():end_pos] if end_pos else match.group(0)
            
            if_node = self.create_node(
                NodeType.CONDITION,
                "if",
                if_content,
                self.current_file_path,
                start_line,
                end_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                0,
                'javascript'
            )
            if_id = self.current_cpg.add_node(if_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, if_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # For loops
        for match in re.finditer(self.patterns['for_loop'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            
            # Find end of loop
            end_pos = self._find_matching_brace(content, match.end() - 1)
            end_line = content[:end_pos].count('\n') + 1 if end_pos else start_line
            
            loop_content = content[match.start():end_pos] if end_pos else match.group(0)
            
            loop_node = self.create_node(
                NodeType.LOOP,
                "for",
                loop_content,
                self.current_file_path,
                start_line,
                end_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                0,
                'javascript',
                loop_type="for"
            )
            loop_id = self.current_cpg.add_node(loop_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, loop_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # While loops
        for match in re.finditer(self.patterns['while_loop'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            
            # Find end of loop
            end_pos = self._find_matching_brace(content, match.end() - 1)
            end_line = content[:end_pos].count('\n') + 1 if end_pos else start_line
            
            loop_content = content[match.start():end_pos] if end_pos else match.group(0)
            
            loop_node = self.create_node(
                NodeType.LOOP,
                "while",
                loop_content,
                self.current_file_path,
                start_line,
                end_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                0,
                'javascript',
                loop_type="while"
            )
            loop_id = self.current_cpg.add_node(loop_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, loop_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Try-catch blocks
        for match in re.finditer(self.patterns['try_catch'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            
            # Find end of try block (simplified)
            end_pos = self._find_matching_brace(content, match.end() - 1)
            end_line = content[:end_pos].count('\n') + 1 if end_pos else start_line
            
            try_content = content[match.start():end_pos] if end_pos else match.group(0)
            
            try_node = self.create_node(
                NodeType.EXCEPTION,
                "try",
                try_content,
                self.current_file_path,
                start_line,
                end_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                0,
                'javascript'
            )
            try_id = self.current_cpg.add_node(try_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, try_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_function_calls(self, content: str, lines: List[str], parent_id: str):
        """Process function calls."""
        for match in re.finditer(self.patterns['function_call'], content, re.MULTILINE):
            func_name = match.group(1)
            start_line = content[:match.start()].count('\n') + 1
            
            call_node = self.create_node(
                NodeType.CALL,
                func_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'javascript',
                function_name=func_name
            )
            call_id = self.current_cpg.add_node(call_node)
            
            # Track variable usage for DFG
            if '.' not in func_name:  # Simple function name
                if func_name not in self.variable_uses:
                    self.variable_uses[func_name] = []
                self.variable_uses[func_name].append(call_id)
            
            # Link to parent
            edge = self.create_edge(parent_id, call_id, EdgeType.AST_CHILD)
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

    def _strip_comments(self, content: str) -> str:
        """Remove JS/TS comments to reduce false regex matches and simplify parsing."""
        # Remove block comments
        without_block = re.sub(r"/\*[^*]*\*+(?:[^/*][^*]*\*+)*/", " ", content, flags=re.DOTALL)
        # Remove line comments
        without_line = re.sub(r"//.*", " ", without_block)
        return without_line
    
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
            language='javascript'
        )
        entry_id = cpg.add_node(entry_node)
        
        exit_node = self.create_node(
            NodeType.EXIT,
            f"{func_node.name}_exit",
            "",
            func_node.file_path,
            func_node.end_line,
            func_node.end_line,
            language='javascript'
        )
        exit_id = cpg.add_node(exit_node)
        
        # Get all child nodes of the function (excluding entry/exit)
        child_nodes = [node for node in list(cpg.get_children(func_node.id))
                      if node.node_type not in [NodeType.ENTRY, NodeType.EXIT]]
        
        # Create control flow edges
        if child_nodes:
            # Connect entry to first statement
            first_stmt = child_nodes[0]
            edge = self.create_edge(entry_id, first_stmt.id, EdgeType.CONTROL_FLOW)
            cpg.add_edge(edge)
            
            # Connect statements sequentially with special handling for control flow
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
    
    def _extract_variables_from_code(self, code: str) -> List[str]:
        """Extract variable names from JavaScript code snippet."""
        import re
        # Simple regex to find variable names (can be improved)
        var_pattern = r'\b[a-zA-Z_$][a-zA-Z0-9_$]*\b'
        variables = []
        
        for match in re.finditer(var_pattern, code):
            var_name = match.group(0)
            # Skip keywords and common tokens
            if var_name not in ['function', 'class', 'if', 'else', 'for', 'while', 'return', 'this', 'new', 'const', 'let', 'var', 'import', 'export', 'from', 'as', 'try', 'catch', 'finally', 'with', 'typeof', 'instanceof', 'in', 'of', 'true', 'false', 'null', 'undefined']:
                variables.append(var_name)
        
        return list(set(variables))
    

