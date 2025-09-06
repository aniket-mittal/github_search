#!/usr/bin/env python3
"""
Go CPG Parser

This module implements a Code Property Graph parser for Go,
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


class GoCPGParser(CPGParser):
    """Go CPG parser using regex patterns."""
    
    def __init__(self):
        """Initialize the Go parser."""
        self.current_cpg = None
        self.current_file_path = ""
        self.variable_definitions = {}
        self.variable_uses = {}
        
        # Go patterns
        self.patterns = {
            'package': r'package\s+(\w+)',
            'import': r'import\s+(?:\(\s*(?:[^)]+)\s*\)|"[^"]+"|`[^`]+`)',
            'import_single': r'import\s+"([^"]+)"',
            'struct': r'type\s+(\w+)\s+struct\s*\{',
            'interface': r'type\s+(\w+)\s+interface\s*\{',
            'type_alias': r'type\s+(\w+)\s+(\w+)',
            'function': r'func\s+(\w+)\s*\([^)]*\)(?:\s*\([^)]*\))?\s*\{',
            'method': r'func\s*\([^)]*\)\s*(\w+)\s*\([^)]*\)(?:\s*\([^)]*\))?\s*\{',
            'variable_decl': r'var\s+(\w+)(?:\s+\w+)?(?:\s*=\s*[^;\n]+)?',
            'short_var_decl': r'(\w+)\s*:=\s*([^;\n]+)',
            'assignment': r'(\w+(?:\.\w+)*)\s*=\s*([^;\n]+)',
            'const_decl': r'const\s+(\w+)(?:\s+\w+)?\s*=\s*([^;\n]+)',
            'function_call': r'(\w+(?:\.\w+)*)\s*\([^)]*\)',
            'if_statement': r'if\s+([^{]+)\s*\{',
            'for_loop': r'for\s+([^{]*)\s*\{',
            'switch_statement': r'switch\s+([^{]*)\s*\{',
            'select_statement': r'select\s*\{',
            'go_routine': r'go\s+(\w+(?:\.\w+)*)\s*\([^)]*\)',
            'channel_op': r'(\w+)\s*<-\s*([^;\n]+)|([^;\n]+)\s*<-\s*(\w+)',
            'return': r'return\s+[^;\n]*',
            'defer': r'defer\s+(\w+(?:\.\w+)*)\s*\([^)]*\)',
        }
    
    def parse(self, content: str, file_path: str) -> CodePropertyGraph:
        """Parse Go source code and generate complete CPG."""
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
        """Build Abstract Syntax Tree for Go code."""
        # Strip comments to avoid false positives in regex matching
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
            'go'
        )
        module_id = self.current_cpg.add_node(module_node)
        
        # Process different constructs
        self._process_package(sanitized, lines, module_id)
        self._process_imports(sanitized, lines, module_id)
        self._process_types(sanitized, lines, module_id)
        self._process_functions(sanitized, lines, module_id)
        self._process_variables(sanitized, lines, module_id)
        self._process_function_calls(sanitized, lines, module_id)
        self._process_control_flow(sanitized, lines, module_id)
        
        return self.current_cpg
    
    def _process_package(self, content: str, lines: List[str], parent_id: str):
        """Process package declaration."""
        for match in re.finditer(self.patterns['package'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            package_name = match.group(1)
            
            package_node = self.create_node(
                NodeType.MODULE,
                package_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'go',
                package_name=package_name
            )
            package_id = self.current_cpg.add_node(package_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, package_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_imports(self, content: str, lines: List[str], parent_id: str):
        """Process import statements."""
        # Single imports
        for match in re.finditer(self.patterns['import_single'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            import_path = match.group(1)
            
            import_node = self.create_node(
                NodeType.IMPORT,
                import_path,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'go',
                import_path=import_path
            )
            import_id = self.current_cpg.add_node(import_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, import_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Multi-line imports
        for match in re.finditer(self.patterns['import'], content, re.MULTILINE):
            if '"' not in match.group(0):  # Skip single imports already processed
                start_line = content[:match.start()].count('\n') + 1
                end_line = content[:match.end()].count('\n') + 1
                
                import_node = self.create_node(
                    NodeType.IMPORT,
                    "import_block",
                    match.group(0),
                    self.current_file_path,
                    start_line,
                    end_line,
                    match.start() - content.rfind('\n', 0, match.start()) - 1,
                    match.end() - content.rfind('\n', 0, match.end()) - 1,
                    'go'
                )
                import_id = self.current_cpg.add_node(import_node)
                
                # Link to parent
                edge = self.create_edge(parent_id, import_id, EdgeType.AST_CHILD)
                self.current_cpg.add_edge(edge)
    
    def _process_types(self, content: str, lines: List[str], parent_id: str):
        """Process struct and interface definitions."""
        # Structs
        for match in re.finditer(self.patterns['struct'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            struct_name = match.group(1)
            
            # Find the end of the struct
            brace_pos = content.find('{', match.start())
            if brace_pos != -1:
                end_pos = self._find_matching_brace(content, brace_pos)
                if end_pos:
                    end_line = content[:end_pos].count('\n') + 1
                    struct_code = content[match.start():end_pos]
                else:
                    end_line = start_line
                    struct_code = match.group(0)
            else:
                end_line = start_line
                struct_code = match.group(0)
            
            struct_node = self.create_node(
                NodeType.CLASS,  # Use CLASS for structs
                struct_name,
                struct_code,
                self.current_file_path,
                start_line,
                end_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                (end_pos if end_pos else match.end()) - content.rfind('\n', 0, end_pos if end_pos else match.end()) - 1,
                'go',
                type_kind='struct'
            )
            struct_id = self.current_cpg.add_node(struct_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, struct_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Interfaces
        for match in re.finditer(self.patterns['interface'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            interface_name = match.group(1)
            
            # Find the end of the interface
            brace_pos = content.find('{', match.start())
            if brace_pos != -1:
                end_pos = self._find_matching_brace(content, brace_pos)
                if end_pos:
                    end_line = content[:end_pos].count('\n') + 1
                    interface_code = content[match.start():end_pos]
                else:
                    end_line = start_line
                    interface_code = match.group(0)
            else:
                end_line = start_line
                interface_code = match.group(0)
            
            interface_node = self.create_node(
                NodeType.CLASS,  # Use CLASS for interfaces
                interface_name,
                interface_code,
                self.current_file_path,
                start_line,
                end_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                (end_pos if end_pos else match.end()) - content.rfind('\n', 0, end_pos if end_pos else match.end()) - 1,
                'go',
                type_kind='interface'
            )
            interface_id = self.current_cpg.add_node(interface_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, interface_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_functions(self, content: str, lines: List[str], parent_id: str):
        """Process function and method definitions."""
        # Functions
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
                'go',
                function_name=func_name
            )
            func_id = self.current_cpg.add_node(func_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, func_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Process function body
            if func_code:
                self._process_function_body(func_code, func_id, start_line)
        
        # Methods
        for match in re.finditer(self.patterns['method'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            method_name = match.group(1)
            
            # Find the end of the method
            brace_pos = content.find('{', match.start())
            if brace_pos != -1:
                end_pos = self._find_matching_brace(content, brace_pos)
                if end_pos:
                    end_line = content[:end_pos].count('\n') + 1
                    method_code = content[match.start():end_pos]
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
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                (end_pos if end_pos else match.end()) - content.rfind('\n', 0, end_pos if end_pos else match.end()) - 1,
                'go',
                method_name=method_name
            )
            method_id = self.current_cpg.add_node(method_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, method_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Process method body
            if method_code:
                self._process_function_body(method_code, method_id, start_line)
    
    def _process_function_body(self, func_content: str, func_id: str, func_start_line: int):
        """Process statements within a function body."""
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
                'go',
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
                'go'
            )
            var_id = self.current_cpg.add_node(var_node)
            
            # Track variable definition for DFG
            if var_name not in self.variable_definitions:
                self.variable_definitions[var_name] = []
            self.variable_definitions[var_name].append(var_id)
            
            # Link to function
            edge = self.create_edge(func_id, var_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Find short variable declarations within function
        for match in re.finditer(self.patterns['short_var_decl'], func_content, re.MULTILINE):
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
                'go',
                declaration_type='short'
            )
            var_id = self.current_cpg.add_node(var_node)
            
            # Track variable definition for DFG
            if var_name not in self.variable_definitions:
                self.variable_definitions[var_name] = []
            self.variable_definitions[var_name].append(var_id)
            
            # Link to function
            edge = self.create_edge(func_id, var_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_variables(self, content: str, lines: List[str], parent_id: str):
        """Process variable declarations and assignments."""
        # Variable declarations
        for match in re.finditer(self.patterns['variable_decl'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            var_name = match.group(1)
            
            var_node = self.create_node(
                NodeType.VARIABLE,
                var_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'go',
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
        
        # Short variable declarations
        for match in re.finditer(self.patterns['short_var_decl'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            var_name = match.group(1)
            
            var_node = self.create_node(
                NodeType.VARIABLE,
                var_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'go',
                variable_name=var_name,
                declaration_type='short'
            )
            var_id = self.current_cpg.add_node(var_node)
            
            # Track variable definition for DFG
            if var_name not in self.variable_definitions:
                self.variable_definitions[var_name] = []
            self.variable_definitions[var_name].append(var_id)
            
            # Link to parent
            edge = self.create_edge(parent_id, var_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Assignments
        for match in re.finditer(self.patterns['assignment'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            var_name = match.group(1).split('.')[0]  # Handle field access
            
            assign_node = self.create_node(
                NodeType.ASSIGNMENT,
                "assignment",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'go',
                target=var_name
            )
            assign_id = self.current_cpg.add_node(assign_node)
            
            # Track assignment for DFG
            if var_name not in self.variable_definitions:
                self.variable_definitions[var_name] = []
            self.variable_definitions[var_name].append(assign_id)
            
            # Link to parent
            edge = self.create_edge(parent_id, assign_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Constants
        for match in re.finditer(self.patterns['const_decl'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            const_name = match.group(1)
            
            const_node = self.create_node(
                NodeType.VARIABLE,
                const_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'go',
                variable_name=const_name,
                is_constant=True
            )
            const_id = self.current_cpg.add_node(const_node)
            
            # Track constant definition for DFG
            if const_name not in self.variable_definitions:
                self.variable_definitions[const_name] = []
            self.variable_definitions[const_name].append(const_id)
            
            # Link to parent
            edge = self.create_edge(parent_id, const_id, EdgeType.AST_CHILD)
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
                'go',
                function_name=func_name
            )
            call_id = self.current_cpg.add_node(call_node)
            
            # Track variable usage for DFG
            base_name = func_name.split('.')[0]
            if base_name not in self.variable_uses:
                self.variable_uses[base_name] = []
            self.variable_uses[base_name].append(call_id)
            
            # Link to parent
            edge = self.create_edge(parent_id, call_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Go routines
        for match in re.finditer(self.patterns['go_routine'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            func_name = match.group(1)
            
            go_node = self.create_node(
                NodeType.CALL,
                f"go_{func_name}",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'go',
                function_name=func_name,
                is_goroutine=True
            )
            go_id = self.current_cpg.add_node(go_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, go_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_control_flow(self, content: str, lines: List[str], parent_id: str):
        """Process control flow statements."""
        # If statements
        for match in re.finditer(self.patterns['if_statement'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            condition = match.group(1).strip()
            
            if_node = self.create_node(
                NodeType.CONDITION,
                "if",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'go',
                condition=condition
            )
            if_id = self.current_cpg.add_node(if_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, if_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # For loops
        for match in re.finditer(self.patterns['for_loop'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            loop_header = match.group(1).strip()
            
            loop_node = self.create_node(
                NodeType.LOOP,
                "for",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'go',
                loop_type='for',
                condition=loop_header
            )
            loop_id = self.current_cpg.add_node(loop_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, loop_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Switch statements
        for match in re.finditer(self.patterns['switch_statement'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            switch_expr = match.group(1).strip()
            
            switch_node = self.create_node(
                NodeType.CONDITION,
                "switch",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'go',
                condition=switch_expr,
                control_type='switch'
            )
            switch_id = self.current_cpg.add_node(switch_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, switch_id, EdgeType.AST_CHILD)
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
        """Remove Go comments (// line and /* */ block) to improve regex accuracy."""
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
            language='go'
        )
        entry_id = cpg.add_node(entry_node)
        
        exit_node = self.create_node(
            NodeType.EXIT,
            f"{func_node.name}_exit",
            "",
            func_node.file_path,
            func_node.end_line,
            func_node.end_line,
            language='go'
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
    def _extract_go_variables_from_code(self, code: str) -> List[str]:
        """Extract variable names from Go code snippet."""
        # Pattern for Go identifiers
        var_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        variables = []
        
        # Go keywords to skip
        go_keywords = {
            'break', 'case', 'chan', 'const', 'continue', 'default', 'defer', 
            'else', 'fallthrough', 'for', 'func', 'go', 'goto', 'if', 'import', 
            'interface', 'map', 'package', 'range', 'return', 'select', 'struct', 
            'switch', 'type', 'var', 'bool', 'byte', 'complex64', 'complex128', 
            'error', 'float32', 'float64', 'int', 'int8', 'int16', 'int32', 
            'int64', 'rune', 'string', 'uint', 'uint8', 'uint16', 'uint32', 
            'uint64', 'uintptr', 'true', 'false', 'iota', 'nil', 'append', 
            'cap', 'close', 'complex', 'copy', 'delete', 'imag', 'len', 'make', 
            'new', 'panic', 'print', 'println', 'real', 'recover'
        }
        
        for match in re.finditer(var_pattern, code):
            var_name = match.group(0)
            if var_name.lower() not in go_keywords:
                variables.append(var_name)
        
        return list(set(variables))
