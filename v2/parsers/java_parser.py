#!/usr/bin/env python3
"""
Java CPG Parser

This module implements a Code Property Graph parser for Java,
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


class JavaCPGParser(CPGParser):
    """Java CPG parser using regex patterns."""
    
    def __init__(self):
        """Initialize the Java parser."""
        self.current_cpg = None
        self.current_file_path = ""
        self.variable_definitions = {}
        self.variable_uses = {}
        self.class_hierarchy = {}
        
        # Java patterns
        self.patterns = {
            'package': r'package\s+([\w\.]+);',
            'import': r'import\s+(?:static\s+)?([\w\.\*]+);',
            'class': r'(?:public\s+|private\s+|protected\s+)?(?:abstract\s+|final\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w\s,]+))?\s*\{',
            'interface': r'(?:public\s+|private\s+|protected\s+)?interface\s+(\w+)(?:\s+extends\s+([\w\s,]+))?\s*\{',
            'enum': r'(?:public\s+|private\s+|protected\s+)?enum\s+(\w+)\s*\{',
            'method': r'(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:final\s+)?(?:abstract\s+)?(\w+(?:<[^>]+>)?)\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w\s,]+)?\s*\{',
            'constructor': r'(?:public\s+|private\s+|protected\s+)?(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w\s,]+)?\s*\{',
            'field': r'(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:final\s+)?(\w+(?:<[^>]+>)?)\s+(\w+)(?:\s*=\s*[^;]+)?;',
            'variable_decl': r'(\w+(?:<[^>]+>)?)\s+(\w+)(?:\s*=\s*[^;]+)?;',
            'assignment': r'(\w+(?:\.\w+)*)\s*=\s*([^;]+);',
            'method_call': r'(\w+(?:\.\w+)*)\s*\([^)]*\)',
            'if_statement': r'if\s*\([^)]+\)\s*\{',
            'for_loop': r'for\s*\([^)]*\)\s*\{',
            'while_loop': r'while\s*\([^)]+\)\s*\{',
            'try_catch': r'try\s*\{',
            'return': r'return\s+[^;]*;',
            'throw': r'throw\s+[^;]*;',
            'annotation': r'@(\w+)(?:\([^)]*\))?',
        }
    
    def parse(self, content: str, file_path: str) -> CodePropertyGraph:
        """Parse Java source code and generate complete CPG."""
        self.current_file_path = file_path
        self.current_cpg = CodePropertyGraph()
        
        # Reset state
        self.variable_definitions.clear()
        self.variable_uses.clear()
        self.class_hierarchy.clear()
        
        # Build AST
        self.build_ast(content, file_path)
        
        # Build CFG
        self.build_cfg(self.current_cpg)
        
        # Build DFG
        self.build_dfg(self.current_cpg)
        
        return self.current_cpg
    
    def build_ast(self, content: str, file_path: str) -> CodePropertyGraph:
        """Build Abstract Syntax Tree for Java code."""
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
            'java'
        )
        module_id = self.current_cpg.add_node(module_node)
        
        # Process different constructs
        self._process_package(content, lines, module_id)
        self._process_imports(content, lines, module_id)
        self._process_classes(content, lines, module_id)
        self._process_interfaces(content, lines, module_id)
        self._process_enums(content, lines, module_id)
        
        return self.current_cpg
    
    def _process_package(self, content: str, lines: List[str], parent_id: str):
        """Process package declaration."""
        match = re.search(self.patterns['package'], content)
        if match:
            package_name = match.group(1)
            start_line = content[:match.start()].count('\n') + 1
            
            package_node = self.create_node(
                NodeType.MODULE,
                package_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'java',
                package_name=package_name
            )
            package_id = self.current_cpg.add_node(package_node)
            
            # Link to module
            edge = self.create_edge(parent_id, package_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_imports(self, content: str, lines: List[str], parent_id: str):
        """Process import statements."""
        for match in re.finditer(self.patterns['import'], content, re.MULTILINE):
            import_name = match.group(1)
            start_line = content[:match.start()].count('\n') + 1
            
            import_node = self.create_node(
                NodeType.IMPORT,
                import_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'java',
                imported_class=import_name
            )
            import_id = self.current_cpg.add_node(import_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, import_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_classes(self, content: str, lines: List[str], parent_id: str):
        """Process class definitions."""
        for match in re.finditer(self.patterns['class'], content, re.MULTILINE):
            class_name = match.group(1)
            extends_class = match.group(2) if match.groups() and len(match.groups()) > 1 else None
            implements_interfaces = match.group(3) if match.groups() and len(match.groups()) > 2 else None
            
            start_line = content[:match.start()].count('\n') + 1
            
            # Find class end
            end_pos = self._find_matching_brace(content, match.end() - 1)
            end_line = content[:end_pos].count('\n') + 1 if end_pos else start_line
            
            # Extract class content
            class_content = content[match.start():end_pos] if end_pos else match.group(0)
            
            # Get annotations
            annotations = self._get_annotations_before(content, match.start())
            
            class_node = self.create_node(
                NodeType.CLASS,
                class_name,
                class_content,
                self.current_file_path,
                start_line,
                end_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                0,
                'java',
                extends_class=extends_class,
                implements_interfaces=implements_interfaces.split(',') if implements_interfaces else [],
                annotations=annotations
            )
            class_id = self.current_cpg.add_node(class_node)
            
            # Track class hierarchy
            if extends_class:
                self.class_hierarchy[class_name] = extends_class
            
            # Link to parent
            edge = self.create_edge(parent_id, class_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Process class members
            self._process_class_members(class_content, class_id, start_line, class_name)
    
    def _process_interfaces(self, content: str, lines: List[str], parent_id: str):
        """Process interface definitions."""
        for match in re.finditer(self.patterns['interface'], content, re.MULTILINE):
            interface_name = match.group(1)
            extends_interfaces = match.group(2) if match.groups() and len(match.groups()) > 1 else None
            
            start_line = content[:match.start()].count('\n') + 1
            
            # Find interface end
            end_pos = self._find_matching_brace(content, match.end() - 1)
            end_line = content[:end_pos].count('\n') + 1 if end_pos else start_line
            
            # Extract interface content
            interface_content = content[match.start():end_pos] if end_pos else match.group(0)
            
            interface_node = self.create_node(
                NodeType.CLASS,  # Using CLASS type for interfaces
                interface_name,
                interface_content,
                self.current_file_path,
                start_line,
                end_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                0,
                'java',
                is_interface=True,
                extends_interfaces=extends_interfaces.split(',') if extends_interfaces else []
            )
            interface_id = self.current_cpg.add_node(interface_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, interface_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Process interface members
            self._process_interface_members(interface_content, interface_id, start_line)
    
    def _process_enums(self, content: str, lines: List[str], parent_id: str):
        """Process enum definitions."""
        for match in re.finditer(self.patterns['enum'], content, re.MULTILINE):
            enum_name = match.group(1)
            start_line = content[:match.start()].count('\n') + 1
            
            # Find enum end
            end_pos = self._find_matching_brace(content, match.end() - 1)
            end_line = content[:end_pos].count('\n') + 1 if end_pos else start_line
            
            # Extract enum content
            enum_content = content[match.start():end_pos] if end_pos else match.group(0)
            
            enum_node = self.create_node(
                NodeType.CLASS,  # Using CLASS type for enums
                enum_name,
                enum_content,
                self.current_file_path,
                start_line,
                end_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                0,
                'java',
                is_enum=True
            )
            enum_id = self.current_cpg.add_node(enum_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, enum_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_class_members(self, class_content: str, class_id: str, 
                              class_start_line: int, class_name: str):
        """Process methods and fields within a class."""
        # Process methods
        for match in re.finditer(self.patterns['method'], class_content, re.MULTILINE):
            return_type = match.group(1)
            method_name = match.group(2)
            
            start_line = class_start_line + class_content[:match.start()].count('\n')
            
            # Find method end
            end_pos = self._find_matching_brace(class_content, match.end() - 1)
            end_line = class_start_line + class_content[:end_pos].count('\n') if end_pos else start_line
            
            # Extract method content
            method_content = class_content[match.start():end_pos] if end_pos else match.group(0)
            
            # Get annotations
            annotations = self._get_annotations_before(class_content, match.start())
            
            method_node = self.create_node(
                NodeType.METHOD,
                method_name,
                method_content,
                self.current_file_path,
                start_line,
                end_line,
                0,
                0,
                'java',
                return_type=return_type,
                annotations=annotations
            )
            method_id = self.current_cpg.add_node(method_node)
            
            # Link to class
            edge = self.create_edge(class_id, method_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Extract parameters from method signature
            self._extract_method_parameters(match.group(0), method_id, start_line)
            
            # Process method body
            self._process_method_body(method_content, method_id, start_line)
        
        # Process constructors
        for match in re.finditer(self.patterns['constructor'], class_content, re.MULTILINE):
            constructor_name = match.group(1)
            
            # Only process if it matches the class name
            if constructor_name == class_name:
                start_line = class_start_line + class_content[:match.start()].count('\n')
                
                # Find constructor end
                end_pos = self._find_matching_brace(class_content, match.end() - 1)
                end_line = class_start_line + class_content[:end_pos].count('\n') if end_pos else start_line
                
                # Extract constructor content
                constructor_content = class_content[match.start():end_pos] if end_pos else match.group(0)
                
                constructor_node = self.create_node(
                    NodeType.METHOD,
                    constructor_name,
                    constructor_content,
                    self.current_file_path,
                    start_line,
                    end_line,
                    0,
                    0,
                    'java',
                    is_constructor=True
                )
                constructor_id = self.current_cpg.add_node(constructor_node)
                
                # Link to class
                edge = self.create_edge(class_id, constructor_id, EdgeType.AST_CHILD)
                self.current_cpg.add_edge(edge)
                
                # Process constructor body
                self._process_method_body(constructor_content, constructor_id, start_line)
        
        # Process fields
        for match in re.finditer(self.patterns['field'], class_content, re.MULTILINE):
            field_type = match.group(1)
            field_name = match.group(2)
            
            start_line = class_start_line + class_content[:match.start()].count('\n')
            
            field_node = self.create_node(
                NodeType.VARIABLE,
                field_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'java',
                field_type=field_type,
                is_field=True
            )
            field_id = self.current_cpg.add_node(field_node)
            
            # Track for DFG
            # Track field definition for DFG
            if field_name not in self.variable_definitions:
                self.variable_definitions[field_name] = []
            self.variable_definitions[field_name].append(field_id)
            
            # Link to class
            edge = self.create_edge(class_id, field_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_interface_members(self, interface_content: str, interface_id: str, 
                                  interface_start_line: int):
        """Process methods within an interface."""
        for match in re.finditer(self.patterns['method'], interface_content, re.MULTILINE):
            return_type = match.group(1)
            method_name = match.group(2)
            
            start_line = interface_start_line + interface_content[:match.start()].count('\n')
            
            method_node = self.create_node(
                NodeType.METHOD,
                method_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'java',
                return_type=return_type,
                is_abstract=True
            )
            method_id = self.current_cpg.add_node(method_node)
            
            # Link to interface
            edge = self.create_edge(interface_id, method_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_method_body(self, method_content: str, method_id: str, method_start_line: int):
        """Process statements within a method body."""
        # Process variable declarations
        for match in re.finditer(self.patterns['variable_decl'], method_content, re.MULTILINE):
            var_type = match.group(1)
            var_name = match.group(2)
            
            start_line = method_start_line + method_content[:match.start()].count('\n')
            
            var_node = self.create_node(
                NodeType.VARIABLE,
                var_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'java',
                variable_type=var_type
            )
            var_id = self.current_cpg.add_node(var_node)
            
            # Track for DFG
            # Track variable definition for DFG
            if var_name not in self.variable_definitions:
                self.variable_definitions[var_name] = []
            self.variable_definitions[var_name].append(var_id)
            
            # Link to method
            edge = self.create_edge(method_id, var_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Process assignments
        for match in re.finditer(self.patterns['assignment'], method_content, re.MULTILINE):
            var_name = match.group(1)
            start_line = method_start_line + method_content[:match.start()].count('\n')
            
            assign_node = self.create_node(
                NodeType.ASSIGNMENT,
                "assignment",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'java',
                target=var_name
            )
            assign_id = self.current_cpg.add_node(assign_node)
            
            # Track for DFG
            # Track assignment for DFG
            if var_name not in self.variable_definitions:
                self.variable_definitions[var_name] = []
            self.variable_definitions[var_name].append(assign_id)
            
            # Track variable usage in assignment expression
            used_vars = self._extract_java_variables_from_code(match.group(0))
            for used_var in used_vars:
                if used_var != var_name:  # Don't track self-reference
                    if used_var not in self.variable_uses:
                        self.variable_uses[used_var] = []
                    self.variable_uses[used_var].append(assign_id)
            
            # Link to method
            edge = self.create_edge(method_id, assign_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Process method calls
        for match in re.finditer(self.patterns['method_call'], method_content, re.MULTILINE):
            call_name = match.group(1)
            start_line = method_start_line + method_content[:match.start()].count('\n')
            
            call_node = self.create_node(
                NodeType.CALL,
                call_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'java',
                method_name=call_name
            )
            call_id = self.current_cpg.add_node(call_node)
            
            # Track variable usage for DFG - extract variables from the call expression
            used_vars = self._extract_java_variables_from_code(match.group(0))
            for var_name in used_vars:
                if var_name not in self.variable_uses:
                    self.variable_uses[var_name] = []
                self.variable_uses[var_name].append(call_id)
            
            # Link to method
            edge = self.create_edge(method_id, call_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Process return statements
        for match in re.finditer(self.patterns['return'], method_content, re.MULTILINE):
            start_line = method_start_line + method_content[:match.start()].count('\n')
            
            return_node = self.create_node(
                NodeType.RETURN,
                "return",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'java'
            )
            return_id = self.current_cpg.add_node(return_node)
            
            # Link to method
            edge = self.create_edge(method_id, return_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Process control flow statements
        self._process_control_flow_in_method(method_content, method_id, method_start_line)
    
    def _extract_method_parameters(self, method_signature: str, method_id: str, start_line: int):
        """Extract parameters from method signature."""
        # Match parameters in parentheses
        param_match = re.search(r'\(([^)]*)\)', method_signature)
        if param_match:
            params_str = param_match.group(1).strip()
            if params_str:
                # Split parameters by comma and process each
                params = [p.strip() for p in params_str.split(',')]
                for param in params:
                    if param:
                        # Extract parameter name (last word)
                        parts = param.split()
                        if len(parts) >= 2:
                            param_name = parts[-1]
                            param_type = ' '.join(parts[:-1])
                            
                            param_node = self.create_node(
                                NodeType.PARAMETER,
                                param_name,
                                param,
                                self.current_file_path,
                                start_line,
                                start_line,
                                0,
                                0,
                                'java',
                                param_type=param_type
                            )
                            param_id = self.current_cpg.add_node(param_node)
                            
                            # Link to method
                            edge = self.create_edge(method_id, param_id, EdgeType.AST_CHILD)
                            self.current_cpg.add_edge(edge)
    
    def _process_control_flow_in_method(self, method_content: str, method_id: str, 
                                       method_start_line: int):
        """Process control flow statements within a method."""
        # If statements
        for match in re.finditer(self.patterns['if_statement'], method_content, re.MULTILINE):
            start_line = method_start_line + method_content[:match.start()].count('\n')
            
            # Find end of if block
            end_pos = self._find_matching_brace(method_content, match.end() - 1)
            end_line = method_start_line + method_content[:end_pos].count('\n') if end_pos else start_line
            
            if_content = method_content[match.start():end_pos] if end_pos else match.group(0)
            
            if_node = self.create_node(
                NodeType.CONDITION,
                "if",
                if_content,
                self.current_file_path,
                start_line,
                end_line,
                0,
                0,
                'java'
            )
            if_id = self.current_cpg.add_node(if_node)
            
            # Link to method
            edge = self.create_edge(method_id, if_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Loops
        for pattern_name in ['for_loop', 'while_loop']:
            for match in re.finditer(self.patterns[pattern_name], method_content, re.MULTILINE):
                start_line = method_start_line + method_content[:match.start()].count('\n')
                
                # Find end of loop
                end_pos = self._find_matching_brace(method_content, match.end() - 1)
                end_line = method_start_line + method_content[:end_pos].count('\n') if end_pos else start_line
                
                loop_content = method_content[match.start():end_pos] if end_pos else match.group(0)
                
                loop_type = pattern_name.split('_')[0]  # 'for' or 'while'
                
                loop_node = self.create_node(
                    NodeType.LOOP,
                    loop_type,
                    loop_content,
                    self.current_file_path,
                    start_line,
                    end_line,
                    0,
                    0,
                    'java',
                    loop_type=loop_type
                )
                loop_id = self.current_cpg.add_node(loop_node)
                
                # Link to method
                edge = self.create_edge(method_id, loop_id, EdgeType.AST_CHILD)
                self.current_cpg.add_edge(edge)
        
        # Try-catch blocks
        for match in re.finditer(self.patterns['try_catch'], method_content, re.MULTILINE):
            start_line = method_start_line + method_content[:match.start()].count('\n')
            
            # Find end of try block
            end_pos = self._find_matching_brace(method_content, match.end() - 1)
            end_line = method_start_line + method_content[:end_pos].count('\n') if end_pos else start_line
            
            try_content = method_content[match.start():end_pos] if end_pos else match.group(0)
            
            try_node = self.create_node(
                NodeType.EXCEPTION,
                "try",
                try_content,
                self.current_file_path,
                start_line,
                end_line,
                0,
                0,
                'java'
            )
            try_id = self.current_cpg.add_node(try_node)
            
            # Link to method
            edge = self.create_edge(method_id, try_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _get_annotations_before(self, content: str, pos: int) -> List[str]:
        """Get annotations that appear before a given position."""
        annotations = []
        
        # Look backwards from the position to find annotations
        text_before = content[:pos]
        lines_before = text_before.split('\n')
        
        # Check the last few lines for annotations
        for line in reversed(lines_before[-5:]):  # Check last 5 lines
            line = line.strip()
            if line.startswith('@'):
                match = re.match(r'@(\w+)(?:\([^)]*\))?', line)
                if match:
                    annotations.insert(0, match.group(1))
            elif line and not line.startswith('//') and not line.startswith('/*'):
                # Stop if we hit non-annotation, non-comment code
                break
        
        return annotations
    
    def _find_matching_brace(self, content: str, start_pos: int) -> Optional[int]:
        """Find the matching closing brace for an opening brace."""
        if start_pos >= len(content) or content[start_pos] != '{':
            return None
        
        brace_count = 1
        pos = start_pos + 1
        in_string = False
        in_char = False
        in_comment = False
        
        while pos < len(content) and brace_count > 0:
            char = content[pos]
            
            # Handle string literals
            if char == '"' and not in_char and not in_comment:
                in_string = not in_string
            elif char == "'" and not in_string and not in_comment:
                in_char = not in_char
            
            # Handle comments (simplified)
            elif not in_string and not in_char:
                if char == '/' and pos + 1 < len(content):
                    if content[pos + 1] == '/':
                        # Single line comment - skip to end of line
                        while pos < len(content) and content[pos] != '\n':
                            pos += 1
                        continue
                    elif content[pos + 1] == '*':
                        # Multi-line comment - skip to */
                        pos += 2
                        while pos + 1 < len(content):
                            if content[pos] == '*' and content[pos + 1] == '/':
                                pos += 2
                                break
                            pos += 1
                        continue
                
                # Count braces only when not in strings or comments
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
            
            pos += 1
        
        return pos if brace_count == 0 else None
    
    def build_cfg(self, cpg: CodePropertyGraph) -> CodePropertyGraph:
        """Build Control Flow Graph from AST."""
        # Find all method nodes
        method_nodes = []
        for node in list(cpg.nodes.values()):
            if node.node_type == NodeType.METHOD:
                method_nodes.append(node)
        
        # Build CFG for each method
        for method_node in method_nodes:
            self._build_method_cfg(method_node, cpg)
        
        return cpg
    
    def _build_method_cfg(self, method_node: CPGNode, cpg: CodePropertyGraph):
        """Build CFG for a single method."""
        # Create entry and exit nodes
        entry_node = self.create_node(
            NodeType.ENTRY,
            f"{method_node.name}_entry",
            "",
            method_node.file_path,
            method_node.start_line,
            method_node.start_line,
            language='java'
        )
        entry_id = cpg.add_node(entry_node)
        
        exit_node = self.create_node(
            NodeType.EXIT,
            f"{method_node.name}_exit",
            "",
            method_node.file_path,
            method_node.end_line,
            method_node.end_line,
            language='java'
        )
        exit_id = cpg.add_node(exit_node)
        
        # Get all child nodes of the method
        child_nodes = cpg.get_children(method_node.id)
        
        # Create control flow edges (simplified)
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
                    
                    false_edge = self.create_edge(current_node.id, exit_id, EdgeType.CONDITIONAL_FALSE)
                    cpg.add_edge(false_edge)
                
                elif current_node.node_type == NodeType.LOOP:
                    # Loop edges
                    loop_edge = self.create_edge(current_node.id, next_node.id, EdgeType.CONTROL_FLOW)
                    cpg.add_edge(loop_edge)
                    
                    # Back edge
                    back_edge = self.create_edge(next_node.id, current_node.id, EdgeType.CONTROL_FLOW)
                    cpg.add_edge(back_edge)
                
                else:
                    # Sequential flow
                    flow_edge = self.create_edge(current_node.id, next_node.id, EdgeType.CONTROL_FLOW)
                    cpg.add_edge(flow_edge)
            
            # Connect last statement to exit
            if child_nodes:
                last_stmt = child_nodes[-1]
                edge = self.create_edge(last_stmt.id, exit_id, EdgeType.CONTROL_FLOW)
                cpg.add_edge(edge)
    
    def build_dfg(self, cpg: CodePropertyGraph) -> CodePropertyGraph:
        """Build Data Flow Graph from AST and CFG - DISABLED for performance."""
        # DFG generation disabled for faster processing
        # Only AST and CFG are generated
        return cpg
    
    def _extract_java_variables_from_code(self, code: str) -> List[str]:
        """Extract variable names from Java code snippet."""
        # Pattern for Java identifiers
        var_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        variables = []
        
        # Java keywords to skip
        java_keywords = {
            'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 
            'char', 'class', 'const', 'continue', 'default', 'do', 'double', 
            'else', 'enum', 'extends', 'final', 'finally', 'float', 'for', 
            'goto', 'if', 'implements', 'import', 'instanceof', 'int', 
            'interface', 'long', 'native', 'new', 'package', 'private', 
            'protected', 'public', 'return', 'short', 'static', 'strictfp', 
            'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 
            'transient', 'try', 'void', 'volatile', 'while'
        }
        
        for match in re.finditer(var_pattern, code):
            var_name = match.group(0)
            if var_name.lower() not in java_keywords:
                variables.append(var_name)
        
        return list(set(variables))
