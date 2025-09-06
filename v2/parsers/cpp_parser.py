#!/usr/bin/env python3
"""
C/C++ CPG Parser

This module implements a Code Property Graph parser for C and C++,
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


class CppCPGParser(CPGParser):
    """C/C++ CPG parser using regex patterns."""
    
    def __init__(self):
        """Initialize the C/C++ parser."""
        self.current_cpg = None
        self.current_file_path = ""
        self.variable_definitions = {}
        self.variable_uses = {}
        self.function_declarations = {}
        
        # C/C++ patterns
        self.patterns = {
            'include': r'#include\s*[<"][^>"]+[>"]',
            'define': r'#define\s+(\w+)(?:\([^)]*\))?\s+[^\n]*',
            'namespace': r'namespace\s+(\w+)\s*\{',
            'class': r'class\s+(\w+)(?:\s*:\s*(?:public|private|protected)\s+(\w+))?\s*\{',
            'struct': r'struct\s+(\w+)\s*\{',
            'enum': r'enum\s+(?:class\s+)?(\w+)\s*\{',
            'function': r'(\w+(?:\s*\*)*)\s+(\w+)\s*\([^)]*\)\s*\{',
            'function_decl': r'(\w+(?:\s*\*)*)\s+(\w+)\s*\([^)]*\);',
            'constructor': r'(\w+)\s*\([^)]*\)\s*(?::\s*[^{]*)?\{',
            'destructor': r'~(\w+)\s*\(\s*\)\s*\{',
            'variable_decl': r'(\w+(?:\s*\*)*)\s+(\w+)(?:\s*=\s*[^;]+)?;',
            'assignment': r'(\w+(?:\.\w+|\->\w+|\[\w+\])*)\s*=\s*([^;]+);',
            'function_call': r'(\w+)\s*\([^)]*\)',
            'if_statement': r'if\s*\([^)]+\)\s*\{',
            'for_loop': r'for\s*\([^)]*\)\s*\{',
            'while_loop': r'while\s*\([^)]+\)\s*\{',
            'do_while': r'do\s*\{',
            'switch': r'switch\s*\([^)]+\)\s*\{',
            'try_catch': r'try\s*\{',
            'return': r'return\s+[^;]*;',
            'throw': r'throw\s+[^;]*;',
            'template': r'template\s*<[^>]+>',
            'typedef': r'typedef\s+[^;]+;',
            'using': r'using\s+[^;]+;',
        }
    
    def parse(self, content: str, file_path: str) -> CodePropertyGraph:
        """Parse C/C++ source code and generate complete CPG."""
        self.current_file_path = file_path
        self.current_cpg = CodePropertyGraph()
        
        # Reset state
        self.variable_definitions.clear()
        self.variable_uses.clear()
        self.function_declarations.clear()
        
        # Build AST
        self.build_ast(content, file_path)
        
        # Build CFG
        self.build_cfg(self.current_cpg)
        
        # Build DFG
        self.build_dfg(self.current_cpg)
        
        return self.current_cpg
    
    def build_ast(self, content: str, file_path: str) -> CodePropertyGraph:
        """Build Abstract Syntax Tree for C/C++ code."""
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
            'cpp' if file_path.endswith(('.cpp', '.cc', '.cxx', '.hpp')) else 'c'
        )
        module_id = self.current_cpg.add_node(module_node)
        
        # Process different constructs
        self._process_preprocessor(content, lines, module_id)
        self._process_namespaces(content, lines, module_id)
        self._process_classes(content, lines, module_id)
        self._process_structs(content, lines, module_id)
        self._process_enums(content, lines, module_id)
        self._process_functions(content, lines, module_id)
        self._process_global_variables(content, lines, module_id)
        
        return self.current_cpg
    
    def _process_preprocessor(self, content: str, lines: List[str], parent_id: str):
        """Process preprocessor directives."""
        # Includes
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
                'cpp'
            )
            include_id = self.current_cpg.add_node(include_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, include_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Defines
        for match in re.finditer(self.patterns['define'], content, re.MULTILINE):
            macro_name = match.group(1)
            start_line = content[:match.start()].count('\n') + 1
            
            define_node = self.create_node(
                NodeType.LITERAL,
                macro_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'cpp',
                macro_name=macro_name
            )
            define_id = self.current_cpg.add_node(define_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, define_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_namespaces(self, content: str, lines: List[str], parent_id: str):
        """Process namespace definitions."""
        for match in re.finditer(self.patterns['namespace'], content, re.MULTILINE):
            namespace_name = match.group(1)
            start_line = content[:match.start()].count('\n') + 1
            
            # Find namespace end
            end_pos = self._find_matching_brace(content, match.end() - 1)
            end_line = content[:end_pos].count('\n') + 1 if end_pos else start_line
            
            # Extract namespace content
            namespace_content = content[match.start():end_pos] if end_pos else match.group(0)
            
            namespace_node = self.create_node(
                NodeType.MODULE,
                namespace_name,
                namespace_content,
                self.current_file_path,
                start_line,
                end_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                0,
                'cpp',
                namespace_name=namespace_name
            )
            namespace_id = self.current_cpg.add_node(namespace_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, namespace_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Process namespace members
            self._process_namespace_members(namespace_content, namespace_id, start_line)
    
    def _process_classes(self, content: str, lines: List[str], parent_id: str):
        """Process class definitions."""
        for match in re.finditer(self.patterns['class'], content, re.MULTILINE):
            class_name = match.group(1)
            base_class = match.group(2) if match.groups() and len(match.groups()) > 1 else None
            
            start_line = content[:match.start()].count('\n') + 1
            
            # Find class end
            end_pos = self._find_matching_brace(content, match.end() - 1)
            end_line = content[:end_pos].count('\n') + 1 if end_pos else start_line
            
            # Extract class content
            class_content = content[match.start():end_pos] if end_pos else match.group(0)
            
            # Check for template
            template_match = self._get_template_before(content, match.start())
            
            class_node = self.create_node(
                NodeType.CLASS,
                class_name,
                class_content,
                self.current_file_path,
                start_line,
                end_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                0,
                'cpp',
                base_class=base_class,
                template=template_match
            )
            class_id = self.current_cpg.add_node(class_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, class_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Process class members
            self._process_class_members(class_content, class_id, start_line, class_name)
    
    def _process_structs(self, content: str, lines: List[str], parent_id: str):
        """Process struct definitions."""
        for match in re.finditer(self.patterns['struct'], content, re.MULTILINE):
            struct_name = match.group(1)
            start_line = content[:match.start()].count('\n') + 1
            
            # Find struct end
            end_pos = self._find_matching_brace(content, match.end() - 1)
            end_line = content[:end_pos].count('\n') + 1 if end_pos else start_line
            
            # Extract struct content
            struct_content = content[match.start():end_pos] if end_pos else match.group(0)
            
            struct_node = self.create_node(
                NodeType.CLASS,  # Using CLASS type for structs
                struct_name,
                struct_content,
                self.current_file_path,
                start_line,
                end_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                0,
                'cpp',
                is_struct=True
            )
            struct_id = self.current_cpg.add_node(struct_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, struct_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Process struct members
            self._process_struct_members(struct_content, struct_id, start_line)
    
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
                'cpp',
                is_enum=True
            )
            enum_id = self.current_cpg.add_node(enum_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, enum_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_functions(self, content: str, lines: List[str], parent_id: str):
        """Process function definitions."""
        # Function definitions
        for match in re.finditer(self.patterns['function'], content, re.MULTILINE):
            return_type = match.group(1).strip()
            func_name = match.group(2)
            
            # Skip if this looks like a class member (will be handled separately)
            if self._is_inside_class_or_struct(content, match.start()):
                continue
            
            start_line = content[:match.start()].count('\n') + 1
            
            # Find function end
            end_pos = self._find_matching_brace(content, match.end() - 1)
            end_line = content[:end_pos].count('\n') + 1 if end_pos else start_line
            
            # Extract function content
            func_content = content[match.start():end_pos] if end_pos else match.group(0)
            
            # Check for template
            template_match = self._get_template_before(content, match.start())
            
            func_node = self.create_node(
                NodeType.FUNCTION,
                func_name,
                func_content,
                self.current_file_path,
                start_line,
                end_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                0,
                'cpp',
                return_type=return_type,
                template=template_match
            )
            func_id = self.current_cpg.add_node(func_node)
            
            # Track function declaration
            self.function_declarations[func_name] = func_id
            
            # Link to parent
            edge = self.create_edge(parent_id, func_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Extract parameters from function signature
            self._extract_function_parameters(match.group(0), func_id, start_line)
            
            # Process function body
            self._process_function_body(func_content, func_id, start_line)
        
        # Function declarations (prototypes)
        for match in re.finditer(self.patterns['function_decl'], content, re.MULTILINE):
            return_type = match.group(1).strip()
            func_name = match.group(2)
            
            start_line = content[:match.start()].count('\n') + 1
            
            func_decl_node = self.create_node(
                NodeType.FUNCTION,
                func_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'cpp',
                return_type=return_type,
                is_declaration=True
            )
            func_decl_id = self.current_cpg.add_node(func_decl_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, func_decl_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_global_variables(self, content: str, lines: List[str], parent_id: str):
        """Process global variable declarations."""
        for match in re.finditer(self.patterns['variable_decl'], content, re.MULTILINE):
            var_type = match.group(1).strip()
            var_name = match.group(2)
            
            # Skip if inside a function or class
            if self._is_inside_function_or_class(content, match.start()):
                continue
            
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
                'cpp',
                variable_type=var_type,
                is_global=True
            )
            var_id = self.current_cpg.add_node(var_node)
            
            # Track variable definition for DFG
            if var_name not in self.variable_definitions:
                self.variable_definitions[var_name] = []
            self.variable_definitions[var_name].append(var_id)
            
            # Link to parent
            edge = self.create_edge(parent_id, var_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_namespace_members(self, namespace_content: str, namespace_id: str, 
                                  namespace_start_line: int):
        """Process members within a namespace."""
        # This is a simplified version - would recursively process all constructs
        pass
    
    def _process_class_members(self, class_content: str, class_id: str, 
                              class_start_line: int, class_name: str):
        """Process methods and fields within a class."""
        # Process member functions
        for match in re.finditer(self.patterns['function'], class_content, re.MULTILINE):
            return_type = match.group(1).strip()
            method_name = match.group(2)
            
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
                'cpp',
                return_type=return_type
            )
            method_id = self.current_cpg.add_node(method_node)
            
            # Link to class
            edge = self.create_edge(class_id, method_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Process method body
            self._process_function_body(method_content, method_id, start_line)
        
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
                    'cpp',
                    is_constructor=True
                )
                constructor_id = self.current_cpg.add_node(constructor_node)
                
                # Link to class
                edge = self.create_edge(class_id, constructor_id, EdgeType.AST_CHILD)
                self.current_cpg.add_edge(edge)
        
        # Process destructors
        for match in re.finditer(self.patterns['destructor'], class_content, re.MULTILINE):
            destructor_name = match.group(1)
            
            if destructor_name == class_name:
                start_line = class_start_line + class_content[:match.start()].count('\n')
                
                # Find destructor end
                end_pos = self._find_matching_brace(class_content, match.end() - 1)
                end_line = class_start_line + class_content[:end_pos].count('\n') if end_pos else start_line
                
                # Extract destructor content
                destructor_content = class_content[match.start():end_pos] if end_pos else match.group(0)
                
                destructor_node = self.create_node(
                    NodeType.METHOD,
                    f"~{destructor_name}",
                    destructor_content,
                    self.current_file_path,
                    start_line,
                    end_line,
                    0,
                    0,
                    'cpp',
                    is_destructor=True
                )
                destructor_id = self.current_cpg.add_node(destructor_node)
                
                # Link to class
                edge = self.create_edge(class_id, destructor_id, EdgeType.AST_CHILD)
                self.current_cpg.add_edge(edge)
        
        # Process member variables
        for match in re.finditer(self.patterns['variable_decl'], class_content, re.MULTILINE):
            var_type = match.group(1).strip()
            var_name = match.group(2)
            
            start_line = class_start_line + class_content[:match.start()].count('\n')
            
            member_var_node = self.create_node(
                NodeType.VARIABLE,
                var_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'cpp',
                variable_type=var_type,
                is_member=True
            )
            member_var_id = self.current_cpg.add_node(member_var_node)
            
            # Track member variable definition for DFG
            if var_name not in self.variable_definitions:
                self.variable_definitions[var_name] = []
            self.variable_definitions[var_name].append(member_var_id)
            
            # Link to class
            edge = self.create_edge(class_id, member_var_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_struct_members(self, struct_content: str, struct_id: str, struct_start_line: int):
        """Process members within a struct."""
        # Process member variables
        for match in re.finditer(self.patterns['variable_decl'], struct_content, re.MULTILINE):
            var_type = match.group(1).strip()
            var_name = match.group(2)
            
            start_line = struct_start_line + struct_content[:match.start()].count('\n')
            
            member_var_node = self.create_node(
                NodeType.VARIABLE,
                var_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'cpp',
                variable_type=var_type,
                is_member=True
            )
            member_var_id = self.current_cpg.add_node(member_var_node)
            
            # Track member variable definition for DFG
            if var_name not in self.variable_definitions:
                self.variable_definitions[var_name] = []
            self.variable_definitions[var_name].append(member_var_id)
            
            # Link to struct
            edge = self.create_edge(struct_id, member_var_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_function_body(self, func_content: str, func_id: str, func_start_line: int):
        """Process statements within a function body."""
        # Process local variables
        for match in re.finditer(self.patterns['variable_decl'], func_content, re.MULTILINE):
            var_type = match.group(1).strip()
            var_name = match.group(2)
            
            start_line = func_start_line + func_content[:match.start()].count('\n')
            
            var_node = self.create_node(
                NodeType.VARIABLE,
                var_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'cpp',
                variable_type=var_type
            )
            var_id = self.current_cpg.add_node(var_node)
            
            # Track variable definition for DFG
            if var_name not in self.variable_definitions:
                self.variable_definitions[var_name] = []
            self.variable_definitions[var_name].append(var_id)
            
            # Link to function
            edge = self.create_edge(func_id, var_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Process assignments
        for match in re.finditer(self.patterns['assignment'], func_content, re.MULTILINE):
            var_name = match.group(1)
            start_line = func_start_line + func_content[:match.start()].count('\n')
            
            assign_node = self.create_node(
                NodeType.ASSIGNMENT,
                "assignment",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'cpp',
                target=var_name
            )
            assign_id = self.current_cpg.add_node(assign_node)
            
            # Track for DFG
            base_var = var_name.split('.')[0].split('[')[0].split('->')[0]
            # Track assignment for DFG
            if base_var not in self.variable_definitions:
                self.variable_definitions[base_var] = []
            self.variable_definitions[base_var].append(assign_id)
            
            # Track variable usage in assignment expression
            used_vars = self._extract_cpp_variables_from_code(match.group(0))
            for used_var in used_vars:
                if used_var != base_var:  # Don't track self-reference
                    if used_var not in self.variable_uses:
                        self.variable_uses[used_var] = []
                    self.variable_uses[used_var].append(assign_id)
            
            # Link to function
            edge = self.create_edge(func_id, assign_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Process function calls
        for match in re.finditer(self.patterns['function_call'], func_content, re.MULTILINE):
            call_name = match.group(1)
            start_line = func_start_line + func_content[:match.start()].count('\n')
            
            call_node = self.create_node(
                NodeType.CALL,
                call_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'cpp',
                function_name=call_name
            )
            call_id = self.current_cpg.add_node(call_node)
            
            # Track variable usage for DFG - extract variables from the call expression
            used_vars = self._extract_cpp_variables_from_code(match.group(0))
            for var_name in used_vars:
                if var_name not in self.variable_uses:
                    self.variable_uses[var_name] = []
                self.variable_uses[var_name].append(call_id)
            
            # Link to function
            edge = self.create_edge(func_id, call_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Process return statements
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
                'cpp'
            )
            return_id = self.current_cpg.add_node(return_node)
            
            # Link to function
            edge = self.create_edge(func_id, return_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Process control flow statements
        self._process_control_flow_in_function(func_content, func_id, func_start_line)
    
    def _extract_cpp_variables_from_code(self, code: str) -> List[str]:
        """Extract variable names from C++ code."""
        variables = []
        
        # Skip C++ keywords
        cpp_keywords = {
            'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default',
            'break', 'continue', 'return', 'class', 'struct', 'enum', 'union',
            'public', 'private', 'protected', 'static', 'const', 'volatile',
            'virtual', 'override', 'final', 'explicit', 'inline', 'extern',
            'namespace', 'using', 'typedef', 'template', 'typename', 'auto',
            'int', 'char', 'float', 'double', 'bool', 'void', 'long', 'short',
            'unsigned', 'signed', 'true', 'false', 'nullptr', 'this', 'new',
            'delete', 'sizeof', 'operator', 'friend', 'mutable', 'register'
        }
        
        # Find variable patterns - simple identifiers
        for match in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', code):
            var_name = match.group(1)
            if var_name not in cpp_keywords and var_name not in variables:
                variables.append(var_name)
        
        return variables
    
    def _extract_function_parameters(self, func_signature: str, func_id: str, start_line: int):
        """Extract parameters from function signature."""
        # Match parameters in parentheses
        param_match = re.search(r'\(([^)]*)\)', func_signature)
        if param_match:
            params_str = param_match.group(1).strip()
            if params_str:
                # Split parameters by comma and process each
                params = [p.strip() for p in params_str.split(',')]
                for param in params:
                    if param:
                        # Extract parameter name (last word after removing pointers/references)
                        param_clean = re.sub(r'[&*]', ' ', param).strip()
                        parts = param_clean.split()
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
                                'cpp',
                                param_type=param_type
                            )
                            param_id = self.current_cpg.add_node(param_node)
                            
                            # Link to function
                            edge = self.create_edge(func_id, param_id, EdgeType.AST_CHILD)
                            self.current_cpg.add_edge(edge)
    
    def _process_control_flow_in_function(self, func_content: str, func_id: str, 
                                         func_start_line: int):
        """Process control flow statements within a function."""
        # If statements
        for match in re.finditer(self.patterns['if_statement'], func_content, re.MULTILINE):
            start_line = func_start_line + func_content[:match.start()].count('\n')
            
            # Find end of if block
            end_pos = self._find_matching_brace(func_content, match.end() - 1)
            end_line = func_start_line + func_content[:end_pos].count('\n') if end_pos else start_line
            
            if_content = func_content[match.start():end_pos] if end_pos else match.group(0)
            
            if_node = self.create_node(
                NodeType.CONDITION,
                "if",
                if_content,
                self.current_file_path,
                start_line,
                end_line,
                0,
                0,
                'cpp'
            )
            if_id = self.current_cpg.add_node(if_node)
            
            # Link to function
            edge = self.create_edge(func_id, if_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Loops
        for pattern_name in ['for_loop', 'while_loop']:
            for match in re.finditer(self.patterns[pattern_name], func_content, re.MULTILINE):
                start_line = func_start_line + func_content[:match.start()].count('\n')
                
                # Find end of loop
                end_pos = self._find_matching_brace(func_content, match.end() - 1)
                end_line = func_start_line + func_content[:end_pos].count('\n') if end_pos else start_line
                
                loop_content = func_content[match.start():end_pos] if end_pos else match.group(0)
                
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
                    'cpp',
                    loop_type=loop_type
                )
                loop_id = self.current_cpg.add_node(loop_node)
                
                # Link to function
                edge = self.create_edge(func_id, loop_id, EdgeType.AST_CHILD)
                self.current_cpg.add_edge(edge)
        
        # Try-catch blocks
        for match in re.finditer(self.patterns['try_catch'], func_content, re.MULTILINE):
            start_line = func_start_line + func_content[:match.start()].count('\n')
            
            # Find end of try block
            end_pos = self._find_matching_brace(func_content, match.end() - 1)
            end_line = func_start_line + func_content[:end_pos].count('\n') if end_pos else start_line
            
            try_content = func_content[match.start():end_pos] if end_pos else match.group(0)
            
            try_node = self.create_node(
                NodeType.EXCEPTION,
                "try",
                try_content,
                self.current_file_path,
                start_line,
                end_line,
                0,
                0,
                'cpp'
            )
            try_id = self.current_cpg.add_node(try_node)
            
            # Link to function
            edge = self.create_edge(func_id, try_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _get_template_before(self, content: str, pos: int) -> Optional[str]:
        """Get template declaration that appears before a given position."""
        # Look backwards from the position to find template
        text_before = content[:pos]
        lines_before = text_before.split('\n')
        
        # Check the last few lines for template
        for line in reversed(lines_before[-3:]):  # Check last 3 lines
            line = line.strip()
            if line.startswith('template'):
                match = re.match(r'template\s*<([^>]+)>', line)
                if match:
                    return match.group(1)
            elif line and not line.startswith('//') and not line.startswith('/*'):
                # Stop if we hit non-template, non-comment code
                break
        
        return None
    
    def _is_inside_class_or_struct(self, content: str, pos: int) -> bool:
        """Check if position is inside a class or struct definition."""
        # Simplified check - count braces before this position
        text_before = content[:pos]
        
        # Find class/struct declarations before this position
        class_matches = list(re.finditer(self.patterns['class'], text_before, re.MULTILINE))
        struct_matches = list(re.finditer(self.patterns['struct'], text_before, re.MULTILINE))
        
        all_matches = class_matches + struct_matches
        all_matches.sort(key=lambda x: x.start())
        
        for match in reversed(all_matches):
            # Check if we're inside the braces of this class/struct
            brace_pos = text_before.find('{', match.end())
            if brace_pos != -1 and brace_pos < pos:
                # Check if the closing brace comes after our position
                closing_brace = self._find_matching_brace(content, brace_pos)
                if closing_brace and closing_brace > pos:
                    return True
        
        return False
    
    def _is_inside_function_or_class(self, content: str, pos: int) -> bool:
        """Check if position is inside a function or class definition."""
        # This is a simplified check
        text_before = content[:pos]
        open_braces = text_before.count('{')
        close_braces = text_before.count('}')
        
        # If we have more open braces than close braces, we're inside something
        return open_braces > close_braces
    
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
                if pos == 0 or content[pos - 1] != '\\':
                    in_string = not in_string
            elif char == "'" and not in_string and not in_comment:
                if pos == 0 or content[pos - 1] != '\\':
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
            language='cpp'
        )
        entry_id = cpg.add_node(entry_node)
        
        exit_node = self.create_node(
            NodeType.EXIT,
            f"{func_node.name}_exit",
            "",
            func_node.file_path,
            func_node.end_line,
            func_node.end_line,
            language='cpp'
        )
        exit_id = cpg.add_node(exit_node)
        
        # Get all child nodes of the function
        child_nodes = cpg.get_children(func_node.id)
        
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
    
    def _extract_cpp_variables_from_code(self, code: str) -> List[str]:
        """Extract variable names from C++ code snippet."""
        # Pattern for C++ identifiers
        var_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        variables = []
        
        # C++ keywords to skip
        cpp_keywords = {
            'alignas', 'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor',
            'bool', 'break', 'case', 'catch', 'char', 'char16_t', 'char32_t', 'class',
            'compl', 'const', 'constexpr', 'const_cast', 'continue', 'decltype', 'default',
            'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum', 'explicit', 'export',
            'extern', 'false', 'float', 'for', 'friend', 'goto', 'if', 'inline', 'int',
            'long', 'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq', 'nullptr',
            'operator', 'or', 'or_eq', 'private', 'protected', 'public', 'register',
            'reinterpret_cast', 'return', 'short', 'signed', 'sizeof', 'static',
            'static_assert', 'static_cast', 'struct', 'switch', 'template', 'this',
            'thread_local', 'throw', 'true', 'try', 'typedef', 'typeid', 'typename',
            'union', 'unsigned', 'using', 'virtual', 'void', 'volatile', 'wchar_t',
            'while', 'xor', 'xor_eq', 'override', 'final'
        }
        
        for match in re.finditer(var_pattern, code):
            var_name = match.group(0)
            if var_name.lower() not in cpp_keywords:
                variables.append(var_name)
        
        return list(set(variables))
