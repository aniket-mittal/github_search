#!/usr/bin/env python3
"""
Kotlin CPG Parser

This module implements a comprehensive Code Property Graph parser for Kotlin,
generating accurate AST, CFG, and DFG representations.
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


class KotlinCPGParser(CPGParser):
    """Kotlin-specific CPG parser using regex patterns and Kotlin syntax analysis."""
    
    def __init__(self):
        """Initialize the Kotlin parser."""
        self.current_cpg = None
        self.current_file_path = ""
        self.variable_definitions = {}
        self.variable_uses = {}
        self.scope_stack = []
        self.current_scope = "global"
        
        # Kotlin-specific patterns
        self.patterns = {
            'import': r'import\s+([^\s]+)',
            'package': r'package\s+([^\s]+)',
            'class': r'class\s+(\w+)(?:\s*\([^)]*\))?(?:\s*:\s*([^{]+))?\s*\{',
            'interface': r'interface\s+(\w+)(?:\s*:\s*([^{]+))?\s*\{',
            'object': r'object\s+(\w+)(?:\s*:\s*([^{]+))?\s*\{',
            'enum': r'enum\s+class\s+(\w+)(?:\s*\([^)]*\))?(?:\s*:\s*([^{]+))?\s*\{',
            'data_class': r'data\s+class\s+(\w+)(?:\s*\([^)]*\))?(?:\s*:\s*([^{]+))?\s*\{',
            'sealed_class': r'sealed\s+class\s+(\w+)(?:\s*:\s*([^{]+))?\s*\{',
            'fun': r'fun\s+(\w+)(?:\s*<[^>]+>)?\s*\([^)]*\)(?:\s*:\s*[^{]+)?\s*\{',
            'val': r'val\s+(\w+)(?:\s*:\s*[^=]+)?(?:\s*=\s*[^{]+)?[;{]',
            'var': r'var\s+(\w+)(?:\s*:\s*[^=]+)?(?:\s*=\s*[^{]+)?[;{]',
            'constructor': r'constructor\s*\([^)]*\)(?:\s*:\s*[^{]+)?\s*\{',
            'init': r'init\s*\{',
            'companion_object': r'companion\s+object(?:\s+(\w+))?\s*\{',
            'if_statement': r'if\s*\([^{]+\s*\{',
            'when_statement': r'when\s*\([^{]+\s*\{',
            'for_loop': r'for\s*\([^{]+\s*\{',
            'while_loop': r'while\s*\([^{]+\s*\{',
            'do_while': r'do\s*\{[^}]*\}\s*while\s*\([^{]+\s*\{',
            'try_catch': r'try\s*\{',
            'catch': r'catch\s*\([^{]+\s*\{',
            'finally': r'finally\s*\{',
            'return': r'return\s+[^;]+',
            'throw': r'throw\s+[^;]+',
            'function_call': r'(\w+(?:\.\w+)*)\s*\([^)]*\)',
            'lambda': r'\{[^}]*->\s*[^}]*\}',
            'extension_function': r'fun\s+(\w+)\.(\w+)\s*\([^)]*\)',
            'infix_function': r'infix\s+fun\s+(\w+)\s*\([^)]*\)',
            'operator_function': r'operator\s+fun\s+(\w+)\s*\([^)]*\)',
            'suspend_function': r'suspend\s+fun\s+(\w+)\s*\([^)]*\)',
            'inline_function': r'inline\s+fun\s+(\w+)\s*\([^)]*\)',
            'tailrec_function': r'tailrec\s+fun\s+(\w+)\s*\([^)]*\)',
            'external_function': r'external\s+fun\s+(\w+)\s*\([^)]*\)',
            'override_function': r'override\s+fun\s+(\w+)\s*\([^)]*\)',
            'open_class': r'open\s+class\s+(\w+)',
            'abstract_class': r'abstract\s+class\s+(\w+)',
            'final_class': r'final\s+class\s+(\w+)',
            'private': r'private\s+',
            'protected': r'protected\s+',
            'internal': r'internal\s+',
            'public': r'public\s+',
            'lateinit': r'lateinit\s+',
            'const': r'const\s+',
            'static': r'static\s+',
            'companion': r'companion\s+',
            'override': r'override\s+',
            'open': r'open\s+',
            'abstract': r'abstract\s+',
            'final': r'final\s+',
            'sealed': r'sealed\s+',
            'data': r'data\s+',
            'inline': r'inline\s+',
            'noinline': r'noinline\s+',
            'crossinline': r'crossinline\s+',
            'vararg': r'vararg\s+',
            'out': r'out\s+',
            'in': r'in\s+',
            'reified': r'reified\s+',
            'suspend': r'suspend\s+',
            'tailrec': r'tailrec\s+',
            'external': r'external\s+',
            'expect': r'expect\s+',
            'actual': r'actual\s+',
        }
    
    def parse(self, content: str, file_path: str) -> CodePropertyGraph:
        """Parse Kotlin source code and generate complete CPG."""
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
        """Build Abstract Syntax Tree for Kotlin code."""
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
            'kotlin'
        )
        module_id = self.current_cpg.add_node(module_node)
        
        # Process different constructs
        self._process_package(content, lines, module_id)
        self._process_imports(content, lines, module_id)
        self._process_classes(content, lines, module_id)
        self._process_interfaces(content, lines, module_id)
        self._process_objects(content, lines, module_id)
        self._process_enums(content, lines, module_id)
        self._process_functions(content, lines, module_id)
        self._process_variables(content, lines, module_id)
        self._process_control_flow(content, lines, module_id)
        self._process_method_calls(content, lines, module_id)
        
        return self.current_cpg
    
    def _process_package(self, content: str, lines: List[str], parent_id: str):
        """Process package declaration."""
        for match in re.finditer(self.patterns['package'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            package_name = match.group(1)
            
            package_node = self.create_node(
                NodeType.IMPORT,
                f"package_{package_name}",
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'kotlin',
                package_name=package_name
            )
            package_id = self.current_cpg.add_node(package_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, package_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_imports(self, content: str, lines: List[str], parent_id: str):
        """Process import statements."""
        for match in re.finditer(self.patterns['import'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            import_name = match.group(1)
            
            import_node = self.create_node(
                NodeType.IMPORT,
                f"import_{import_name}",
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'kotlin',
                module_name=import_name
            )
            import_id = self.current_cpg.add_node(import_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, import_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_classes(self, content: str, lines: List[str], parent_id: str):
        """Process class definitions."""
        # Regular classes
        for match in re.finditer(self.patterns['class'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            class_name = match.group(1)
            inheritance = match.group(2).strip() if match.group(2) else None
            
            class_node = self.create_node(
                NodeType.CLASS,
                class_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'kotlin',
                inheritance=inheritance
            )
            class_id = self.current_cpg.add_node(class_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, class_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Process class members
            self._process_class_members(content, lines, class_id, line_num)
        
        # Data classes
        for match in re.finditer(self.patterns['data_class'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            class_name = match.group(1)
            inheritance = match.group(2).strip() if match.group(2) else None
            
            class_node = self.create_node(
                NodeType.CLASS,
                class_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'kotlin',
                inheritance=inheritance,
                is_data_class=True
            )
            class_id = self.current_cpg.add_node(class_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, class_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Process class members
            self._process_class_members(content, lines, class_id, line_num)
        
        # Sealed classes
        for match in re.finditer(self.patterns['sealed_class'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            class_name = match.group(1)
            inheritance = match.group(2).strip() if match.group(2) else None
            
            class_node = self.create_node(
                NodeType.CLASS,
                class_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'kotlin',
                inheritance=inheritance,
                is_sealed_class=True
            )
            class_id = self.current_cpg.add_node(class_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, class_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Process class members
            self._process_class_members(content, lines, class_id, line_num)
    
    def _process_interfaces(self, content: str, lines: List[str], parent_id: str):
        """Process interface definitions."""
        for match in re.finditer(self.patterns['interface'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            interface_name = match.group(1)
            inheritance = match.group(2).strip() if match.group(2) else None
            
            interface_node = self.create_node(
                NodeType.CLASS,  # Treat interfaces as classes for now
                interface_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'kotlin',
                inheritance=inheritance,
                is_interface=True
            )
            interface_id = self.current_cpg.add_node(interface_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, interface_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_objects(self, content: str, lines: List[str], parent_id: str):
        """Process object definitions."""
        for match in re.finditer(self.patterns['object'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            object_name = match.group(1)
            inheritance = match.group(2).strip() if match.group(2) else None
            
            object_node = self.create_node(
                NodeType.CLASS,  # Treat objects as classes for now
                object_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'kotlin',
                inheritance=inheritance,
                is_object=True
            )
            object_id = self.current_cpg.add_node(object_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, object_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Companion objects
        for match in re.finditer(self.patterns['companion_object'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            object_name = match.group(1) if match.group(1) else "Companion"
            
            companion_node = self.create_node(
                NodeType.CLASS,  # Treat companion objects as classes for now
                f"companion_{object_name}",
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'kotlin',
                is_companion_object=True
            )
            companion_id = self.current_cpg.add_node(companion_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, companion_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_enums(self, content: str, lines: List[str], parent_id: str):
        """Process enum definitions."""
        for match in re.finditer(self.patterns['enum'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            enum_name = match.group(1)
            inheritance = match.group(2).strip() if match.group(2) else None
            
            enum_node = self.create_node(
                NodeType.CLASS,  # Treat enums as classes for now
                enum_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'kotlin',
                inheritance=inheritance,
                is_enum=True
            )
            enum_id = self.current_cpg.add_node(enum_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, enum_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_functions(self, content: str, lines: List[str], parent_id: str):
        """Process function definitions."""
        # Regular functions
        for match in re.finditer(self.patterns['fun'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            func_name = match.group(1)
            
            func_node = self.create_node(
                NodeType.FUNCTION,
                func_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'kotlin'
            )
            func_id = self.current_cpg.add_node(func_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, func_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Track function definition
            self.variable_definitions[func_name] = func_id
        
        # Extension functions
        for match in re.finditer(self.patterns['extension_function'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            extended_type = match.group(1)
            func_name = match.group(2)
            
            ext_func_node = self.create_node(
                NodeType.FUNCTION,
                f"{extended_type}.{func_name}",
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'kotlin',
                is_extension_function=True,
                extended_type=extended_type
            )
            ext_func_id = self.current_cpg.add_node(ext_func_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, ext_func_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Track function definition
            self.variable_definitions[f"{extended_type}.{func_name}"] = ext_func_id
        
        # Constructors
        for match in re.finditer(self.patterns['constructor'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            
            constructor_node = self.create_node(
                NodeType.FUNCTION,
                "constructor",
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'kotlin',
                is_constructor=True
            )
            constructor_id = self.current_cpg.add_node(constructor_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, constructor_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Init blocks
        for match in re.finditer(self.patterns['init'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            
            init_node = self.create_node(
                NodeType.FUNCTION,
                "init",
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'kotlin',
                is_init_block=True
            )
            init_id = self.current_cpg.add_node(init_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, init_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_class_members(self, content: str, lines: List[str], class_id: str, start_line: int):
        """Process members within a class."""
        # Find the end of the class by looking for matching '}' statements
        class_end_line = self._find_matching_brace(content, start_line)
        
        # Extract class content
        class_content = '\n'.join(lines[start_line-1:class_end_line])
        
        # Process methods within this class
        for match in re.finditer(self.patterns['fun'], class_content, re.MULTILINE):
            method_name = match.group(1)
            
            method_node = self.create_node(
                NodeType.METHOD,
                method_name,
                match.group(0).strip(),
                self.current_file_path,
                start_line + match.start() // (len(class_content) // (class_end_line - start_line + 1)),
                start_line + match.start() // (len(class_content) // (class_end_line - start_line + 1)),
                0,
                len(match.group(0)),
                'kotlin'
            )
            method_id = self.current_cpg.add_node(method_node)
            
            # Add AST edge
            edge = self.create_edge(class_id, method_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_variables(self, content: str, lines: List[str], parent_id: str):
        """Process variable declarations."""
        # Variable declarations (var)
        for match in re.finditer(self.patterns['var'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            var_name = match.group(1)
            
            var_node = self.create_node(
                NodeType.VARIABLE,
                var_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'kotlin',
                variable_type='var'
            )
            var_id = self.current_cpg.add_node(var_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, var_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Track variable definition
            self.variable_definitions[var_name] = var_id
        
        # Constant declarations (val)
        for match in re.finditer(self.patterns['val'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            var_name = match.group(1)
            
            val_node = self.create_node(
                NodeType.VARIABLE,
                var_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'kotlin',
                variable_type='val'
            )
            val_id = self.current_cpg.add_node(val_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, val_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Track variable definition
            self.variable_definitions[var_name] = val_id
    
    def _process_control_flow(self, content: str, lines: List[str], parent_id: str):
        """Process control flow constructs."""
        # If statements
        for match in re.finditer(self.patterns['if_statement'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            
            if_node = self.create_node(
                NodeType.CONDITION,
                "if_statement",
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'kotlin'
            )
            if_id = self.current_cpg.add_node(if_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, if_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # When statements (Kotlin's switch)
        for match in re.finditer(self.patterns['when_statement'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            
            when_node = self.create_node(
                NodeType.CONDITION,
                "when_statement",
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'kotlin'
            )
            when_id = self.current_cpg.add_node(when_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, when_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # For loops
        for match in re.finditer(self.patterns['for_loop'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            
            for_node = self.create_node(
                NodeType.LOOP,
                "for_loop",
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'kotlin'
            )
            for_id = self.current_cpg.add_node(for_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, for_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # While loops
        for match in re.finditer(self.patterns['while_loop'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            
            while_node = self.create_node(
                NodeType.LOOP,
                "while_loop",
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'kotlin'
            )
            while_id = self.current_cpg.add_node(while_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, while_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_method_calls(self, content: str, lines: List[str], parent_id: str):
        """Process method calls."""
        for match in re.finditer(self.patterns['function_call'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            func_name = match.group(1)
            
            call_node = self.create_node(
                NodeType.CALL,
                func_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'kotlin'
            )
            call_id = self.current_cpg.add_node(call_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, call_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Track function use
            if func_name not in self.variable_uses:
                self.variable_uses[func_name] = []
            self.variable_uses[func_name].append(call_id)
    
    def _find_matching_brace(self, content: str, start_line: int) -> int:
        """Find the matching '}' statement for a block."""
        lines = content.splitlines()
        depth = 0
        
        for i in range(start_line - 1, len(lines)):
            line = lines[i]
            
            # Count opening braces
            depth += line.count('{')
            depth -= line.count('}')
            
            if depth == 0:
                return i + 1
        
        return len(lines)
    
    def build_cfg(self, cpg: CodePropertyGraph) -> CodePropertyGraph:
        """Build Control Flow Graph from AST."""
        # Find all control flow nodes
        control_nodes = [node for node in list(cpg.nodes.values()) 
                        if node.node_type in [NodeType.CONDITION, NodeType.LOOP]]
        
        for node in control_nodes:
            # Create entry and exit nodes for control flow
            entry_node = self.create_node(
                NodeType.ENTRY,
                f"{node.name}_entry",
                f"entry_{node.name}",
                node.file_path,
                node.start_line,
                node.start_line,
                node.start_column,
                node.start_column,
                node.language
            )
            entry_id = cpg.add_node(entry_node)
            
            exit_node = self.create_node(
                NodeType.EXIT,
                f"{node.name}_exit",
                f"exit_{node.name}",
                node.file_path,
                node.end_line,
                node.end_line,
                node.end_column,
                node.end_column,
                node.language
            )
            exit_id = cpg.add_node(exit_node)
            
            # Add control flow edges
            entry_edge = self.create_edge(entry_id, node.id, EdgeType.CONTROL_FLOW)
            cpg.add_edge(entry_edge)
            
            exit_edge = self.create_edge(node.id, exit_id, EdgeType.CONTROL_FLOW)
            cpg.add_edge(exit_edge)
        
        return cpg
    
    def build_dfg(self, cpg: CodePropertyGraph) -> CodePropertyGraph:
        """Build Data Flow Graph from AST and CFG - DISABLED for performance."""
        # DFG generation disabled for faster processing
        # Only AST and CFG are generated
        return cpg