#!/usr/bin/env python3
"""
Swift CPG Parser

This module implements a comprehensive Code Property Graph parser for Swift,
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


class SwiftCPGParser(CPGParser):
    """Swift-specific CPG parser using regex patterns and Swift syntax analysis."""
    
    def __init__(self):
        """Initialize the Swift parser."""
        self.current_cpg = None
        self.current_file_path = ""
        self.variable_definitions = {}
        self.variable_uses = {}
        self.scope_stack = []
        self.current_scope = "global"
        
        # Swift-specific patterns
        self.patterns = {
            'import': r'import\s+(\w+)',
            'class': r'class\s+(\w+)(?:\s*:\s*([^{]+))?\s*\{',
            'struct': r'struct\s+(\w+)(?:\s*:\s*([^{]+))?\s*\{',
            'enum': r'enum\s+(\w+)(?:\s*:\s*([^{]+))?\s*\{',
            'protocol': r'protocol\s+(\w+)(?:\s*:\s*([^{]+))?\s*\{',
            'extension': r'extension\s+(\w+)(?:\s*:\s*([^{]+))?\s*\{',
            'func': r'func\s+(\w+)(?:\s*<[^>]+>)?\s*\([^)]*\)(?:\s*->\s*[^{]+)?\s*\{',
            'init': r'init(?:\s*<[^>]+>)?\s*\([^)]*\)(?:\s*throws)?\s*\{',
            'deinit': r'deinit\s*\{',
            'var': r'var\s+(\w+)(?:\s*:\s*[^=]+)?(?:\s*=\s*[^{]+)?[;{]',
            'let': r'let\s+(\w+)(?:\s*:\s*[^=]+)?(?:\s*=\s*[^{]+)?[;{]',
            'static_var': r'static\s+var\s+(\w+)',
            'static_let': r'static\s+let\s+(\w+)',
            'computed_property': r'var\s+(\w+)\s*:\s*[^{]+\s*\{',
            'subscript': r'subscript\s*\([^)]+\)\s*->\s*[^{]+\s*\{',
            'if_statement': r'if\s+[^{]+\s*\{',
            'guard_statement': r'guard\s+[^{]+\s*else\s*\{',
            'switch_statement': r'switch\s+[^{]+\s*\{',
            'case': r'case\s+[^{]+:',
            'default': r'default\s*:',
            'for_loop': r'for\s+[^{]+\s*\{',
            'while_loop': r'while\s+[^{]+\s*\{',
            'repeat_while': r'repeat\s*\{[^}]*\}\s*while\s+[^{]+\s*\{',
            'do_catch': r'do\s*\{',
            'catch': r'catch\s+[^{]+\s*\{',
            'defer': r'defer\s*\{',
            'return': r'return\s+[^;]+',
            'throw': r'throw\s+[^;]+',
            'try': r'try\s+[^;]+',
            'try_question': r'try\?\s+[^;]+',
            'try_bang': r'try!\s+[^;]+',
            'function_call': r'(\w+(?:\.\w+)*)\s*\([^)]*\)',
            'method_call': r'\.(\w+)\s*\([^)]*\)',
            'property_access': r'\.(\w+)',
            'optional_chaining': r'\?\.(\w+)',
            'force_unwrap': r'!\.(\w+)',
            'nil_coalescing': r'\?\?',
            'ternary': r'\?\s*[^:]+\s*:',
            'closure': r'\{[^}]*in\s*[^}]*\}',
            'trailing_closure': r'\{[^}]*\}',
            'generics': r'<[^>]+>',
            'where_clause': r'where\s+[^{]+',
            'access_control': r'(public|private|internal|fileprivate|open)\s+',
            'override': r'override\s+',
            'final': r'final\s+',
            'mutating': r'mutating\s+',
            'static': r'static\s+',
            'class_method': r'class\s+func\s+(\w+)',
            'weak': r'weak\s+',
            'unowned': r'unowned\s+',
            'lazy': r'lazy\s+',
            'optional': r'\?',
            'implicitly_unwrapped': r'!',
        }
    
    def parse(self, content: str, file_path: str) -> CodePropertyGraph:
        """Parse Swift source code and generate complete CPG."""
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
        """Build Abstract Syntax Tree for Swift code."""
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
            'swift'
        )
        module_id = self.current_cpg.add_node(module_node)
        
        # Process different constructs
        self._process_imports(content, lines, module_id)
        self._process_classes(content, lines, module_id)
        self._process_structs(content, lines, module_id)
        self._process_enums(content, lines, module_id)
        self._process_protocols(content, lines, module_id)
        self._process_extensions(content, lines, module_id)
        self._process_functions(content, lines, module_id)
        self._process_variables(content, lines, module_id)
        self._process_control_flow(content, lines, module_id)
        self._process_method_calls(content, lines, module_id)
        
        return self.current_cpg
    
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
                'swift',
                module_name=import_name
            )
            import_id = self.current_cpg.add_node(import_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, import_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_classes(self, content: str, lines: List[str], parent_id: str):
        """Process class definitions."""
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
                'swift',
                inheritance=inheritance
            )
            class_id = self.current_cpg.add_node(class_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, class_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Process class members
            self._process_class_members(content, lines, class_id, line_num)
    
    def _process_structs(self, content: str, lines: List[str], parent_id: str):
        """Process struct definitions."""
        for match in re.finditer(self.patterns['struct'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            struct_name = match.group(1)
            conformance = match.group(2).strip() if match.group(2) else None
            
            struct_node = self.create_node(
                NodeType.CLASS,  # Treat structs as classes for now
                struct_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'swift',
                is_struct=True,
                conformance=conformance
            )
            struct_id = self.current_cpg.add_node(struct_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, struct_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Process struct members
            self._process_class_members(content, lines, struct_id, line_num)
    
    def _process_enums(self, content: str, lines: List[str], parent_id: str):
        """Process enum definitions."""
        for match in re.finditer(self.patterns['enum'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            enum_name = match.group(1)
            raw_type = match.group(2).strip() if match.group(2) else None
            
            enum_node = self.create_node(
                NodeType.CLASS,  # Treat enums as classes for now
                enum_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'swift',
                is_enum=True,
                raw_type=raw_type
            )
            enum_id = self.current_cpg.add_node(enum_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, enum_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_protocols(self, content: str, lines: List[str], parent_id: str):
        """Process protocol definitions."""
        for match in re.finditer(self.patterns['protocol'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            protocol_name = match.group(1)
            inheritance = match.group(2).strip() if match.group(2) else None
            
            protocol_node = self.create_node(
                NodeType.CLASS,  # Treat protocols as classes for now
                protocol_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'swift',
                is_protocol=True,
                inheritance=inheritance
            )
            protocol_id = self.current_cpg.add_node(protocol_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, protocol_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_extensions(self, content: str, lines: List[str], parent_id: str):
        """Process extension definitions."""
        for match in re.finditer(self.patterns['extension'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            extended_type = match.group(1)
            conformance = match.group(2).strip() if match.group(2) else None
            
            extension_node = self.create_node(
                NodeType.CLASS,  # Treat extensions as classes for now
                f"extension_{extended_type}",
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'swift',
                is_extension=True,
                extended_type=extended_type,
                conformance=conformance
            )
            extension_id = self.current_cpg.add_node(extension_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, extension_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_functions(self, content: str, lines: List[str], parent_id: str):
        """Process function definitions."""
        # Regular functions
        for match in re.finditer(self.patterns['func'], content, re.MULTILINE):
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
                'swift'
            )
            func_id = self.current_cpg.add_node(func_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, func_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Track function definition
            self.variable_definitions[func_name] = func_id
        
        # Initializers
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
                'swift',
                is_initializer=True
            )
            init_id = self.current_cpg.add_node(init_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, init_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Deinitializers
        for match in re.finditer(self.patterns['deinit'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            
            deinit_node = self.create_node(
                NodeType.FUNCTION,
                "deinit",
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'swift',
                is_deinitializer=True
            )
            deinit_id = self.current_cpg.add_node(deinit_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, deinit_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_class_members(self, content: str, lines: List[str], class_id: str, start_line: int):
        """Process members within a class/struct."""
        # Find the end of the class by looking for matching '}' statements
        class_end_line = self._find_matching_brace(content, start_line)
        
        # Extract class content
        class_content = '\n'.join(lines[start_line-1:class_end_line])
        
        # Process methods within this class
        for match in re.finditer(self.patterns['func'], class_content, re.MULTILINE):
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
                'swift'
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
                'swift',
                variable_type='var'
            )
            var_id = self.current_cpg.add_node(var_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, var_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Track variable definition
            self.variable_definitions[var_name] = var_id
        
        # Constant declarations (let)
        for match in re.finditer(self.patterns['let'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            var_name = match.group(1)
            
            let_node = self.create_node(
                NodeType.VARIABLE,
                var_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'swift',
                variable_type='let'
            )
            let_id = self.current_cpg.add_node(let_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, let_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Track variable definition
            self.variable_definitions[var_name] = let_id
        
        # Computed properties
        for match in re.finditer(self.patterns['computed_property'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            prop_name = match.group(1)
            
            prop_node = self.create_node(
                NodeType.VARIABLE,
                prop_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'swift',
                variable_type='computed_property'
            )
            prop_id = self.current_cpg.add_node(prop_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, prop_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Track variable definition
            self.variable_definitions[prop_name] = prop_id
    
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
                'swift'
            )
            if_id = self.current_cpg.add_node(if_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, if_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Guard statements
        for match in re.finditer(self.patterns['guard_statement'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            
            guard_node = self.create_node(
                NodeType.CONDITION,
                "guard_statement",
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'swift'
            )
            guard_id = self.current_cpg.add_node(guard_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, guard_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Switch statements
        for match in re.finditer(self.patterns['switch_statement'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            
            switch_node = self.create_node(
                NodeType.CONDITION,
                "switch_statement",
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'swift'
            )
            switch_id = self.current_cpg.add_node(switch_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, switch_id, EdgeType.AST_CHILD)
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
                'swift'
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
                'swift'
            )
            while_id = self.current_cpg.add_node(while_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, while_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_method_calls(self, content: str, lines: List[str], parent_id: str):
        """Process method calls."""
        # Function calls
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
                'swift'
            )
            call_id = self.current_cpg.add_node(call_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, call_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Track function use
            if func_name not in self.variable_uses:
                self.variable_uses[func_name] = []
            self.variable_uses[func_name].append(call_id)
        
        # Method calls
        for match in re.finditer(self.patterns['method_call'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            method_name = match.group(1)
            
            call_node = self.create_node(
                NodeType.CALL,
                method_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'swift'
            )
            call_id = self.current_cpg.add_node(call_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, call_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Track method use
            if method_name not in self.variable_uses:
                self.variable_uses[method_name] = []
            self.variable_uses[method_name].append(call_id)
    
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