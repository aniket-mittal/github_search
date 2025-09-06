#!/usr/bin/env python3
"""
Rust CPG Parser

This module implements a Code Property Graph parser for Rust,
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


class RustCPGParser(CPGParser):
    """Rust CPG parser using regex patterns."""
    
    def __init__(self):
        """Initialize the Rust parser."""
        self.current_cpg = None
        self.current_file_path = ""
        self.variable_definitions = {}
        self.variable_uses = {}
        
        # Rust patterns
        self.patterns = {
            'use': r'use\s+([^;]+);',
            'mod': r'mod\s+(\w+)(?:\s*\{|\s*;)',
            'struct': r'(?:pub\s+)?struct\s+(\w+)(?:<[^>]+>)?\s*\{',
            'enum': r'(?:pub\s+)?enum\s+(\w+)(?:<[^>]+>)?\s*\{',
            'trait': r'(?:pub\s+)?trait\s+(\w+)(?:<[^>]+>)?\s*\{',
            'impl': r'impl(?:<[^>]+>)?\s+(?:(\w+)(?:<[^>]+>)?\s+for\s+)?(\w+)(?:<[^>]+>)?\s*\{',
            'function': r'(?:pub\s+)?fn\s+(\w+)(?:<[^>]+>)?\s*\([^)]*\)(?:\s*->\s*[^{]+)?\s*\{',
            'method': r'(?:pub\s+)?fn\s+(\w+)(?:<[^>]+>)?\s*\(&[^)]*\)(?:\s*->\s*[^{]+)?\s*\{',
            'let_binding': r'let\s+(?:mut\s+)?(\w+)(?:\s*:\s*[^=]+)?\s*=\s*([^;]+);',
            'const': r'const\s+(\w+)\s*:\s*[^=]+\s*=\s*([^;]+);',
            'static': r'static\s+(?:mut\s+)?(\w+)\s*:\s*[^=]+\s*=\s*([^;]+);',
            'assignment': r'(\w+(?:\.\w+)*)\s*=\s*([^;]+);',
            'function_call': r'(\w+(?:::\w+)*(?:\.\w+)*)\s*\([^)]*\)',
            'macro_call': r'(\w+)!\s*\([^)]*\)',
            'if_statement': r'if\s+([^{]+)\s*\{',
            'match_statement': r'match\s+([^{]+)\s*\{',
            'for_loop': r'for\s+(\w+)\s+in\s+([^{]+)\s*\{',
            'while_loop': r'while\s+([^{]+)\s*\{',
            'loop': r'loop\s*\{',
            'return': r'return\s+[^;]*;?',
        }
    
    def parse(self, content: str, file_path: str) -> CodePropertyGraph:
        """Parse Rust source code and generate complete CPG."""
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
        """Build Abstract Syntax Tree for Rust code."""
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
            'rust'
        )
        module_id = self.current_cpg.add_node(module_node)
        
        # Process different constructs
        self._process_uses(content, lines, module_id)
        self._process_modules(content, lines, module_id)
        self._process_types(content, lines, module_id)
        self._process_functions(content, lines, module_id)
        self._process_variables(content, lines, module_id)
        self._process_function_calls(content, lines, module_id)
        self._process_control_flow(content, lines, module_id)
        
        return self.current_cpg
    
    def _process_uses(self, content: str, lines: List[str], parent_id: str):
        """Process use statements."""
        for match in re.finditer(self.patterns['use'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            use_path = match.group(1)
            
            use_node = self.create_node(
                NodeType.IMPORT,
                use_path,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'rust',
                use_path=use_path
            )
            use_id = self.current_cpg.add_node(use_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, use_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_modules(self, content: str, lines: List[str], parent_id: str):
        """Process module declarations."""
        for match in re.finditer(self.patterns['mod'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            mod_name = match.group(1)
            
            mod_node = self.create_node(
                NodeType.MODULE,
                mod_name,
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'rust',
                module_name=mod_name
            )
            mod_id = self.current_cpg.add_node(mod_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, mod_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_types(self, content: str, lines: List[str], parent_id: str):
        """Process struct, enum, and trait definitions."""
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
                'rust',
                type_kind='struct'
            )
            struct_id = self.current_cpg.add_node(struct_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, struct_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Enums
        for match in re.finditer(self.patterns['enum'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            enum_name = match.group(1)
            
            # Find the end of the enum
            brace_pos = content.find('{', match.start())
            if brace_pos != -1:
                end_pos = self._find_matching_brace(content, brace_pos)
                if end_pos:
                    end_line = content[:end_pos].count('\n') + 1
                    enum_code = content[match.start():end_pos]
                else:
                    end_line = start_line
                    enum_code = match.group(0)
            else:
                end_line = start_line
                enum_code = match.group(0)
            
            enum_node = self.create_node(
                NodeType.CLASS,  # Use CLASS for enums
                enum_name,
                enum_code,
                self.current_file_path,
                start_line,
                end_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                (end_pos if end_pos else match.end()) - content.rfind('\n', 0, end_pos if end_pos else match.end()) - 1,
                'rust',
                type_kind='enum'
            )
            enum_id = self.current_cpg.add_node(enum_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, enum_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Traits
        for match in re.finditer(self.patterns['trait'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            trait_name = match.group(1)
            
            # Find the end of the trait
            brace_pos = content.find('{', match.start())
            if brace_pos != -1:
                end_pos = self._find_matching_brace(content, brace_pos)
                if end_pos:
                    end_line = content[:end_pos].count('\n') + 1
                    trait_code = content[match.start():end_pos]
                else:
                    end_line = start_line
                    trait_code = match.group(0)
            else:
                end_line = start_line
                trait_code = match.group(0)
            
            trait_node = self.create_node(
                NodeType.CLASS,  # Use CLASS for traits
                trait_name,
                trait_code,
                self.current_file_path,
                start_line,
                end_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                (end_pos if end_pos else match.end()) - content.rfind('\n', 0, end_pos if end_pos else match.end()) - 1,
                'rust',
                type_kind='trait'
            )
            trait_id = self.current_cpg.add_node(trait_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, trait_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_functions(self, content: str, lines: List[str], parent_id: str):
        """Process function definitions."""
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
                'rust',
                function_name=func_name
            )
            func_id = self.current_cpg.add_node(func_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, func_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Process statements within function
            if func_code:
                self._process_statements_in_function(func_code, func_id, start_line)
        
        # Methods (in impl blocks)
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
                'rust',
                method_name=method_name
            )
            method_id = self.current_cpg.add_node(method_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, method_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Process statements within method
            if method_code:
                self._process_statements_in_function(method_code, method_id, start_line)
    
    def _process_statements_in_function(self, func_content: str, func_id: str, func_start_line: int):
        """Process statements within a function body."""
        # Find let bindings within function
        for match in re.finditer(self.patterns['let_binding'], func_content, re.MULTILINE):
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
                'rust',
                variable_name=var_name
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
            var_name = match.group(1).split('.')[0]  # Handle field access
            
            assign_node = self.create_node(
                NodeType.ASSIGNMENT,
                "assignment",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'rust',
                target=var_name
            )
            assign_id = self.current_cpg.add_node(assign_node)
            
            # Track assignment for DFG
            if var_name not in self.variable_definitions:
                self.variable_definitions[var_name] = []
            self.variable_definitions[var_name].append(assign_id)
            
            # Link to function
            edge = self.create_edge(func_id, assign_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Find control flow statements within function
        for match in re.finditer(self.patterns['if_statement'], func_content, re.MULTILINE):
            start_line = func_start_line + func_content[:match.start()].count('\n')
            condition = match.group(1).strip()
            
            if_node = self.create_node(
                NodeType.CONDITION,
                "if",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'rust',
                condition=condition
            )
            if_id = self.current_cpg.add_node(if_node)
            
            # Link to function
            edge = self.create_edge(func_id, if_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Find match statements within function
        for match in re.finditer(self.patterns['match_statement'], func_content, re.MULTILINE):
            start_line = func_start_line + func_content[:match.start()].count('\n')
            match_expr = match.group(1).strip()
            
            match_node = self.create_node(
                NodeType.CONDITION,
                "match",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'rust',
                condition=match_expr,
                control_type='match'
            )
            match_id = self.current_cpg.add_node(match_node)
            
            # Link to function
            edge = self.create_edge(func_id, match_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Find loops within function
        for match in re.finditer(self.patterns['for_loop'], func_content, re.MULTILINE):
            start_line = func_start_line + func_content[:match.start()].count('\n')
            loop_var = match.group(1)
            
            loop_node = self.create_node(
                NodeType.LOOP,
                "for",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'rust',
                loop_type='for',
                loop_variable=loop_var
            )
            loop_id = self.current_cpg.add_node(loop_node)
            
            # Link to function
            edge = self.create_edge(func_id, loop_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        for match in re.finditer(self.patterns['while_loop'], func_content, re.MULTILINE):
            start_line = func_start_line + func_content[:match.start()].count('\n')
            condition = match.group(1).strip()
            
            loop_node = self.create_node(
                NodeType.LOOP,
                "while",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'rust',
                loop_type='while',
                condition=condition
            )
            loop_id = self.current_cpg.add_node(loop_node)
            
            # Link to function
            edge = self.create_edge(func_id, loop_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        for match in re.finditer(self.patterns['loop'], func_content, re.MULTILINE):
            start_line = func_start_line + func_content[:match.start()].count('\n')
            
            loop_node = self.create_node(
                NodeType.LOOP,
                "loop",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                0,
                0,
                'rust',
                loop_type='infinite'
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
                'rust'
            )
            return_id = self.current_cpg.add_node(return_node)
            
            # Link to function
            edge = self.create_edge(func_id, return_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_variables(self, content: str, lines: List[str], parent_id: str):
        """Process variable declarations and assignments."""
        # Let bindings
        for match in re.finditer(self.patterns['let_binding'], content, re.MULTILINE):
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
                'rust',
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
        
        # Constants
        for match in re.finditer(self.patterns['const'], content, re.MULTILINE):
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
                'rust',
                variable_name=const_name,
                is_constant=True
            )
            const_id = self.current_cpg.add_node(const_node)
            
            # Track constant definition for DFG
            if const_name not in self.variable_definitions:
                self.variable_definitions[const_name] = []
            self.variable_definitions[var_name].append(const_id)
            
            # Link to parent
            edge = self.create_edge(parent_id, const_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_function_calls(self, content: str, lines: List[str], parent_id: str):
        """Process function calls and macro calls."""
        # Function calls
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
                'rust',
                function_name=func_name
            )
            call_id = self.current_cpg.add_node(call_node)
            
            # Track variable usage for DFG
            base_name = func_name.split('::')[0].split('.')[0]
            if base_name not in self.variable_uses:
                self.variable_uses[base_name] = []
            self.variable_uses[base_name].append(call_id)
            
            # Link to parent
            edge = self.create_edge(parent_id, call_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Macro calls
        for match in re.finditer(self.patterns['macro_call'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            macro_name = match.group(1)
            
            macro_node = self.create_node(
                NodeType.CALL,
                f"{macro_name}!",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'rust',
                function_name=macro_name,
                is_macro=True
            )
            macro_id = self.current_cpg.add_node(macro_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, macro_id, EdgeType.AST_CHILD)
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
                'rust',
                condition=condition
            )
            if_id = self.current_cpg.add_node(if_node)
            
            # Link to parent
            edge = self.create_edge(parent_id, if_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Loops
        for match in re.finditer(self.patterns['for_loop'], content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            loop_var = match.group(1)
            
            loop_node = self.create_node(
                NodeType.LOOP,
                "for",
                match.group(0),
                self.current_file_path,
                start_line,
                start_line,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.end()) - 1,
                'rust',
                loop_type='for',
                loop_variable=loop_var
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
            language='rust'
        )
        entry_id = cpg.add_node(entry_node)
        
        exit_node = self.create_node(
            NodeType.EXIT,
            f"{func_node.name}_exit",
            "",
            func_node.file_path,
            func_node.end_line,
            func_node.end_line,
            language='rust'
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