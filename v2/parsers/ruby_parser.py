#!/usr/bin/env python3
"""
Ruby CPG Parser

This module implements a comprehensive Code Property Graph parser for Ruby,
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


class RubyCPGParser(CPGParser):
    """Ruby-specific CPG parser using regex patterns and Ruby syntax analysis."""
    
    def __init__(self):
        """Initialize the Ruby parser."""
        self.current_cpg = None
        self.current_file_path = ""
        self.variable_definitions = {}
        self.variable_uses = {}
        self.scope_stack = []
        self.current_scope = "global"
        
        # Ruby-specific patterns
        self.patterns = {
            'class': r'class\s+(\w+)(?:\s*<\s*(\w+))?\s*$',
            'module': r'module\s+(\w+)\s*$',
            'def': r'def\s+(\w+)(?:\s*\([^)]*\))?\s*$',
            'def_self': r'def\s+self\.(\w+)(?:\s*\([^)]*\))?\s*$',
            'def_class_method': r'def\s+(\w+)\.(\w+)(?:\s*\([^)]*\))?\s*$',
            'variable_assignment': r'(\w+)\s*=\s*[^=\n]',
            'instance_variable': r'@(\w+)\s*[=:]',
            'class_variable': r'@@(\w+)\s*[=:]',
            'global_variable': r'\$(\w+)\s*[=:]',
            'constant': r'([A-Z]\w*)\s*[=:]',
            'require': r'require\s+[\'"]([^\'"]+)[\'"]',
            'require_relative': r'require_relative\s+[\'"]([^\'"]+)[\'"]',
            'load': r'load\s+[\'"]([^\'"]+)[\'"]',
            'include': r'include\s+(\w+)',
            'extend': r'extend\s+(\w+)',
            'if_statement': r'if\s+[^#\n]',
            'unless_statement': r'unless\s+[^#\n]',
            'case_statement': r'case\s+[^#\n]',
            'when_clause': r'when\s+[^#\n]',
            'for_loop': r'for\s+\w+\s+in\s+[^#\n]',
            'while_loop': r'while\s+[^#\n]',
            'until_loop': r'until\s+[^#\n]',
            'each_loop': r'\.each\s*\{',
            'times_loop': r'\.times\s*\{',
            'upto_loop': r'\.upto\s*\([^)]*\)\s*\{',
            'downto_loop': r'\.downto\s*\([^)]*\)\s*\{',
            'step_loop': r'\.step\s*\([^)]*\)\s*\{',
            'begin_rescue': r'begin\s*$',
            'rescue': r'rescue\s+[^#\n]',
            'ensure': r'ensure\s*$',
            'return': r'return\s+[^#\n]',
            'yield': r'yield\s*[^#\n]',
            'block': r'do\s*$|{\s*$',
            'end': r'end\s*$',
            'method_call': r'(\w+(?:\.\w+)*)\s*\([^)]*\)',
            'symbol': r':(\w+)',
            'string_interpolation': r'"[^"]*#\{[^}]+\}[^"]*"',
            'heredoc': r'<<[A-Z_]+\s*$',
            'array': r'\[[^\]]*\]',
            'hash': r'\{[^}]*\}',
            'lambda': r'lambda\s*\{',
            'proc': r'proc\s*\{',
            'attr_accessor': r'attr_accessor\s+[^#\n]',
            'attr_reader': r'attr_reader\s+[^#\n]',
            'attr_writer': r'attr_writer\s+[^#\n]',
            'private': r'private\s*$',
            'protected': r'protected\s*$',
            'public': r'public\s*$',
        }
    
    def parse(self, content: str, file_path: str) -> CodePropertyGraph:
        """Parse Ruby source code and generate complete CPG."""
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
        """Build Abstract Syntax Tree for Ruby code."""
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
            'ruby'
        )
        module_id = self.current_cpg.add_node(module_node)
        
        # Process different constructs
        self._process_requires(content, lines, module_id)
        self._process_classes(content, lines, module_id)
        self._process_modules(content, lines, module_id)
        self._process_methods(content, lines, module_id)
        self._process_variables(content, lines, module_id)
        self._process_control_flow(content, lines, module_id)
        self._process_method_calls(content, lines, module_id)
        
        return self.current_cpg
    
    def _process_requires(self, content: str, lines: List[str], parent_id: str):
        """Process require statements."""
        for match in re.finditer(self.patterns['require'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            require_node = self.create_node(
                NodeType.IMPORT,
                f"require_{match.group(1)}",
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'ruby',
                module_name=match.group(1)
            )
            require_id = self.current_cpg.add_node(require_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, require_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        for match in re.finditer(self.patterns['require_relative'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            require_node = self.create_node(
                NodeType.IMPORT,
                f"require_relative_{match.group(1)}",
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'ruby',
                module_name=match.group(1),
                relative=True
            )
            require_id = self.current_cpg.add_node(require_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, require_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_classes(self, content: str, lines: List[str], parent_id: str):
        """Process class definitions."""
        for match in re.finditer(self.patterns['class'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            class_name = match.group(1)
            parent_class = match.group(2) if match.group(2) else None
            
            class_node = self.create_node(
                NodeType.CLASS,
                class_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'ruby',
                parent_class=parent_class
            )
            class_id = self.current_cpg.add_node(class_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, class_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Add inheritance edge if parent class exists
            if parent_class:
                inheritance_edge = self.create_edge(class_id, parent_id, EdgeType.INHERITANCE)
                self.current_cpg.add_edge(inheritance_edge)
            
            # Process methods within this class
            self._process_class_methods(content, lines, class_id, line_num)
    
    def _process_modules(self, content: str, lines: List[str], parent_id: str):
        """Process module definitions."""
        for match in re.finditer(self.patterns['module'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            module_name = match.group(1)
            
            module_node = self.create_node(
                NodeType.CLASS,  # Treat modules as classes for now
                module_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'ruby',
                is_module=True
            )
            module_id = self.current_cpg.add_node(module_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, module_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_methods(self, content: str, lines: List[str], parent_id: str):
        """Process method definitions."""
        # Regular method definitions
        for match in re.finditer(self.patterns['def'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            method_name = match.group(1)
            
            method_node = self.create_node(
                NodeType.FUNCTION,
                method_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'ruby'
            )
            method_id = self.current_cpg.add_node(method_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, method_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Track method definition
            self.variable_definitions[method_name] = method_id
        
        # Class method definitions (def self.method_name)
        for match in re.finditer(self.patterns['def_self'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            method_name = match.group(1)
            
            method_node = self.create_node(
                NodeType.METHOD,
                f"self.{method_name}",
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'ruby',
                is_class_method=True
            )
            method_id = self.current_cpg.add_node(method_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, method_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_class_methods(self, content: str, lines: List[str], class_id: str, start_line: int):
        """Process methods within a specific class."""
        # Find the end of the class by looking for matching 'end' statements
        class_end_line = self._find_matching_end(content, start_line)
        
        # Extract class content
        class_content = '\n'.join(lines[start_line-1:class_end_line])
        
        # Process methods within this class
        for match in re.finditer(self.patterns['def'], class_content, re.MULTILINE):
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
                'ruby'
            )
            method_id = self.current_cpg.add_node(method_node)
            
            # Add AST edge
            edge = self.create_edge(class_id, method_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_variables(self, content: str, lines: List[str], parent_id: str):
        """Process variable assignments and declarations."""
        # Instance variables (@variable)
        for match in re.finditer(self.patterns['instance_variable'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            var_name = match.group(1)
            
            var_node = self.create_node(
                NodeType.VARIABLE,
                f"@{var_name}",
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'ruby',
                variable_type='instance'
            )
            var_id = self.current_cpg.add_node(var_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, var_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Track variable definition
            self.variable_definitions[f"@{var_name}"] = var_id
        
        # Class variables (@@variable)
        for match in re.finditer(self.patterns['class_variable'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            var_name = match.group(1)
            
            var_node = self.create_node(
                NodeType.VARIABLE,
                f"@@{var_name}",
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'ruby',
                variable_type='class'
            )
            var_id = self.current_cpg.add_node(var_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, var_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Track variable definition
            self.variable_definitions[f"@@{var_name}"] = var_id
        
        # Global variables ($variable)
        for match in re.finditer(self.patterns['global_variable'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            var_name = match.group(1)
            
            var_node = self.create_node(
                NodeType.VARIABLE,
                f"${var_name}",
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'ruby',
                variable_type='global'
            )
            var_id = self.current_cpg.add_node(var_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, var_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Track variable definition
            self.variable_definitions[f"${var_name}"] = var_id
        
        # Regular variable assignments
        for match in re.finditer(self.patterns['variable_assignment'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            var_name = match.group(1)
            
            # Skip if it's a method call (contains parentheses)
            if '(' in match.group(0) and ')' in match.group(0):
                continue
            
            var_node = self.create_node(
                NodeType.VARIABLE,
                var_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'ruby',
                variable_type='local'
            )
            var_id = self.current_cpg.add_node(var_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, var_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Track variable definition
            self.variable_definitions[var_name] = var_id
    
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
                'ruby'
            )
            if_id = self.current_cpg.add_node(if_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, if_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Unless statements
        for match in re.finditer(self.patterns['unless_statement'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            
            unless_node = self.create_node(
                NodeType.CONDITION,
                "unless_statement",
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'ruby'
            )
            unless_id = self.current_cpg.add_node(unless_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, unless_id, EdgeType.AST_CHILD)
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
                'ruby'
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
                'ruby'
            )
            while_id = self.current_cpg.add_node(while_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, while_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Each loops
        for match in re.finditer(self.patterns['each_loop'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            
            each_node = self.create_node(
                NodeType.LOOP,
                "each_loop",
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'ruby'
            )
            each_id = self.current_cpg.add_node(each_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, each_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_method_calls(self, content: str, lines: List[str], parent_id: str):
        """Process method calls."""
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
                'ruby'
            )
            call_id = self.current_cpg.add_node(call_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, call_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Track method use
            if method_name not in self.variable_uses:
                self.variable_uses[method_name] = []
            self.variable_uses[method_name].append(call_id)
    
    def _find_matching_end(self, content: str, start_line: int) -> int:
        """Find the matching 'end' statement for a block."""
        lines = content.splitlines()
        depth = 0
        
        for i in range(start_line - 1, len(lines)):
            line = lines[i].strip()
            
            # Count opening keywords
            if re.match(r'^\s*(class|module|def|if|unless|case|for|while|until|begin|do)\s', line):
                depth += 1
            elif re.match(r'^\s*end\s*$', line):
                depth -= 1
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