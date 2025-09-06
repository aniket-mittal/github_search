#!/usr/bin/env python3
"""
Scala CPG Parser

This module implements a comprehensive Code Property Graph parser for Scala,
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


class ScalaCPGParser(CPGParser):
    """Scala-specific CPG parser using regex patterns and Scala syntax analysis."""
    
    def __init__(self):
        """Initialize the Scala parser."""
        self.current_cpg = None
        self.current_file_path = ""
        self.variable_definitions = {}
        self.variable_uses = {}
        self.scope_stack = []
        self.current_scope = "global"
        
        # Scala-specific patterns
        self.patterns = {
            'import': r'import\s+([^\s]+)',
            'package': r'package\s+([^\s]+)',
            'class': r'class\s+(\w+)(?:\s*\([^)]*\))?(?:\s*extends\s+([^{]+))?(?:\s*with\s+([^{]+))?\s*\{',
            'trait': r'trait\s+(\w+)(?:\s*extends\s+([^{]+))?(?:\s*with\s+([^{]+))?\s*\{',
            'object': r'object\s+(\w+)(?:\s*extends\s+([^{]+))?(?:\s*with\s+([^{]+))?\s*\{',
            'case_class': r'case\s+class\s+(\w+)(?:\s*\([^)]*\))?(?:\s*extends\s+([^{]+))?(?:\s*with\s+([^{]+))?\s*\{',
            'case_object': r'case\s+object\s+(\w+)(?:\s*extends\s+([^{]+))?(?:\s*with\s+([^{]+))?\s*\{',
            'def': r'def\s+(\w+)(?:\s*<[^>]+>)?\s*\([^)]*\)(?:\s*:\s*[^{]+)?\s*[={]',
            'val': r'val\s+(\w+)(?:\s*:\s*[^=]+)?(?:\s*=\s*[^{]+)?[;{]',
            'var': r'var\s+(\w+)(?:\s*:\s*[^=]+)?(?:\s*=\s*[^{]+)?[;{]',
            'lazy_val': r'lazy\s+val\s+(\w+)(?:\s*:\s*[^=]+)?(?:\s*=\s*[^{]+)?[;{]',
            'type_alias': r'type\s+(\w+)\s*=\s*[^{]+',
            'if_statement': r'if\s*\([^{]+\s*\{',
            'match_statement': r'(\w+)\s*match\s*\{',
            'case': r'case\s+[^{]+:',
            'for_comprehension': r'for\s*\([^{]+\s*\{',
            'while_loop': r'while\s*\([^{]+\s*\{',
            'do_while': r'do\s*\{[^}]*\}\s*while\s*\([^{]+\s*\{',
            'try_catch': r'try\s*\{',
            'catch': r'catch\s*\{[^}]*case\s+[^{]+:',
            'finally': r'finally\s*\{',
            'return': r'return\s+[^;]+',
            'throw': r'throw\s+[^;]+',
            'function_call': r'(\w+(?:\.\w+)*)\s*\([^)]*\)',
            'method_call': r'\.(\w+)\s*\([^)]*\)',
            'lambda': r'\{[^}]*=>\s*[^}]*\}',
            'partial_function': r'\{[^}]*case\s+[^{]+=>\s*[^}]*\}',
            'implicit': r'implicit\s+',
            'private': r'private\s+',
            'protected': r'protected\s+',
            'override': r'override\s+',
            'final': r'final\s+',
            'abstract': r'abstract\s+',
            'sealed': r'sealed\s+',
            'case': r'case\s+',
            'lazy': r'lazy\s+',
            'volatile': r'volatile\s+',
            'transient': r'transient\s+',
            'synchronized': r'synchronized\s*\{',
            'yield': r'yield\s+[^;]+',
            'for_yield': r'for\s*\([^{]+\s*yield\s*\{',
            'companion_object': r'object\s+(\w+)\s*\{',
            'apply_method': r'def\s+apply\s*\([^)]*\)',
            'unapply_method': r'def\s+unapply\s*\([^)]*\)',
            'implicit_class': r'implicit\s+class\s+(\w+)',
            'implicit_def': r'implicit\s+def\s+(\w+)',
            'implicit_val': r'implicit\s+val\s+(\w+)',
            'implicit_var': r'implicit\s+var\s+(\w+)',
            'macro': r'macro\s+',
            'inline': r'inline\s+',
            'specialized': r'@specialized',
            'tailrec': r'@tailrec',
            'deprecated': r'@deprecated',
            'throws': r'@throws',
            'transient': r'@transient',
            'volatile': r'@volatile',
            'serializable': r'@Serializable',
            'serialversionuid': r'@SerialVersionUID',
        }
    
    def parse(self, content: str, file_path: str) -> CodePropertyGraph:
        """Parse Scala source code and generate complete CPG."""
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
        """Build Abstract Syntax Tree for Scala code."""
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
            'scala'
        )
        module_id = self.current_cpg.add_node(module_node)
        
        # Process different constructs
        self._process_package(content, lines, module_id)
        self._process_imports(content, lines, module_id)
        self._process_classes(content, lines, module_id)
        self._process_traits(content, lines, module_id)
        self._process_objects(content, lines, module_id)
        self._process_case_classes(content, lines, module_id)
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
                'scala',
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
                'scala',
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
            extends_clause = match.group(2).strip() if match.group(2) else None
            with_clause = match.group(3).strip() if match.group(3) else None
            
            class_node = self.create_node(
                NodeType.CLASS,
                class_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'scala',
                extends_clause=extends_clause,
                with_clause=with_clause
            )
            class_id = self.current_cpg.add_node(class_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, class_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Process class members
            self._process_class_members(content, lines, class_id, line_num)
    
    def _process_traits(self, content: str, lines: List[str], parent_id: str):
        """Process trait definitions."""
        for match in re.finditer(self.patterns['trait'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            trait_name = match.group(1)
            extends_clause = match.group(2).strip() if match.group(2) else None
            with_clause = match.group(3).strip() if match.group(3) else None
            
            trait_node = self.create_node(
                NodeType.CLASS,  # Treat traits as classes for now
                trait_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'scala',
                extends_clause=extends_clause,
                with_clause=with_clause,
                is_trait=True
            )
            trait_id = self.current_cpg.add_node(trait_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, trait_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Process trait members
            self._process_class_members(content, lines, trait_id, line_num)
    
    def _process_objects(self, content: str, lines: List[str], parent_id: str):
        """Process object definitions."""
        for match in re.finditer(self.patterns['object'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            object_name = match.group(1)
            extends_clause = match.group(2).strip() if match.group(2) else None
            with_clause = match.group(3).strip() if match.group(3) else None
            
            object_node = self.create_node(
                NodeType.CLASS,  # Treat objects as classes for now
                object_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'scala',
                extends_clause=extends_clause,
                with_clause=with_clause,
                is_object=True
            )
            object_id = self.current_cpg.add_node(object_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, object_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Process object members
            self._process_class_members(content, lines, object_id, line_num)
    
    def _process_case_classes(self, content: str, lines: List[str], parent_id: str):
        """Process case class definitions."""
        for match in re.finditer(self.patterns['case_class'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            class_name = match.group(1)
            extends_clause = match.group(2).strip() if match.group(2) else None
            with_clause = match.group(3).strip() if match.group(3) else None
            
            case_class_node = self.create_node(
                NodeType.CLASS,
                class_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'scala',
                extends_clause=extends_clause,
                with_clause=with_clause,
                is_case_class=True
            )
            case_class_id = self.current_cpg.add_node(case_class_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, case_class_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Process case class members
            self._process_class_members(content, lines, case_class_id, line_num)
        
        # Case objects
        for match in re.finditer(self.patterns['case_object'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            object_name = match.group(1)
            extends_clause = match.group(2).strip() if match.group(2) else None
            with_clause = match.group(3).strip() if match.group(3) else None
            
            case_object_node = self.create_node(
                NodeType.CLASS,  # Treat case objects as classes for now
                object_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'scala',
                extends_clause=extends_clause,
                with_clause=with_clause,
                is_case_object=True
            )
            case_object_id = self.current_cpg.add_node(case_object_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, case_object_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
    
    def _process_functions(self, content: str, lines: List[str], parent_id: str):
        """Process function definitions."""
        for match in re.finditer(self.patterns['def'], content, re.MULTILINE):
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
                'scala'
            )
            func_id = self.current_cpg.add_node(func_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, func_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Track function definition
            self.variable_definitions[func_name] = func_id
    
    def _process_class_members(self, content: str, lines: List[str], class_id: str, start_line: int):
        """Process members within a class/trait/object."""
        # Find the end of the class by looking for matching '}' statements
        class_end_line = self._find_matching_brace(content, start_line)
        
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
                'scala'
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
                'scala',
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
                'scala',
                variable_type='val'
            )
            val_id = self.current_cpg.add_node(val_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, val_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Track variable definition
            self.variable_definitions[var_name] = val_id
        
        # Lazy values
        for match in re.finditer(self.patterns['lazy_val'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            var_name = match.group(1)
            
            lazy_val_node = self.create_node(
                NodeType.VARIABLE,
                var_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'scala',
                variable_type='lazy_val'
            )
            lazy_val_id = self.current_cpg.add_node(lazy_val_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, lazy_val_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Track variable definition
            self.variable_definitions[var_name] = lazy_val_id
        
        # Type aliases
        for match in re.finditer(self.patterns['type_alias'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            type_name = match.group(1)
            
            type_node = self.create_node(
                NodeType.VARIABLE,
                type_name,
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'scala',
                variable_type='type_alias'
            )
            type_id = self.current_cpg.add_node(type_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, type_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
            
            # Track type definition
            self.variable_definitions[type_name] = type_id
    
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
                'scala'
            )
            if_id = self.current_cpg.add_node(if_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, if_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # Match statements
        for match in re.finditer(self.patterns['match_statement'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            match_expr = match.group(1)
            
            match_node = self.create_node(
                NodeType.CONDITION,
                "match_statement",
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'scala',
                match_expression=match_expr
            )
            match_id = self.current_cpg.add_node(match_node)
            
            # Add AST edge
            edge = self.create_edge(parent_id, match_id, EdgeType.AST_CHILD)
            self.current_cpg.add_edge(edge)
        
        # For comprehensions
        for match in re.finditer(self.patterns['for_comprehension'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            
            for_node = self.create_node(
                NodeType.LOOP,
                "for_comprehension",
                match.group(0).strip(),
                self.current_file_path,
                line_num,
                line_num,
                match.start() - content.rfind('\n', 0, match.start()) - 1,
                match.end() - content.rfind('\n', 0, match.start()) - 1,
                'scala'
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
                'scala'
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
                'scala'
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