"""
Objective-C CPG Parser

Specialized parser for Objective-C language constructs.
Conforms to the core CPGParser API.
"""

import re
from typing import List, Optional

from cpg_core import CPGParser, CodePropertyGraph, NodeType, EdgeType


class ObjectiveCCPGParser(CPGParser):
    """Objective-C specific CPG parser."""
    
    def __init__(self):
        self.name = "objective-c"
        
        # Objective-C specific patterns
        self.patterns = {
            'interface': re.compile(r'@interface\s+(\w+)(?:\s*:\s*(\w+))?(?:\s*<([^>]+)>)?', re.MULTILINE),
            'implementation': re.compile(r'@implementation\s+(\w+)', re.MULTILINE),
            'protocol': re.compile(r'@protocol\s+(\w+)(?:\s*<([^>]+)>)?', re.MULTILINE),
            'category': re.compile(r'@interface\s+(\w+)\s*\((\w+)\)', re.MULTILINE),
            'method_decl': re.compile(r'[+-]\s*\([^)]+\)\s*(\w+)(?:\s*:\s*\([^)]+\)\s*(\w+))?', re.MULTILINE),
            'method_impl': re.compile(r'[+-]\s*\([^)]+\)\s*(\w+)', re.MULTILINE),
            'property': re.compile(r'@property\s*(?:\([^)]+\))?\s*([\w\*]+)\s*(\w+)', re.MULTILINE),
            'ivar_ptr': re.compile(r'(\w+)\s*\*\s*(\w+)\s*[=;]', re.MULTILINE),
            'ivar': re.compile(r'(\w+)\s+(\w+)\s*[=;]', re.MULTILINE),
            'if': re.compile(r'\bif\s*\(([^)]+)\)', re.MULTILINE),
            'for': re.compile(r'\bfor\s*\(([^;]+);([^;]+);([^)]+)\)', re.MULTILINE),
            'while': re.compile(r'\bwhile\s*\(([^)]+)\)', re.MULTILINE),
            'switch': re.compile(r'\bswitch\s*\(([^)]+)\)', re.MULTILINE),
            'ret': re.compile(r'\breturn\s+([^;]+);', re.MULTILINE),
            'objc_msg': re.compile(r'\[([^\]]+)\s+(\w+)(?::\s*[^\]]+)?\]', re.MULTILINE),
            'c_call': re.compile(r'\b(\w+)\s*\(([^)]*)\)', re.MULTILINE),
            'import': re.compile(r'#import\s*[<"]([^>"]+)[>"]', re.MULTILINE),
            'include': re.compile(r'#include\s*[<"]([^>"]+)[>"]', re.MULTILINE),
        }

    def parse(self, content: str, file_path: str) -> CodePropertyGraph:
        cpg = CodePropertyGraph()

        # Module node
        lines = content.splitlines()
        module_node = self.create_node(
            NodeType.MODULE,
            name=file_path.split('/')[-1],
            code=content,
            file_path=file_path,
            start_line=1,
            end_line=len(lines),
            language='objective-c',
        )
        module_id = cpg.add_node(module_node)

        # Imports
        for pat in ['import', 'include']:
            for m in self.patterns[pat].finditer(content):
                start_line = content[:m.start()].count('\n') + 1
                imp_node = self.create_node(
                    NodeType.IMPORT,
                    name=m.group(1),
                    code=m.group(0),
                    file_path=file_path,
                    start_line=start_line,
                    end_line=start_line,
                    language='objective-c',
                    import_type=pat,
                )
                imp_id = cpg.add_node(imp_node)
                cpg.add_edge(self.create_edge(module_id, imp_id, EdgeType.AST_CHILD))

        # Interfaces / classes
        class_name_to_id = {}
        for m in self.patterns['interface'].finditer(content):
            class_name = m.group(1)
            parent_class = m.group(2)
            protocols = [p.strip() for p in (m.group(3) or '').split(',') if p and p.strip()]
            start_line = content[:m.start()].count('\n') + 1
            cls_node = self.create_node(
                NodeType.CLASS,
                name=class_name,
                code=m.group(0),
                file_path=file_path,
                start_line=start_line,
                end_line=start_line,
                language='objective-c',
                is_interface=True,
                protocols=protocols,
            )
            cls_id = cpg.add_node(cls_node)
            class_name_to_id[class_name] = cls_id
            cpg.add_edge(self.create_edge(module_id, cls_id, EdgeType.AST_CHILD))

            if parent_class:
                # Relationship edge
                cpg.add_edge(self.create_edge(cls_id, cls_id, EdgeType.TYPE_RELATION, parent=parent_class))

        # Implementations
        for m in self.patterns['implementation'].finditer(content):
            class_name = m.group(1)
            start_line = content[:m.start()].count('\n') + 1
            impl_node = self.create_node(
                NodeType.CLASS,
                name=f"{class_name}__impl",
                code=m.group(0),
                file_path=file_path,
                start_line=start_line,
                end_line=start_line,
                language='objective-c',
                is_implementation=True,
            )
            impl_id = cpg.add_node(impl_node)
            cpg.add_edge(self.create_edge(module_id, impl_id, EdgeType.AST_CHILD))

        # Properties and ivars
        for m in self.patterns['property'].finditer(content):
            start_line = content[:m.start()].count('\n') + 1
            prop_node = self.create_node(
                NodeType.VARIABLE,
                name=m.group(2),
                code=m.group(0),
                file_path=file_path,
                start_line=start_line,
                end_line=start_line,
                language='objective-c',
                var_type=m.group(1),
                is_property=True,
            )
            prop_id = cpg.add_node(prop_node)
            cpg.add_edge(self.create_edge(module_id, prop_id, EdgeType.AST_CHILD))

        for pat in ['ivar_ptr', 'ivar']:
            for m in self.patterns[pat].finditer(content):
                start_line = content[:m.start()].count('\n') + 1
                var_node = self.create_node(
                    NodeType.VARIABLE,
                    name=m.group(2),
                    code=m.group(0),
                file_path=file_path,
                    start_line=start_line,
                    end_line=start_line,
                    language='objective-c',
                    var_type=m.group(1),
                )
                var_id = cpg.add_node(var_node)
                cpg.add_edge(self.create_edge(module_id, var_id, EdgeType.AST_CHILD))

        # Methods (declarations and implementations)
        for pat in ['method_decl', 'method_impl']:
            for m in self.patterns[pat].finditer(content):
                method_name = m.group(1)
                start_line = content[:m.start()].count('\n') + 1
                meth_node = self.create_node(
                    NodeType.METHOD,
                    name=method_name,
                    code=m.group(0),
                    file_path=file_path,
                    start_line=start_line,
                    end_line=start_line,
                    language='objective-c',
                    is_declaration=(pat == 'method_decl'),
                )
                meth_id = cpg.add_node(meth_node)
                cpg.add_edge(self.create_edge(module_id, meth_id, EdgeType.AST_CHILD))

        # Control flow and calls (coarse)
        for m in self.patterns['if'].finditer(content):
            start_line = content[:m.start()].count('\n') + 1
            cond_node = self.create_node(
                NodeType.CONDITION,
                name='if',
                code=m.group(0),
                file_path=file_path,
                start_line=start_line,
                end_line=start_line,
                language='objective-c',
            )
            cond_id = cpg.add_node(cond_node)
            cpg.add_edge(self.create_edge(module_id, cond_id, EdgeType.AST_CHILD))

        for pat in ['for', 'while']:
            for m in self.patterns[pat].finditer(content):
                start_line = content[:m.start()].count('\n') + 1
                loop_node = self.create_node(
                    NodeType.LOOP,
                    name=pat,
                    code=m.group(0),
                file_path=file_path,
                    start_line=start_line,
                    end_line=start_line,
                    language='objective-c',
                )
                loop_id = cpg.add_node(loop_node)
                cpg.add_edge(self.create_edge(module_id, loop_id, EdgeType.AST_CHILD))

        for m in self.patterns['ret'].finditer(content):
            start_line = content[:m.start()].count('\n') + 1
            ret_node = self.create_node(
                NodeType.RETURN,
                name='return',
                code=m.group(0),
                file_path=file_path,
                start_line=start_line,
                end_line=start_line,
                language='objective-c',
            )
            ret_id = cpg.add_node(ret_node)
            cpg.add_edge(self.create_edge(module_id, ret_id, EdgeType.AST_CHILD))

        for pat in ['objc_msg', 'c_call']:
            for m in self.patterns[pat].finditer(content):
                start_line = content[:m.start()].count('\n') + 1
                name = m.group(2) if pat == 'objc_msg' else m.group(1)
                call_node = self.create_node(
                    NodeType.CALL,
                    name=name,
                    code=m.group(0),
                file_path=file_path,
                    start_line=start_line,
                    end_line=start_line,
                    language='objective-c',
                )
                call_id = cpg.add_node(call_node)
                cpg.add_edge(self.create_edge(module_id, call_id, EdgeType.AST_CHILD))

        # Build minimal CFG within module scope (coarse ordering)
        child_nodes = cpg.get_children(module_id)
        if child_nodes:
            entry = self.create_node(NodeType.ENTRY, name='entry', file_path=file_path, language='objective-c')
            entry_id = cpg.add_node(entry)
            cpg.add_edge(self.create_edge(module_id, entry_id, EdgeType.AST_CHILD))
            cpg.add_edge(self.create_edge(entry_id, child_nodes[0].id, EdgeType.CONTROL_FLOW))
            for i in range(len(child_nodes) - 1):
                cpg.add_edge(self.create_edge(child_nodes[i].id, child_nodes[i + 1].id, EdgeType.CONTROL_FLOW))
            exit_node = self.create_node(NodeType.EXIT, name='exit', file_path=file_path, language='objective-c')
            exit_id = cpg.add_node(exit_node)
            cpg.add_edge(self.create_edge(module_id, exit_id, EdgeType.AST_CHILD))
            cpg.add_edge(self.create_edge(child_nodes[-1].id, exit_id, EdgeType.CONTROL_FLOW))

        # DFG is non-trivial for ObjC; skip for now (would require real parsing)
        return cpg

    # Abstract method implementations (minimal to satisfy interface)
    def build_ast(self, content: str, file_path: str) -> CodePropertyGraph:
        # Use the same coarse parsing as parse but without CFG/DFG wiring
        return self.parse(content, file_path)

    def build_cfg(self, cpg: CodePropertyGraph) -> CodePropertyGraph:
        # No-op: parse already wires a simple CFG at module scope
        return cpg

    def build_dfg(self, cpg: CodePropertyGraph) -> CodePropertyGraph:
        """Build Data Flow Graph from AST and CFG - DISABLED for performance."""
        # DFG generation disabled for faster processing
        # Only AST and CFG are generated
        return cpg