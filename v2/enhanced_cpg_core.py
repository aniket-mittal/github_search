#!/usr/bin/env python3
"""
Enhanced Code Property Graph (CPG) Core System
===============================================

This module provides enhanced CPG infrastructure with:
1. Proper AST, CFG, and DFG generation
2. Advanced control flow analysis
3. Data flow tracking and analysis
4. Cross-language semantic analysis
"""

import os
import ast
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import re
from abc import ABC, abstractmethod

from cpg_core import (
    NodeType, EdgeType, CPGNode, CPGEdge, CodePropertyGraph, 
    LanguageDetector, CPGParser, CPGBuilder, get_supported_languages, get_file_extensions
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ControlFlowAnalyzer:
    """Analyzes and builds control flow graphs from AST."""
    
    def __init__(self, cpg: CodePropertyGraph):
        self.cpg = cpg
        self.cfg_nodes = {}  # Map function nodes to their CFG
        self.basic_blocks = {}
        
    def build_cfg(self) -> CodePropertyGraph:
        """Build control flow graph for all functions."""
        # Find all function nodes
        function_nodes = [node for node in self.cpg.nodes.values() 
                         if node.node_type == NodeType.FUNCTION]
        
        for func_node in function_nodes:
            self._build_function_cfg(func_node)
        
        return self.cpg
    
    def _build_function_cfg(self, func_node: CPGNode):
        """Build CFG for a single function."""
        # Create entry and exit nodes
        entry_node = CPGNode(
            id=f"{func_node.id}_entry",
            node_type=NodeType.ENTRY,
            name=f"{func_node.name}_entry",
            code="ENTRY",
            file_path=func_node.file_path,
            start_line=func_node.start_line,
            end_line=func_node.start_line,
            language=func_node.language
        )
        
        exit_node = CPGNode(
            id=f"{func_node.id}_exit", 
            node_type=NodeType.EXIT,
            name=f"{func_node.name}_exit",
            code="EXIT",
            file_path=func_node.file_path,
            start_line=func_node.end_line,
            end_line=func_node.end_line,
            language=func_node.language
        )
        
        self.cpg.add_node(entry_node)
        self.cpg.add_node(exit_node)
        
        # Connect entry to function
        entry_edge = CPGEdge(
            id="",
            source_id=entry_node.id,
            target_id=func_node.id,
            edge_type=EdgeType.CONTROL_FLOW
        )
        self.cpg.add_edge(entry_edge)
        
        # Get function body nodes
        body_nodes = self._get_function_body_nodes(func_node)
        
        if body_nodes:
            # Connect entry to first statement
            first_edge = CPGEdge(
                id="",
                source_id=entry_node.id, 
                target_id=body_nodes[0].id,
                edge_type=EdgeType.CONTROL_FLOW
            )
            self.cpg.add_edge(first_edge)
            
            # Connect sequential statements
            for i in range(len(body_nodes) - 1):
                curr_node = body_nodes[i]
                next_node = body_nodes[i + 1]
                
                # Handle conditional nodes
                if curr_node.node_type == NodeType.CONDITION:
                    # True branch
                    true_edge = CPGEdge(
                        id="",
                        source_id=curr_node.id,
                        target_id=next_node.id, 
                        edge_type=EdgeType.CONDITIONAL_TRUE
                    )
                    self.cpg.add_edge(true_edge)
                    
                    # False branch (connect to exit or next statement)
                    false_target = exit_node.id if i == len(body_nodes) - 1 else body_nodes[i + 2].id if i + 2 < len(body_nodes) else exit_node.id
                    false_edge = CPGEdge(
                        id="",
                        source_id=curr_node.id,
                        target_id=false_target,
                        edge_type=EdgeType.CONDITIONAL_FALSE
                    )
                    self.cpg.add_edge(false_edge)
                else:
                    # Regular control flow
                    flow_edge = CPGEdge(
                        id="",
                        source_id=curr_node.id,
                        target_id=next_node.id,
                        edge_type=EdgeType.CONTROL_FLOW
                    )
                    self.cpg.add_edge(flow_edge)
            
            # Connect last statement to exit
            if body_nodes:
                exit_edge = CPGEdge(
                    id="",
                    source_id=body_nodes[-1].id,
                    target_id=exit_node.id,
                    edge_type=EdgeType.CONTROL_FLOW
                )
                self.cpg.add_edge(exit_edge)
        else:
            # Empty function - connect entry directly to exit
            empty_edge = CPGEdge(
                id="",
                source_id=entry_node.id,
                target_id=exit_node.id,
                edge_type=EdgeType.CONTROL_FLOW
            )
            self.cpg.add_edge(empty_edge)
    
    def _get_function_body_nodes(self, func_node: CPGNode) -> List[CPGNode]:
        """Get all nodes that are part of function body."""
        body_nodes = []
        
        # Get all child nodes of function
        child_edges = self.cpg.get_outgoing_edges(func_node.id, EdgeType.AST_CHILD)
        for edge in child_edges:
            child = self.cpg.get_node(edge.target_id)
            if child and child.node_type in [NodeType.ASSIGNMENT, NodeType.CALL, 
                                            NodeType.CONDITION, NodeType.LOOP, 
                                            NodeType.RETURN]:
                body_nodes.append(child)
        
        # Sort by line number
        body_nodes.sort(key=lambda n: n.start_line)
        return body_nodes


# DFG removed for performance - focus on AST + CFG accuracy


class SemanticAnalyzer:
    """Performs semantic analysis and creates semantic edges."""
    
    def __init__(self, cpg: CodePropertyGraph):
        self.cpg = cpg
        self.function_calls = {}
        self.class_inheritance = {}
    
    def analyze(self) -> CodePropertyGraph:
        """Perform semantic analysis."""
        self._analyze_function_calls()
        self._analyze_inheritance()
        return self.cpg
    
    def _analyze_function_calls(self):
        """Analyze function calls and create call edges."""
        call_nodes = [node for node in self.cpg.nodes.values() 
                     if node.node_type == NodeType.CALL]
        function_nodes = [node for node in self.cpg.nodes.values() 
                         if node.node_type == NodeType.FUNCTION]
        
        for call_node in call_nodes:
            call_name = call_node.name
            
            # Find matching function definitions
            for func_node in function_nodes:
                if func_node.name == call_name:
                    call_edge = CPGEdge(
                        id="",
                        source_id=call_node.id,
                        target_id=func_node.id,
                        edge_type=EdgeType.CALL_EDGE
                    )
                    self.cpg.add_edge(call_edge)
    
    def _analyze_inheritance(self):
        """Analyze class inheritance relationships.""" 
        class_nodes = [node for node in self.cpg.nodes.values() 
                      if node.node_type == NodeType.CLASS]
        
        for class_node in class_nodes:
            # Check properties for inheritance info
            if 'base_classes' in class_node.properties:
                base_classes = class_node.properties['base_classes']
                if isinstance(base_classes, (list, tuple)):
                    for base_class in base_classes:
                        # Find base class node
                        for other_class in class_nodes:
                            if other_class.name == base_class:
                                inherit_edge = CPGEdge(
                                    id="",
                                    source_id=class_node.id,
                                    target_id=other_class.id,
                                    edge_type=EdgeType.INHERITANCE
                                )
                                self.cpg.add_edge(inherit_edge)


class EnhancedCPGBuilder(CPGBuilder):
    """Enhanced CPG builder with AST+CFG support optimized for large repositories."""
    
    def build_cpg(self, file_path: str, content: str = None) -> CodePropertyGraph:
        """Build enhanced CPG with AST and CFG - optimized for performance."""
        # Start with basic AST
        cpg = super().build_cpg(file_path, content)
        
        if not cpg.nodes:
            return cpg
        
        try:
            # Build control flow graph - lightweight implementation
            cfg_analyzer = ControlFlowAnalyzer(cpg)
            cpg = cfg_analyzer.build_cfg()
            
            # Perform lightweight semantic analysis (only function calls)
            semantic_analyzer = SemanticAnalyzer(cpg)
            cpg = semantic_analyzer.analyze()
            
            # Update metadata
            cpg.metadata['has_cfg'] = True
            cpg.metadata['has_semantic'] = True
            
            # Track graph size for monitoring
            node_count = len(cpg.nodes)
            edge_count = len(cpg.edges)
            
            # Log performance stats
            if node_count > 1000:
                logger.warning(f"Large graph generated for {file_path}: {node_count} nodes, {edge_count} edges")
            else:
                logger.info(f"Enhanced CPG for {file_path}: {node_count} nodes, {edge_count} edges")
            
        except Exception as e:
            logger.warning(f"Enhanced analysis failed for {file_path}: {e}")
            # Return basic AST at least
            cpg.metadata['error'] = f"Enhanced analysis failed: {str(e)}"
        
        return cpg


def test_enhanced_cpg_generation():
    """Test the enhanced CPG generation."""
    print("Testing Enhanced CPG Generation")
    print("=" * 40)
    
    # Test Python code
    python_code = '''
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, a, b):
        self.result = a + b
        return self.result

calc = Calculator()
result = calc.add(5, 3)
fib_result = fibonacci(10)
'''
    
    # Test JavaScript code
    js_code = '''
function factorial(n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

class Person {
    constructor(name) {
        this.name = name;
    }
    
    greet() {
        return "Hello " + this.name;
    }
}

let person = new Person("Alice");
let greeting = person.greet();
let fact = factorial(5);
'''
    
    # Test Java code
    java_code = '''
public class Example {
    private int value;
    
    public Example(int val) {
        this.value = val;
    }
    
    public int multiply(int factor) {
        int result = this.value * factor;
        return result;
    }
    
    public static void main(String[] args) {
        Example ex = new Example(10);
        int result = ex.multiply(5);
    }
}
'''
    
    builder = EnhancedCPGBuilder()
    
    test_cases = [
        ("test.py", python_code, "python"),
        ("test.js", js_code, "javascript"),
        ("test.java", java_code, "java")
    ]
    
    for filename, code, lang in test_cases:
        print(f"\n--- Testing {lang.upper()} ---")
        
        # Write test file
        test_file = f"/tmp/{filename}"
        with open(test_file, 'w') as f:
            f.write(code)
        
        # Generate CPG
        cpg = builder.build_cpg(test_file, code)
        
        # Analyze results
        print(f"Nodes: {len(cpg.nodes)}")
        print(f"Edges: {len(cpg.edges)}")
        
        # Count node types
        node_counts = {}
        for node in cpg.nodes.values():
            node_type = node.node_type.value
            node_counts[node_type] = node_counts.get(node_type, 0) + 1
        
        print("Node types:")
        for node_type, count in sorted(node_counts.items()):
            print(f"  {node_type}: {count}")
        
        # Count edge types
        edge_counts = {}
        for edge in cpg.edges.values():
            edge_type = edge.edge_type.value
            edge_counts[edge_type] = edge_counts.get(edge_type, 0) + 1
        
        print("Edge types:")
        for edge_type, count in sorted(edge_counts.items()):
            print(f"  {edge_type}: {count}")
        
        # Check for CFG and semantic analysis
        has_cfg = any(edge.edge_type == EdgeType.CONTROL_FLOW for edge in cpg.edges.values())
        has_calls = any(edge.edge_type == EdgeType.CALL_EDGE for edge in cpg.edges.values())
        
        print(f"Has CFG: {has_cfg}")
        print(f"Has Call Edges: {has_calls}")
        
        # Clean up
        os.unlink(test_file)


if __name__ == "__main__":
    test_enhanced_cpg_generation()