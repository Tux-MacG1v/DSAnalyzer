import ast
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Set
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import traceback
import importlib.util
import sys
import networkx as nx
from collections import defaultdict

class BaseAnalyzer:
    """Base class for all analyzers"""
    def __init__(self):
        self.analysis_results = {}
        self.visualization_data = {}
        self.metrics = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"analysis_output_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = os.path.join(self.output_dir, f"{self.__class__.__name__}_analysis.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        self.log_info(f"Logging initialized for {self.__class__.__name__}")
        self.log_info(f"Log file: {log_file}")
    
    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(f"Info in : {message}")
    
    def log_error(self, error: Exception, context: str):
        """Log error message"""
        self.logger.error(f"Error in {context}: {str(error)}")
        self.logger.error(traceback.format_exc())
    
    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(f"Warning in : {message}")
    
    def log_debug(self, message: str):
        """Log debug message"""
        self.logger.debug(f"Debug in : {message}")
    
    def analyze(self, ast_tree: ast.AST) -> Dict[str, Any]:
        """Analyze the AST tree"""
        raise NotImplementedError
    
    def visualize(self):
        """Create visualizations"""
        raise NotImplementedError
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get analysis metrics"""
        return self.metrics
    
    def export_results(self, format: str = 'json'):
        """Export analysis results"""
        try:
            self.log_info("Exporting results in json format")
            output_file = os.path.join(self.output_dir, f"{self.__class__.__name__}_{self.timestamp}.json")
            
            if format == 'json':
                import json
                with open(output_file, 'w') as f:
                    json.dump(self.metrics, f, indent=2)
            
            self.log_info(f"Exported JSON results to {output_file}")
            
        except Exception as e:
            self.log_error(e, "export_results")

class TimeComplexityAnalyzer(BaseAnalyzer):
    """Analyzer for time complexity"""
    def __init__(self):
        super().__init__()
        self.function_complexities = {}
        self.loop_complexities = {}
        self.recursive_functions = []
        self.nested_loops = {}
        self.operation_counts = {}
        self.log_info("TimeComplexityAnalyzer initialized")
    
    def analyze(self, ast_tree: ast.AST) -> Dict[str, Any]:
        """Analyze time complexity of the code"""
        try:
            self.log_info("Starting time complexity analysis")
            
            # Analyze functions
            functions = [node for node in ast.walk(ast_tree) if isinstance(node, ast.FunctionDef)]
            self.log_info(f"Found {len(functions)} functions to analyze")
            
            for node in tqdm(functions, desc="Analyzing functions"):
                self._analyze_function(node)
            
            # Calculate metrics
            self.metrics = {
                'function_complexities': self.function_complexities,
                'loop_complexities': self.loop_complexities,
                'recursive_functions': self.recursive_functions,
                'nested_loops': self.nested_loops,
                'operation_counts': self.operation_counts
            }
            
            self.log_info(f"Analysis completed. Found {len(self.function_complexities)} function complexities")
            return self.metrics
            
        except Exception as e:
            self.log_error(e, "analyze")
            raise
    
    def _analyze_function(self, node: ast.FunctionDef):
        """Analyze time complexity of a function"""
        try:
            function_name = node.name
            self.log_debug(f"Analyzing function: {function_name}")
            
            # Initialize complexity tracking
            complexity = 'O(1)'  # Default complexity
            loop_count = 0
            max_nesting = 0
            current_nesting = 0
            
            # Analyze loops and nesting
            for child in ast.walk(node):
                if isinstance(child, (ast.For, ast.While)):
                    loop_count += 1
                    current_nesting += 1
                    max_nesting = max(max_nesting, current_nesting)
                elif isinstance(child, ast.FunctionDef):
                    current_nesting = 0
            
            # Update complexity based on loops
            if max_nesting > 1:
                complexity = f'O(n^{max_nesting})'
            elif loop_count > 0:
                complexity = 'O(n)'
            
            # Store results
            self.function_complexities[function_name] = complexity
            self.loop_complexities[function_name] = loop_count
            self.nested_loops[function_name] = max_nesting
            
            self.log_debug(f"Function {function_name} analyzed. Complexity: {complexity}, Loops: {loop_count}, Max nesting: {max_nesting}")
            
        except Exception as e:
            self.log_error(e, f"_analyze_function: {node.name}")
    
    def visualize(self):
        """Create visualizations of time complexity analysis"""
        try:
            self.log_info("Creating time complexity visualizations")
            
            if not self.function_complexities:
                self.log_warning("No complexity data to visualize")
                return
            
            # Create complexity distribution plot
            complexity_counts = pd.Series(self.function_complexities.values()).value_counts()
            fig1 = go.Figure(data=[
                go.Bar(x=complexity_counts.index, y=complexity_counts.values)
            ])
            fig1.update_layout(
                title='Function Time Complexity Distribution',
                xaxis_title='Complexity',
                yaxis_title='Count'
            )
            self.visualization_data['complexity_distribution'] = fig1
            
            # Create nested loops plot
            fig2 = go.Figure(data=[
                go.Bar(x=list(self.nested_loops.keys()), 
                      y=list(self.nested_loops.values()))
            ])
            fig2.update_layout(
                title='Maximum Loop Nesting Level by Function',
                xaxis_title='Function',
                yaxis_title='Nesting Level'
            )
            self.visualization_data['nested_loops'] = fig2
            
            self.log_info("Time complexity visualizations created")
            
        except Exception as e:
            self.log_error(e, "visualize")

class SpaceComplexityAnalyzer(BaseAnalyzer):
    """Analyzer for space complexity"""
    def __init__(self):
        super().__init__()
        self.function_space_complexities = {}
        self.variable_sizes = {}
        self.data_structure_usage = {}
        self.memory_estimates = {}
        self.log_info("SpaceComplexityAnalyzer initialized")
    
    def analyze(self, ast_tree: ast.AST) -> Dict[str, Any]:
        """Analyze space complexity of the code"""
        try:
            self.log_info("Starting space complexity analysis")
            
            # Analyze functions
            functions = [node for node in ast.walk(ast_tree) if isinstance(node, ast.FunctionDef)]
            self.log_info(f"Found {len(functions)} functions to analyze")
            
            for node in tqdm(functions, desc="Analyzing functions"):
                self._analyze_function(node)
            
            # Calculate metrics
            self.metrics = {
                'function_space_complexities': self.function_space_complexities,
                'variable_sizes': self.variable_sizes,
                'data_structure_usage': self.data_structure_usage,
                'memory_estimates': self.memory_estimates
            }
            
            self.log_info(f"Analysis completed. Found {len(self.function_space_complexities)} function complexities")
            self.log_info(f"Data structure usage: {self.data_structure_usage}")
            return self.metrics
            
        except Exception as e:
            self.log_error(e, "analyze")
            raise
    
    def _analyze_function(self, node: ast.FunctionDef):
        """Analyze space complexity of a function"""
        try:
            function_name = node.name
            self.log_debug(f"Analyzing function: {function_name}")
            
            complexity = 'O(1)'  # Default complexity
            max_data_structure_size = 0
            data_structures = []
            
            # Analyze variable declarations and data structures
            assignments = [child for child in ast.walk(node) if isinstance(child, ast.Assign)]
            self.log_debug(f"Found {len(assignments)} assignments in {function_name}")
            
            for child in assignments:
                size, ds_type = self._analyze_assignment(child)
                max_data_structure_size = max(max_data_structure_size, size)
                if ds_type:
                    data_structures.append(ds_type)
            
            # Analyze function arguments
            for arg in node.args.args:
                if arg.annotation:
                    if isinstance(arg.annotation, ast.Name):
                        if arg.annotation.id in ['list', 'dict', 'set']:
                            data_structures.append(arg.annotation.id)
                            max_data_structure_size = max(max_data_structure_size, 2)
            
            # Update complexity based on data structures
            if len(data_structures) > 0:
                if any(ds in ['list', 'dict', 'set'] for ds in data_structures):
                    complexity = 'O(n)'
                elif max_data_structure_size > 1:
                    complexity = 'O(n)'
            
            self.function_space_complexities[function_name] = complexity
            self.memory_estimates[function_name] = max_data_structure_size
            
            self.log_debug(f"Function {function_name} analyzed. Complexity: {complexity}, Max size: {max_data_structure_size}")
            self.log_debug(f"Data structures found: {data_structures}")
            
        except Exception as e:
            self.log_error(e, f"_analyze_function: {node.name}")
    
    def _analyze_assignment(self, node: ast.Assign) -> tuple:
        """Analyze the size of data structures in assignments"""
        try:
            size = 1  # Default size for simple variables
            ds_type = None
            
            if isinstance(node.value, ast.List):
                # Check for list comprehensions or large list literals
                size = max(size, len(node.value.elts))
                self.data_structure_usage['list'] = self.data_structure_usage.get('list', 0) + 1
                ds_type = 'list'
                self.log_debug(f"Found list with {len(node.value.elts)} elements")
                
            elif isinstance(node.value, ast.Dict):
                # Check for dictionary literals
                size = max(size, len(node.value.keys))
                self.data_structure_usage['dict'] = self.data_structure_usage.get('dict', 0) + 1
                ds_type = 'dict'
                self.log_debug(f"Found dict with {len(node.value.keys)} keys")
                
            elif isinstance(node.value, ast.Call):
                # Check for function calls that might create data structures
                if isinstance(node.value.func, ast.Name):
                    if node.value.func.id in ['list', 'dict', 'set']:
                        size = 2  # Assume these create data structures
                        self.data_structure_usage[node.value.func.id] = self.data_structure_usage.get(node.value.func.id, 0) + 1
                        ds_type = node.value.func.id
                        self.log_debug(f"Found {node.value.func.id} constructor call")
            
            # Store variable size information
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.variable_sizes[target.id] = size
                    self.log_debug(f"Variable {target.id} size: {size}")
            
            return size, ds_type
            
        except Exception as e:
            self.log_error(e, "_analyze_assignment")
            return 1, None
    
    def visualize(self):
        """Create visualizations of space complexity analysis"""
        try:
            self.log_info("Creating space complexity visualizations")
            
            if not self.function_space_complexities:
                self.log_warning("No complexity data to visualize")
                return
            
            # Create complexity distribution plot
            complexity_counts = pd.Series(self.function_space_complexities.values()).value_counts()
            fig1 = go.Figure(data=[
                go.Bar(x=complexity_counts.index, y=complexity_counts.values)
            ])
            fig1.update_layout(
                title='Function Space Complexity Distribution',
                xaxis_title='Complexity',
                yaxis_title='Count'
            )
            self.visualization_data['complexity_distribution'] = fig1
            
            # Create data structure usage plot
            if self.data_structure_usage:
                fig2 = go.Figure(data=[
                    go.Bar(x=list(self.data_structure_usage.keys()), 
                          y=list(self.data_structure_usage.values()))
                ])
                fig2.update_layout(
                    title='Data Structure Usage',
                    xaxis_title='Data Structure',
                    yaxis_title='Usage Count'
                )
                self.visualization_data['data_structure_usage'] = fig2
            
            # Create memory estimates plot
            fig3 = go.Figure(data=[
                go.Bar(x=list(self.memory_estimates.keys()), 
                      y=list(self.memory_estimates.values()))
            ])
            fig3.update_layout(
                title='Memory Usage Estimates by Function',
                xaxis_title='Function',
                yaxis_title='Estimated Size'
            )
            self.visualization_data['memory_estimates'] = fig3
            
            self.log_info("Space complexity visualizations created")
            
        except Exception as e:
            self.log_error(e, "visualize")

class RuntimeAnalyzer(BaseAnalyzer):
    """Analyzer for runtime performance"""
    def __init__(self):
        super().__init__()
        self.function_runtimes = {}
        self.call_counts = {}
        self.avg_runtimes = {}
        self.log_info("RuntimeAnalyzer initialized")
    
    def analyze(self, ast_tree: ast.AST) -> Dict[str, Any]:
        try:
            self.log_info("Starting runtime analysis")
            
            # Get all function definitions
            functions = [node for node in ast.walk(ast_tree) if isinstance(node, ast.FunctionDef)]
            self.log_info(f"Found {len(functions)} functions to analyze")
            
            # Analyze each function
            for node in tqdm(functions, desc="Analyzing functions"):
                self._analyze_function(node)
            
            # Calculate metrics
            self.metrics = {
                'function_runtimes': self.function_runtimes,
                'call_counts': self.call_counts,
                'avg_runtimes': self.avg_runtimes
            }
            
            self.log_info(f"Analysis completed. Found {len(self.function_runtimes)} function runtimes")
            return self.metrics
            
        except Exception as e:
            self.log_error(e, "analyze")
            raise
    
    def _analyze_function(self, node: ast.FunctionDef):
        try:
            function_name = node.name
            self.log_debug(f"Analyzing function: {function_name}")
            
            # Count function calls
            call_count = 0
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    call_count += 1
            
            # Estimate runtime based on operations
            operations = 0
            for child in ast.walk(node):
                if isinstance(child, (ast.For, ast.While)):
                    operations += 10  # Loop operations
                elif isinstance(child, (ast.If, ast.Compare)):
                    operations += 2   # Conditional operations
                elif isinstance(child, ast.Call):
                    operations += 5   # Function call overhead
            
            # Store results
            self.function_runtimes[function_name] = operations
            self.call_counts[function_name] = call_count
            self.avg_runtimes[function_name] = operations / max(1, call_count)
            
            self.log_debug(f"Function {function_name} analyzed. Operations: {operations}, Calls: {call_count}")
            
        except Exception as e:
            self.log_error(e, f"_analyze_function: {node.name}")
    
    def visualize(self):
        try:
            self.log_info("Creating runtime visualizations")
            
            if not self.function_runtimes:
                self.log_warning("No runtime data to visualize")
                return
            
            # Create runtime distribution plot
            fig1 = go.Figure(data=[
                go.Bar(x=list(self.function_runtimes.keys()), 
                      y=list(self.function_runtimes.values()))
            ])
            fig1.update_layout(
                title='Function Runtime Estimates',
                xaxis_title='Function',
                yaxis_title='Estimated Operations'
            )
            self.visualization_data['runtime_distribution'] = fig1
            
            # Create call count plot
            fig2 = go.Figure(data=[
                go.Bar(x=list(self.call_counts.keys()), 
                      y=list(self.call_counts.values()))
            ])
            fig2.update_layout(
                title='Function Call Counts',
                xaxis_title='Function',
                yaxis_title='Number of Calls'
            )
            self.visualization_data['call_counts'] = fig2
            
            self.log_info("Runtime visualizations created")
            
        except Exception as e:
            self.log_error(e, "visualize")

class FunctionTreeAnalyzer(BaseAnalyzer):
    """Analyzer for function call tree and structure"""
    def __init__(self):
        super().__init__()
        self.function_calls = defaultdict(set)
        self.function_tree = nx.DiGraph()
        self.log_info("FunctionTreeAnalyzer initialized")
    
    def analyze(self, ast_tree: ast.AST) -> Dict[str, Any]:
        try:
            self.log_info("Starting function tree analysis")
            
            # Get all function definitions
            functions = [node for node in ast.walk(ast_tree) if isinstance(node, ast.FunctionDef)]
            self.log_info(f"Found {len(functions)} functions to analyze")
            
            # Build function call graph
            for node in tqdm(functions, desc="Analyzing functions"):
                self._analyze_function(node)
            
            # Calculate metrics
            self.metrics = {
                'function_calls': {k: list(v) for k, v in self.function_calls.items()},
                'function_tree': nx.node_link_data(self.function_tree)
            }
            
            self.log_info(f"Analysis completed. Found {len(self.function_calls)} function relationships")
            return self.metrics
            
        except Exception as e:
            self.log_error(e, "analyze")
            raise
    
    def _analyze_function(self, node: ast.FunctionDef):
        try:
            function_name = node.name
            self.log_debug(f"Analyzing function: {function_name}")
            
            # Add function to graph
            self.function_tree.add_node(function_name)
            
            # Find function calls
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):
                        called_func = child.func.id
                        self.function_calls[function_name].add(called_func)
                        self.function_tree.add_edge(function_name, called_func)
            
            self.log_debug(f"Function {function_name} analyzed. Calls: {self.function_calls[function_name]}")
            
        except Exception as e:
            self.log_error(e, f"_analyze_function: {node.name}")
    
    def visualize(self):
        try:
            self.log_info("Creating function tree visualizations")
            if not self.function_tree.nodes():
                self.log_warning("No function tree data to visualize")
                return

            # Find root nodes (functions not called by any other function)
            called = set()
            for _, v in self.function_tree.edges():
                called.add(v)
            roots = [n for n in self.function_tree.nodes() if n not in called]
            if not roots:
                roots = list(self.function_tree.nodes())[:1]  # fallback

            # Assign levels to nodes for hierarchy
            levels = {}
            def assign_levels(node, level):
                if node in levels and levels[node] <= level:
                    return
                levels[node] = level
                for child in self.function_tree.successors(node):
                    assign_levels(child, level + 1)
            for root in roots:
                assign_levels(root, 0)

            # Group nodes by level
            level_nodes = {}
            for node, lvl in levels.items():
                level_nodes.setdefault(lvl, []).append(node)
            max_width = max(len(nodes) for nodes in level_nodes.values())
            max_level = max(level_nodes.keys())

            # Assign x/y positions for a tree layout
            pos = {}
            for lvl in range(max_level + 1):
                nodes = level_nodes.get(lvl, [])
                for i, node in enumerate(nodes):
                    x = (i + 1) * (1.0 / (len(nodes) + 1))
                    y = -lvl
                    pos[node] = (x, y)

            # Edges
            edge_x = []
            edge_y = []
            for src, dst in self.function_tree.edges():
                x0, y0 = pos[src]
                x1, y1 = pos[dst]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                mode='lines')

            # Nodes
            node_x = []
            node_y = []
            node_text = []
            for node in self.function_tree.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                calls = list(self.function_calls[node])
                node_text.append(f"{node}<br>Calls: {', '.join(calls) if calls else 'None'}")

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=[n for n in self.function_tree.nodes()],
                textposition="bottom center",
                marker=dict(
                    size=30,
                    color='skyblue',
                    line=dict(width=2, color='navy'),
                ),
                customdata=node_text,
                hovertemplate='%{customdata}<extra></extra>'
            )

            fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Function Call Tree (Hierarchical)',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=40,l=40,r=40,t=60),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=120 * (max_level + 2),
                )
            )
            self.visualization_data['function_tree'] = fig
            self.log_info("Function tree visualization created (hierarchical layout)")
        except Exception as e:
            self.log_error(e, "visualize")

class DSAnalyzer:
    """Main analyzer class that combines all analyzers"""
    def __init__(self):
        self.analyzers = {}
        self.ast_tree = None
        self.analysis_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"analysis_output_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_logger()
        self.log_info("DSAnalyzer initialized")
    
    def _setup_logger(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger("DSAnalyzer")
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = os.path.join(self.output_dir, "dsanalyzer.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        self.log_info(f"Log file: {log_file}")
    
    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def log_error(self, error: Exception, context: str):
        """Log error message"""
        self.logger.error(f"Error in {context}: {str(error)}")
        self.logger.error(traceback.format_exc())
    
    def log_debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def register_analyzer(self, analyzer: BaseAnalyzer):
        """Register a new analyzer"""
        try:
            analyzer_name = analyzer.__class__.__name__
            self.analyzers[analyzer_name] = analyzer
            self.log_info(f"Registered analyzer: {analyzer_name}")
        except Exception as e:
            self.log_error(e, f"register_analyzer: {analyzer_name}")
    
    def load_analyzers_from_directory(self, directory: str = '.'):
        """Load analyzer modules from a directory"""
        try:
            self.log_info(f"Loading analyzers from directory: {directory}")
            
            for filename in os.listdir(directory):
                if filename.endswith('.py') and not filename.startswith('__'):
                    module_name = filename[:-3]
                    if module_name not in ['DSAnalyzer', 'analyzer_base']:
                        try:
                            self.log_debug(f"Loading module: {module_name}")
                            spec = importlib.util.spec_from_file_location(module_name, os.path.join(directory, filename))
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            
                            # Find analyzer classes in the module
                            for attr_name in dir(module):
                                attr = getattr(module, attr_name)
                                if isinstance(attr, type) and issubclass(attr, BaseAnalyzer) and attr != BaseAnalyzer:
                                    analyzer = attr()
                                    self.register_analyzer(analyzer)
                                    
                        except Exception as e:
                            self.log_error(e, f"load_analyzers_from_directory: {module_name}")
            
        except Exception as e:
            self.log_error(e, "load_analyzers_from_directory")
    
    def analyze_file(self, file_path: str):
        """Analyze a Python file"""
        try:
            self.log_info(f"Starting analysis of file: {file_path}")
            
            # Read and parse the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.log_debug(f"File content length: {len(content)} characters")
            
            self.ast_tree = ast.parse(content)
            self.log_debug("AST tree parsed successfully")
            
            # Run analyzers in parallel
            def run_analyzer(analyzer_name, analyzer):
                try:
                    results = analyzer.analyze(self.ast_tree)
                    analyzer.visualize()
                    return analyzer_name, results
                except Exception as e:
                    self.log_error(e, f"run_analyzer: {analyzer_name}")
                    return analyzer_name, None
            
            # Run analyzers
            for analyzer_name, analyzer in self.analyzers.items():
                _, results = run_analyzer(analyzer_name, analyzer)
                if results:
                    self.analysis_results[analyzer_name] = results
                    self.log_info(f"Completed analysis with {analyzer_name}")
            
            # Print summary
            self.print_analysis_summary()
            
            # Create combined dashboard
            self.create_combined_dashboard()
            
            # Export results
            self.export_all_results()
            
        except Exception as e:
            self.log_error(e, "analyze_file")
    
    def print_analysis_summary(self):
        """Print a summary of all analysis results"""
        try:
            self.log_info("Printing analysis summary")
            print("\n=== DSAnalyzer Summary Report ===\n")
            
            for analyzer_name, results in self.analysis_results.items():
                print(f"\n{analyzer_name} Analysis:")
                print("Metrics:")
                for metric_name, metric_value in results.items():
                    print(f"  - {metric_name}: {metric_value}")
                
                analyzer = self.analyzers[analyzer_name]
                if analyzer.visualization_data:
                    print("\nVisualizations available:")
                    for viz_name in analyzer.visualization_data.keys():
                        print(f"  - {viz_name}")
                print()
            
        except Exception as e:
            self.log_error(e, "print_analysis_summary")
    
    def create_combined_dashboard(self):
        try:
            self.log_info("Creating combined dashboard")
            all_figs = []
            subplot_titles = []
            for analyzer_name, analyzer in self.analyzers.items():
                for viz_name, fig in analyzer.visualization_data.items():
                    all_figs.append(fig)
                    subplot_titles.append(f"{analyzer_name}: {viz_name}")

            # --- Big-O Reference Table ---
            bigo_table_header = dict(values=[
                "Algorithm", "Best Time", "Average Time", "Worst Time", "Worst Space"
            ], fill_color='yellow', align='left')
            bigo_table_cells = dict(values=[
                ["Linear Search", "Binary Search", "Bubble Sort", "Selection Sort", "Insertion Sort", "Merge Sort", "Quick Sort", "Heap Sort", "Bucket Sort", "Radix Sort", "Tim Sort", "Shell Sort"],
                ["O(1)", "O(1)", "O(n)", "O(n^2)", "O(n)", "O(nlogn)", "O(nlogn)", "O(nlogn)", "O(n+k)", "O(nk)", "O(n)", "O(n)",],
                ["O(n)", "O(log n)", "O(n^2)", "O(n^2)", "O(n^2)", "O(nlogn)", "O(nlogn)", "O(nlogn)", "O(n+k)", "O(nk)", "O(nlogn)", "O((nlog(n))^2)"],
                ["O(n)", "O(log n)", "O(n^2)", "O(n^2)", "O(n^2)", "O(nlogn)", "O(n^2)", "O(nlogn)", "O(n^2)", "O(nk)", "O(nlogn)", "O((nlog(n))^2)"],
                ["O(1)", "O(1)", "O(1)", "O(1)", "O(1)", "O(n)", "O(n)", "O(log n)", "O(n)", "O(n+k)", "O(n)", "O(1)"]
            ], align='left')
            bigo_table = go.Figure(data=[go.Table(header=bigo_table_header, cells=bigo_table_cells)])
            bigo_table.update_layout(title="Big-O Reference Table")
            all_figs.insert(0, bigo_table)
            subplot_titles.insert(0, "Big-O Reference Table")

            # --- Big-O Growth Curve Plot ---
            n = np.arange(1, 101)
            curves = {
                'O(1)': np.ones_like(n),
                'O(log n)': np.log2(n),
                'O(n)': n,
                'O(n log n)': n * np.log2(n),
                'O(n^2)': n ** 2,
                'O(2^n)': 2 ** n,
                'O(n!)': [np.math.factorial(i) for i in n]
            }
            bigo_curve_fig = go.Figure()
            colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            for i, (label, y) in enumerate(curves.items()):
                # Limit y for O(2^n) and O(n!) to avoid overflow
                y = np.array(y)
                y = np.where(y > 1000, np.nan, y)
                bigo_curve_fig.add_trace(go.Scatter(x=n, y=y, mode='lines', name=label, line=dict(color=colors[i % len(colors)])))
            bigo_curve_fig.update_layout(title="Big-O Complexity Growth Curves", xaxis_title="Elements", yaxis_title="Operations", legend_title="Complexity")
            all_figs.insert(1, bigo_curve_fig)
            subplot_titles.insert(1, "Big-O Growth Curves")

            if all_figs:
                combined_fig = make_subplots(rows=len(all_figs), cols=1, subplot_titles=subplot_titles)
                for i, fig in enumerate(all_figs, 1):
                    for trace in fig.data:
                        combined_fig.add_trace(trace, row=i, col=1)
                combined_fig.update_layout(height=350 * len(all_figs), title_text="Combined Analysis Dashboard", showlegend=True)
                output_file = os.path.join(self.output_dir, f"combined_dashboard_{self.timestamp}.html")
                combined_fig.write_html(output_file)
                self.log_info(f"Combined dashboard saved to {output_file}")
        except Exception as e:
            self.log_error(e, "create_combined_dashboard")
    
    def export_all_results(self):
        """Export results from all analyzers"""
        try:
            self.log_info("Exporting all results in json format")
            for analyzer in self.analyzers.values():
                analyzer.export_results()
        except Exception as e:
            self.log_error(e, "export_all_results")

def main():
    """Main function to run the analyzer"""
    if len(sys.argv) != 2:
        print("Usage: python DSAnalyzer.py <python_file>")
        sys.exit(1)
    
    # Create and configure the analyzer
    analyzer = DSAnalyzer()
    
    # Register built-in analyzers
    analyzer.register_analyzer(TimeComplexityAnalyzer())
    analyzer.register_analyzer(SpaceComplexityAnalyzer())
    analyzer.register_analyzer(RuntimeAnalyzer())
    analyzer.register_analyzer(FunctionTreeAnalyzer())
    
    # Load additional analyzers from the current directory
    analyzer.load_analyzers_from_directory()
    
    # Analyze the file
    analyzer.analyze_file(sys.argv[1])

if __name__ == "__main__":
    main()
