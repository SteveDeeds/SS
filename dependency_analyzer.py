#!/usr/bin/env python3
"""
Dependency Analyzer for Trading System

Recursively analyzes local dependencies for specified files to determine
what code is actually needed for the core functionality.
"""

import os
import ast
import re
from typing import Set, List, Dict
from pathlib import Path


class DependencyAnalyzer:
    """Analyzes Python file dependencies recursively"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.analyzed_files = set()
        self.dependencies = set()
        self.import_map = {}
        
    def analyze_file(self, file_path: str) -> Set[str]:
        """Analyze a single Python file for imports"""
        file_path = Path(file_path)
        
        # Convert to absolute path if relative
        if not file_path.is_absolute():
            file_path = self.project_root / file_path
            
        # Skip if already analyzed
        if str(file_path) in self.analyzed_files:
            return set()
            
        self.analyzed_files.add(str(file_path))
        
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            return set()
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST for imports
            tree = ast.parse(content, filename=str(file_path))
            imports = self._extract_imports_from_ast(tree)
            
            # Also look for sys.path.append imports (manual imports)
            manual_imports = self._extract_manual_imports(content)
            imports.update(manual_imports)
            
            print(f"📁 {file_path.relative_to(self.project_root)}")
            
            # Process each import
            local_deps = set()
            for imp in imports:
                if self._is_local_import(imp):
                    dep_path = self._resolve_import_path(imp, file_path)
                    if dep_path:
                        local_deps.add(str(dep_path))
                        print(f"   └── {dep_path.relative_to(self.project_root)}")
                        
            self.dependencies.update(local_deps)
            return local_deps
            
        except Exception as e:
            print(f"❌ Error analyzing {file_path}: {e}")
            return set()
    
    def _extract_imports_from_ast(self, tree: ast.AST) -> Set[str]:
        """Extract import statements from AST"""
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Handle "from module import name"
                    for alias in node.names:
                        if alias.name == '*':
                            imports.add(node.module)
                        else:
                            imports.add(f"{node.module}.{alias.name}")
                    # Also add the module itself
                    imports.add(node.module)
                    
        return imports
    
    def _extract_manual_imports(self, content: str) -> Set[str]:
        """Extract manual imports via sys.path manipulation"""
        imports = set()
        
        # Look for sys.path.append followed by from X import Y
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'sys.path.append' in line and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith('from ') and ' import ' in next_line:
                    # Extract module name
                    match = re.match(r'from\s+([^\s]+)\s+import', next_line)
                    if match:
                        imports.add(match.group(1))
                        
        return imports
    
    def _is_local_import(self, import_name: str) -> bool:
        """Check if import is local to the project"""
        # Skip standard library and third-party packages
        stdlib_modules = {
            'os', 'sys', 'datetime', 'typing', 'pathlib', 'json', 'csv', 
            'uuid', 'abc', 'asyncio', 'collections', 'functools', 'itertools',
            'math', 'random', 're', 'time', 'warnings', 'weakref'
        }
        
        third_party = {
            'numpy', 'pandas', 'matplotlib', 'yfinance', 'plotly', 'seaborn',
            'scipy', 'sklearn', 'requests', 'flask', 'django', 'pytest'
        }
        
        root_module = import_name.split('.')[0]
        
        return (root_module not in stdlib_modules and 
                root_module not in third_party and
                not root_module.startswith('_'))
    
    def _resolve_import_path(self, import_name: str, from_file: Path) -> Path:
        """Resolve import to actual file path"""
        # Handle relative imports
        if import_name.startswith('.'):
            # Relative import - resolve relative to current file's directory
            current_dir = from_file.parent
            parts = import_name.split('.')
            # Count leading dots
            level = 0
            for part in parts:
                if part == '':
                    level += 1
                else:
                    break
            
            # Go up 'level' directories
            target_dir = current_dir
            for _ in range(level - 1):
                target_dir = target_dir.parent
                
            # Add remaining path parts
            remaining_parts = [p for p in parts if p]
            for part in remaining_parts:
                target_dir = target_dir / part
                
            # Try to find the file
            candidates = [
                target_dir.with_suffix('.py'),
                target_dir / '__init__.py'
            ]
            
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            return None
        
        # Absolute import - search from project root and common paths
        search_paths = [
            self.project_root,
            self.project_root / 'src',
            from_file.parent
        ]
        
        parts = import_name.split('.')
        
        for search_path in search_paths:
            # Try as module path
            module_path = search_path
            for part in parts:
                module_path = module_path / part
                
            candidates = [
                module_path.with_suffix('.py'),
                module_path / '__init__.py'
            ]
            
            for candidate in candidates:
                if candidate.exists() and self._is_within_project(candidate):
                    return candidate
                    
        return None
    
    def _is_within_project(self, path: Path) -> bool:
        """Check if path is within the project directory"""
        try:
            path.relative_to(self.project_root)
            return True
        except ValueError:
            return False
    
    def analyze_recursive(self, initial_files: List[str]) -> Dict[str, Set[str]]:
        """Recursively analyze dependencies starting from initial files"""
        print("🔍 Starting recursive dependency analysis...")
        print("=" * 60)
        
        to_analyze = set(initial_files)
        analyzed = set()
        dependency_map = {}
        
        while to_analyze:
            current_file = to_analyze.pop()
            if current_file in analyzed:
                continue
                
            analyzed.add(current_file)
            deps = self.analyze_file(current_file)
            dependency_map[current_file] = deps
            
            # Add new dependencies to analyze
            for dep in deps:
                if dep not in analyzed:
                    to_analyze.add(dep)
                    
        return dependency_map
    
    def get_flat_dependency_list(self) -> List[str]:
        """Get a flat list of all dependencies"""
        return sorted(list(self.dependencies))
    
    def print_summary(self, initial_files: List[str]):
        """Print a summary of the analysis"""
        print("\n" + "=" * 60)
        print("📊 DEPENDENCY ANALYSIS SUMMARY")
        print("=" * 60)
        
        print(f"\n🎯 Target Files ({len(initial_files)}):")
        for file in initial_files:
            rel_path = Path(file).relative_to(self.project_root) if Path(file).is_absolute() else file
            print(f"   • {rel_path}")
        
        print(f"\n📁 Total Dependencies Found: {len(self.dependencies)}")
        
        flat_deps = self.get_flat_dependency_list()
        print(f"\n📋 Complete Dependency List:")
        for i, dep in enumerate(flat_deps, 1):
            rel_path = Path(dep).relative_to(self.project_root)
            print(f"   {i:2d}. {rel_path}")
            
        # Categorize by directory
        print(f"\n📂 Dependencies by Directory:")
        by_dir = {}
        for dep in flat_deps:
            rel_path = Path(dep).relative_to(self.project_root)
            dir_name = str(rel_path.parent) if rel_path.parent != Path('.') else 'root'
            if dir_name not in by_dir:
                by_dir[dir_name] = []
            by_dir[dir_name].append(rel_path.name)
            
        for dir_name, files in sorted(by_dir.items()):
            print(f"   📁 {dir_name}/ ({len(files)} files)")
            for file in sorted(files):
                print(f"      • {file}")


def main():
    """Main analysis function"""
    # Project root
    project_root = r"c:\Users\stdeeds\Documents\GitHub\SS"
    
    # Target files to analyze
    target_files = [
        "heatmap_visualizer.py",
        "mobile_server.py.py",
        "mobile_signals.py",
        "examples/generic_strategy_optimization.py",
        "examples/enhanced_strategy_visualizer.py",
        "strategies/adaptive_ma_crossover.py",
        "strategies/bollinger_bands_strategy.py",
        "strategies/rsi_strategy.py"
    ]
    
    print("🔬 Trading System Dependency Analyzer")
    print("=" * 60)
    print(f"Project Root: {project_root}")
    
    # Create analyzer
    analyzer = DependencyAnalyzer(project_root)
    
    # Perform analysis
    dependency_map = analyzer.analyze_recursive(target_files)
    
    # Print results
    analyzer.print_summary(target_files)
    
    # Export to file for reference
    output_file = Path(project_root) / "dependency_analysis_results.txt"
    with open(output_file, 'w') as f:
        f.write("Trading System Dependency Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Target Files:\n")
        for file in target_files:
            f.write(f"  • {file}\n")
        f.write("\n")
        
        flat_deps = analyzer.get_flat_dependency_list()
        f.write(f"Required Dependencies ({len(flat_deps)}):\n")
        for dep in flat_deps:
            rel_path = Path(dep).relative_to(project_root)
            f.write(f"  • {rel_path}\n")
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return analyzer.get_flat_dependency_list()


if __name__ == "__main__":
    main()
