#!/usr/bin/env python3
"""
Streamlit Frontend for AI Documentation Generator
Upload code files or paste code to generate comprehensive documentation
"""

import streamlit as st
import os
import tempfile
import zipfile
import io
from pathlib import Path
import requests
import ast
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

# Set page config
st.set_page_config(
    page_title="AI Documentation Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class FileInfo:
    """Information about a code file"""
    path: str
    language: str
    content: str
    functions: List[dict]
    classes: List[dict]
    imports: List[str]

class LLMClient:
    """Client for interacting with Ollama LLM"""
    
    def __init__(self, model_name: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate text using the LLM"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "max_tokens": 2048
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            st.error(f"Error calling LLM: {e}")
            return ""
    
    def is_available(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            models = response.json().get("models", [])
            return any(model["name"].startswith(self.model_name.split(":")[0]) for model in models)
        except:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except:
            return []

class CodeAnalyzer:
    """Analyzes code files and extracts structural information"""
    
    SUPPORTED_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.cs': 'csharp',
        '.go': 'go',
        '.rs': 'rust',
        '.php': 'php',
        '.rb': 'ruby',
        '.kt': 'kotlin',
        '.swift': 'swift'
    }
    
    def analyze_content(self, content: str, filename: str) -> Optional[FileInfo]:
        """Analyze code content"""
        file_ext = Path(filename).suffix.lower()
        
        if file_ext not in self.SUPPORTED_EXTENSIONS:
            return None
        
        language = self.SUPPORTED_EXTENSIONS[file_ext]
        
        if language == 'python':
            return self._analyze_python(filename, content, language)
        else:
            return self._analyze_generic(filename, content, language)
    
    def _analyze_python(self, file_path: str, content: str, language: str) -> FileInfo:
        """Analyze Python code using AST"""
        functions = []
        classes = []
        imports = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        'name': node.name,
                        'line': node.lineno,
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node) or "",
                        'decorators': []
                    })
                
                elif isinstance(node, ast.ClassDef):
                    methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append({
                                'name': item.name,
                                'line': item.lineno,
                                'args': [arg.arg for arg in item.args.args],
                                'docstring': ast.get_docstring(item) or ""
                            })
                    
                    classes.append({
                        'name': node.name,
                        'line': node.lineno,
                        'docstring': ast.get_docstring(node) or "",
                        'methods': methods,
                        'bases': []
                    })
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    else:
                        module = node.module or ""
                        for alias in node.names:
                            imports.append(f"{module}.{alias.name}" if module else alias.name)
        
        except SyntaxError as e:
            st.warning(f"Syntax error in {file_path}: {e}")
        
        return FileInfo(file_path, language, content, functions, classes, imports)
    
    def _analyze_generic(self, file_path: str, content: str, language: str) -> FileInfo:
        """Analyze non-Python files using regex patterns"""
        functions = []
        classes = []
        imports = []
        
        # Function patterns for different languages
        func_patterns = {
            'javascript': r'(?:function\s+|const\s+|let\s+|var\s+)(\w+)\s*(?:\([^)]*\)|\s*=\s*(?:\([^)]*\)\s*=>|\([^)]*\)\s*{))',
            'typescript': r'(?:function\s+|const\s+|let\s+)(\w+)\s*(?:\([^)]*\)|\s*=\s*(?:\([^)]*\)\s*=>|\([^)]*\)\s*{))',
            'java': r'(?:public|private|protected|static|\s)*\s+\w+\s+(\w+)\s*\([^)]*\)\s*{',
            'cpp': r'(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*{',
            'c': r'(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*{',
            'go': r'func\s+(\w+)\s*\([^)]*\)',
            'rust': r'fn\s+(\w+)\s*\([^)]*\)',
            'php': r'function\s+(\w+)\s*\([^)]*\)',
            'ruby': r'def\s+(\w+)(?:\([^)]*\))?',
            'kotlin': r'fun\s+(\w+)\s*\([^)]*\)',
            'swift': r'func\s+(\w+)\s*\([^)]*\)'
        }
        
        lines = content.split('\n')
        
        # Extract functions
        if language in func_patterns:
            pattern = func_patterns[language]
            for i, line in enumerate(lines, 1):
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    functions.append({
                        'name': match,
                        'line': i,
                        'args': [],
                        'docstring': ""
                    })
        
        return FileInfo(file_path, language, content, functions, classes, imports)

class DocumentationGenerator:
    """Generates various types of documentation using LLM"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def generate_function_docstring(self, func_info: dict, file_info: FileInfo) -> str:
        """Generate docstring for a function"""
        system_prompt = f"""You are a documentation expert. Generate clear, concise docstrings for {file_info.language} functions.
Follow the standard documentation format for {file_info.language}.
Include: brief description, parameters, return value, and any important notes."""
        
        prompt = f"""Generate a docstring for this {file_info.language} function:

Function name: {func_info['name']}
Parameters: {', '.join(func_info.get('args', []))}
Line number: {func_info['line']}

Context from file: {file_info.path}

Provide only the docstring text, properly formatted for {file_info.language}."""
        
        return self.llm.generate(prompt, system_prompt)
    
    def generate_class_documentation(self, class_info: dict, file_info: FileInfo) -> str:
        """Generate documentation for a class"""
        system_prompt = f"""You are a documentation expert. Generate comprehensive class documentation for {file_info.language}.
Include: class purpose, key methods, usage examples, and important notes."""
        
        methods_info = "\n".join([f"- {m['name']}({', '.join(m.get('args', []))})" for m in class_info.get('methods', [])])
        
        prompt = f"""Generate documentation for this {file_info.language} class:

Class name: {class_info['name']}
Line number: {class_info['line']}
Methods:
{methods_info}

File: {file_info.path}

Provide comprehensive class documentation including purpose, key methods, and usage."""
        
        return self.llm.generate(prompt, system_prompt)
    
    def generate_code_documentation(self, file_info: FileInfo) -> str:
        """Generate complete documentation for a code file"""
        system_prompt = f"""You are a technical documentation expert. Generate comprehensive documentation for this {file_info.language} code file.
Include:
1. File overview and purpose
2. Key functions and classes
3. Usage examples
4. Dependencies and imports
5. Code structure explanation

Format in clean Markdown."""
        
        functions_summary = "\n".join([f"- {f['name']}() - Line {f['line']}" for f in file_info.functions])
        classes_summary = "\n".join([f"- {c['name']} - Line {c['line']}" for c in file_info.classes])
        imports_summary = "\n".join([f"- {imp}" for imp in file_info.imports[:10]])  # Limit imports
        
        prompt = f"""Generate comprehensive documentation for this {file_info.language} file:

**File:** {file_info.path}
**Language:** {file_info.language}

**Functions ({len(file_info.functions)}):**
{functions_summary or "None"}

**Classes ({len(file_info.classes)}):**
{classes_summary or "None"}

**Key Imports:**
{imports_summary or "None"}

**Code Preview:**
```{file_info.language}
{file_info.content[:1000]}{'...' if len(file_info.content) > 1000 else ''}
```

Generate detailed documentation explaining what this code does, how it works, and how to use it."""
        
        return self.llm.generate(prompt, system_prompt)
    
    def generate_readme(self, file_infos: List[FileInfo], project_name: str) -> str:
        """Generate README.md file for multiple files"""
        system_prompt = """You are creating a README.md file for a software project.
Create a comprehensive, professional README that includes:
1. Project title and description
2. Features and functionality
3. Installation instructions
4. Usage examples
5. File structure overview
6. API reference (if applicable)
7. Contributing guidelines

Format in Markdown with proper headers, code blocks, and formatting."""
        
        languages = list(set(fi.language for fi in file_infos))
        total_functions = sum(len(fi.functions) for fi in file_infos)
        total_classes = sum(len(fi.classes) for fi in file_infos)
        
        files_overview = []
        for fi in file_infos:
            files_overview.append(f"- **{fi.path}** ({fi.language}): {len(fi.functions)} functions, {len(fi.classes)} classes")
        
        files_summary = "\n".join(files_overview)
        
        prompt = f"""Generate a comprehensive README.md for this project:

**Project Name:** {project_name}
**Languages:** {', '.join(languages)}
**Total Files:** {len(file_infos)}
**Total Functions:** {total_functions}
**Total Classes:** {total_classes}

**Files Overview:**
{files_summary}

Create a professional README.md that clearly explains what this project does, how to install and use it, and provides good documentation for developers."""
        
        return self.llm.generate(prompt, system_prompt)

# Initialize session state
if 'file_infos' not in st.session_state:
    st.session_state.file_infos = []
if 'generated_docs' not in st.session_state:
    st.session_state.generated_docs = {}

def main():
    st.title("🚀 AI Documentation Generator")
    st.markdown("Generate comprehensive documentation from your code using AI")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        llm_client = LLMClient()
        available_models = llm_client.get_available_models()
        
        if available_models:
            selected_model = st.selectbox(
                "Select LLM Model",
                available_models,
                index=0 if available_models else 0,
                help="Choose the AI model for documentation generation"
            )
            llm_client.model_name = selected_model
        else:
            st.error("⚠️ No Ollama models found!")
            st.markdown("Please install Ollama and pull a model:")
            st.code("ollama pull llama3.1:8b")
            return
        
        # Connection status
        if llm_client.is_available():
            st.success(f"✅ Connected to {selected_model}")
        else:
            st.error(f"❌ Cannot connect to {selected_model}")
            st.markdown("Make sure Ollama is running: `ollama serve`")
        
        st.markdown("---")
        
        # Documentation options
        st.header("📄 Documentation Options")
        generate_readme = st.checkbox("Generate README.md", value=True)
        generate_individual = st.checkbox("Generate Individual File Docs", value=True)
        generate_functions = st.checkbox("Generate Function Documentation", value=True)
        generate_classes = st.checkbox("Generate Class Documentation", value=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📂 Input Code")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Upload Files", "Paste Code", "Upload ZIP"],
            horizontal=True
        )
        
        analyzer = CodeAnalyzer()
        
        if input_method == "Upload Files":
            uploaded_files = st.file_uploader(
                "Upload code files",
                accept_multiple_files=True,
                type=['py', 'js', 'ts', 'jsx', 'tsx', 'java', 'cpp', 'c', 'cs', 'go', 'rs', 'php', 'rb', 'kt', 'swift']
            )
            
            if uploaded_files:
                st.session_state.file_infos = []
                for uploaded_file in uploaded_files:
                    content = uploaded_file.read().decode('utf-8', errors='ignore')
                    file_info = analyzer.analyze_content(content, uploaded_file.name)
                    if file_info:
                        st.session_state.file_infos.append(file_info)
        
        elif input_method == "Paste Code":
            filename = st.text_input("Filename (with extension)", value="main.py")
            code_content = st.text_area(
                "Paste your code here",
                height=400,
                placeholder="def hello_world():\n    print('Hello, World!')"
            )
            
            if st.button("Analyze Code") and code_content.strip():
                file_info = analyzer.analyze_content(code_content, filename)
                if file_info:
                    st.session_state.file_infos = [file_info]
                else:
                    st.error("Unsupported file type")
        
        elif input_method == "Upload ZIP":
            zip_file = st.file_uploader("Upload ZIP file", type=['zip'])
            
            if zip_file:
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Extract ZIP
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Analyze all files
                    st.session_state.file_infos = []
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()
                                relative_path = os.path.relpath(file_path, temp_dir)
                                file_info = analyzer.analyze_content(content, relative_path)
                                if file_info:
                                    st.session_state.file_infos.append(file_info)
                            except Exception as e:
                                continue
        
        # Display analyzed files
        if st.session_state.file_infos:
            st.success(f"✅ Analyzed {len(st.session_state.file_infos)} files")
            
            with st.expander("📊 Analysis Summary"):
                for file_info in st.session_state.file_infos:
                    st.markdown(f"**{file_info.path}** ({file_info.language})")
                    st.markdown(f"- Functions: {len(file_info.functions)}")
                    st.markdown(f"- Classes: {len(file_info.classes)}")
                    st.markdown(f"- Imports: {len(file_info.imports)}")
                    st.markdown("---")
    
    with col2:
        st.header("📝 Generated Documentation")
        
        if st.session_state.file_infos and llm_client.is_available():
            generator = DocumentationGenerator(llm_client)
            
            if st.button("🚀 Generate Documentation"):
                with st.spinner("Generating documentation..."):
                    st.session_state.generated_docs = {}
                    
                    # Generate README
                    if generate_readme:
                        project_name = st.text_input("Project Name", value="My Project") or "My Project"
                        readme_content = generator.generate_readme(st.session_state.file_infos, project_name)
                        st.session_state.generated_docs['README'] = readme_content
                    
                    # Generate individual file documentation
                    if generate_individual:
                        for file_info in st.session_state.file_infos:
                            doc_content = generator.generate_code_documentation(file_info)
                            st.session_state.generated_docs[f'DOC_{file_info.path}'] = doc_content
                    
                    # Generate function documentation
                    if generate_functions:
                        for file_info in st.session_state.file_infos:
                            for func_info in file_info.functions:
                                func_doc = generator.generate_function_docstring(func_info, file_info)
                                key = f'FUNC_{file_info.path}_{func_info["name"]}'
                                st.session_state.generated_docs[key] = func_doc
                    
                    # Generate class documentation
                    if generate_classes:
                        for file_info in st.session_state.file_infos:
                            for class_info in file_info.classes:
                                class_doc = generator.generate_class_documentation(class_info, file_info)
                                key = f'CLASS_{file_info.path}_{class_info["name"]}'
                                st.session_state.generated_docs[key] = class_doc
                
                st.success("✅ Documentation generated successfully!")
        
        # Display generated documentation
        if st.session_state.generated_docs:
            doc_type = st.selectbox(
                "Select documentation to view:",
                list(st.session_state.generated_docs.keys())
            )
            
            if doc_type:
                st.markdown("### Generated Documentation")
                st.markdown(st.session_state.generated_docs[doc_type])
                
                # Download button
                st.download_button(
                    label=f"📥 Download {doc_type}",
                    data=st.session_state.generated_docs[doc_type],
                    file_name=f"{doc_type.lower().replace(' ', '_')}.md",
                    mime="text/markdown"
                )
        
        # Download all documentation as ZIP
        if st.session_state.generated_docs:
            if st.button("📦 Download All Documentation"):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for doc_name, doc_content in st.session_state.generated_docs.items():
                        filename = f"{doc_name.lower().replace(' ', '_')}.md"
                        zip_file.writestr(filename, doc_content)
                
                st.download_button(
                    label="📥 Download Documentation ZIP",
                    data=zip_buffer.getvalue(),
                    file_name="generated_documentation.zip",
                    mime="application/zip"
                )

if __name__ == "__main__":
    main()
