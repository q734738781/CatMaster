# CatMaster LLM-Driven Intelligent Catalysis Agent Engine

## Overview

CatMaster is an intelligent catalysis computation system based on Large Language Models (LLMs), implementing automated computational task planning, execution, and analysis through four collaborative agents.

## System Architecture

### Four Core Agents

1. **Planning Agent**
   - Receives natural language requests from users
   - Decomposes into specific computational steps
   - Selects appropriate computational methods and resources

2. **Task Orchestrator Agent**
   - Manages task execution order
   - Handles data dependencies between tasks
   - Coordinates resource allocation

3. **Task Execution Agent**  
   - Invokes specific computational tools
   - Executes tasks on different devices (local/GPU/CPU)
   - Handles file transfers and result collection

4. **Task Summary Agent**
   - Analyzes computational results
   - Generates easy-to-understand reports
   - Provides scientific insights and recommendations

### Technology Stack

- **LLM Framework**: LangChain + LangGraph
- **Remote Execution**: jobflow-remote
- **Computational Engines**: VASP (DFT), MACE (Machine Learning Force Field)
- **File Transfer**: Paramiko (SSH/SFTP)

## Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements_llm.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key"
```

### 2. Configure jobflow-remote

Ensure `~/.jfremote/catmaster.yaml` is properly configured for CPU and GPU workers.

### 3. Run O2 Molecule Calculation Demo

```bash
# Basic usage
python demo_o2_calculation.py

# Use GPT-4 model
python demo_o2_calculation.py --model gpt-4

# Custom request
python demo_o2_calculation.py --request "Calculate the energy and structure of O2 molecule"

# Show detailed logs
python demo_o2_calculation.py --verbose
```

## Execution Flow

1. **User Input**: "Please calculate the optimized structure and accurate energy of O2 molecule"

2. **Planning Phase**: 
   - Create initial O2 molecule structure
   - MACE pre-optimization (GPU)
   - VASP input preparation
   - VASP accurate calculation (CPU)

3. **Execution Phase**:
   - Create O2 molecule locally (bond length 1.21Å)
   - Run MACE optimization on GPU
   - Prepare VASP input files
   - Run VASP calculation on CPU cluster

4. **Result Analysis**:
   - Extract final energy
   - Calculate O-O bond length
   - Compare with experimental values
   - Generate report

## Output Files

- `o2_calculation_report_*.txt`: Human-readable computation report
- `o2_calculation_results_*.json`: Detailed results in JSON format
- `catmaster_workspace/`: Working directory containing all intermediate files
- `catmaster_demo_*.log`: Detailed execution logs

## Extended Usage

### Custom Computation Tasks

```python
from catmaster.agents.workflow import create_workflow
from langchain_openai import ChatOpenAI

# Create workflow
llm = ChatOpenAI(model="gpt-4")
workflow = create_workflow(llm)

# Execute computation
summary = workflow.run("Calculate vibrational frequencies of H2O molecule")
```

### Adding New Computational Methods

1. Implement new tools in `catmaster/tools/`
2. Add invocation logic in `ExecutionAgent`
3. Update `PlanningAgent` prompts

## Notes

1. **Resource Requirements**:
   - GPU worker requires MACE and PyTorch installation
   - CPU worker requires VASP license and binaries
   
2. **Network Requirements**:
   - SSH connectivity from local to remote servers
   - MongoDB cloud database access

3. **File Synchronization**:
   - Use rsync to sync catmaster code to remote
   - Ensure PYTHONPATH is correctly set

## Troubleshooting

1. **SSH Connection Failed**:
   ```bash
   # Test SSH connections
   ssh cpu-worker  # Should connect successfully
   ssh gpu-worker
   ```

2. **jobflow-remote Issues**:
   ```bash
   # Check worker status
   jf project check
   ```

3. **VASP Execution Failed**:
   - Check VASP_STD_BIN environment variable
   - Confirm VASP license is valid

## Development Guide

### Adding a New Agent

```python
from catmaster.agents.base import BaseAgent

class CustomAgent(BaseAgent):
    def process(self, input_data):
        # Implement custom logic
        pass
```

### Modifying Workflow

Edit the `_build_workflow` method in `catmaster/agents/workflow.py`.

## Performance Optimization

1. **Parallel Execution**: Workflow supports parallel task execution
2. **Caching Mechanism**: Repeated calculations utilize existing results
3. **Resource Scheduling**: Automatically selects appropriate workers

## Future Plans

- [ ] Support more molecular systems
- [ ] Integrate more computational methods (Gaussian, CP2K, etc.)
- [ ] Web interface
- [ ] Batch computation support
- [ ] Automatic parameter optimization

## Contributing

Issues and Pull Requests are welcome!

## License

MIT License