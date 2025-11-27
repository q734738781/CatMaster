#!/usr/bin/env python3
"""
Demo script for O2 molecule geometry optimization and energy calculation using CatMaster.

This demo demonstrates:
1. Creating an O2 molecule structure
2. ML pre-optimization with MACE (Might optional decision by LLM)
3. VASP DFT calculation (geometry optimization)
4. VASP DFT single-point energy calculation
5. Summary of results

All using the LLM-driven workflow with proper tool registry integration.
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from catmaster.agents.workflow import CatMasterWorkflow
from langchain_openai import ChatOpenAI

def main():
    """Main demo function"""
    # Setup logging
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_dir / f"catmaster_demo_{timestamp}.log"
    
    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.info(f"Log file created: {log_filename}")
    
    logger.info("=" * 60)
    logger.info("CatMaster Demo: O2 DFT Calculation")
    logger.info("=" * 60)
    
    # Configuration
    config = {
        "model": "gpt-5",  # Change to your preferred model
        "temperature": 0,
        "workspace": "./demo_workspace",
    }
    # Clear workspace if it exists
    if os.path.exists(config["workspace"]):
        shutil.rmtree(config["workspace"])
        os.makedirs(config["workspace"])
    
    # User request
    user_request = """
    Please perform a complete VASP DFT calculation for O2 molecule, using spin-polarized calculations appropriate for triplet O2. 
    """
    
    logger.info("Configuration:")
    logger.info(f"  Model: {config['model']}")
    logger.info(f"  Workspace: {config['workspace']}")
    
    logger.info("Request:")
    logger.info(user_request.strip())
    logger.info("=" * 60)
    
    # Create workspace directory
    workspace_path = Path(config["workspace"])
    workspace_path.mkdir(parents=True, exist_ok=True)
    
    # Change to workspace directory
    original_cwd = os.getcwd()
    os.chdir(workspace_path)
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=config["model"],
        temperature=config["temperature"]
    )
    
    # Initialize CatMaster workflow
    workflow = CatMasterWorkflow(llm=llm)
    
    # Execute the task
    logger.info("[Starting CatMaster Workflow]")
    
    try:
        # Run the workflow
        result = workflow.run(user_request)
        
        logger.info("=" * 60)
        logger.info("[Workflow Completed Successfully]")
        logger.info("=" * 60)
        
        # Display results
        if result:
            logger.info("Summary:")
            logger.info(f"  Title: {result.title}")
            logger.info(f"  Objective: {result.objective}")
            
            if result.key_results:
                logger.info("Key Results:")
                for key, value in result.key_results.items():
                    logger.info(f"  - {key}: {value}")
            
            if result.conclusions:
                logger.info("Conclusions:")
                for conclusion in result.conclusions:
                    logger.info(f"  - {conclusion}")
            
            if result.recommendations:
                logger.info("Recommendations:")
                for rec in result.recommendations:
                    logger.info(f"  - {rec}")
            
            if result.execution_summary:
                logger.info("Execution Summary:")
                for key, value in result.execution_summary.items():
                    logger.info(f"  - {key}: {value}")
            
            if result.detailed_results:
                logger.info("Detailed Results:")
                # Show structure files if available
                if "output_files" in result.detailed_results:
                    logger.info("  Output Files:")
                    for category, files in result.detailed_results["output_files"].items():
                        logger.info(f"    {category}:")
                        for f in files:
                            logger.info(f"      - {f}")
        
        else:
            logger.info("No result returned from workflow")
            
    except Exception as e:
        logger.error("=" * 60)
        logger.error("[Workflow Failed]")
        logger.error("=" * 60)
        logger.error(f"Error: {str(e)}")
        
        import traceback
        logger.error("Traceback:")
        logger.error(traceback.format_exc())
    
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
    
    logger.info("=" * 60)
    logger.info("Demo completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
