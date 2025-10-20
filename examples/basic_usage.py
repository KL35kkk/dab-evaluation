#!/usr/bin/env python3
"""
DAB Evaluation SDK Basic Usage Example
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dab_eval import DABEvaluator, AgentMetadata, TaskCategory, TaskType, EvaluationMethod

async def basic_evaluation_example():
    """Basic evaluation example"""
    # LLM configuration
    llm_config = {
        "model": "doubao-seed-1-6",
        "temperature": 0.3,
        "max_tokens": 2000
    }
    
    # Create SDK instance with LLM config and output path
    evaluator = DABEvaluator(llm_config, "output")
    
    # Define Agent
    agent = AgentMetadata(
        url="http://localhost:8002",  # Handcrafted Agent
        capabilities=["web3_analysis", "blockchain_exploration"],
        timeout=30,
        close_endpoint="http://localhost:8002/close"
    )
    
    # Evaluation question
    question = "What is the date (UTC) when the US SEC approved Bitcoin spot ETF?"
    
    # Execute evaluation using enums
    result = await evaluator.evaluate_agent(
        question=question,
        agent_metadata=agent,
        task_type=TaskType.SINGLE_DOC_FACT_QA,
        category=TaskCategory.WEB_RETRIEVAL,
        context={"category": "regulatory_events"}
    )
    
    return result

async def multiple_agents_evaluation_example():
    """Multiple agents evaluation example"""
    # LLM configuration
    llm_config = {
        "model": "doubao-seed-1-6",
        "temperature": 0.3,
        "max_tokens": 2000
    }
    
    # Create evaluator
    evaluator = DABEvaluator(llm_config, "output")
    
    # Define multiple Agents
    agents = [
        AgentMetadata(
            url="http://localhost:8002",
            capabilities=["web3_analysis"],
            timeout=30,
            close_endpoint="http://localhost:8002/close"
        ),
        AgentMetadata(
            url="http://localhost:8003",
            capabilities=["web3_analysis", "multi_step_reasoning"],
            timeout=30,
            close_endpoint="http://localhost:8003/close"
        )
    ]
    
    # Evaluation question
    question = "Briefly describe the differences between Optimistic Rollup and ZK-Rollup in terms of exit time and security assumptions"
    
    # Execute multi-agent evaluation
    results = []
    for agent in agents:
        result = await evaluator.evaluate_agent(
            question=question,
            agent_metadata=agent,
            task_type=TaskType.ABSTRACTIVE_WITH_EVIDENCE,
            category=TaskCategory.WEB_RETRIEVAL,
            context={"category": "scaling_solutions"}
        )
        results.append(result)
    
    return results

async def export_results_example():
    """Export results example"""
    # LLM configuration
    llm_config = {
        "model": "doubao-seed-1-6",
        "temperature": 0.3,
        "max_tokens": 2000
    }
    
    # Create evaluator with custom output path
    evaluator = DABEvaluator(llm_config, "custom_output")
    
    # Execute some evaluations
    agent = AgentMetadata(
        url="http://localhost:8002",
        capabilities=["web3_analysis"],
        timeout=30
    )
    
    questions = [
        "What is the date (UTC) when the US SEC approved Bitcoin spot ETF?",
        "What is the topic0 (event signature hash) of ERC-20 Transfer event?"
    ]
    
    results = []
    for question in questions:
        result = await evaluator.evaluate_agent(
            question=question,
            agent_metadata=agent,
            task_type=TaskType.SINGLE_DOC_FACT_QA,
            category=TaskCategory.WEB_RETRIEVAL
        )
        results.append(result)
    
    # Export JSON format (automatically saved to custom_output directory)
    json_results = evaluator.export_results("json")
    
    # Export CSV format (automatically saved to custom_output directory)
    csv_results = evaluator.export_results("csv")
    
    return json_results, csv_results

async def main():
    """Main function"""
    try:
        # Basic evaluation
        await basic_evaluation_example()
        
        # Multi-agent evaluation
        await multiple_agents_evaluation_example()
        
        # Export results
        await export_results_example()
        
    except Exception as e:
        # Handle errors silently
        pass

if __name__ == "__main__":
    asyncio.run(main())