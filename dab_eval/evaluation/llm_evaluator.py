"""
LLM Evaluator for DAB Evaluation SDK
LLM evaluator for DAB Evaluation SDK
"""

import os
import json
from typing import Dict, Any, List, Optional
from .base_evaluator import BaseEvaluator

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class LLMEvaluator(BaseEvaluator):
    """LLM evaluator"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model_name = config.get("model") if config and config.get("model") else None
        self.temperature = config.get("temperature", 0.3) if config else 0.3
        self.max_tokens = config.get("max_tokens", 2000) if config else 2000
        
        self.client = None
        if config and config.get("client"):
            self.client = config["client"]
        elif OPENAI_AVAILABLE and config and config.get("api_key"):
            base_url = config.get("base_url")
            if base_url:
                self.client = OpenAI(base_url=base_url, api_key=config["api_key"])
            else:
                self.client = OpenAI(api_key=config["api_key"])
    
    async def evaluate(self, 
                      question: str,
                      agent_response: str,
                      expected_answer: Optional[str] = None,
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Use LLM to evaluate Agent response"""
        
        if not self.client:
            return {
                "score": 0.0,
                "reasoning": "LLM client not configured; provide llm_config['client'] or an API key",
                "details": {"error": "OpenAI client not initialized"}
            }
        if not self.model_name:
            return {
                "score": 0.0,
                "reasoning": "Model name missing in llm_config; please set llm_config['model']",
                "details": {"error": "LLM model not specified"}
            }
        
        try:
            # Build evaluation prompt
            evaluation_prompt = self._build_evaluation_prompt(
                question, agent_response, expected_answer, context
            )
            
            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a professional Web3 Agent evaluation expert."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Parse LLM response
            llm_response = response.choices[0].message.content
            
            # Try to parse JSON format response
            try:
                result = json.loads(llm_response)
                score = float(result.get("score", 0.0))
                reasoning = result.get("reasoning", "LLM evaluation completed")
            except:
                # If not JSON format, use simple scoring logic
                score = 0.7 if len(llm_response) > 50 else 0.5
                reasoning = llm_response[:200]
            
            return {
                "score": min(1.0, max(0.0, score)),
                "reasoning": reasoning,
                "details": {
                    "model_used": self.model_name,
                    "llm_response": llm_response,
                    "evaluation_method": "llm_based"
                }
            }
            
        except Exception as e:
            return {
                "score": 0.0,
                "reasoning": f"LLM evaluation failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    def _build_evaluation_prompt(self, 
                                question: str, 
                                agent_response: str, 
                                expected_answer: Optional[str] = None,
                                context: Optional[Dict[str, Any]] = None) -> str:
        """Build evaluation prompt"""
        
        prompt = f"""
Please evaluate the following Web3 Agent response quality:

Question: {question}

Agent Response: {agent_response}
"""
        
        if expected_answer:
            prompt += f"\nExpected Answer: {expected_answer}"
        
        if context:
            prompt += f"\nContext Information: {json.dumps(context, ensure_ascii=False)}"
        
        prompt += """

Please evaluate the Agent response from the following dimensions:
1. Accuracy: Whether the response accurately answers the question
2. Completeness: Whether the response is complete and contains necessary information
3. Professionalism: Whether the response demonstrates Web3 professional knowledge
4. Usefulness: Whether the response is helpful to users

Please return evaluation results in JSON format:
{
    "score": Score between 0.0-1.0,
    "reasoning": "Detailed evaluation reasoning"
}
"""
        
        return prompt
    
    def get_capabilities(self) -> List[str]:
        """Get evaluator capabilities"""
        return [
            "web3_analysis",
            "blockchain_exploration", 
            "defi_analysis",
            "nft_evaluation",
            "smart_contract",
            "technical_knowledge",
            "general_evaluation"
        ]
