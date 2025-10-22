"""
LLM Evaluator for DAB Evaluation SDK
LLM evaluator for DAB Evaluation SDK
"""

import json
from typing import Dict, Any, List, Optional, Tuple
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
            evaluation_prompt = self._build_evaluation_prompt(
                question, agent_response, expected_answer, context
            )

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a professional Web3 Agent evaluation expert."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            llm_response = response.choices[0].message.content
            parsed = self._parse_llm_response(llm_response)

            return {
                "score": parsed["score"],
                "reasoning": parsed["reasoning"],
                "details": {
                    "model_used": self.model_name,
                    "llm_response": llm_response,
                    "confidence": parsed["confidence"],
                    "flags": parsed["flags"],
                    "evaluation_method": "llm_based"
                }
            }

        except Exception as e:
            return {
                "score": 0.0,
                "reasoning": f"LLM evaluation failed: {str(e)}",
                "details": {"error": str(e), "evaluation_method": "llm_based"}
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
    "score": 0.0-1.0,
    "confidence": 0.0-1.0,
    "reasoning": "Detailed evaluation reasoning",
    "flags": ["optional issues or notes"]
}
"""
        
        return prompt

    def _parse_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse and validate LLM response enforcing schema."""

        fallback_score = 0.5 if len(llm_response or "") < 50 else 0.7
        fallback_reason = (llm_response or "")[:200]

        try:
            result = json.loads(llm_response)
        except json.JSONDecodeError:
            return {
                "score": fallback_score,
                "confidence": 0.4,
                "reasoning": f"LLM response not valid JSON; fallback applied. Snippet: {fallback_reason}",
                "flags": ["invalid_json"],
            }

        score, score_flag = self._extract_float(result, "score", 0.0, 1.0, fallback_score)
        confidence, confidence_flag = self._extract_float(result, "confidence", 0.0, 1.0, 0.6)
        reasoning = self._extract_string(result, "reasoning", fallback_reason)
        flags = self._extract_list(result, "flags", default=[])

        schema_flags = []
        for flag in (score_flag, confidence_flag):
            if flag:
                schema_flags.append(flag)
        flags.extend(schema_flags)

        if self._contains_uncertainty(reasoning):
            confidence = min(confidence, 0.4)
            flags.append("low_confidence_reasoning")

        return {
            "score": score,
            "confidence": confidence,
            "reasoning": reasoning,
            "flags": flags,
        }

    def _extract_float(self, payload: Dict[str, Any], key: str, min_value: float, max_value: float, default: float) -> Tuple[float, Optional[str]]:
        value = payload.get(key)
        try:
            num = float(value)
            if not (min_value <= num <= max_value):
                return default, f"{key}_out_of_range"
            return num, None
        except (TypeError, ValueError):
            return default, f"{key}_invalid"

    def _extract_string(self, payload: Dict[str, Any], key: str, default: str) -> str:
        value = payload.get(key)
        return str(value) if isinstance(value, str) and value.strip() else default

    def _extract_list(self, payload: Dict[str, Any], key: str, default: Optional[List[str]] = None) -> List[str]:
        value = payload.get(key)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return value
        return default or []

    def _contains_uncertainty(self, text: str) -> bool:
        lowered = (text or "").lower()
        uncertainty_tokens = [
            "unsure",
            "uncertain",
            "cannot determine",
            "not enough information",
            "insufficient data",
            "no confidence",
        ]
        return any(token in lowered for token in uncertainty_tokens)
    
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
