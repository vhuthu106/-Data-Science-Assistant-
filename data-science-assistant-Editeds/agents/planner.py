import os
import json
from dotenv import load_dotenv
import requests
from qwen_agent.agents import Assistant
from qwen_agent.llm import get_chat_model
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Planner:
    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize Qwen as fallback
        llm_config = {
            'model': 'qwen-turbo',
            'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        }
        llm = get_chat_model(llm_config)
        self.qwen_agent = Assistant(llm=llm)
        
        self.use_openai = True if self.openai_api_key else False
        self.fallback_used = False

    def plan(self, user_query: str):
        if self.use_openai:
            try:
                return self._plan_with_openai(user_query)
            except Exception as e:
                if any(error_str in str(e).lower() for error_str in ["quota", "429", "rate", "insufficient_quota", "expecting value"]):
                    logger.warning(f"OpenAI failed ({e}). Switching to Qwen...")
                    self.use_openai = False
                    self.fallback_used = True
                    return self._plan_with_qwen(user_query)
                else:
                    logger.error(f"OpenAI error: {e} for prompt ; {user_query}")
                    raise e
        else:
            return self._plan_with_qwen(user_query)
    
    def _plan_with_openai(self, user_query: str):
        """Use OpenAI for planning"""
        if not self.openai_api_key:
            logger.warning("OpenAI API key not found. Using Qwen instead.")
            self.use_openai = False
            return self._plan_with_qwen(user_query)
            
        prompt = f"""
You are a planner agent for data science tasks. Convert the user request into a valid JSON plan with EXACTLY this structure:
{{
  "intent": "brief_description_of_the_task",
  "steps": [
    {{"action": "specific_action_name", "args": {{"query": "user_query_here"}}}}
  ],
  "analysis": ["steps for data prep or stats"],
  "visualization": ["steps for plots"],
  "validation": ["checks for correctness/errors"],
  "tools": ["pandas", "matplotlib", "seaborn"]
}}

Available actions: create_plot, data_analysis, data_cleaning, correlation_analysis, group_analysis, statistical_test

User request: "{user_query}"

Return ONLY the JSON object without any additional text, explanations, or code blocks.
"""
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }
        
        data = {
            "model": "gpt-4.1-mini",
            "messages": [
                {"role": "system", "content": "You are a planning assistant that returns only valid JSON. No additional text."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0,
            "max_tokens": 600
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")

            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            content = self._clean_json_response(content)

            if not content or content.isspace():
                raise Exception("Empty response from OpenAI")

            plan = json.loads(content)

            # ✅ Ensure steps exist
            if "steps" not in plan or not isinstance(plan["steps"], list) or len(plan["steps"]) == 0:
                inferred_action = "data_analysis"
                q = user_query.lower()
                if any(x in q for x in ["plot", "chart", "visualize"]):
                    inferred_action = "create_plot"
                elif any(x in q for x in ["group", "average", "mean"]):
                    inferred_action = "group_analysis"
                elif any(x in q for x in ["correlation", "relationship"]):
                    inferred_action = "correlation_analysis"
                elif any(x in q for x in ["clean", "missing", "null"]):
                    inferred_action = "data_cleaning"

                plan["steps"] = [{"action": inferred_action, "args": {"query": user_query}}]
                logger.info(f"✅ Auto-added steps to plan: {plan['steps']}")

            return plan

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}. Content: {content if 'content' in locals() else 'None'}")
            return self._create_fallback_plan(user_query)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise e

    def _clean_json_response(self, content):
        """Clean the response to extract pure JSON"""
        if not content:
            return "{}"
        
        # Remove markdown code blocks
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
        
        # Remove any non-JSON content before the first {
        if '{' in content:
            content = content[content.find('{'):]
        
        # Remove any non-JSON content after the last }
        if '}' in content:
            content = content[:content.rfind('}') + 1]
        
        return content
    
    def _plan_with_qwen(self, user_query: str):
        """Use Qwen as fallback for planning"""
        prompt = f"""
You are a planner agent for data science tasks. Convert the user request into a valid JSON plan with EXACTLY this structure:
{{
  "intent": "brief_description",
  "steps": [
    {{"action": "specific_action_name", "args": {{"query": "user_query_here"}}}}
  ],
  "tools": ["pandas", "matplotlib", "seaborn"]
}}

Available actions: create_plot, data_analysis, data_cleaning, correlation_analysis, group_analysis, statistical_test

User request: "{user_query}"

Return ONLY the JSON object without any additional text or explanations.
"""
        
        try:
            response = self.qwen_agent.run(prompt)
            
            if hasattr(response, 'content'):
                content = response.content
            elif isinstance(response, dict) and 'content' in response:
                content = response['content']
            else:
                content = str(response)
            
            content = self._clean_json_response(content)
            
            if not content or content.isspace():
                return self._create_fallback_plan(user_query)
                
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Qwen planning failed: {e}")
            return self._create_fallback_plan(user_query)
    
    def _create_fallback_plan(self, user_query: str):
        """Rule-based fallback planning"""
        user_query_lower = user_query.lower()
        
        if any(word in user_query_lower for word in ['plot', 'chart', 'graph', 'visualize']):
            return {
                "intent": "data_visualization",
                "steps": [{"action": "create_plot", "args": {"query": user_query}}],
                "tools": ["matplotlib", "seaborn"]
            }
        elif any(word in user_query_lower for word in ['analyze', 'statistics', 'describe', 'summary']):
            return {
                "intent": "data_analysis",
                "steps": [{"action": "data_analysis", "args": {"query": user_query}}],
                "tools": ["pandas", "numpy"]
            }
        elif any(word in user_query_lower for word in ['clean', 'process', 'transform', 'filter']):
            return {
                "intent": "data_processing",
                "steps": [{"action": "data_cleaning", "args": {"query": user_query}}],
                "tools": ["pandas"]
            }
        elif any(word in user_query_lower for word in ['correlation', 'relationship']):
            return {
                "intent": "correlation_analysis",
                "steps": [{"action": "correlation_analysis", "args": {"query": user_query}}],
                "tools": ["pandas", "seaborn"]
            }
        elif any(word in user_query_lower for word in ['average', 'mean', 'group by']):
            return {
                "intent": "group_analysis",
                "steps": [{"action": "group_analysis", "args": {"query": user_query}}],
                "tools": ["pandas", "matplotlib"]
            }
        else:
            return {
                "intent": "general_analysis",
                "steps": [{"action": "data_analysis", "args": {"query": user_query}}],
                "tools": ["pandas", "matplotlib"]
            }
    
    def get_planning_mode(self):
        if self.use_openai:
            return "OpenAI"
        else:
            return "Qwen (fallback)" if self.fallback_used else "Qwen"
        
    def critique(self, plan: dict):
        """Ask the model to review its own plan and suggest improvements"""
        critique_prompt = f"""
You are a senior data science reviewer. Review the following plan for errors, missing steps, or unclear reasoning.
Suggest improvements without changing the intent. Return JSON like:
{{
  "issues": ["list of problems"],
  "improvements": ["suggested fixes"],
  "confidence": "high/medium/low"
}}

Plan: {json.dumps(plan, indent=2)}
"""
        try:
            if self.use_openai:
                response = self._plan_with_openai(critique_prompt)
            else:
                response = self._plan_with_qwen(critique_prompt)
            return response
        except Exception as e:
            logger.error(f"Critique failed: {e}")
            return {"issues": ["Critique unavailable"], "improvements": [], "confidence": "low"}
