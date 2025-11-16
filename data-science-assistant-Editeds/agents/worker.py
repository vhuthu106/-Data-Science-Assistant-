import os
from openai import OpenAI
from dotenv import load_dotenv
from qwen_agent.agents import Assistant
from qwen_agent.llm import get_chat_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tempfile
import json
import logging
import re  # Added for regex pattern matching
import warnings  # Added for warning handling

# Import entropy logger + planner for feedback loop
from core.entropy_logger import EntropyLogger
from agents.planner import Planner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Worker:
    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=self.openai_api_key) if self.openai_api_key else None

        # Initialize Qwen
        llm_config = {
            'model': 'qwen-turbo',
            'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        }
        llm = get_chat_model(llm_config)
        self.qwen_agent = Assistant(llm=llm)

        # Tools
        self.use_openai = self.openai_api_key is not None
        self.logger = EntropyLogger()   # ✅ database logger
        self.planner = Planner()        # ✅ feedback planner

    def handle(self, plan, dataset_manager, max_retries=2):
        outputs = {"images": [], "tables": [], "results": {}}
        df = dataset_manager.get_dataset()
        if df is None:
            return {"error": "No dataset loaded"}

        user_query = plan.get("intent", "unknown_task")

        for step in plan["steps"]:
            action = step["action"]
            args = step.get("args", {})
            retries = 0

            while retries <= max_retries:
                try:
                    # --- Generate code
                    if self.use_openai:
                        generated_code = self._generate_with_openai(action, args, df)
                    else:
                        generated_code = self._generate_with_qwen(action, args, df)

                    # --- Execute
                    result = self._execute_code(generated_code, df)

                    if result.get("success"):
                        outputs["results"][action] = result
                        if result.get("image"):
                            outputs["images"].append(result["image"])
                        if result.get("table") is not None:
                            outputs["tables"].append(result["table"])
                        
                        # ✅ Log success
                        #self.logger.log_interaction(user_query, f"{action} success", error_count=retries)
                        break

                    else:
                        raise Exception(result.get("error", "Unknown error"))

                except Exception as e:
                    retries += 1
                    error_msg = str(e)

                    # ❌ Log failure
                    #self.logger.log_interaction(user_query, f"{action} failed: {error_msg}", error_count=retries)

                    if retries <= max_retries:
                        # 🔄 Ask planner to revise plan
                        revised_plan = self._feedback_to_planner(plan, action, error_msg)
                        if revised_plan:
                            plan = revised_plan
                            step = plan["steps"][0]  # take revised first step
                            action, args = step["action"], step.get("args", {})
                            continue
                    else:
                        # 🛑 Final fallback
                        fallback_plan = self.planner._create_fallback_plan(user_query)
                        outputs["results"][action] = {
                            "success": False,
                            "error": error_msg,
                            "plan_used": "rule_based_fallback",
                            "fallback_plan": fallback_plan
                        }
                        break

        return outputs
    
    def _feedback_to_planner(self, plan, action, error_msg):
        """Ask planner to fix the plan if worker execution failed"""
        try:
            critique = self.planner.critique(plan)
            logger.info(f"Planner critique: {critique}")
            revised = self.planner.plan(f"Revise due to error: {error_msg}")
            return revised
        except Exception as e:
            logger.error(f"Feedback loop failed: {e}")
            return None    
    
    def _generate_with_openai(self, action, args, df):
        """Generate code using OpenAI"""
        prompt = self._create_code_prompt(action, args, df)
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a helpful code generation assistant. Return only executable Python code. Avoid pandas inplace operations with chained assignment as they cause warnings."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        code = response.choices[0].message.content.strip()
        code = self._clean_code(code)
        code = self._validate_and_fix_code(code, df)
        return code
    
    def _generate_with_qwen(self, action, args, df):
        """Generate code using Qwen"""
        prompt = self._create_code_prompt(action, args, df)
        
        try:
            # Get the response from Qwen agent
            response = self.qwen_agent.run(prompt)
            
            # Handle different response types more robustly
            content = self._extract_content_from_response(response)
            
            code = self._clean_code(content)
            code = self._validate_and_fix_code(code, df)
            return code
            
        except Exception as e:
            logger.error(f"Qwen code generation failed: {e}")
            # Fallback to simple code generation
            return self._generate_fallback_code(action, args, df)
    
    def _extract_content_from_response(self, response):
        """Extract content from various Qwen response formats"""
        try:
            # If response is a string, return it directly
            if isinstance(response, str):
                return response
            
            # If response has a content attribute
            if hasattr(response, 'content'):
                return response.content
            
            # If response is a dictionary with content
            if isinstance(response, dict):
                if 'content' in response:
                    return response['content']
                elif 'text' in response:
                    return response['text']
                elif 'response' in response:
                    return response['response']
            
            # If response is iterable (like a generator)
            if hasattr(response, '__iter__') and not isinstance(response, (str, dict)):
                content = ""
                for chunk in response:
                    if hasattr(chunk, 'content'):
                        content += str(chunk.content)
                    elif isinstance(chunk, dict):
                        if 'content' in chunk:
                            content += str(chunk['content'])
                        elif 'text' in chunk:
                            content += str(chunk['text'])
                    elif isinstance(chunk, str):
                        content += chunk
                    else:
                        content += str(chunk)
                return content
            
            # Fallback: convert to string
            return str(response)
            
        except Exception as e:
            logger.error(f"Error extracting content from response: {e}")
            return f"# Error processing response: {str(e)}"
    
    def _create_code_prompt(self, action, args, df):
        query = args.get('query', '')
        return f"""
        Generate Python code to perform: {action}
        Query: {query}
        
        Dataframe info:
        - Name: {getattr(df, 'name', 'Unknown')}
        - Shape: {df.shape}
        - Columns: {list(df.columns)}
        - Data types: {df.dtypes.to_dict()}
        
        **IMPORTANT REQUIREMENTS:**
        1. Return ONLY Python code without any explanations
        2. Use modern pandas practices (avoid inplace=True with chained assignment)
        3. Never use: df['col'].method(inplace=True) - this causes warnings
        4. Instead use: df = df.method() or df['col'] = df['col'].method()
        5. Handle missing values appropriately with .copy() when needed
        6. Include necessary imports
        7. Make sure the code is complete and executable
        
        **BAD EXAMPLE (causes warnings):**
        df['Age'].fillna(df['Age'].mean(), inplace=True)
        
        **GOOD EXAMPLE:**
        df = df.copy()
        df['Age'] = df['Age'].fillna(df['Age'].mean())
        
        Example for data cleaning:
        import pandas as pd
        import numpy as np
        
        # Create a copy to avoid warnings
        df_clean = df.copy()
        
        # Handle missing values properly
        df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].mean())
        df_clean['Embarked'] = df_clean['Embarked'].fillna('Unknown')
        
        # Return cleaned dataframe
        result = df_clean
        """
    
    def _generate_fallback_code(self, action, args, df):
        """Generate fallback code based on action type"""
        action_lower = action.lower()
        query = args.get('query', '').lower()
        
        if 'correlation' in query or 'relationship' in query:
            return """
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create copy to avoid warnings
df_clean = df.copy()

# Select only numeric columns for correlation
numeric_df = df_clean.select_dtypes(include=[np.number])

# Calculate correlation matrix
correlation_matrix = numeric_df.corr()

# Create heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.savefig('temp_plot.png', bbox_inches='tight', dpi=100)
plt.close()

# Return the correlation matrix
result = correlation_matrix
"""
        elif 'clean' in query or 'missing' in query:
            return """
import pandas as pd
import numpy as np

# Create copy to avoid warnings
df_clean = df.copy()

# Handle missing values properly
print("Missing values before cleaning:")
print(df_clean.isnull().sum())

# Fill numeric missing values with mean
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

# Fill categorical missing values with mode
categorical_cols = df_clean.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')

print("\\nMissing values after cleaning:")
print(df_clean.isnull().sum())

result = df_clean
"""
        elif 'average' in query and ('class' in query or 'pclass' in query):
            return """
import pandas as pd
import matplotlib.pyplot as plt

# Create copy to avoid warnings
df_clean = df.copy()

# Calculate average age by class
average_age_by_class = df_clean.groupby('Pclass')['Age'].mean().reset_index()
average_age_by_class.columns = ['Pclass', 'AverageAge']

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(average_age_by_class['Pclass'].astype(str), average_age_by_class['AverageAge'])
plt.xlabel('Passenger Class')
plt.ylabel('Average Age')
plt.title('Average Age by Passenger Class')
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, v in enumerate(average_age_by_class['AverageAge']):
    plt.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom')

plt.savefig('temp_plot.png', bbox_inches='tight', dpi=100)
plt.close()

# Return the result
result = average_age_by_class
"""
        else:
            return """
import pandas as pd
import matplotlib.pyplot as plt

# Create copy to avoid warnings
df_clean = df.copy()

# Basic data analysis
print("Dataset shape:", df_clean.shape)
print("\\nColumn names:", list(df_clean.columns))
print("\\nData types:\\n", df_clean.dtypes)
print("\\nMissing values:\\n", df_clean.isnull().sum())

# Show basic statistics
print("\\nBasic statistics:\\n", df_clean.describe())

result = df_clean
"""
    
    def _validate_and_fix_code(self, code, df):
        """Validate generated code and fix common pandas issues"""
        # List of problematic patterns and their fixes
        patterns_and_fixes = [
            # Fix inplace operations with chained assignment
            (r"df\[['\"](.*?)['\"]\]\.(.*?)\(.*?inplace\s*=\s*True", r"df = df.copy()\ndf['\1'] = df['\1'].\2("),
            (r"df\.(.*?)\(.*?inplace\s*=\s*True", r"df = df.\1("),
            
            # Ensure copy is used when modifying
            (r"^(?!.*\.copy\(\))(.*df\[.*\] =.*)", r"df = df.copy()\n\1"),
            
            # Add missing imports
            (r"^import pandas as pd", r"import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns"),
        ]
        
        # Apply fixes
        for pattern, fix in patterns_and_fixes:
            code = re.sub(pattern, fix, code, flags=re.MULTILINE)
        
        # Ensure copy is used if dataframe is modified
        if any(mod_pattern in code for mod_pattern in ["df[", "df.", "= df", "df ="]):
            if "df.copy()" not in code and "df = df.copy()" not in code:
                code = "df = df.copy()\n" + code
        
        return code
    
    def _clean_code(self, code):
        """Clean generated code"""
        if not code:
            return "print('No code generated')"
            
        if '```python' in code:
            code = code.split('```python')[1].split('```')[0].strip()
        elif '```' in code:
            code = code.split('```')[1].split('```')[0].strip()
        
        # Remove any non-code content
        lines = code.split('\n')
        clean_lines = []
        for line in lines:
            if not line.strip().startswith(('//', '/*', '*/', '#')) or line.strip().startswith('# '):
                clean_lines.append(line)
        
        return '\n'.join(clean_lines)
    
    def _execute_code(self, code, df):
        try:
            df_copy = df.copy()
            
            # Create a function to capture print output
            print_output = []
            def custom_print(*args, **kwargs):
                output = " ".join(str(arg) for arg in args)
                print_output.append(output)
            
            # Capture warnings too
            warning_messages = []
            
            def custom_warning(message, category, filename, lineno, file=None, line=None):
                warning_messages.append(f"{category.__name__}: {message}")
            
            # Set up execution environment
            local_vars = {
                'df': df_copy,
                'pd': pd,
                'plt': plt,
                'np': np,
                'sns': sns,
                'print': custom_print,
                'warnings': warnings
            }
            
            # Execute with warnings captured
            with warnings.catch_warnings(record=True) as w:
                warnings.showwarning = custom_warning
                exec(code, {}, local_vars)
                
                # Capture any warnings
                for warning in w:
                    warning_messages.append(f"{warning.category.__name__}: {warning.message}")
            
            result = {
                "success": True, 
                "code": code,
                "print_output": print_output,
                "warnings": warning_messages  # Include warnings in result
            }
            
            # Check for plots
            if plt.gcf().get_axes():
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    plt.savefig(tmp.name, bbox_inches='tight', dpi=100)
                    plt.close()
                    result["image"] = tmp.name
            
            # Check for any returned results
            for var_name in local_vars:
                if var_name not in ['df', 'pd', 'plt', 'np', 'sns', 'print', 'custom_print', 'warnings'] and not var_name.startswith('_'):
                    var_value = local_vars[var_name]
                    if isinstance(var_value, (pd.DataFrame, pd.Series)):
                        result["table"] = var_value
                    elif isinstance(var_value, (str, int, float, list, dict)):
                        result["output"] = var_value
            
            return result
            
        except Exception as e:
            return {"error": str(e), "code": code, "success": False}