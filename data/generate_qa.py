import json
import re
from typing import Dict, List, Tuple

# D_e maps attribute keys to values
EntityDict = Dict[str, str]

def realization_operator(template: str, entity: EntityDict) -> str:
    """
    Implements Realize(T, D_e) and the substitution function sigma_e.
    Replaces placeholders like 'X' and 'X.k' with their dictionary values.
    """
    def substitution_function(match: re.Match) -> str:
        # match.group(1) captures the 'k' in 'X.k'. If None, the placeholder is 'X'.
        key = match.group(1) 
        
        if key is None:
            # sigma_e(X) = D_e(name)
            return entity.get("name", "[MISSING_NAME]")
        else:
            # sigma_e(X.k) = D_e(k)
            return entity.get(key, f"[MISSING_{key.upper()}]")
            
    # Regex explanation:
    # \bX      : Matches 'X' as a whole word
    # (?:\.    : Starts a non-capturing group for the '.k' part
    # (\w+)    : Captures the attribute key 'k' (alphanumeric characters and underscores)
    # )?       : Makes the '.k' part optional
    placeholder_pattern = re.compile(r'\bX(?:\.(\w+))?')
    
    return placeholder_pattern.sub(substitution_function, template)

def generate_from_json(filepath: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Loads tasks from a JSON file and generates all (q, a) pairs per task.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    all_tasks_data = {}
    
    # Iterate over T
    for task in data.get("tasks", []):
        task_name = task.get("name", "Unnamed Task")
        entities = task.get("entities", [])
        templates = task.get("templates", [])
        
        task_pairs = []
        
        # Cross product of E_t and T_t
        for entity in entities:
            for template in templates:
                t_q = template["T_Q"]
                t_a = template["T_A"]
                
                q = realization_operator(t_q, entity)
                a = realization_operator(t_a, entity)
                task_pairs.append((q, a))
                
        all_tasks_data[task_name] = task_pairs
        
    return all_tasks_data

if __name__ == "__main__":
    # Ensure tasks_data.json is in the same directory, or provide the full path
    try:
        generated_data = generate_from_json('tasks_data.json')
        
        for task_name, qa_pairs in generated_data.items():
            print(f"--- {task_name.upper()} ---")
            for idx, (q, a) in enumerate(qa_pairs, 1):
                print(f"Pair {idx}:")
                print(f"  q = {q}")
                print(f"  a = {a}\n")
                
    except FileNotFoundError:
        print("Error: Could not find 'tasks_data.json'. Please ensure the file exists.")
