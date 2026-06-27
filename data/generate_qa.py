import json
import re
from typing import Dict, List, Tuple

EntityDict = Dict[str, str]

def realization_operator(template: str, entity: EntityDict) -> str:
    def substitution_function(match: re.Match) -> str:
        key = match.group(1) 
        if key is None:
            return entity.get("name", "[MISSING_NAME]")
        else:
            return entity.get(key, f"[MISSING_{key.upper()}]")
            
    placeholder_pattern = re.compile(r'\bX(?:\.(\w+))?')
    return placeholder_pattern.sub(substitution_function, template)

def generate_from_json(filepath: str) -> Dict[str, List[Tuple[str, str]]]:
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    all_tasks_data = {}
    
    for task in data.get("tasks", []):
        task_name = task.get("name", "Unnamed Task")
        entities = task.get("entities", [])
        templates = task.get("templates", [])
        
        task_pairs = []
        
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
    input_filename = 'tasks_data.json'
    output_filename = 'qa_dataset.txt'
    
    try:
        # 1. Generate the data
        generated_data = generate_from_json(input_filename)
        
        # 2. Write it to a text file
        with open(output_filename, 'w', encoding='utf-8') as out_file:
            total_pairs = 0
            for task_name, qa_pairs in generated_data.items():
                out_file.write(f"--- {task_name.upper()} ---\n")
                
                for q, a in qa_pairs:
                    out_file.write(f"Q: {q}\n")
                    out_file.write(f"A: {a}\n\n")
                    total_pairs += 1
                    
        print(f"Success! Generated {total_pairs} QA pairs.")
        print(f"Saved to: {output_filename}")
                
    except FileNotFoundError:
        print(f"Error: Could not find '{input_filename}'. Please ensure both files are in the same folder.")
