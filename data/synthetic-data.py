"""
Generate synthetic disease risk prediction data using decision tree logic.
This file creates `data/synthetic_disease_data.json` by default and
contains the sentence-generation logic (inputs=30, outputs=10).
"""

import json
import os
import random
from typing import List, Dict
from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier
import numpy as np

@dataclass
class HealthProfile:
    name: str
    age: int
    smoker: bool
    overweight: bool
    diabetes: bool
    high_bp: bool
    high_cholesterol: bool
    sedentary: bool
    family_history_heart: bool
    family_history_cancer: bool
    family_history_diabetes: bool
    alcohol_consumption: str
    exercise_frequency: str
    diet_quality: str


class DiseaseRiskPredictor:
    DISEASES = [
        'diabetes', 'heart_disease', 'lung_cancer', 'stroke',
        'hypertension', 'obesity_complications', 'liver_disease',
        'kidney_disease', 'dementia', 'copd'
    ]

    def __init__(self):
        self.models = {}
        self._train_models()

    def _train_models(self):
        n_samples = 500
        X = []
        disease_targets = {d: [] for d in self.DISEASES}

        for _ in range(n_samples):
            p = self._generate_random_profile()
            X.append(self._profile_to_features(p))
            disease_targets['diabetes'].append(1 if (p.overweight and p.age > 40) or p.family_history_diabetes else 0)
            disease_targets['heart_disease'].append(1 if (p.high_bp and p.high_cholesterol) or (p.smoker and p.age > 45) else 0)
            disease_targets['lung_cancer'].append(1 if p.smoker and p.age > 50 else 0)
            disease_targets['stroke'].append(1 if (p.high_bp and p.age > 55) or (p.high_cholesterol and p.overweight) else 0)
            disease_targets['hypertension'].append(1 if p.high_bp or (p.overweight and p.age > 35) else 0)
            disease_targets['obesity_complications'].append(1 if p.overweight and p.sedentary else 0)
            disease_targets['liver_disease'].append(1 if p.alcohol_consumption == 'heavy' and p.age > 40 else 0)
            disease_targets['kidney_disease'].append(1 if (p.diabetes or p.high_bp) and p.age > 45 else 0)
            disease_targets['dementia'].append(1 if p.age > 65 and (p.sedentary or p.alcohol_consumption == 'heavy') else 0)
            disease_targets['copd'].append(1 if (p.smoker and p.age > 40) or (p.sedentary and p.age > 60) else 0)

        X = np.array(X)
        for d in self.DISEASES:
            y = np.array(disease_targets[d])
            clf = DecisionTreeClassifier(max_depth=4, random_state=42)
            clf.fit(X, y)
            self.models[d] = clf

    def _generate_random_profile(self) -> HealthProfile:
        names = ['John', 'Jane', 'Michael', 'Sarah', 'Robert', 'Emma', 'David', 'Lisa']
        return HealthProfile(
            name=random.choice(names),
            age=random.randint(20, 80),
            smoker=random.choice([True, False]),
            overweight=random.choice([True, False]),
            diabetes=random.choice([True, False]),
            high_bp=random.choice([True, False]),
            high_cholesterol=random.choice([True, False]),
            sedentary=random.choice([True, False]),
            family_history_heart=random.choice([True, False]),
            family_history_cancer=random.choice([True, False]),
            family_history_diabetes=random.choice([True, False]),
            alcohol_consumption=random.choice(['none', 'moderate', 'heavy']),
            exercise_frequency=random.choice(['none', 'moderate', 'high']),
            diet_quality=random.choice(['poor', 'average', 'good'])
        )

    def _profile_to_features(self, p: HealthProfile) -> List[float]:
        return [
            p.age,
            1.0 if p.smoker else 0.0,
            1.0 if p.overweight else 0.0,
            1.0 if p.diabetes else 0.0,
            1.0 if p.high_bp else 0.0,
            1.0 if p.high_cholesterol else 0.0,
            1.0 if p.sedentary else 0.0,
            1.0 if p.family_history_heart else 0.0,
            1.0 if p.family_history_cancer else 0.0,
            1.0 if p.family_history_diabetes else 0.0,
            1.0 if p.alcohol_consumption == 'moderate' else 0.0,
            1.0 if p.alcohol_consumption == 'heavy' else 0.0,
            1.0 if p.exercise_frequency == 'moderate' else 0.0,
            1.0 if p.exercise_frequency == 'high' else 0.0,
            1.0 if p.diet_quality == 'average' else 0.0,
            1.0 if p.diet_quality == 'good' else 0.0,
        ]

    def predict_risks(self, p: HealthProfile) -> Dict[str, bool]:
        f = np.array(self._profile_to_features(p)).reshape(1, -1)
        return {d: bool(self.models[d].predict(f)[0]) for d in self.DISEASES}

    def generate_input_statements(self, p: HealthProfile) -> str:
        s = []
        s.append(f"{p.name} is {p.age} years old.")
        s.append(f"{p.name} is {'a smoker' if p.smoker else 'not a smoker'}.")
        s.append(f"{p.name} is {'overweight' if p.overweight else 'not overweight'}.")
        s.append(f"{p.name} has {'high blood pressure' if p.high_bp else 'normal blood pressure'}.")
        s.append(f"{p.name} has {'high cholesterol' if p.high_cholesterol else 'normal cholesterol levels'}.")
        s.append(f"{p.name} {'has' if p.diabetes else 'does not have'} diabetes.")
        s.append(f"{p.name} {'leads a sedentary lifestyle' if p.sedentary else 'is moderately active'}.")
        exercise_map = {'none': f"{p.name} does not exercise.", 'moderate': f"{p.name} exercises moderately.", 'high': f"{p.name} exercises frequently."}
        s.append(exercise_map[p.exercise_frequency])
        diet_map = {'poor': f"{p.name} has a poor diet.", 'average': f"{p.name} has an average diet.", 'good': f"{p.name} follows a healthy diet."}
        s.append(diet_map[p.diet_quality])
        alcohol_map = {'none': f"{p.name} does not consume alcohol.", 'moderate': f"{p.name} drinks alcohol moderately.", 'heavy': f"{p.name} drinks alcohol heavily."}
        s.append(alcohol_map[p.alcohol_consumption])
        s.append(f"{p.name} {'has' if p.family_history_heart else 'does not have'} a family history of heart disease.")
        s.append(f"{p.name} {'has' if p.family_history_cancer else 'does not have'} a family history of cancer.")
        s.append(f"{p.name} {'has' if p.family_history_diabetes else 'does not have'} a family history of diabetes.")
        age_group = 'in his 20s' if p.age < 30 else 'in his 30s' if p.age < 40 else 'in his 40s' if p.age < 50 else 'in his 50s' if p.age < 60 else 'in his 60s' if p.age < 70 else 'over 70'
        s.append(f"{p.name} is {age_group}.")
        health_conditions = []
        if p.diabetes: health_conditions.append('diabetes')
        if p.high_bp: health_conditions.append('high blood pressure')
        if p.high_cholesterol: health_conditions.append('high cholesterol')
        if p.overweight: health_conditions.append('being overweight')
        s.append(f"{p.name} is managing {', '.join(health_conditions)}." if health_conditions else f"{p.name} has no major chronic conditions.")
        s.append(f"{p.name} spends most of the day sitting." if p.sedentary else f"{p.name} maintains an active lifestyle.")
        s.append(f"{p.name} reports {'high' if random.choice([True, False]) else 'moderate'} stress levels.")
        s.append(f"{p.name} has {random.choice(['poor','average','good'])} sleep quality.")
        if p.overweight: s.append(f"{p.name}'s BMI indicates overweight category.")
        s.append(f"{p.name} has {random.choice(['high','moderate','low'])} perception of health risks.")
        s.append(f"{p.name} has {random.choice(['annual','biennial','irregular'])} medical checkups.")
        s.append(f"{p.name} is {'currently on' if random.choice([True, False]) else 'not on'} any medications.")
        s.append(f"{p.name} has {random.choice(['strong','moderate','limited'])} social support.")
        s.append(f"{p.name}'s work is {random.choice(['sedentary','moderately active','physically demanding'])}.")
        s.append(f"{p.name} commutes by {random.choice(['car','public transport','walking/biking'])}.")
        s.append(f"{p.name} has {random.choice(['adequate','inadequate'])} water intake.")
        s.append(f"{p.name} consumes fruits and vegetables {random.choice(['regularly','occasionally','rarely'])}.")
        s.append(f"{p.name} consumes processed foods {random.choice(['frequently','moderately','rarely'])}.")
        s.append(f"{p.name} is {random.choice(['actively','not'])} managing weight.")
        s.append(f"{p.name} has {random.choice(['easy','moderate','difficult'])} access to healthcare.")
        return "\n".join(s[:30])

    def generate_output_statements(self, p: HealthProfile, risks: Dict[str, bool]) -> str:
        rm = {
            'diabetes':'diabetes','heart_disease':'heart disease','lung_cancer':'lung cancer','stroke':'stroke',
            'hypertension':'hypertension','obesity_complications':'obesity-related complications','liver_disease':'liver disease',
            'kidney_disease':'kidney disease','dementia':'dementia','copd':'COPD (chronic obstructive pulmonary disease)'
        }
        out = []
        for k,v in rm.items():
            out.append(f"{p.name} is at risk of {v}." if risks.get(k, False) else f"{p.name} is not at risk of {v}.")
        return "\n".join(out[:10])


def generate_dataset(num_samples: int = 100, output_file: str = 'data/synthetic_disease_data.json') -> None:
    pred = DiseaseRiskPredictor()
    data = []
    for i in range(num_samples):
        p = pred._generate_random_profile()
        risks = pred.predict_risks(p)
        data.append({'id': i, 'profile': p.__dict__, 'inputs': pred.generate_input_statements(p), 'outputs': pred.generate_output_statements(p, risks), 'risks': risks})
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} samples to {output_file}")


if __name__ == '__main__':
    generate_dataset(num_samples=100, output_file='data/synthetic_disease_data.json')
