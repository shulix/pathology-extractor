import pandas as pd
import re
from typing import List

class PathologyReportAnalyzer:
    def __init__(self, reports: List[str]):
        self.reports = reports
        
    def extract_number(self, text: str) -> float:
        """Extract number from text."""
        match = re.search(r'(\d+\.?\d*)', text)
        return float(match.group(1)) if match else None
    
    def get_differentiation(self, text: str) -> str:
        """Extract tumor differentiation."""
        if 'well-differentiated' in text.lower():
            return 'Well'
        elif 'poorly differentiated' in text.lower():
            return 'Poor'
        elif 'moderately differentiated' in text.lower():
            return 'Moderate'
        return 'Unknown'
    
    def get_invasion_status(self, text: str) -> dict:
        """Extract invasion information."""
        return {
            'lymphovascular': 'lymphovascular invasion' in text.lower(),
            'perineural': 'perineural invasion' in text.lower()
        }
    
    def determine_stage(self, tumor_depth: str, positive_nodes: int) -> str:
        """Determine cancer stage based on T and N status."""
        if 'confined to submucosa' in tumor_depth.lower():
            t_stage = 1
        elif 'muscularis propria' in tumor_depth.lower():
            t_stage = 2
        elif 'pericolonic' in tumor_depth.lower() or 'subsero' in tumor_depth.lower():
            t_stage = 3
        elif 'adjacent structure' in tumor_depth.lower():
            t_stage = 4
        else:
            t_stage = 0
            
        if positive_nodes == 0:
            n_stage = 0
        elif 1 <= positive_nodes <= 3:
            n_stage = 1
        else:
            n_stage = 2
            
        if t_stage <= 2 and n_stage == 0:
            return 'Stage I'
        elif t_stage <= 3 and n_stage == 0:
            return 'Stage II'
        elif n_stage == 1:
            return 'Stage III-A'
        elif n_stage == 2:
            return 'Stage III-B'
        return 'Unknown'

    def get_dataframe(self) -> pd.DataFrame:
        """Convert reports to structured DataFrame."""
        data = []
        
        for report in self.reports:
            # Extract tumor size
            size_match = re.search(r'(\d+\.?\d*)\s*cm', report)
            tumor_size = float(size_match.group(1)) if size_match else None
            
            # Extract lymph node information
            total_nodes_match = re.search(r'(\d+)\s+lymph nodes? (?:are|is|were|was) (?:examined|retrieved|evaluated)', report)
            positive_nodes_match = re.search(r'(\d+)\s+(?:are|is|contain|showing|with|positive)', report)
            
            total_nodes = int(total_nodes_match.group(1)) if total_nodes_match else 0
            positive_nodes = int(positive_nodes_match.group(1)) if positive_nodes_match and 'positive' in report else 0
            
            # Get differentiation
            differentiation = self.get_differentiation(report)
            
            # Get invasion status
            invasion = self.get_invasion_status(report)
            
            # Determine stage
            stage = self.determine_stage(report, positive_nodes)
            
            data.append({
                'tumor_size': tumor_size,
                'differentiation': differentiation,
                'total_lymph_nodes': total_nodes,
                'positive_lymph_nodes': positive_nodes,
                'lymphovascular_invasion': invasion['lymphovascular'],
                'perineural_invasion': invasion['perineural'],
                'stage': stage
            })
            
        return pd.DataFrame(data)
    
    def answer_question(self, question: str, df: pd.DataFrame) -> str:
        """Simple question answering based on the data."""
        question = question.lower()
        
        if 'average' in question and 'size' in question:
            return f"The average tumor size is {df['tumor_size'].mean():.2f} cm"
        elif 'most common' in question and 'stage' in question:
            return f"The most common stage is {df['stage'].mode().iloc[0]}"
        elif 'lymph nodes' in question:
            return f"On average, {df['positive_lymph_nodes'].mean():.1f} lymph nodes are positive out of {df['total_lymph_nodes'].mean():.1f} examined"
        elif 'differentiation' in question:
            return f"The distribution of differentiation is: {df['differentiation'].value_counts().to_dict()}"
        else:
            return "I'm sorry, I don't understand that question. Try asking about tumor size, stage, lymph nodes, or differentiation." 