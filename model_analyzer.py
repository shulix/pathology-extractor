import pandas as pd
import re
from typing import List
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer

class PathologyReportAnalyzer:
    def __init__(self, reports: List[str]):
        self.reports = reports
        # Load pre-trained BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
        self.model = AutoModelForQuestionAnswering.from_pretrained('dmis-lab/biobert-v1.1')
        self.nlp = pipeline('question-answering', model=self.model, tokenizer=self.tokenizer)
        
    def extract_number(self, text: str) -> float:
        """Extract number from text."""
        match = re.search(r'(\d+\.?\d*)', text)
        return float(match.group(1)) if match else None
    
    def analyze_text_with_bert(self, text: str, aspect: str) -> str:
        """Use BERT to analyze specific aspects of the report."""
        questions = {
            'differentiation': 'What is the differentiation status of the tumor?',
            'invasion': 'Is there any invasion present?',
            'stage': 'What characteristics indicate the cancer stage?'
        }
        
        if aspect in questions:
            result = self.nlp(question=questions[aspect], context=text)
            return result['answer']
        return None

    def get_differentiation(self, text: str) -> str:
        """Extract tumor differentiation using BERT."""
        bert_result = self.analyze_text_with_bert(text, 'differentiation')
        if 'well' in bert_result.lower():
            return 'Well'
        elif 'poor' in bert_result.lower():
            return 'Poor'
        elif 'moderate' in bert_result.lower():
            return 'Moderate'
        return 'Unknown'
    
    def get_invasion_status(self, text: str) -> dict:
        """Extract invasion information using BERT."""
        bert_result = self.analyze_text_with_bert(text, 'invasion')
        return {
            'lymphovascular': 'lymphovascular' in bert_result.lower(),
            'perineural': 'perineural' in bert_result.lower()
        }
    
    def determine_stage(self, tumor_depth: str, positive_nodes: int) -> str:
        """Determine cancer stage based on BERT analysis and node status."""
        bert_result = self.analyze_text_with_bert(tumor_depth, 'stage')
        
        # Determine T stage
        if 'submucosa' in bert_result.lower():
            t_stage = 1
        elif 'muscularis propria' in bert_result.lower():
            t_stage = 2
        elif 'pericolonic' in bert_result.lower() or 'subsero' in bert_result.lower():
            t_stage = 3
        elif 'adjacent' in bert_result.lower():
            t_stage = 4
        else:
            t_stage = 0
            
        # Determine N stage
        if positive_nodes == 0:
            n_stage = 0
        elif 1 <= positive_nodes <= 3:
            n_stage = 1
        else:
            n_stage = 2
            
        # Determine final stage
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
        """Convert reports to structured DataFrame using BERT analysis."""
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
            
            # Get differentiation using BERT
            differentiation = self.get_differentiation(report)
            
            # Get invasion status using BERT
            invasion = self.get_invasion_status(report)
            
            # Determine stage using BERT
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
        """Use BERT for more sophisticated question answering."""
        # Combine all reports into a single context
        context = " ".join(self.reports)
        
        try:
            # First try to get answer from BERT
            bert_answer = self.nlp(question=question, context=context)
            
            # If confidence is low, fall back to statistical analysis
            if bert_answer['score'] < 0.5:
                return self.statistical_answer(question, df)
                
            return bert_answer['answer']
            
        except:
            return self.statistical_answer(question, df)
    
    def statistical_answer(self, question: str, df: pd.DataFrame) -> str:
        """Fallback statistical analysis for questions."""
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