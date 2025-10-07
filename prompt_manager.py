from typing import Dict, Any, Optional
from prompt import rec_ent_prompt, rel_score_prompt, create_triple_prompt, datasets_prompt, confidence_judgment_prompt, Temporary_prompt, insufficient_info_prompt, insufficient_info_prompt_pub

class PromptManager:
    def __init__(self, dataset_name: str, sub_type: Optional[str] = None):
        """
        Initialize PromptManager
        
        :param dataset_name: Name of the dataset
        :param sub_type: Subtype (e.g., social science for scienceQA)
        """
        self.dataset_name = dataset_name
        self.sub_type = sub_type
        
    def get_entity_recognition_prompt(self) -> Dict[str, str]:
        """Get entity recognition prompt"""
        if self.dataset_name == "scienceQA":
            return rec_ent_prompt[self.dataset_name][self.sub_type]
        return rec_ent_prompt[self.dataset_name]
    
    def get_relation_scoring_prompt(self) -> Dict[str, str]:
        """Get relation scoring prompt"""
        if self.dataset_name == "scienceQA":
            return rel_score_prompt[self.dataset_name][self.sub_type]
        return rel_score_prompt[self.dataset_name]
    
    def get_triple_creation_prompt(self) -> Dict[str, str]:
        """Get triple creation prompt"""
        if self.dataset_name == "scienceQA":
            return create_triple_prompt[self.dataset_name][self.sub_type]
        return create_triple_prompt[self.dataset_name]
    
    def get_dataset_prompt(self) -> Dict[str, str]:
        """Get dataset-specific prompt"""
        if self.dataset_name == "scienceQA":
            return datasets_prompt[self.dataset_name][self.sub_type]
        return datasets_prompt[self.dataset_name]
    
    def format_entity_prompt(self, question: str) -> str:
        """Format entity recognition prompt"""
        return self.get_entity_recognition_prompt()["request"].format(question)
    
    def format_relation_prompt(self, question: str, retrieve_result: str, choice: Optional[str] = None) -> str:
        """Format relation scoring prompt"""
        if choice is not None:
            return self.get_relation_scoring_prompt()["request"].format(q=question, c=choice, t=str(retrieve_result))
        return self.get_relation_scoring_prompt()["request"].format(q=question, t=str(retrieve_result))
    
    def format_triple_prompt(self, question: str, choices: list, answer: str) -> str:
        """Format triple creation prompt"""
        if self.dataset_name == "medmc_qa_multi":
            return self.get_triple_creation_prompt()["request"].format(question, choices, answer)
        return self.get_triple_creation_prompt()["request"].format(question, answer)
    
    def format_dataset_prompt(self, question: str, knowledge: str, choices: Optional[str] = None) -> str:
        """Format dataset-specific prompt"""
        if choices is not None:
            return self.get_dataset_prompt()["request"].format(q=question, i=knowledge, c=choices)
        return self.get_dataset_prompt()["request"].format(q=question, i=knowledge)
    
    def get_temporary_prompt(self) -> Dict[str, str]:
        """Get temporary prompt"""
        if self.dataset_name == "scienceQA":
            return Temporary_prompt[self.dataset_name][self.sub_type]
        return Temporary_prompt[self.dataset_name]
    
    def format_temporary_prompt(self, question: str, choices: Optional[str] = None) -> str:
        """Format temporary prompt"""
        if choices is not None:
            return self.get_temporary_prompt()["request"].format(q=question, c=choices)
        return self.get_temporary_prompt()["request"].format(q=question)
    
    def get_confidence_judgment_prompt(self) -> Dict[str, str]:
        """Get confidence judgment prompt"""
        if self.dataset_name == "scienceQA":
            return confidence_judgment_prompt[self.dataset_name][self.sub_type]
        return confidence_judgment_prompt[self.dataset_name]
    
    def format_confidence_judgment_prompt(self, question: str, choices: Optional[str] = None, knowledge: Optional[str] = None) -> str:
        """Format confidence judgment prompt"""
        if choices is not None:
            return self.get_confidence_judgment_prompt()["request"].format(q=question, c=choices, i=knowledge)
        return self.get_confidence_judgment_prompt()["request"].format(q=question, i=knowledge)
    
    def get_insufficient_info_prompt(self) -> Dict[str, str]:
        """Get insufficient information judgment prompt"""
        if self.dataset_name == "pubmed_qa":
            return insufficient_info_prompt_pub
        return insufficient_info_prompt