"""
Response Generation System
Handles LLM-based response generation and refinement
"""
import openai
from typing import Dict, Optional, List
from dataclasses import dataclass
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .student_simulator_config import AppConfig
from .student_persona_manager import StudentPersona
from .logger import get_logger

logger = get_logger(__name__)

@dataclass
class ResponseContext:
    """Context for response generation"""
    question: str
    raw_answer: str
    understanding: str
    emotion: str
    reasoning: str
    persona_code: str

class ResponseGenerator:
    """Handles response generation and refinement"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self._validate_openai_config()
    
    def _validate_openai_config(self) -> None:
        """Validate OpenAI configuration"""
        if not self.config.model.openai_api_key:
            raise ValueError("OpenAI API key not provided in configuration")
        
        openai.api_key = self.config.model.openai_api_key
        logger.info("OpenAI configuration validated")
    
    def generate_refined_response(self, context: ResponseContext) -> str:
        """Generate a refined response based on the context"""
        try:
            prompt = self._build_refinement_prompt(context)
            
            response = openai.ChatCompletion.create(
                model=self.config.model.openai_model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt(context)},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.model.max_tokens,
                temperature=self.config.model.temperature
            )
            
            refined_answer = response.choices[0].message["content"].strip()
            logger.info(f"Generated refined response for persona {context.persona_code}")
            return refined_answer
            
        except Exception as e:
            logger.error(f"Error generating refined response: {e}")
            return context.raw_answer  # Fallback to raw answer
    
    def _build_refinement_prompt(self, context: ResponseContext) -> str:
        """Build the refinement prompt based on context"""
        return f"""User question: {context.question}

Context Information:
- Understanding Level: {context.understanding}
- Emotional State: {context.emotion}
- Reasoning Style: {context.reasoning}

Raw Answer from Dataset: {context.raw_answer}

Please refine this answer to match the student's understanding level, emotional state, and reasoning style. 
The response should sound like it comes from a student with these characteristics."""
    
    def _get_system_prompt(self, context: ResponseContext) -> str:
        """Get system prompt based on student persona"""
        understanding = context.understanding.lower()
        emotion = context.emotion.lower()
        reasoning = context.reasoning.lower()
        
        base_prompt = f"""You are a {understanding} student who is {emotion} and uses {reasoning} reasoning. 
Your responses should reflect your understanding level, emotional state, and reasoning style."""
        
        # Add specific guidance based on understanding level
        if "low" in understanding:
            base_prompt += " You often give incorrect or incomplete answers, showing confusion about concepts."
        elif "medium" in understanding:
            base_prompt += " You have partial understanding and sometimes make mistakes or give incomplete answers."
        else:  # high understanding
            base_prompt += " You have good understanding and can provide accurate answers."
        
        # Add emotional context
        if emotion == "engaged":
            base_prompt += " You are interested and eager to participate."
        elif emotion == "distressed":
            base_prompt += " You are feeling stressed or overwhelmed."
        elif emotion == "fatigued":
            base_prompt += " You are tired and may not be thinking clearly."
        elif emotion == "anxious":
            base_prompt += " You are nervous and uncertain about your answers."
        elif emotion == "bored":
            base_prompt += " You are not very interested and may give short or dismissive answers."
        elif emotion == "fed up":
            base_prompt += " You are frustrated and may be sarcastic or dismissive."
        
        # Add reasoning style context
        if reasoning == "deductive":
            base_prompt += " You tend to use logical reasoning and draw conclusions from general principles."
        elif reasoning == "analogical":
            base_prompt += " You tend to use analogies and comparisons to understand concepts."
        
        return base_prompt
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of the input text"""
        try:
            sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
            return sentiment_scores['compound']
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0
    
    def format_response(self, response: str, sentiment_score: float) -> str:
        """Format the final response with sentiment score"""
        return f"{response}_{sentiment_score:.2f}"
    
    def generate_fallback_response(self, question: str, persona: StudentPersona) -> str:
        """Generate a fallback response when no matching data is found"""
        try:
            fallback_prompts = {
                "low": "I don't really understand this question. Can you explain it differently?",
                "medium": "I think I know something about this, but I'm not sure. Maybe it's related to...",
                "high": "I'm not sure about this specific question, but I know that..."
            }
            
            understanding_level = persona.understanding.lower()
            if "low" in understanding_level:
                base_response = fallback_prompts["low"]
            elif "medium" in understanding_level:
                base_response = fallback_prompts["medium"]
            else:
                base_response = fallback_prompts["high"]
            
            # Add emotional context to fallback
            if persona.emotion == "bored":
                base_response = "I guess... " + base_response.lower()
            elif persona.emotion == "fed up":
                base_response = "Ugh, " + base_response.lower()
            elif persona.emotion == "anxious":
                base_response = "I'm not sure, but... " + base_response.lower()
            
            logger.info(f"Generated fallback response for persona {persona.code}")
            return base_response
            
        except Exception as e:
            logger.error(f"Error generating fallback response: {e}")
            return "I don't know how to answer that."
    
    def validate_response(self, response: str) -> bool:
        """Validate if a response is appropriate"""
        if not response or len(response.strip()) == 0:
            return False
        
        # Check for common error patterns
        error_patterns = ["Error:", "Exception:", "Failed to", "Unable to"]
        if any(pattern in response for pattern in error_patterns):
            return False
        
        return True
