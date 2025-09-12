"""
Student Persona Management System
Handles student context mapping and persona definitions
"""
from typing import Dict, Optional, List
from dataclasses import dataclass
from .logger import get_logger

logger = get_logger(__name__)

@dataclass
class StudentPersona:
    """Represents a student persona with understanding, emotion, and reasoning"""
    understanding: str
    emotion: str
    reasoning: str
    code: str
    description: str = ""

class StudentPersonaManager:
    """Manages student personas and context mapping"""
    
    def __init__(self):
        self.personas = self._initialize_personas()
        self.default_persona = StudentPersona(
            understanding="High understanding",
            emotion="engaged",
            reasoning="deductive",
            code="HED",
            description="Default engaged high-achieving student"
        )
    
    def _initialize_personas(self) -> Dict[str, StudentPersona]:
        """Initialize all available student personas"""
        personas = {}
        
        # Define all possible combinations
        understanding_levels = ["High", "Medium", "Low"]
        emotions = ["engaged", "distressed", "fatigued", "anxious", "bored", "fed up"]
        reasoning_styles = ["deductive", "analogical"]
        
        for understanding in understanding_levels:
            for emotion in emotions:
                for reasoning in reasoning_styles:
                    # Create code (e.g., HED = High understanding + Engaged + Deductive)
                    code = f"{understanding[0]}{emotion[0].upper()}{reasoning[0].upper()}"
                    
                    # Handle special cases for fed up (FED)
                    if emotion == "fed up":
                        code = f"{understanding[0]}FED{reasoning[0].upper()}"
                    
                    persona = StudentPersona(
                        understanding=f"{understanding} understanding",
                        emotion=emotion,
                        reasoning=reasoning,
                        code=code,
                        description=f"{understanding} understanding, {emotion}, {reasoning} reasoning"
                    )
                    personas[code] = persona
        
        logger.info(f"Initialized {len(personas)} student personas")
        return personas
    
    def get_persona(self, code: str) -> Optional[StudentPersona]:
        """Get student persona by code"""
        code = code.upper()
        persona = self.personas.get(code)
        
        if persona:
            logger.debug(f"Retrieved persona: {code} - {persona.description}")
        else:
            logger.warning(f"Unknown persona code: {code}, using default")
            persona = self.default_persona
        
        return persona
    
    def get_context(self, code: str) -> Dict[str, str]:
        """Get context dictionary for a persona code"""
        persona = self.get_persona(code)
        return {
            "Understanding": persona.understanding,
            "Emotion": persona.emotion,
            "Reasoning": persona.reasoning
        }
    
    def list_personas(self) -> List[StudentPersona]:
        """Get list of all available personas"""
        return list(self.personas.values())
    
    def get_personas_by_understanding(self, understanding: str) -> List[StudentPersona]:
        """Get personas by understanding level"""
        return [p for p in self.personas.values() if understanding.lower() in p.understanding.lower()]
    
    def get_personas_by_emotion(self, emotion: str) -> List[StudentPersona]:
        """Get personas by emotion"""
        return [p for p in self.personas.values() if emotion.lower() in p.emotion.lower()]
    
    def validate_code(self, code: str) -> bool:
        """Validate if a persona code exists"""
        return code.upper() in self.personas
    
    def get_available_codes(self) -> List[str]:
        """Get list of all available persona codes"""
        return list(self.personas.keys())
    
    def search_personas(self, **criteria) -> List[StudentPersona]:
        """Search personas by criteria"""
        results = []
        for persona in self.personas.values():
            match = True
            for key, value in criteria.items():
                if hasattr(persona, key):
                    if value.lower() not in getattr(persona, key).lower():
                        match = False
                        break
                else:
                    match = False
                    break
            if match:
                results.append(persona)
        return results
