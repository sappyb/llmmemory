from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
class NoOpLLMChain(LLMChain):
   """No-op LLM chain."""
   def __init__(self):
       """Initialize."""
       super().__init__(llm=ChatOpenAI(), prompt=PromptTemplate(template="", input_variables=[]))

   def run(self, question: str, *args, **kwargs) -> str:
       return question