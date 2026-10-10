import os
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

# getting the api key
api_key = os.getenv("GROQ_API_KEY")

# Defining the custom groq wrapper for DeepEval
class GroqEvaluator(DeepEvalBaseLLM):
    def __init__(self, model_name="openai/gpt-oss-120b"):
        self.model = ChatGroq(model=model_name, temperature=0)

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        res = self.model.invoke(prompt)
        return res.content

    async def a_generate(self, prompt: str) -> str:
        res = await self.model.ainvoke(prompt)
        return res.content

    def get_model_name(self) -> str:
        return "Groq Evaluator"


test_case = LLMTestCase(
    input = "Where do penguins live?",
    actual_output="Penguins are found throughout the Southern Hemisphere.  According to the PDF, they live not only on the icy coasts of Antarctica but also in a variety of other southern locales, including the cool shores of South America, the southern coast of Africa, and even the Galapagos Islands near the equator.",
    retrieval_context=["Penguins are flightless seabirds that are specially adapted for life in the Southern Hemisphere. "]
)

groq_model = GroqEvaluator()

metric = ContextualRelevancyMetric(threshold=0.7, model=groq_model)

metric.measure(test_case=test_case)

print("Score: ", metric.score)
print("Reason: ", metric.reason)
print("Passed: ", metric.is_successful())