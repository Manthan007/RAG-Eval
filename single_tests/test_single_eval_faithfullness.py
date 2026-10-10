import os
from deepeval.metrics import FaithfulnessMetric
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
    retrieval_context=["Penguins are flightless seabirds that are specially adapted for life in the Southern Hemisphere. \nAlthough people often associate them only with Antarctica, penguins live in a wide range of \nenvironments, including the icy coasts of Antarctica, the cool shores of South America and \nsouthern Africa, and even the Galapagos Islands near the equator. Their upright posture, \ncompact bodies, and distinctive black-and-white coloring make them among the most \nrecognizable birds in the world.", "Penguins are also well insulated against cold water. Many species have a thick layer of fat \nbeneath the skin and densely packed feathers that trap air close to the body. Penguins regularly \npreen their feathers to keep them waterproof and properly aligned. This combination of fat, \nfeathers, and trapped air helps them conserve heat while spending long periods swimming in \nchilly seas.", "Breeding behavior is another fascinating part of penguin life. Many species gather in large \ncolonies where thousands of birds may nest close together. Courtship often involves calls, \npostures, and repeated movements that help partners recognize one another. Depending on the \nspecies, nests may be made from pebbles, grass, or shallow scrapes, while emperor penguins \nfamously balance a single egg on their feet beneath a warm fold of skin."]
)

groq_model = GroqEvaluator()

metric = FaithfulnessMetric(threshold=0.7, model=groq_model)

metric.measure(test_case=test_case)

print("Score: ", metric.score)
print("Reason: ", metric.reason)
print("Passed: ", metric.is_successful())
