import os
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
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
    input="Where do penguins live?",
    actual_output="Penguins are found throughout the Southern Hemisphere.  According to the PDF, they live not only on the icy coasts of Antarctica but also in a variety of other southern locales, including the cool shores of South America, the southern coast of Africa, and even the Galapagos Islands near the equator.",
    expected_output="Penguins live in the Southern Hemisphere, including Antarctica, South America, southern Africa, and the Galapagos Islands."
)

groq_model = GroqEvaluator()

metric = GEval(
    name="Correctness",
    criteria=(
        "Determine whether the acutal output is factually correct based on the expect output"
    ),
    evaluation_params=[
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT
    ],
    model=groq_model
)

metric.measure(test_case=test_case)

print("Score: ", metric.score)
print("Reason: ", metric.reason)
