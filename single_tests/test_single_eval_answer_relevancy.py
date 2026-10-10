import os
from deepeval.metrics import AnswerRelevancyMetric
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

# Create the test case
test_case = LLMTestCase(
    input="What do penguins eat?",
    actual_output=("Penguins primarily feed on marine animals such as krill and tiny fishes")
)

# Instantiate the custom Groq model
groq_model = GroqEvaluator()

# Passing the model to the metric
metric = AnswerRelevancyMetric(threshold=0.7, model=groq_model)

# Executing the evaluation
metric.measure(test_case)

print("Score: ", metric.score)
print("Reason: ", metric.reason)
print("Passed: ", metric.is_successful())