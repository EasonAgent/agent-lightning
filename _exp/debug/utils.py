from utu.agents import LLMAgent
from utu.utils import LLMOutputParser
from utu.config import ConfigLoader

# -----------------------------------------------------------------------------------------------------------
LLM = LLMAgent(ConfigLoader.load_model_config("base"))
TEMPLATE = """Please grade the following answer based on the question.

question: {question}
ground truth answer: {gt}
student's answer: {pred}

Please give a score from 0 to 1, where 1 means the model answer is completely correct, and 0 means it is completely incorrect.
Only output the score as a float number, e.g. 0.75, 1.0, 0.0
"""

async def evaluate(question: str, gt: str, pred: str) -> float:
    prompt = TEMPLATE.format(question=question, gt=gt, pred=pred)
    result = await LLM.run(prompt)
    try:
        score = LLMOutputParser.extract_float_number(result.final_output.strip()) or 0.0
        return max(0.0, min(1.0, score))
    except ValueError:
        return 0.0

if __name__ == '__main__':
    import asyncio

    async def test():
        res = await evaluate(
            question="What is the capital of France?",
            gt="Paris",
            pred="The capital of France is Paris.",
        )
        print("Evaluation score:", res)
    asyncio.run(test())
