from agentlightning.types import Dataset

TRAIN_DATA: Dataset[dict[str,str]] = [
    {
        "question": "What is the capital of France?",
        "answer": "Paris",
    },
    {
        "question": "Who wrote 'To Kill a Mockingbird'?",
        "answer": "Harper Lee",
    }
] * 20
