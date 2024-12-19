from typing import Dict, List
from datasets import load_dataset

from helm.common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output


class MMLUProScenario(Scenario):
    """
    The MMLU-Pro dataset is an advanced version of the Massive Multitask Language Understanding (MMLU)
    benchmark, created to push the boundaries of language models' reasoning and comprehension skills.
    Designed as a more challenging evaluation, it increases the answer options per question from four
    to ten, significantly reducing the likelihood of correct random guesses. This update makes the
    dataset better at distinguishing the capabilities of models on complex tasks.

    MMLU-Pro emphasizes reasoning over simple factual recall by integrating diverse, intricate questions
    across 14 domains, including subjects like biology, economics, law, and psychology. In addition, it
    addresses limitations in the original MMLU by filtering out trivial questions, making it a more
    robust benchmark. Performance comparisons suggest that models benefit from reasoning-based
    approaches (such as Chain of Thought, or CoT) on MMLU-Pro, which contrasts with the original
    MMLU where CoT didn’t show as much benefit. This makes MMLU-Pro especially suitable for evaluating
    advanced models that rely on nuanced reasoning and comprehension skills​.

    Dataset: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
    Paper: https://arxiv.org/abs/2406.01574
    """

    name = "mmlu_pro"
    description = "Enhanced Massive Multitask Language Understanding with increased options and reasoning"
    tags = ["knowledge", "multiple_choice", "reasoning"]

    def __init__(self, subject: str):
        super().__init__()
        self.subject: str = subject

    def process_csv(self, data, split: str) -> List[Instance]:
        instances: List[Instance] = []
        hlog(f"Processing data for {split} split")
        for row in data:
            question = row["question"]
            answers = row["options"][:10]  # Limit to 10 answers if necessary
            correct_choice = row["answer"]
            answers_dict = dict(zip(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"], answers))
            correct_answer = answers_dict[correct_choice]

            def answer_to_reference(answer: str) -> Reference:
                return Reference(Output(text=answer), tags=[CORRECT_TAG] if answer == correct_answer else [])

            instance = Instance(
                input=Input(text=question),
                references=list(map(answer_to_reference, answers)),
                split=split,
            )
            instances.append(instance)
        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        # Load the MMLU-Pro dataset from Hugging Face
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")

        # Process all the instances
        instances: List[Instance] = []
        splits: Dict[str, str] = {
            "validation": TRAIN_SPLIT,
            "test": TEST_SPLIT,
        }
        for hf_split, split in splits.items():
            data = dataset[hf_split].filter(lambda x: x["category"] == self.subject)
            print(f"Filtered instances in {hf_split}: {len(data)}")
            instances.extend(self.process_csv(data, split))

        return instances