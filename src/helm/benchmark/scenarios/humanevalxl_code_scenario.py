import io
import json
import os
import sys
from typing import List, Dict, Iterable, Optional, cast

from helm.common.general import ensure_file_downloaded
from helm.common.hierarchical_logger import hlog
from helm.benchmark.scenarios.code_scenario_helper import run as run_reindent
from helm.benchmark.scenarios.code_scenario_apps_pinned_file_order import apps_listdir_with_pinned_order
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)
from datasets import Dataset
from datasets import load_dataset

class CodeReference(Reference):
    # Extra none-string metadata, e.g., paths.
    test_cases: Optional[Dict] = None

    def __init__(self, test_cases=None, **kw):
        self.test_cases = test_cases
        super(CodeReference, self).__init__(**kw)


class CodeInstance(Instance):
    reference: CodeReference

    # Extra none-string metadata, e.g., paths.
    metadata: Optional[Dict] = None

    def __init__(self, metadata=None, **kw):
        self.metadata = metadata
        super(CodeInstance, self).__init__(**kw)


# === HumanEval-XL ===
def _read_and_preprocess_human_eval_xl(
    dataset: Dataset, num_train_instances: int, num_val_instances: int, num_test_instances: int
) -> List[CodeInstance]:
    problems = load_dataset("FloatAI/humaneval-xl", "python")
    dataset = problems["Portuguese"]
    instances = []
    for sample_idx, task_id in enumerate(dataset):
        if sample_idx < num_train_instances:
            split = TRAIN_SPLIT
        elif sample_idx < num_train_instances + num_val_instances:
            split = VALID_SPLIT
        else:
            split = TEST_SPLIT

        instance = CodeInstance(
            input=Input(text=dataset[task_id]["prompt"]),
            references=[
                CodeReference(
                    output=Output(text=dataset[task_id]["canonical_solution"]),
                    test_cases=dataset[task_id],
                    tags=[CORRECT_TAG],
                )
            ],
            split=split,
        )
        instances.append(instance)
    return instances

class HumanEvalXLCodeScenario(Scenario):
    name = "humanevalxlcode"
    description = "Code Generation"
    tags = ["Reasoning", "Code Generation"]

    def __init__(self, dataset: str):
        super().__init__()
        self.dataset = dataset

        self.human_eval_hparams = dict(num_train_instances=0, num_val_instances=0, num_test_instances=164)

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset: any
        dataset = load_dataset("FloatAI/humaneval-xl", "python")
        dataset = dataset["Portuguese"]
        instances = _read_and_preprocess_human_eval_xl(dataset=dataset, **self.human_eval_hparams)
        
        return instances