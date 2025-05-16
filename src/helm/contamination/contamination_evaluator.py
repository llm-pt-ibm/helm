from helm.contamination.ts_guessing_question_based import TSGuessingQuestionBasedContaminationEvaluator
from helm.contamination.ts_guessing_question_multichoice import TSGuessingQuestionMultiChoiceContaminationEvaluator
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.common.hierarchical_logger import hlog


class ContaminationEvaluator:
    """
    Class responsible for evaluating contamination using different strategies
    based on the specified method.
    """

    def evaluate(self, executor, method: str, benchmark_path: str, scenario_state, language: str, tokenizer_service) -> list[dict]:
        """
        Evaluate contamination using the specified method.

        Args:
            method: The contamination evaluation method to use.
            benchmark_path: Path to the benchmark data.
            scenario_state: The current scenario state.
            language: defines the prompt language.

        Returns:
            List containing contamination evaluation statistics.
        """
        # Select the appropriate evaluator based on the method
        if method == "ts_guessing_question_base":
            evaluator = TSGuessingQuestionBasedContaminationEvaluator()
        elif method == "ts_guessing_question_multichoice":
            evaluator = TSGuessingQuestionMultiChoiceContaminationEvaluator()
        else:
            hlog(f"Unknown contamination evaluation method: {method}")
            return []

        # Run the selected evaluator
        return evaluator.evaluate(
            executor=executor, 
            benchmark_path=benchmark_path, 
            scenario_state=scenario_state, 
            language=language,
            tokenizer_service=tokenizer_service
        )
