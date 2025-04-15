from helm.benchmark.metrics.metric import MetricResult
from helm.contamination.ts_guessing_question_based import TSGuessingQuestionBasedContaminationEvaluator
from helm.contamination.ts_guessing_question_multichoice import TSGuessingQuestionMultiChoiceContaminationEvaluator


class ContaminationEvaluator:
    """
    Class responsible for evaluating contamination using different strategies
    based on the specified method.
    """
    
    def evaluate(
        self,
        executor,
        method: str,
        benchmark_path: str,
        scenario_state,
        language: str
    ) -> MetricResult:
        """
        Evaluate contamination using the specified method.
        
        Args:
            method: The contamination evaluation method to use.
            model_path: Path to the model.
            benchmark_path: Path to the benchmark data.
            scenario_state: The current scenario state.
            metric_service: Service for computing metrics.
            eval_cache_path: Path for caching evaluation results.
            parallelism: Number of parallel workers.
            
        Returns:
            MetricResult containing contamination evaluation statistics.
        """
        # Select the appropriate evaluator based on the method
        if method == "ts_guessing_question_base":
            evaluator = TSGuessingQuestionBasedContaminationEvaluator()
        if method == "ts_guessing_question_multichoice":
            evaluator = TSGuessingQuestionMultiChoiceContaminationEvaluator()
        # Add more evaluators as needed
        else:
            raise ValueError(f"Unknown contamination evaluation method: {method}")
        
        # Run the selected evaluator
        return evaluator.evaluate(
            executor=executor,
            benchmark_path=benchmark_path,
            scenario_state=scenario_state,
            language=language
        )