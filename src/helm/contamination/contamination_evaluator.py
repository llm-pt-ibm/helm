from helm.benchmark.metrics.metric import MetricResult
from helm.contamination.ts_guessing_question_based import TSGuessingQuestionBasedContaminationEvaluator


class ContaminationEvaluator:
    """
    Class responsible for evaluating contamination using different strategies
    based on the specified method.
    """
    
    def evaluate(
        self,
        executor,
        method: str,
        model_path: str,
        benchmark_path: str,
        scenario_state,
        metric_service,
        eval_cache_path: str,
        parallelism: int
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
        # Add more evaluators as needed
        else:
            raise ValueError(f"Unknown contamination evaluation method: {method}")
        
        # Run the selected evaluator
        return evaluator.evaluate(
            ex=executor,
            model_path=model_path,
            benchmark_path=benchmark_path,
            scenario_state=scenario_state,
            metric_service=metric_service,
            eval_cache_path=eval_cache_path,
            parallelism=parallelism
        )