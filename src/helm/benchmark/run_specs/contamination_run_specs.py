from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.benchmark.adaptation.common_adapter_specs import get_generation_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs, get_basic_metric_specs


@run_spec_function("contamination")
def get_contamination_spec(dataset: str, strategy: str, language: str) -> RunSpec:
    valid_strategies = {"ts_guessing_question_base", "ts_guessing_question_multichoice"}
    if strategy not in valid_strategies:
        raise ValueError(f"Unknown strategy '{strategy}'. Valid: {sorted(valid_strategies)}")

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.contamination.contamination_scenario.ContaminationScenario",
        args={"dataset": dataset, "strategy": strategy, "language": language},
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="",
        input_noun="Input",
        output_noun="Result",
        max_tokens=100,
        temperature=0.0,
        stop_sequences=["\n"],
    )

    if strategy == "ts_guessing_question_base":
        metric_specs = get_exact_match_metric_specs()
    elif strategy == "ts_guessing_question_multichoice":
        metric_specs = get_basic_metric_specs(
            [
                "exact_match",
                "quasi_exact_match",
                "prefix_exact_match",
                "quasi_prefix_exact_match",
                "rouge_l",
            ]
        )
    else:
        metric_specs = get_exact_match_metric_specs()

    return RunSpec(
        name=f"contamination:dataset={dataset},strategy={strategy},language={language}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["contamination"],
    )
