import pytest
from tempfile import TemporaryDirectory
from helm.benchmark.scenarios.contamination.contamination_scenario import ContaminationScenario
from helm.benchmark.scenarios.scenario import TEST_SPLIT
from helm.benchmark.scenarios.bluex_scenario import BLUEXScenario


@pytest.mark.scenarios
def test_contamination_scenario_bluex_ts_guessing_multichoice():
    scenario = ContaminationScenario(dataset="bluex", strategy="ts_guessing_question_multichoice", language="pt")

    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) > 100
    assert instances[100].split == TEST_SPLIT
    assert instances[0].input.text is not None
    assert len(instances[0].input.text) > 0
    assert len(instances[0].references) > 0

    correct_refs = [ref for ref in instances[0].references if ref.is_correct]
    assert len(correct_refs) > 0


@pytest.mark.scenarios
def test_contamination_scenario_bluex_ts_guessing_base():
    scenario = ContaminationScenario(dataset="bluex", strategy="ts_guessing_question_base", language="pt")

    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) > 100
    assert instances[100].split == TEST_SPLIT
    assert instances[0].input.text is not None
    assert len(instances[0].input.text) > 0
    assert len(instances[0].references) > 0


@pytest.mark.scenarios
def test_contamination_scenario_with_dataset_params():
    scenario = ContaminationScenario(dataset="bluex", strategy="ts_guessing_question_multichoice", language="pt")

    function_name, params = scenario._parse_dataset_string(scenario.dataset)
    assert function_name == "bluex"
    assert params == {}


@pytest.mark.scenarios
def test_contamination_scenario_invalid_dataset():
    """Test contamination scenario with invalid dataset name."""
    scenario = ContaminationScenario(
        dataset="nonexistent_dataset", strategy="ts_guessing_question_multichoice", language="pt"
    )

    with TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Unknown dataset function"):
            scenario.get_instances(tmpdir)


@pytest.mark.scenarios
def test_contamination_scenario_invalid_strategy():
    """Test contamination scenario with invalid strategy name."""
    scenario = ContaminationScenario(dataset="bluex", strategy="nonexistent_strategy", language="pt")

    with TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Unknown contamination strategy"):
            scenario.get_instances(tmpdir)


@pytest.mark.scenarios
def test_contamination_scenario_preserves_instance_count():
    """Test that contamination preserves the number of instances."""
    scenario = ContaminationScenario(dataset="bluex", strategy="ts_guessing_question_multichoice", language="pt")

    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)
        original_scenario = BLUEXScenario()
        original_instances = original_scenario.get_instances(tmpdir)

        assert len(instances) == len(original_instances)
