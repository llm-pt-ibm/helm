from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.adaptation.common_adapter_specs import get_multiple_choice_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec

@run_spec_function("hellaswag_pt")
def get_hellaswag_pt_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.hella_swag_pt_scenario.HellaSwag_PT_Scenario",args={}
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions=(
            "Escolha a alternativa que continua o contexto de forma mais plausível. "
            "Responda apenas com a letra da alternativa correta (A, B, C ou D).\n\n"
            "Exemplo:\n"
            "Contexto: Ao terminar de pintar uma pequena caixa de madeira, "
            "Lívia percebe que a tinta ainda está fresca. "
            "Para evitar manchas, ela\n"
            "A. enrola a caixa imediatamente em uma toalha.\n"
            "B. esfrega a superfície pintada com as mãos.\n"
            "C. deixa a caixa em um local protegido até a tinta secar.\n"
            "D. coloca livros sobre a superfície recém-pintada.\n"
            "Resposta: C\n\n"
            "Agora responda à próxima questão."
        ),
        input_noun="Contexto",
        output_noun="Resposta",
    )

    return RunSpec(
        name="hellaswag_pt",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["hellaswag_pt"],
    )