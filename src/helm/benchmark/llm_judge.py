import json
import os
from typing import List, Dict, Any

from helm.common.request import Request
from helm.common.authentication import Authentication

class LLMJudger:
    def __init__(self, executor_service, judge_model: str = "openai/gpt2"):
        """
        executor_service: self.executor.service vindo do Runner
        """
        self.executor_service = executor_service
        self.judge_model = judge_model

    # def judge_predictions(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    #     judgements = []

    #     for prediction in predictions:
    #         input_text = prediction.get("input", "")
    #         model_response = prediction.get("prediction", "")

    #         prompt = (
    #             f"Você é um avaliador. Leia o texto de entrada e a resposta gerada.\n\n"
    #             f"Entrada: {input_text}\n\n"
    #             f"Resposta: {model_response}\n\n"
    #             f"Se a resposta está adequada e correta, responda apenas com '1'. "
    #             f"Se estiver errada ou inadequada, responda apenas com '0'."
    #         )

    #         # Chamada ao modelo julgador
    #         judged_value = self.call_llm(prompt)

    #         judgements.append({
    #             "instance_id": prediction.get("instance_id"),
    #             "input": input_text,
    #             "prediction": model_response,
    #             "judgement": judged_value,
    #         })

    #     return judgements

    def judge_predictions(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        judgements = []

        for prediction in predictions:
            input_text = prediction.get("input", "")
            model_response = prediction.get("prediction", "")

            prompt = (
                f"Você é um avaliador. Leia o texto de entrada e a resposta gerada.\n\n"
                f"Entrada: {input_text}\n\n"
                f"Resposta: {model_response}\n\n"
                f"Responda:\n"
                f"- '1 - (explique brevemente porque a resposta está correta)' se a resposta estiver correta;\n"
                f"- '0 - (explique brevemente porque a resposta está incorreta)' se a resposta estiver errada.\n"
                f"Explique em apenas uma frase."
            )


            # Chamada ao modelo julgador
            judged_value, explanation = self.call_llm(prompt)

            judgements.append({
                "instance_id": prediction.get("instance_id"),
                "input": input_text,
                "prediction": model_response,
                "judgement": judged_value,
                "explanation": explanation,
            })

        return judgements

    def call_llm(self, prompt: str) -> (int, str):
        from helm.common.request import Request

        request = Request(
            model=self.judge_model,
            model_deployment=self._resolve_model_deployment(),  
            prompt=prompt,
            temperature=0.7,
            max_tokens=50,  # suporta resposta + explicação
        )

        result = self.executor_service.make_request(Authentication(""), request)
        print(f"Resultado: {result}")

        if not result.success:
            raise Exception(f"LLM Judge request failed: {result.error}")

        if result.completions:
            text = result.completions[0].text.strip()

            # Agora queremos 0/1 + explicação
            if text.startswith("1"):
                judgement = 1
            elif text.startswith("0"):
                judgement = 0
            else:
                judgement = 0  # fallback se vier algo estranho

            parts = text.split("-", 1)  # separa no hífen
            explanation = parts[1].strip() if len(parts) > 1 else ""

            return judgement, explanation

        return 0, ""


    # def call_llm(self, prompt: str) -> int:
    #     """
    #     Faz uma chamada ao modelo para obter 0 ou 1 como resposta.
    #     """
    #     # Cria uma Request
    #     request = Request(
    #         model= self.judge_model,
    #         model_deployment="huggingface/gpt2",  # Pode deixar um nome simbólico aqui por enquanto
    #         prompt=prompt,
    #         temperature=0.0,
    #         max_tokens=100,
    #     )

    #     # Autenticação vazia, porque localmente já está autenticado
    #     result = self.executor_service.make_request(Authentication(""), request)

    #     if not result.success:
    #         raise Exception(f"LLM Judge request failed: {result.error}")

    #     if result.completions:
    #         text = result.completions[0].text.strip()
    #         if "1" in text:
    #             return 1
    #         else:
    #             return 0

    #     return 0  # fallback conservador

    def judge_and_save(self, predictions_path: str, output_path: str):
        """
        Lê predictions.json, aplica julgamento e salva llm_judgements.json.
        """
        with open(predictions_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        judgements = self.judge_predictions(predictions)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(judgements, f, indent=2)

        print(f"Julgmentos salvos em: {output_path}")

    def _resolve_model_deployment(self) -> str:
        print("CHEGUEI AQUI NO MODEL DEPLOYMENT====================================================================")

        from helm.benchmark.model_deployment_registry import get_default_model_deployment_for_model
    
        deployment_name = get_default_model_deployment_for_model(self.judge_model)
        if not deployment_name:
            raise Exception(f"Não foi possível encontrar um model deployment para '{self.judge_model}'")
        return deployment_name
