# helm/benchmark/llm_judger/llm_judger.py

import json
import os
from typing import List, Dict, Any

class LLMJudger:
    def __init__(self, executor_service):
        """
        executor_service: geralmente é self.executor.service vindo do Runner,
        usado para fazer chamadas ao modelo.
        """
        self.executor_service = executor_service

    def judge_predictions(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        judgements = []

        for prediction in predictions:
            input_text = prediction.get("input", "")
            model_response = prediction.get("prediction", "")

            prompt = (
                f"Você é um avaliador. Leia o texto de entrada e a resposta gerada.\n\n"
                f"Entrada: {input_text}\n\n"
                f"Resposta: {model_response}\n\n"
                f"Se a resposta está adequada e correta, responda apenas com '1'. "
                f"Se estiver errada ou inadequada, responda apenas com '0'."
            )

            # Aqui fazemos a chamada para o modelo (usando o executor já configurado)
            result = self.executor_service.make_request(prompt)

            # Limpar a resposta para extrair só '0' ou '1'
            judged_value = self._extract_binary_result(result)

            judgements.append({
                "instance_id": prediction.get("instance_id"),
                "input": input_text,
                "prediction": model_response,
                "judgement": judged_value,
            })

        return judgements

    def _extract_binary_result(self, result: str) -> int:
        """
        Trata a resposta retornada do modelo para garantir que seja 0 ou 1.
        """
        result = result.strip()
        if "1" in result:
            return 1
        elif "0" in result:
            return 0
        else:
            return 0  # fallback conservador

    def judge_and_save(self, predictions_path: str, output_path: str):
        """
        Lê o arquivo de predições, julga e salva o arquivo de julgamentos.
        """
        with open(predictions_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        judgements = self.judge_predictions(predictions)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(judgements, f, indent=2)

        print(f"Julgmentos salvos em: {output_path}")
