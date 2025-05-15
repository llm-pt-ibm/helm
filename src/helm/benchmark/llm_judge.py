import json
import os
import re
from typing import List, Dict, Any

from helm.common.request import Request
from helm.common.authentication import Authentication
from helm.benchmark.model_deployment_registry import get_default_model_deployment_for_model

from helm.common.request import Request
from helm.common.authentication import Authentication

class LLMJudger:
    def __init__(self, executor_service, judge_model: str = "openai/gpt2", prompt_file: str = "default_prompt.txt"):
        """
        executor_service: self.executor.service vindo do Runner
        """
        self.executor_service = executor_service
        self.judge_model = judge_model
        self.prompt_file = prompt_file

    def judge_predictions(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        judgements = []

        for prediction in predictions:
            input_text = prediction.get("input", "")
            model_response = prediction.get("prediction", "")

            # Criação do prompt para o modelo julgador
            # prompt = (
            #     "Você é um avaliador rigoroso de respostas geradas por modelos de linguagem. "
            #     "Sua tarefa é avaliar, de forma objetiva, se a resposta fornecida está correta em relação ao texto de entrada. "
            #     "Corretude significa o quanto a resposta condiz com a intenção e o conteúdo do texto fornecido como entrada.\n\n"
            #     f"Entrada:\n{input_text}\n\n"
            #     f"Resposta do modelo:\n{model_response}\n\n"
            #     "Agora, com base na entrada e na resposta do modelo, retorne sua avaliação **estritamente** no seguinte formato JSON:\n\n"
            #     "{\n"
            #     '  "judgement": 1(quando concorda com o modelo principal) ou 0(quando discorda do modelo principal),\n'
            #     '  "explanation": "explicação do porquê a resposta está correta ou incorreta"\n'
            #     "}\n\n"
            #     "Importante: não adicione nenhum comentário, rótulo, explicação fora do JSON. Apenas imprima esse JSON diretamente como resposta."
            # )

            prompt_tamplate = self._load_prompt_template(self.prompt_file)
            prompt = prompt_tamplate.replace("{input}", input_text).replace("{response}", model_response)

            print(f"==============Prompt: {prompt}====================")

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
    
    def call_llm(self, prompt: str) -> tuple[int, str]:
        

        request = Request(
            model=self.judge_model,
            model_deployment=self._resolve_model_deployment(),
            prompt=prompt,
            temperature=0.7,
            max_tokens=300,
        )

        result = self.executor_service.make_request(Authentication(""), request)
        print(f"Result: {result}")

        if not result.success:
            raise Exception(f"LLM Judge request failed: {result.error}")

        if result.completions:
            text = result.completions[0].text.strip()

            # Extract fields from JSON-like output using regular expressions
            judgement_match = re.search(r'"judgement"\s*:\s*(\d)', text)
            explanation_match = re.search(r'"explanation"\s*:\s*"(.+?)"', text, re.DOTALL)

            if judgement_match and explanation_match:
                judgement = int(judgement_match.group(1))
                explanation = explanation_match.group(1).strip()
                return judgement, explanation
            else:
                print("WARNING: Could not extract expected fields.")
                print("Raw model output:\n", text)
                return 0, "Malformed or incomplete response."

        return 0, "No response from LLM judge."


    def judge_and_save(self, predictions_path: str, output_path: str):
        """
        Reads predictions.json, applies judgment and saves llm_judgements.json.
        """
        with open(predictions_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        judgements = self.judge_predictions(predictions)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(judgements, f, indent=2)

        print(f"Julgmentos salvos em: {output_path}")

    # Verify if the model deployment is available
    def _resolve_model_deployment(self) -> str:
        deployment_name = get_default_model_deployment_for_model(self.judge_model)
        if not deployment_name:
            raise Exception(f"Could not find a model deployment for judge model '{self.judge_model}'. "
                            f"Make sure the model is correctly registered in the HELM model YAML.")
        return deployment_name
    
    # Load the prompt template from a file
    def _load_prompt_template(self, prompt_file: str) -> str:
        """
        Load the prompt template from a .txt file located in customizable_prompts/
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_dir, "customizable_prompts", prompt_file)

        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt file '{prompt_path}' not found. Make sure the file exists.")

        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
