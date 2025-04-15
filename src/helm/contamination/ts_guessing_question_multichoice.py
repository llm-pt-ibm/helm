import os
import numpy as np
import asyncio
from googletrans import Translator
from rouge_score import rouge_scorer
from dataclasses import replace

from helm.common.hierarchical_logger import hlog, htrack_block
from nltk.tokenize import sent_tokenize
from .utils_contamination import UtilsContamination

class TSGuessingQuestionMultiChoiceContaminationEvaluator:
    """
    Implements question-based multichoice guessing test for contamination detection.
    """
    
    def __init__(self):
        self.language = "en"
        self.translator = Translator()

    def evaluate(
        self,
        executor,
        benchmark_path: str,
        scenario_state,
        language: str 
    ) -> float:
        """
        Evaluate contamination using the TS-Guessing question-multichoice approach.
        """
        return asyncio.run(self._evaluate_async(
            executor, 
            benchmark_path, 
            scenario_state, 
            language
        ))

    async def _evaluate_async(
        self,
        executor,
        benchmark_path: str,
        scenario_state,
        language: str 
    ) -> float:
        self.language = language
        with htrack_block("TS-Guessing (question-multichoice) contamination evaluation"):
            
            eval_data_name = os.path.basename(benchmark_path).split(":")[0]

            if scenario_state.adapter_spec.method != "multiple_choice_joint":
                hlog(f"The selected benchmark \"{eval_data_name}\" does not qualify for the verification strategy TS-Guessing question-multichoice")
                return

            # Filter and prepare data
            data_points = self._filter_data(scenario_state)
            hlog(f"Filtered to {len(data_points)} data points")

            n_eval_data_points = min(100, len(data_points))
            if n_eval_data_points == 0:
                return 0.0

            p = np.random.permutation(len(data_points))
            data_points = [data_points[p[i]] for i in range(n_eval_data_points)]

            answers, wrong_letters = [], []

            # Build prompts and update scenario_state
            for i, request_state in enumerate(scenario_state.request_states):
                if i < len(data_points):
                    data_point = data_points[i]
                    prompt, answer, wrong_letter = await self._build_prompt(data_point, eval_data_name)

                    if prompt != "failed":
                        new_input = replace(request_state.instance.input, text=prompt)
                        new_instance = replace(request_state.instance, input=new_input)
                        new_request = replace(
                            request_state.request,
                            prompt=prompt,
                            max_tokens=100,
                            temperature=0.0
                        )
                        scenario_state.request_states[i] = replace(
                            request_state,
                            instance=new_instance,
                            request=new_request
                        )
                        answers.append(answer)
                        wrong_letters.append(wrong_letter)
                    else:
                        answers.append("")
                        wrong_letters.append("")
                else:
                    answers.append("")
                    wrong_letters.append("")

            response_scenario_state = self._query_model(scenario_state, executor)

            # Process results
            results = []
            for i, rs in enumerate(response_scenario_state.request_states):
                if i < len(answers) and answers[i] != "":
                    if hasattr(rs, "result") and hasattr(rs.result, "completions"):
                        response_text = rs.result.completions[0].text.strip()
                        processed_response = self._process_response(response_text, wrong_letters[i])
                        results.append({
                            "id": f"instance_{i}",
                            "answer": answers[i].lower(),
                            "response": processed_response.lower()
                        })

            if not results:
                return 0.0

            answers_list = [x["answer"] for x in results]
            responses_list = [x["response"] for x in results]

            # Metrics
            exact_match = sum(
                1 for i in range(len(responses_list))
                if responses_list[i] == answers_list[i]
            ) / len(responses_list)

            scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
            rouge_l = np.mean([
                scorer.score(responses_list[i], answers_list[i])["rougeLsum"].fmeasure
                for i in range(len(responses_list))
            ])

            hlog("TS-Guessing (question-multichoice) results:")
            hlog(f"  Exact Match (EM): {exact_match:.2f}")
            hlog(f"  ROUGE-L F1: {rouge_l:.2f}")

            return {"exact_match": exact_match, "rouge_l": rouge_l}

    def _filter_data(self, scenario_state):
        """Extract valid multiple choice instances from scenario state."""
        data_points = []
        for rs in scenario_state.request_states:
            question = UtilsContamination.get_question_text(rs.instance)
            choices = UtilsContamination.get_choices(rs.instance)
            answer_index = UtilsContamination.get_answer_index(rs.instance)
            if choices and 0 <= answer_index < len(choices):
                data_points.append({
                    "id": getattr(rs.instance, "id", None),
                    "text": question,
                    "question": question,
                    "choices": choices,
                    "answer_index": answer_index
                })
        return data_points

    async def _build_prompt(self, example, eval_data_name):
        """Build a multiple-choice prompt for TS-Guessing."""
        text = example.get("text") or example.get("question", "")
        choices = example.get("choices", [])
        answer_index = example.get("answer_index", -1)

        if not choices or answer_index < 0 or answer_index >= len(choices):
            return "failed", "", ""

        answer = choices[answer_index]
        wrong_indices = [i for i in range(len(choices)) if i != answer_index]
        if not wrong_indices:
            return "failed", "", ""

        wrong_index = np.random.choice(wrong_indices)
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        wrong_letter = alphabet[wrong_index]

        # Original prompt parts in English
        # Usamos '__MASK_TOKEN__' como um placeholder para evitar que [MASK] seja traduzido
        part_one = "Please fill in the __MASK_TOKEN__ in option"  # 'wrong_letter' será adicionado depois
        part_two = "based on your benchmark knowledge."
        part_three = "The crucial rule is that you must provide a different answer in all other options."
        part_four = "Question:"  # 'text' será adicionado depois
        part_five = "Options:"
        part_six = "Reply with the answer only."

        # Translate prompt parts if language is not English
        try:
            if self.language != "en" and self.language:
                try:
                    translated_one = await self.translator.translate(part_one, src='en', dest=self.language)
                    translated_two = await self.translator.translate(part_two, src='en', dest=self.language)
                    translated_three = await self.translator.translate(part_three, src='en', dest=self.language)
                    translated_four = await self.translator.translate(part_four, src='en', dest=self.language)
                    translated_five = await self.translator.translate(part_five, src='en', dest=self.language)
                    translated_six = await self.translator.translate(part_six, src='en', dest=self.language)
                    
                    # Substituir o placeholder pelo token [MASK] original
                    part_one = translated_one.text.replace("__MASK_TOKEN__", "[MASK]")
                    part_two = translated_two.text
                    part_three = translated_three.text
                    part_four = translated_four.text
                    part_five = translated_five.text
                    part_six = translated_six.text
                except Exception as e:
                    print(f"Translation error: {e}")
                    # Substituir o placeholder pelo token [MASK] original em caso de falha
                    part_one = part_one.replace("__MASK_TOKEN__", "[MASK]")
        except Exception as e:
            print(f"Error in prompt translation: {e}")
            # Substituir o placeholder pelo token [MASK] original em caso de falha
            part_one = part_one.replace("__MASK_TOKEN__", "[MASK]")

        # Build the final prompt, preserving wrong_letter, [MASK], and text variables
        prompt = f"{part_one} {wrong_letter} {part_two}\n\n{part_three}\n\n{part_four} {text}\n{part_five}"
        for i, choice in enumerate(choices):
            letter = alphabet[i]
            content = "[MASK]" if i == wrong_index else choice
            prompt += f"\n{letter}: {content}"
        prompt += f"\n\n{part_six}"

        return prompt, answer, wrong_letter

    def _process_response(self, response, wrong_letter):
        """Clean and normalize model's response."""
        symbol = wrong_letter + ":"
        if symbol in response:
            response = response.split(symbol)[1]

        try:
            sents = sent_tokenize(response)
            if sents:
                response = sents[0]
        except (ImportError, LookupError):
            for delimiter in ['.', '!', '?']:
                if delimiter in response:
                    response = response.split(delimiter)[0] + delimiter
                    break

        return response.strip().replace("[", "").replace("]", "").replace("MASK", "")

    def _query_model(self, scenario_state, executor):
        """Query the model with updated scenario state."""
        return executor.execute(scenario_state)