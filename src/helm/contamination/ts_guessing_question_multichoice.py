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
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def evaluate(
        self,
        executor,
        benchmark_path: str,
        scenario_state,
        language: str
    ) -> dict:
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
    ) -> dict:
        self.language = language
        with htrack_block("TS-Guessing (question-multichoice) contamination evaluation"):

            eval_data_name = os.path.basename(benchmark_path).split(":")[0]

            if not hasattr(scenario_state, "adapter_spec") or \
               not hasattr(scenario_state.adapter_spec, "method") or \
               scenario_state.adapter_spec.method != "multiple_choice_joint":
                hlog(f"The selected benchmark \"{eval_data_name}\" does not qualify for the verification strategy TS-Guessing question-multichoice.")
                return {"error": "Benchmark not suitable", "exact_match": 0.0, "rouge_l": 0.0}

            # Filter and prepare data
            data_points = self._filter_data(scenario_state)
            hlog(f"Filtered to {len(data_points)} data points")

            if not data_points:
                hlog("No data points after filtering. Skipping evaluation.")
                return {"error": "No data points", "exact_match": 0.0, "rouge_l": 0.0}

            p = np.random.permutation(len(data_points))
            data_points = [data_points[i] for i in p[:len(data_points)]]

            answers_for_eval, wrong_letters_for_eval = [], []
            valid_request_states_indices = []

            prompt_parts = await self._prompt_default()
            part_one, part_two, part_three, part_four, part_five, part_six = prompt_parts

            # Build prompts and update scenario_state
            for i, request_state in enumerate(scenario_state.request_states):
                if i < len(data_points):
                    data_point = data_points[i]
                    try:
                        instruction_text, user_text, answer, wrong_letter = self._build_prompt(
                            data_point, part_one, part_two, part_three, part_four, part_five, part_six
                        )
                        if instruction_text != "failed":
                            new_adapter_spec = replace(
                                scenario_state.adapter_spec,
                                method='generation',
                                instructions=instruction_text,
                                input_prefix='',
                                input_suffix='',
                                output_prefix='Answer: ',
                                max_tokens=100
                            )

                            new_input = replace(request_state.instance.input, text=user_text)
                            new_instance = replace(request_state.instance, input=new_input)
                            new_request = replace(
                                request_state.request,
                                prompt=user_text,
                                max_tokens=100
                            )

                            scenario_state.adapter_spec = new_adapter_spec
                            scenario_state.request_states[i] = replace(
                                request_state,
                                instance=new_instance,
                                request=new_request
                            )
                            answers_for_eval.append(answer)
                            wrong_letters_for_eval.append(wrong_letter)
                            valid_request_states_indices.append(i)
                        else:
                            hlog(f"Failed to build prompt for data point {i} (original index in scenario_state)")
                    except Exception as e:
                        hlog(f"Error building prompt or updating state for data point {i}: {e}")
            
            if not valid_request_states_indices:
                hlog("No valid prompts were constructed for any data points.")
                return {"error": "No valid prompts constructed", "exact_match": 0.0, "rouge_l": 0.0}

            try:
                response_scenario_state = self._query_model(scenario_state, executor)
            except Exception as e:
                hlog(f"Error querying model: {e}")
                return {"error": "Model query failed", "exact_match": 0.0, "rouge_l": 0.0}

            # Process results
            results = []
            for idx_in_eval_lists, original_rs_index in enumerate(valid_request_states_indices):
                rs = response_scenario_state.request_states[original_rs_index]
                current_answer = answers_for_eval[idx_in_eval_lists]
                current_wrong_letter = wrong_letters_for_eval[idx_in_eval_lists]

                try:
                    if hasattr(rs, "result") and rs.result is not None and \
                       hasattr(rs.result, "completions") and rs.result.completions:
                        response_text = rs.result.completions[0].text.strip()
                        processed_response = self._process_response(response_text, current_wrong_letter)
                        results.append({
                            "id": f"instance_{original_rs_index}",
                            "answer": current_answer.lower(),
                            "response": processed_response.lower()
                        })
                    else:
                        hlog(f"No valid result/completion for instance index {original_rs_index}")
                except Exception as e:
                    hlog(f"Error processing result for instance index {original_rs_index}: {e}")

            if not results:
                hlog("No valid results to evaluate after model query")
                return {"error": "No results to evaluate", "exact_match": 0.0, "rouge_l": 0.0}

            answers_list = [x["answer"] for x in results]
            responses_list = [x["response"] for x in results]

            # Metrics
            exact_match = sum(
                1 for i in range(len(responses_list))
                if responses_list[i] == answers_list[i]
            ) / len(responses_list) if responses_list else 0.0

            rouge_l = 0.0
            if responses_list:
                try:
                    scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
                    rouge_scores = []
                    for i in range(len(responses_list)):
                        resp_str = str(responses_list[i])
                        ans_str = str(answers_list[i])
                        try:
                            score = scorer.score(resp_str, ans_str)["rougeLsum"].fmeasure
                            rouge_scores.append(score)
                        except Exception as e_rouge_item:
                            hlog(f"Error calculating ROUGE for instance {results[i]['id']}: {e_rouge_item}. Response: '{resp_str}', Answer: '{ans_str}'")
                            rouge_scores.append(0.0) 

                    if rouge_scores:
                        rouge_l = np.mean(rouge_scores)

                except Exception as e_rouge_scorer:
                    hlog(f"Error initializing ROUGE scorer or in overall ROUGE calculation: {e_rouge_scorer}")
                    rouge_l = 0.0

            hlog(f"TS-Guessing Results: EM: {exact_match:.4f}, ROUGE-L: {rouge_l:.4f}")
            return {"exact_match": exact_match, "rouge_l": rouge_l}

    def _filter_data(
        self, 
        scenario_state
    ):
        """Extract valid multiple choice instances from scenario state."""
        data_points = []
        for i, rs in enumerate(scenario_state.request_states):
            try:
                question = UtilsContamination.get_question_text(rs.instance)
                choices = UtilsContamination.get_choices(rs.instance)
                answer_index = UtilsContamination.get_answer_index(rs.instance)

                if question and choices and isinstance(answer_index, int) and 0 <= answer_index < len(choices):
                    data_points.append({
                        "id": getattr(rs.instance, "id", f"instance_{i}"),
                        "text": question,
                        "choices": choices,
                        "answer_index": answer_index,
                        "original_request_state_index": i
                    })
                else:
                    hlog(f"Skipping data point at index {i} due to missing info or invalid answer_index.")
            except Exception as e:
                hlog(f"Error filtering data point at index {i}: {e}")
        return data_points

    async def _prompt_default(
        self
    ):
        """Get default prompt parts with translation support."""
        parts = [
            "Please fill in the [] in option",
            "based on your benchmark knowledge.",
            "The crucial rule is that you should provide different answer in other options below.",
            "Question:",
            "Options:",
            "Reply with the answer only."
        ]

        if self.language != "en" and self.language:
            try:
                hlog(f"Translating prompt parts to: {self.language}")
                tasks = [self._safe_translate(part) for part in parts]
                translated_parts = await asyncio.gather(*tasks, return_exceptions=True)

                final_parts = []
                for i, result in enumerate(translated_parts):
                    if isinstance(result, Exception):
                        hlog(f"Translation failed for part {i+1} ('{parts[i]}'): {result}. Using original English text.")
                        final_parts.append(parts[i])
                    else:
                        final_parts.append(result)
                return tuple(final_parts)
            except Exception as e:
                hlog(f"Error during translation process: {e}. Using original English text for all parts.")
                return tuple(parts)

        return tuple(parts)

    async def _safe_translate(
        self, 
        text
    ):
        """Safely translate text with error handling."""
        try:
            result = await self.translator.translate(text, src="en", dest=self.language)
            return result.text
        except Exception as e:
            hlog(f"Translation error: {e}")
            raise e

    def _build_prompt(
        self, 
        example, 
        part_one, 
        part_two, 
        part_three, 
        part_four, 
        part_five, 
        part_six
    ):
        """Build a multiple-choice prompt for TS-Guessing."""
        try:
            text = example.get("text") or example.get("question", "")
            choices = example.get("choices", [])
            answer_index = example.get("answer_index", -1)

            if (
                not text
                or not choices
                or not isinstance(answer_index, int)
                or answer_index < 0
                or answer_index >= len(choices)
            ):
                hlog(f"Invalid example data for prompt building: {example.get('id', 'Unknown ID')}")
                return "failed", "", "", ""

            answer = str(choices[answer_index])
            wrong_indices = [i for i in range(len(choices)) if i != answer_index]

            if not wrong_indices:
                hlog(f"No wrong choices available for example: {example.get('id', 'Unknown ID')}")
                return "failed", "", "", ""

            wrong_choice_index = np.random.choice(wrong_indices)

            # Ensure we don't exceed alphabet length
            if wrong_choice_index >= len(self.alphabet):
                hlog(f"Warning: wrong_choice_index {wrong_choice_index} exceeds alphabet length {len(self.alphabet)}")
                wrong_choice_index = wrong_choice_index % len(self.alphabet)

            if wrong_choice_index >= len(self.alphabet):
                hlog(f"Warning: wrong_choice_index {wrong_choice_index} exceeds alphabet length {len(self.alphabet)}. Using modulo.")
                wrong_letter = self.alphabet[wrong_choice_index % len(self.alphabet)]
            else:
                wrong_letter = self.alphabet[wrong_choice_index]

            # Build the final prompt
            instruction_text = f"{part_one} {wrong_letter} {part_two}\n\n{part_three}"

            user_text = f"{part_four} {text}\n{part_five}"
            for i, choice_item in enumerate(choices):
                choice_text = str(choice_item)
                if i >= len(self.alphabet):
                    hlog(f"Warning: choice index {i} for example {example.get('id', 'Unknown ID')} exceeds alphabet length. Skipping this choice in prompt.")
                    continue

                letter = self.alphabet[i]
                content = "[MASK]" if i == wrong_choice_index else f"[{choice_text}]"
                user_text += f"\n{letter}: {content}"

            user_text += f"\n\n{part_six}"

            return instruction_text, user_text, answer, wrong_letter

        except Exception as e:
            hlog(f"Error building prompt for example {example.get('id', 'Unknown ID')}: {e}")
            return "failed", "", "", ""

    def _process_response(self, response, wrong_letter):
        try:
            if not response:
                return ""

            processed_response_text = str(response)
            symbol = wrong_letter + ":"
            if symbol in processed_response_text:
                parts = processed_response_text.split(symbol, 1)
                if len(parts) > 1:
                    processed_response_text = parts[1]
            
            try:
                sents = sent_tokenize(processed_response_text)
                if sents:
                    processed_response_text = sents[0]
            except Exception as e:
                hlog(f"Sentence tokenization failed: {e}")
                for delimiter in [".", "!", "?"]:
                    if delimiter in processed_response_text:
                        processed_response_text = processed_response_text.split(delimiter)[0] + delimiter
                        break
            
            return processed_response_text.strip().replace("[", "").replace("]", "").replace("MASK", "")

        except Exception as e:
            hlog(f"Error processing response: {e}")
            return ""

    def _query_model(self, scenario_state, executor):
        """Query the model with updated scenario state."""
        print(f"SCENARIOOOOOOOOOO: {scenario_state}")

        try:
            return executor.execute(scenario_state)
        except Exception as e:
            hlog(f"Model query failed: {e}")
            raise e
