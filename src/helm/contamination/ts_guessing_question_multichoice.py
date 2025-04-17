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

            if not hasattr(scenario_state, "adapter_spec") or not hasattr(scenario_state.adapter_spec, "method") or scenario_state.adapter_spec.method != "multiple_choice_joint":
                hlog(f"The selected benchmark \"{eval_data_name}\" does not qualify for the verification strategy TS-Guessing question-multichoice")
                return 0.0

            # Filter and prepare data
            data_points = self._filter_data(scenario_state)
            hlog(f"Filtered to {len(data_points)} data points")


            p = np.random.permutation(len(data_points))
            data_points = [data_points[p[i]] for i in range(len(data_points))]

            answers, wrong_letters = [], []
            part_one, part_two, part_three, part_four, part_five, part_six = await self._prompt_default()
            
            # Build prompts and update scenario_state             
            for i, request_state in enumerate(scenario_state.request_states):                 
                if i < len(data_points):                     
                    data_point = data_points[i]                     
                    try:                         
                        prompt, answer, wrong_letter = self._build_prompt(data_point, eval_data_name, part_one, part_two, part_three, part_four, part_five, part_six)                         
                        if prompt != "failed":                             
                            new_input = replace(request_state.instance.input, text=prompt)                             
                            new_instance = replace(request_state.instance, input=new_input)                             
                            new_request = replace(                                 
                                request_state.request,                                 
                                prompt=prompt,                                 
                                max_tokens=100,                                 
                                temperature=0.0                         
                            )
                            
                            # Update adapter_spec
                            if hasattr(scenario_state, "adapter_spec"):                                             
                                try:                                                 
                                    new_adapter_spec = replace(                                                     
                                        scenario_state.adapter_spec,                                                      
                                        method='generation',
                                        instructions=prompt,
                                        input_prefix='',                                                      
                                        output_prefix='Answer: ',                                                      
                                        max_tokens=100                                                                                                 
                                    )                                                 
                                    scenario_state.adapter_spec = new_adapter_spec                                             
                                except Exception as e:                                                 
                                    hlog(f"Error updating adapter_spec: {e}")
                            
                            scenario_state.request_states[i] = replace(                                 
                                request_state,                                 
                                instance=new_instance,                                 
                                request=new_request                             
                            )                             
                            answers.append(answer)                             
                            wrong_letters.append(wrong_letter)                         
                        else:                             
                            hlog(f"Failed to build prompt for data point {i}")                             
                            answers.append("")                             
                            wrong_letters.append("")                     
                    except Exception as e:                         
                        hlog(f"Error building prompt for data point {i}: {e}")                         
                        answers.append("")                         
                        wrong_letters.append("")                 
                else:                     
                    answers.append("")                     
                    wrong_letters.append("")

            try:
                print("SCENARIO: ", scenario_state)
                response_scenario_state = self._query_model(scenario_state, executor)
            except Exception as e:
                hlog(f"Error querying model: {e}")
                return 0.0

            # Process results
            results = []
            for i, rs in enumerate(response_scenario_state.request_states):
                if i < len(answers) and answers[i] != "":
                    try:
                        if hasattr(rs, "result") and rs.result is not None and hasattr(rs.result, "completions") and rs.result.completions:
                            response_text = rs.result.completions[0].text.strip()
                            processed_response = self._process_response(response_text, wrong_letters[i])
                            results.append({
                                "id": f"instance_{i}",
                                "answer": answers[i].lower(),
                                "response": processed_response.lower()
                            })
                    except Exception as e:
                        hlog(f"Error processing result for instance {i}: {e}")

            if not results:
                hlog("No valid results to evaluate")
                return 0.0

            answers_list = [x["answer"] for x in results]
            responses_list = [x["response"] for x in results]

            # Metrics
            exact_match = sum(
                1 for i in range(len(responses_list))
                if responses_list[i] == answers_list[i]
            ) / len(responses_list)

            try:
                scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
                rouge_scores = []
                for i in range(len(responses_list)):
                    try:
                        score = scorer.score(responses_list[i], answers_list[i])["rougeLsum"].fmeasure
                        rouge_scores.append(score)
                    except Exception as e:
                        hlog(f"Error calculating ROUGE for instance {i}: {e}")
                
                rouge_l = np.mean(rouge_scores) if rouge_scores else 0.0
            except Exception as e:
                hlog(f"Error calculating ROUGE metrics: {e}")
                rouge_l = 0.0

            return {"exact_match": exact_match, "rouge_l": rouge_l}

    def _filter_data(self, scenario_state):
        """Extract valid multiple choice instances from scenario state."""
        data_points = []
        for rs in scenario_state.request_states:
            try:
                question = UtilsContamination.get_question_text(rs.instance)
                choices = UtilsContamination.get_choices(rs.instance)
                answer_index = UtilsContamination.get_answer_index(rs.instance)
                
                # Only include valid data points
                if choices and isinstance(answer_index, int) and 0 <= answer_index < len(choices):
                    data_points.append({
                        "id": getattr(rs.instance, "id", f"instance_{len(data_points)}"),
                        "text": question,
                        "question": question,
                        "choices": choices,
                        "answer_index": answer_index
                    })
            except Exception as e:
                hlog(f"Error filtering data point: {e}")
        return data_points
    
    async def _prompt_default(self):
        """Get default prompt parts with translation support."""
        part_one = "Please fill in the __MASK_TOKEN__ in option"
        part_two = "based on your benchmark knowledge."
        part_three = "The crucial rule is that you must provide a different answer in all other options."
        part_four = "Question:"
        part_five = "Options:"
        part_six = "Reply with the answer only."

        # Translation
        try:
            if self.language != "en" and self.language:
                tasks = [
                    self._safe_translate(part_one),
                    self._safe_translate(part_two),
                    self._safe_translate(part_three),
                    self._safe_translate(part_four),
                    self._safe_translate(part_five),
                    self._safe_translate(part_six)
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle results safely
                translated_parts = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        hlog(f"Translation failed for part {i+1}: {result}")
                        translated_parts.append([part_one, part_two, part_three, part_four, part_five, part_six][i])
                    else:
                        translated_parts.append(result)
                
                return tuple(translated_parts)
        except Exception as e:
            hlog(f"Error in translation: {e}")
        
        return part_one, part_two, part_three, part_four, part_five, part_six
    
    async def _safe_translate(self, text):
        """Safely translate text with error handling."""
        try:
            result = await self.translator.translate(text, src='en', dest=self.language)
            return result.text
        except Exception as e:
            hlog(f"Translation error: {e}")
            raise e
        
    def _build_prompt(self, example, eval_data_name, part_one, part_two, part_three, part_four, part_five, part_six):
        """Build a multiple-choice prompt for TS-Guessing."""
        try:
            text = example.get("text") or example.get("question", "")
            choices = example.get("choices", [])
            answer_index = example.get("answer_index", -1)

            if not text or not choices or not isinstance(answer_index, int) or answer_index < 0 or answer_index >= len(choices):
                return "failed", "", ""

            answer = choices[answer_index]
            wrong_indices = [i for i in range(len(choices)) if i != answer_index]
            if not wrong_indices:
                return "failed", "", ""

            wrong_index = np.random.choice(wrong_indices)
            
            # Ensure we don't exceed alphabet length
            if wrong_index >= len(self.alphabet):
                hlog(f"Warning: wrong_index {wrong_index} exceeds alphabet length {len(self.alphabet)}")
                wrong_index = wrong_index % len(self.alphabet)
                
            wrong_letter = self.alphabet[wrong_index]
        
            masked_part_one = part_one.replace("__mask_token__", "[MASK]")

            # Build the final prompt
            prompt = f"{masked_part_one} {wrong_letter} {part_two}\n\n{part_three}\n\n{part_four} {text}\n{part_five}"
            
            for i, choice in enumerate(choices):
                if i >= len(self.alphabet):
                    hlog(f"Warning: choice index {i} exceeds alphabet length {len(self.alphabet)}")
                    continue
                    
                letter = self.alphabet[i]
                content = "[MASK]" if i == wrong_index else choice
                prompt += f"\n{letter}: {content}"
                
            prompt += f"\n\n{part_six}"

            return prompt, answer, wrong_letter
            
        except Exception as e:
            hlog(f"Error building prompt: {e}")
            return "failed", "", ""

    def _process_response(self, response, wrong_letter):
        """Clean and normalize model's response."""
        try:
            if not response:
                return ""
                
            # Safely check for the wrong_letter in response
            symbol = wrong_letter + ":"
            if symbol in response:
                parts = response.split(symbol)
                if len(parts) > 1:
                    response = parts[1]

            # Try to extract first sentence
            try:
                sents = sent_tokenize(response)
                if sents:
                    response = sents[0]
            except Exception as e:
                hlog(f"Sentence tokenization failed: {e}")
                # Fallback to simple delimiter-based extraction
                for delimiter in ['.', '!', '?']:
                    if delimiter in response:
                        response = response.split(delimiter)[0] + delimiter
                        break

            # Clean up the response
            return response.strip().replace("[", "").replace("]", "").replace("MASK", "")
            
        except Exception as e:
            hlog(f"Error processing response: {e}")
            return ""

    def _query_model(self, scenario_state, executor):
        """Query the model with updated scenario state."""
        try:
            return executor.execute(scenario_state)
        except Exception as e:
            hlog(f"Model query failed: {e}")
            raise e