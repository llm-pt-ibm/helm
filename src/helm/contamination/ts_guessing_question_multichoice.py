import os
import numpy as np
from rouge_score import rouge_scorer
from dataclasses import replace

from helm.benchmark.metrics.metric import MetricResult
from helm.common.hierarchical_logger import hlog, htrack_block
from nltk.tokenize import sent_tokenize
from .utils_contamination import UtilsContamination


class TSGuessingQuestionMultiChoiceContaminationEvaluator:
    """Implements question-based multichoice guessing test for contamination detection."""
    
    def evaluate(self, executor, benchmark_path: str, scenario_state) -> MetricResult:
        """Evaluate contamination using the TS guessing question-based multichoice approach."""
        with htrack_block("ts_guessing_question_multichoice contamination evaluation"):
            if scenario_state.adapter_spec.method != "multiple_choice_joint":
                return

            eval_data_name = os.path.basename(benchmark_path).split(':')[0]

            # Filter and prepare data
            data_points = self._filter_data(scenario_state)
            hlog(f"Left with {len(data_points)} data points after filtering")

            # Subsample if needed
            n_eval_data_points = min(100, len(data_points))
            if n_eval_data_points == 0:
                return 0.0
                
            p = np.random.permutation(len(data_points))
            data_points = [data_points[p[i]] for i in range(n_eval_data_points)]

            # Modify prompt and update directly in scenario_state
            answers = []
            wrong_letters = []

            for i, request_state in enumerate(scenario_state.request_states):
                if i < len(data_points):
                    data_point = data_points[i]
                    prompt, answer, wrong_letter = self._build_prompt(data_point, eval_data_name)

                    if prompt != "failed":
                        new_input = replace(request_state.instance.input, text=prompt)
                        new_instance = replace(request_state.instance, input=new_input)
                        new_request = replace(
                            request_state.request,
                            prompt=prompt,
                            max_tokens=100,
                            temperature=0.0 
                        )

                        # Update request_state with changes
                        scenario_state.request_states[i] = replace(request_state, 
                                                                  instance=new_instance, 
                                                                  request=new_request)
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
                    if hasattr(rs, 'result') and hasattr(rs.result, 'completions'):
                        response_text = rs.result.completions[0].text.strip()
                        processed_response = self._process_response(response_text, wrong_letters[i])
                        results.append({
                            "id": f"instance_{i}",
                            "answer": answers[i].lower(),
                            "response": processed_response.lower()
                        })

            # Calculate metrics
            if not results:
                return 0.0
                
            answers_list = [x["answer"] for x in results]
            responses_list = [x["response"] for x in results]
            
            # Exact match
            exact_match = sum(1 for i in range(len(responses_list)) 
                           if responses_list[i] == answers_list[i]) / len(responses_list)
            
            # Rouge-L score
            scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
            rouge_l = np.mean([
                scorer.score(responses_list[i], answers_list[i])['rougeLsum'].fmeasure 
                for i in range(len(responses_list))
            ])
            
            hlog(f"Question-based multichoice contamination evaluation")
            hlog(f"Exact Match (EM): {exact_match:.2f}, ROUGE-L F1: {rouge_l:.2f}")
            
            return exact_match
    
    def _filter_data(self, scenario_state):
        """Convert and filter HELM instances to a dictionary format compatible with this evaluator."""
        data_points = []
    
        for rs in scenario_state.request_states:
            # Extract required information
            question = UtilsContamination.get_question_text(rs.instance)
            choices = UtilsContamination.get_choices(rs.instance)
            answer_index = UtilsContamination.get_answer_index(rs.instance)
            
            # Only add valid data points
            if choices and 0 <= answer_index < len(choices):
                data_points.append({
                    "id": getattr(rs.instance, "id", None),
                    "text": question,
                    "question": question,
                    "choices": choices,
                    "answer_index": answer_index
                })

        return data_points
    
    def _build_prompt(self, example, eval_data_name):
        """Build a prompt with a multichoice question for testing."""
        text = example.get("text") or example.get("question", "")
        choices = example.get("choices", [])
        answer_index = example.get("answer_index", -1)

        if not choices or answer_index < 0 or answer_index >= len(choices):
            return "failed", "", ""

        answer = choices[answer_index]
        wrong_choices_indices = [i for i in range(len(choices)) if i != answer_index]

        if not wrong_choices_indices:
            return "failed", "", ""

        wrong_choice_index = np.random.choice(wrong_choices_indices)
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        wrong_letter = alphabet[wrong_choice_index]

        prompt = "Please fill in the [] in option " + wrong_letter + " based on your benchmark knowledge."
        prompt += "\n\nThe crucial rule is that you should provide different answer in other options below."
        prompt += f"\n\nQuestion: {text}"
        prompt += "\nOptions:"

        for i, choice in enumerate(choices):
            letter = alphabet[i]
            masked_choice = "[MASK]" if i == wrong_choice_index else choice
            prompt += f"\n{letter}: {masked_choice}"

        prompt += "\n\nReply with answer only."

        return prompt, answer, wrong_letter
    
    def _process_response(self, response, wrong_letter):
        """Process the model's response to extract the answer."""
        symbol = wrong_letter + ":"
        if symbol in response:
            response = response.split(symbol)[1]

        try:
            sents = sent_tokenize(response)
            if sents:
                response = sents[0]
        except (ImportError, LookupError):
            # Fallback: simple sentence splitting by punctuation
            for delimiter in ['.', '!', '?']:
                if delimiter in response:
                    response = response.split(delimiter)[0] + delimiter
                    break
                
        # Clean up the response
        response = response.strip()
        response = response.replace("[", "").replace("]", "").replace("MASK", "")
        
        return response
    
    def _query_model(self, scenario_state, executor):
        """Query the model with the modified scenario state."""
        return executor.execute(scenario_state)