import os
import numpy as np
import spacy
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
from dataclasses import replace

from helm.benchmark.metrics.metric import MetricResult, Stat, PerInstanceStats
from helm.benchmark.metrics.metric_name import MetricName
from helm.common.hierarchical_logger import hlog, htrack_block

class TSGuessingQuestionBasedContaminationEvaluator:
    """
    Implementation of question-based guessing test for contamination detection.
    Modified to update all prompts before a single model query.
    """
    
    def __init__(self):
        pass
        
    def evaluate(
        self,
        ex,
        benchmark_path: str,
        scenario_state,
    ) -> MetricResult:
        """
        Evaluate contamination using the TS guessing question-based approach.
        """
        with htrack_block("ts_guessing_question_based contamination evaluation"):

            # Get instances from scenario state
            instances = [rs.instance for rs in scenario_state.request_states]
            eval_data_name = os.path.basename(benchmark_path).split(':')[0]

            # Initialize tagger
            tagger = self._get_spacy_tagger()

            # Filter and prepare data
            data_points = self._filter_data(instances, eval_data_name)
            hlog(f"Left with {len(data_points)} data points after filtering")

            # Subsample if needed
            n_eval_data_points = min(100, len(data_points))
            if n_eval_data_points > 0:
                p = np.random.permutation(len(data_points))
                data_points = [data_points[p[i]] for i in range(n_eval_data_points)]

            # Modify prompt and update directly in scenario_state
            masked_words = []

            for i, request_state in enumerate(scenario_state.request_states):
                if i < len(data_points):
                    data_point = data_points[i]
                    prompt, masked_word = self._build_prompt(data_point, tagger, eval_data_name)

                    if prompt != "failed":
                        new_input = replace(request_state.instance.input, text=prompt)
                        new_instance = replace(request_state.instance, input=new_input)
                        new_request = replace(
                            request_state.request,
                            prompt=prompt,
                            max_tokens=10,
                            temperature=0.0 
                        )

                        # Creates new request_state with the changes
                        novo_request_state = replace(request_state, instance=new_instance, request=new_request)
                        scenario_state.request_states[i] = novo_request_state
                        masked_words.append(masked_word)
                    else:
                        masked_words.append("")
                else:
                    masked_words.append("")

            response_scenario_state = self._query_model(scenario_state, ex)

            # Process results
            results = []
            for i, rs in enumerate(response_scenario_state.request_states):
                if i < len(masked_words) and masked_words[i] != "":
                    if hasattr(rs, 'result') and hasattr(rs.result, 'completions'):
                        response_text = rs.result.completions[0].text.strip()
                        results.append({
                            "id": f"instance_{i}",
                            "masked_word": masked_words[i].lower(),
                            "response": response_text.lower()
                        })
            
            print(results)
            # Calculate metrics
            if results:
                masked_words = [x["masked_word"] for x in results]
                responses = [x["response"] for x in results]
                exact_match = sum(1 for i in range(len(responses)) 
                               if responses[i] == masked_words[i]) / len(responses)
            else:
                exact_match = 0.0
            
            hlog(f"Question-based completion contamination evaluation")
            hlog(f"Exact Match (EM): {exact_match:.2f}")
            
            
            return exact_match
    
    def _get_spacy_tagger(self):
        """Initialize and return a spaCy language model for POS tagging."""
        return spacy.load("en_core_web_sm")
    
    def _filter_data(self, eval_data, eval_data_name):
        """Filter data points based on criteria from the original implementation."""
        data_points = []
        if eval_data_name == "truthful_qa":
            for x in eval_data:
                # Convert instance to dictionary format if needed
                x_dict = self._instance_to_dict(x)
                
                # Remove questions with 4 or less words
                n_words = len(word_tokenize(x_dict["text"]))
                if n_words <= 4:
                    continue
                # Remove questions of 'Indexical Error' category
                if 'Indexical Error' in x_dict.get("category", ""):
                    continue 

                data_points.append(x_dict)
        else:
            scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
            for x in eval_data:
                # Convert instance to dictionary format if needed
                x_dict = self._instance_to_dict(x)
                
                # Process based on non-truthful_qa data
                data_points.append(x_dict)
                
        return data_points
    
    def _instance_to_dict(self, instance):
        """Convert a HELM instance to a dictionary format compatible with this evaluator."""
        if hasattr(instance, '__dict__'):
            result = instance.__dict__.copy()
        else:
            # Create a dictionary with common fields
            result = {
                "id": getattr(instance, "id", None),
                "text": getattr(instance, "input", ""),
                "references": getattr(instance, "references", []),
            }
        
        return result
    
    def _build_prompt(self, example, tagger, eval_data_name, type_hint=False, category_hint=False, url_hint=False):
        """Build a prompt with a masked word for testing."""
        text = example["input"].text
        doc = tagger(text)

        # Select words with relevant POS
        words = [token for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB']]
        if len(words) == 0:
            return "failed", ""
        
        idx = np.random.randint(len(words))
        word = words[idx].text

        if word in text:
            text = text.replace(word, "[MASK]", 1)
        else:
            return "failed", ""

        prompt = "Preencha o espaço marcado com [MASK] na frase abaixo com uma única palavra que faça sentido no contexto:"
        prompt += f"\n\n\"{text}\""
        prompt += "\n\nResposta: "
        
        return prompt, word
    
    def _query_model(self, scenario_state, executor):
        """Query the model with the modified scenario state."""
        # Execute the scenario
        response_scenario_state = executor.execute(scenario_state)

        return response_scenario_state