import os
import numpy as np
import spacy
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize

from concurrent.futures import ThreadPoolExecutor


from helm.benchmark.metrics.metric import MetricResult, Stat, PerInstanceStats
from helm.benchmark.metrics.metric_name import MetricName
from helm.common.hierarchical_logger import hlog, htrack_block

class TSGuessingQuestionBasedContaminationEvaluator:
    """
    Implementation of question-based guessing test for contamination detection.
    Adapted from the ts_guessing_question_based method.
    """
    
    def __init__(self):
        # Initialize any required resources
        pass
        
    def evaluate(
        self,
        ex,
        model_path: str,
        benchmark_path: str,
        scenario_state,
        metric_service,
        eval_cache_path: str,
        parallelism: int
    ) -> MetricResult:
        """
        Evaluate contamination using the TS guessing question-based approach.
        
        Args:
            model_path: Path to the model.
            benchmark_path: Path to the benchmark data.
            scenario_state: The current scenario state.
            metric_service: Service for computing metrics.
            eval_cache_path: Path for caching evaluation results.
            parallelism: Number of parallel workers.
            
        Returns:
            MetricResult containing contamination evaluation statistics.
        """
        with htrack_block("ts_guessing_question_based contamination evaluation"):
            # Get instances from scenario state
            instances = [rs.instance for rs in scenario_state.request_states]
            eval_data_name = os.path.basename(benchmark_path).split(':')[0]
            
            # Filter and prepare data
            data_points = self._filter_data(instances, eval_data_name)
            hlog(f"Left with {len(data_points)} data points after filtering")
            
            # Subsample if needed
            n_eval_data_points = min(100, len(data_points))
            if n_eval_data_points > 0:
                p = np.random.permutation(len(data_points))
                data_points = [data_points[x] for x in p[:n_eval_data_points]]
            
            # Initialize tagger
            tagger = self._get_spacy_tagger()
            
            # Process data points
            results = []
            with ThreadPoolExecutor(max_workers=parallelism) as executor:
                futures = []
                for data_point in data_points:
                    futures.append(
                        executor.submit(
                            self._process_data_point,
                            data_point,
                            eval_data_name,
                            tagger,
                            model_path,
                            scenario_state,
                            ex
                        )
                    )
                
                for future in tqdm(futures, desc="Processing data points"):
                    result = future.result()
                    if result["response"] != "failed":
                        results.append(result)
            
            # Calculate metrics
            masked_words = [x["masked_word"].lower() for x in results]
            responses = [x["response"].lower() for x in results]
            exact_match = len([i for i in range(len(responses)) 
                             if responses[i] == masked_words[i]]) / max(len(responses), 1)
            
            hlog(f"Question-based completion contamination evaluation")
            hlog(f"Exact Match (EM): {exact_match:.2f}")
            
            # Create stats
            stats = [
                Stat(
                    name=MetricName("contamination_ts_guessing_exact_match"),
                    value=exact_match,
                    higher_is_better=False,  # Lower exact match implies less contamination
                    description="Exact match rate in masked word prediction (lower implies less contamination)"
                )
            ]
            
            # Create per-instance stats if needed
            per_instance_stats = [
                PerInstanceStats(
                    instance_id=results[i].get("id", f"instance_{i}"),
                    stats=[
                        Stat(
                            name=MetricName("contamination_ts_guessing_match"),
                            value=1.0 if responses[i] == masked_words[i] else 0.0,
                            higher_is_better=False,
                            description="Binary indicator of exact match (1=match, 0=no match)"
                        )
                    ]
                )
                for i in range(len(results))
            ]
            
            return MetricResult(aggregated_stats=stats, per_instance_stats=per_instance_stats)
    
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
                # Implementation would need to be adapted for HELM's data structures
                data_points.append(x_dict)
                
        return data_points
    
    def _instance_to_dict(self, instance):
        """Convert a HELM instance to a dictionary format compatible with this evaluator."""
        # This implementation would depend on HELM's exact data structures
        # Here's a simplified version
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

        # Seleciona palavras com POS relevantes
        words = [token for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB']]
        if len(words) == 0:
            return "failed", ""
        
        idx = np.random.randint(len(words))
        word = words[idx].text

        # Substitui a primeira ocorrência da palavra por [MASK]
        if word in text:
            text = text.replace(word, "[MASK]", 1)

        prompt = "Complete a frase em uma palavra:"
        prompt += f"\n\n{text}"

        # Add hints if specified
        if type_hint and eval_data_name == "truthful_qa" and "type" in example:
            prompt += f"\ndica: {example['type']}"
        if category_hint and eval_data_name == "truthful_qa" and "category" in example:
            prompt += f"\ndica: {example['category']}"
        if url_hint and eval_data_name == "truthful_qa" and "source" in example:
            prompt += f"\ndica: {example['source']}"
            
        prompt += "\nResponda apenas a resposta."
        
        return prompt, word

    
    def _process_data_point(self, data_point, eval_data_name, tagger, model_path, scenario_state, executor):
        """Process a single data point for contamination evaluation."""
        # Build the prompt with masked word
        prompt, masked_word = self._build_prompt(data_point, tagger, eval_data_name)
        
        if prompt == "failed":
            data_point["masked_word"] = ""
            data_point["response"] = "failed"
            return data_point
        
        response = self._query_model(prompt, model_path, scenario_state, executor)
        
        # Process the response
        processed_response = word_tokenize(response)[0] if response else ""
        
        # Update data point with results
        data_point["masked_word"] = masked_word
        data_point["response"] = processed_response

        #print("PALAVRA OCULTA: ", data_point["masked_word"], "\nRESPOSTA: ", data_point["response"])
        
        return data_point
    
    def _query_model(self, prompt, model_path, scenario_state, executor):
        scenario_state.prompt = prompt
        # Execute (fill up results)
        response = executor.execute(scenario_state)
        print(response.result.completions[0].text)
        # Annotate (post-process the results)
        #scenario_state = executor.execute(scenario_state)

        return "simulated word response"