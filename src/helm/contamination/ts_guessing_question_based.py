import os
import numpy as np
import spacy
import asyncio
from googletrans import Translator
from nltk.tokenize import word_tokenize
from dataclasses import replace

from helm.common.hierarchical_logger import hlog, htrack_block

class TSGuessingQuestionBasedContaminationEvaluator:
    """
    Implementation of question-based guessing test for contamination detection.
    Modified to wrap async calls in synchronous methods.
    """
    
    def __init__(self):
        self.language = "en"
        self.translator = Translator()
        
    def evaluate(
        self,
        ex,
        benchmark_path: str,
        scenario_state,
        language: str 
    ) -> float:
        """
        Evaluate contamination using the TS guessing question-based approach.
        """
        return asyncio.run(self._evaluate_async(ex, benchmark_path, scenario_state, language))
        
    async def _evaluate_async(
        self,
        ex,
        benchmark_path: str,
        scenario_state,
        language: str 
    ) -> float:
        self.language = language
        with htrack_block("ts_guessing_question_based contamination evaluation"):

            # Get instances from scenario state
            instances = [rs.instance for rs in scenario_state.request_states]
            eval_data_name = os.path.basename(benchmark_path).split(':')[0]

            # Initialize tagger
            tagger = self._get_spacy_tagger()

            # Filter and prepare data
            data_points = self._filter_data(instances, eval_data_name)
            hlog(f"Left with {len(data_points)} data points after filtering")

            # Modify prompt and update directly in scenario_state
            masked_words = []

            for i, request_state in enumerate(scenario_state.request_states):
                if i < len(data_points):
                    data_point = data_points[i]
                    prompt, masked_word = await self._build_prompt(data_point, tagger, eval_data_name)
                    if prompt != "failed":
                        # Modifies the instance input
                        new_input = replace(request_state.instance.input, text=prompt)
                        new_instance = replace(request_state.instance, input=new_input)
                        
                        # Modifies the request prompt
                        new_request = replace(
                            request_state.request,
                            prompt=prompt,
                            max_tokens=10,
                            temperature=0.0,
                            stop_sequences=[]
                        )
                        
                        # Update instructions in adapter_spec
                        new_adapter_spec = replace(
                            scenario_state.adapter_spec, 
                            instructions=prompt,
                            input_prefix='', 
                            output_prefix='Answer: ', 
                            max_tokens=10, 
                            stop_sequences=[]
                        )
                        scenario_state.adapter_spec = new_adapter_spec
                        
                        # Creates new request_state
                        novo_request_state = replace(request_state, instance=new_instance, request=new_request)
                        scenario_state.request_states[i] = novo_request_state
                        masked_words.append(masked_word)
                    else:
                        masked_words.append("")
                else:
                    masked_words.append("")

            response_scenario_state = ex.execute(scenario_state)
            # Process results
            results = []
            for i, rs in enumerate(response_scenario_state.request_states):
                if i < len(masked_words) and masked_words[i] != "":
                    if hasattr(rs, 'result') and hasattr(rs.result, 'completions'):
                        full_response = rs.result.completions[0].text.strip().lower()
                        first_word = full_response.split()[0].strip('"""\'\'.,;!?¿¡#').lower() if full_response else "" 
                        results.append({
                            "id": f"instance_{i}",
                            "masked_word": masked_words[i].lower(),
                            "response": first_word
                        })

            # Calculate metrics
            if results:
                exact_match = sum(1 for r in results if r["response"] == r["masked_word"]) / len(results)
            else:
                exact_match = 0.0
            
            return exact_match, "Exact Match"
    
    def _get_spacy_tagger(self):
        """Initialize and return a spaCy language model for POS tagging."""
        return spacy.load("en_core_web_sm")
    
    def _filter_data(self, eval_data, eval_data_name):
        """Filter data points"""
        data_points = []
        if eval_data_name == "truthful_qa":
            for x in eval_data:
                x_dict = self._instance_to_dict(x)
                n_words = len(word_tokenize(x_dict["text"]))
                if n_words <= 4:
                    continue
                if 'Indexical Error' in x_dict.get("category", ""):
                    continue 

                data_points.append(x_dict)
        else:
            for x in eval_data:
                # Convert instance to dictionary format if needed
                x_dict = self._instance_to_dict(x)
                
                # Process based on non-truthful_qa data
                data_points.append(x_dict)
                
        return data_points
    
    async def _build_prompt(self, example, tagger, eval_data_name):
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

        part_one = "Fill in the space marked with"
        part_two = " in the sentence below with a single word that makes sense in the context:"
        part_three = f"\n\n\"{text}\"\n\n"
        part_four = "Answer: "

        # Translation
        try:
            if self.language != "en" and self.language:
                try:
                    translated_one = await self.translator.translate(part_one, src='en', dest=self.language)
                    translated_two = await self.translator.translate(part_two, src='en', dest=self.language)
                    translated_four = await self.translator.translate(part_four, src='en', dest=self.language)
                    
                    prompt = f"{translated_one.text} [MASK] {translated_two.text}{part_three}{translated_four.text}"
                except Exception as e:
                    print(f"Translation error: {e}")
                    prompt = f"{part_one} [MASK] {part_two}{part_three}{part_four}"
            else:
                prompt = f"{part_one} [MASK] {part_two}{part_three}{part_four}"
                
            return prompt, word.lower()
        except Exception as e:
            print(f"Error in prompt building: {e}")
            return prompt, word.lower()

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