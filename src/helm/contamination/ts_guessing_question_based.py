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
        Evaluate contamination using the TS guessing question-based approach.
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
        try:
            self.language = language
            with htrack_block("ts_guessing_question_based contamination evaluation"):

                # Get instances from scenario state
                instances = [rs.instance for rs in scenario_state.request_states 
                            if hasattr(rs, "instance")]
                
                if not instances:
                    hlog("No valid instances found in scenario state")
                    return {"exact_match": 0.0}
                    
                eval_data_name = os.path.basename(benchmark_path).split(':')[0]

                # Initialize tagger
                try:
                    tagger = self._get_spacy_tagger()
                except Exception as e:
                    hlog(f"Failed to load spaCy tagger: {e}")
                    return {"exact_match": 0.0}

                # Filter and prepare data
                data_points = self._filter_data(instances, eval_data_name)
                hlog(f"Left with {len(data_points)} data points after filtering")
                
                if not data_points:
                    hlog("No data points remained after filtering")
                    return {"exact_match": 0.0}

                # Modify prompt and update directly in scenario_state
                masked_words = []
                part_one, part_two, part_four = await self._prompt_default()
                
                for i, request_state in enumerate(scenario_state.request_states):
                    if i < len(data_points):
                        data_point = data_points[i]
                        try:
                            prompt, masked_word = self._build_prompt(
                                data_point, tagger, eval_data_name, part_one, part_two, part_four
                            )

                            if prompt != "failed":
                                # Safely modify the instance input
                                if hasattr(request_state, "instance") and hasattr(request_state.instance, "input"):
                                    new_input = replace(request_state.instance.input, text=prompt)
                                    new_instance = replace(request_state.instance, input=new_input)
                                    
                                    # Safely modify the request prompt
                                    if hasattr(request_state, "request"):
                                        new_request = replace(
                                            request_state.request,
                                            prompt=prompt,
                                            max_tokens=15
                                        )
                                        
                                        # Update instructions in adapter_spec if it exists
                                        if hasattr(scenario_state, "adapter_spec"):
                                            try:
                                                new_adapter_spec = replace(
                                                    scenario_state.adapter_spec,
                                                    method='generation',
                                                    instructions='',
                                                    input_prefix='', 
                                                    output_prefix='Answer: ', 
                                                    max_tokens=15
                                                )
                                                scenario_state.adapter_spec = new_adapter_spec
                                            except Exception as e:
                                                hlog(f"Error updating adapter_spec: {e}")
                                        
                                        # Creates new request_state
                                        novo_request_state = replace(request_state, instance=new_instance, request=new_request)
                                        scenario_state.request_states[i] = novo_request_state
                                        masked_words.append(masked_word)
                                    else:
                                        hlog(f"Request attribute missing in request_state {i}")
                                        masked_words.append("")
                                else:
                                    hlog(f"Instance or input attribute missing in request_state {i}")
                                    masked_words.append("")
                            else:
                                masked_words.append("")
                        except Exception as e:
                            hlog(f"Error building prompt for instance {i}: {e}")
                            masked_words.append("")
                    else:
                        masked_words.append("")

                # Execute model queries
                try:
                    response_scenario_state = executor.execute(scenario_state)
                except Exception as e:
                    hlog(f"Error executing model queries: {e}")
                    return {"exact_match": 0.0}
                
                # Process results
                results = []
                for i, rs in enumerate(response_scenario_state.request_states):
                    if i < len(masked_words):
                        try:
                            if (hasattr(rs, 'result') and rs.result is not None and 
                                hasattr(rs.result, 'completions') and rs.result.completions):

                                full_response = rs.result.completions[0].text.strip().lower()
                                
                                # Safely extract first word
                                response_words = full_response.split()
                                first_word = response_words[0].strip('"""\'\'.,;!?¿¡#').lower() if response_words else ""
                                
                                results.append({
                                    "id": f"instance_{i}",
                                    "masked_word": masked_words[i].lower(),
                                    "response": first_word
                                })
                        except Exception as e:
                            hlog(f"Error processing result for instance {i}: {e}")


                # Calculate metrics
                if results:
                    exact_match = sum(1 for r in results if r["response"] == r["masked_word"]) / len(results)
                else:
                    hlog("No valid results to calculate exact match")
                    exact_match = 0.0
                 
                return {"exact_match": exact_match}
        except Exception as e:
            hlog(f"Unhandled error in _evaluate_async: {e}")
            return {"exact_match": 0.0}
    
    def _get_spacy_tagger(self):
        """Initialize and return a spaCy language model for POS tagging."""
        try:
            return spacy.load("en_core_web_sm")
        except Exception as e:
            hlog(f"Failed to load spaCy model: {e}")
            # Try with a different model name if available
            try:
                return spacy.load("en")
            except:
                raise Exception("Could not load any spaCy model")
    
    def _filter_data(self, eval_data, eval_data_name):
        """Filter data points"""
        data_points = []
        try:
            if eval_data_name == "truthful_qa":
                for x in eval_data:
                    try:
                        x_dict = self._instance_to_dict(x)
                        if "text" not in x_dict or not x_dict["text"]:
                            continue
                            
                        text = x_dict.get("text", "")
                        if isinstance(text, str):
                            try:
                                n_words = len(word_tokenize(text))
                                if n_words <= 4:
                                    continue
                            except Exception as e:
                                hlog(f"Error tokenizing text: {e}")
                                continue
                                
                        # Skip instances with 'Indexical Error' category
                        if 'Indexical Error' in x_dict.get("category", ""):
                            continue 

                        data_points.append(x_dict)
                    except Exception as e:
                        hlog(f"Error processing data point: {e}")
            else:
                for x in eval_data:
                    try:
                        # Convert instance to dictionary format
                        x_dict = self._instance_to_dict(x)
                        
                        # Basic validation
                        if "input" in x_dict and hasattr(x_dict["input"], "text") and x_dict["input"].text:
                            data_points.append(x_dict)
                    except Exception as e:
                        hlog(f"Error processing non-truthful_qa data point: {e}")
        except Exception as e:
            hlog(f"Error in _filter_data: {e}")
                
        return data_points
    
    async def _prompt_default(self):
        """Get default prompt parts with translation support."""
        part_one = "Fill in the space marked with"
        part_two = " in the sentence below with a single word that makes sense in the context:"
        part_four = "Answer: "

        # Translation
        try:
            if self.language != "en" and self.language:
                tasks = [
                    self._safe_translate(part_one),
                    self._safe_translate(part_two),
                    self._safe_translate(part_four)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle results safely
                translated_parts = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        hlog(f"Translation failed for part {i+1}: {result}")
                        translated_parts.append([part_one, part_two, part_four][i])
                    else:
                        translated_parts.append(result)
                
                return tuple(translated_parts)
        except Exception as e:
            hlog(f"Error in translation: {e}")
            
        return part_one, part_two, part_four
    
    async def _safe_translate(self, text):
        """Safely translate text with error handling."""
        try:
            result = await self.translator.translate(text, src='en', dest=self.language)
            return result.text
        except Exception as e:
            hlog(f"Translation error: {e}")
            raise e
        
    def _build_prompt(self, example, tagger, eval_data_name, part_one, part_two, part_four):
        """Build a prompt with a masked word for testing."""
        try:
            # Safely access the text content
            if "input" in example and hasattr(example["input"], "text"):
                text = example["input"].text
            elif "text" in example:
                text = example["text"]
            else:
                hlog("No text found in example")
                return "failed", ""
                
            if not text or not isinstance(text, str):
                hlog("Invalid text content")
                return "failed", ""
                
            # Process with spaCy
            try:
                doc = tagger(text)
            except Exception as e:
                hlog(f"Error processing text with spaCy: {e}")
                return "failed", ""

            # Select words with relevant POS
            words = [token for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB']]
            if not words:
                hlog("No suitable words found in text")
                return "failed", ""
            
            # Safely select a random word
            try:
                idx = np.random.randint(len(words))
                word = words[idx].text
            except Exception as e:
                hlog(f"Error selecting random word: {e}")
                return "failed", ""

            # Check if the word exists in the text
            if word in text:
                # Replace only the first occurrence
                try:
                    masked_text = text.replace(word, "[MASK]", 1)
                except Exception as e:
                    hlog(f"Error replacing word in text: {e}")
                    return "failed", ""
            else:
                hlog("Selected word not found in text")
                return "failed", ""

            # Build the final prompt
            try:
                part_three = f"\n\n\"{masked_text}\"\n\n"
                prompt = f"{part_one} [MASK] {part_two}{part_three}{part_four}"
                return prompt, word.lower()
            except Exception as e:
                hlog(f"Error building final prompt: {e}")
                return "failed", ""
                
        except Exception as e:
            hlog(f"Unhandled error in _build_prompt: {e}")
            return "failed", ""

    def _instance_to_dict(self, instance):
        """Convert a HELM instance to a dictionary format compatible with this evaluator."""
        try:
            if hasattr(instance, '__dict__'):
                result = instance.__dict__.copy()
            else:
                # Create a dictionary with common fields
                result = {
                    "id": getattr(instance, "id", None),
                    "input": getattr(instance, "input", None),
                    "references": getattr(instance, "references", []),
                }
            
            return result
        except Exception as e:
            hlog(f"Error converting instance to dict: {e}")
            return {"id": None, "input": None, "references": []}