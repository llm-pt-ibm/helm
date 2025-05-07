class UtilsContamination:
    @staticmethod
    def get_choices(example):
        if hasattr(example, "references") and example.references:
            return [ref.output.text for ref in example.references]

        if hasattr(example, "output_mapping") and example.output_mapping:
            return list(example.output_mapping.values())

        if isinstance(example, dict):
            if "choices" in example:
                if isinstance(example["choices"], dict) and "text" in example["choices"]:
                    return example["choices"]["text"]
                elif isinstance(example["choices"], list):
                    return example["choices"]
            if "endings" in example:
                return example["endings"]
            if "correct_answers" in example:
                return example["correct_answers"]
            if "option1" in example and "option2" in example:
                return [example["option1"], example["option2"]]

        return []

    @staticmethod
    def get_answer_index(example):
        alphabet = "abcdefghijklmnopqrstuvwxyz"

        if hasattr(example, "references") and example.references:
            for i, ref in enumerate(example.references):
                if hasattr(ref, "tags") and "correct" in ref.tags:
                    return i

        if hasattr(example, "output_mapping") and hasattr(example, "references"):
            for ref in example.references:
                if hasattr(ref, "tags") and "correct" in ref.tags:
                    correct_text = ref.output.text
                    for letter, text in example.output_mapping.items():
                        if text == correct_text:
                            try:
                                return alphabet.index(letter.lower())
                            except ValueError:
                                pass

        if isinstance(example, dict):
            if "answerKey" in example:
                key = str(example["answerKey"]).lower()
                if key.isdigit():
                    return int(key) - 1
                elif key in alphabet:
                    return alphabet.index(key)
            if "label" in example:
                try:
                    return int(example["label"])
                except ValueError:
                    return -1
            if "answer" in example:
                if isinstance(example["answer"], int):
                    return example["answer"]
                if isinstance(example["answer"], str) and example["answer"].isdigit():
                    return int(example["answer"]) - 1
            if "best_answer" in example and "correct_answers" in example:
                try:
                    return example["correct_answers"].index(example["best_answer"])
                except ValueError:
                    return -1

        return -1

    @staticmethod
    def get_question_text(example):
        if hasattr(example, "input") and hasattr(example.input, "text"):
            return example.input.text

        if isinstance(example, dict):
            for key in ["question", "text", "query", "prompt"]:
                if key in example:
                    return example[key]

        return "Unknown question"
