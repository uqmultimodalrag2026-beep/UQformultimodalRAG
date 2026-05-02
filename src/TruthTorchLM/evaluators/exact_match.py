from .correctness_evaluator import CorrectnessEvaluator

class ExactMatch(CorrectnessEvaluator):
    def __init__(self):
        super().__init__()
    def __call__(
        self,
        question_text: str,
        generated_text: str,
        ground_truths: list[str],
        context: str = "",
        seed: int = None,
    ) -> bool:
        for gt in ground_truths:
            
            matched = generated_text.strip().lower() == str(gt).strip().lower() #fixed the bug here
            if matched:
                return 1
        return 0

    def __str__(self):
        return "Exact Match"