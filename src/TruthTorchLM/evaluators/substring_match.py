from .correctness_evaluator import CorrectnessEvaluator
import ast
class SubstringMatch(CorrectnessEvaluator):
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
        # Normalize to lowercase for case-insensitive match
        gen_text_lower = generated_text.lower()
        #try:
        #    tmp = ast.literal_eval(ground_truths[0])
        #    ground_truths = tmp
        #except:
        #    pass
        for gt in ground_truths:
            if gt.lower() in gen_text_lower:
                return 1
        return 0

    def __str__(self):
        return "Substring Match"