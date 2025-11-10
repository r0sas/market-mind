class Reporting:
    """
    Reporting methods for fit scores and explanations.
    """

    @staticmethod
    def print_analysis(fit_scores: dict, explanations: dict):
        print("=== Fit Scores ===")
        for model, score in fit_scores.items():
            print(f"{model}: {score:.2f}")
            for category, notes in explanations.get(model, {}).items():
                for note in notes:
                    print(f"  [{category}] {note}")
        print("=================")

    @staticmethod
    def summary(fit_scores: dict, top_n: int = 2):
        sorted_models = sorted(fit_scores.items(), key=lambda x: x[1], reverse=True)
        return [model for model, _ in sorted_models[:top_n]]
