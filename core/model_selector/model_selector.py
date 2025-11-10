from .metrics_extractor import MetricsExtractor
from .scorer import Scorer
from .reporting import Reporting

class ModelSelector:
    """
    Main class: orchestrates metric extraction, scoring, and reporting.
    """

    def __init__(self, financial_data: dict):
        self.data = financial_data
        self.metrics = {}
        self.fit_scores = {}
        self.explanations = {}

    def analyze(self):
        # Step 1: Extract all metrics
        extractor = MetricsExtractor(self.data)
        self.metrics = extractor.extract_all_metrics()

        # Step 2: Score all models
        scorer = Scorer(self.metrics)
        self.fit_scores = {
            'ddm_single': scorer.score_ddm_single(),
            'ddm_multi': scorer.score_ddm_multi(),
            'dcf': scorer.score_dcf(),
            'pe_model': scorer.score_pe_model(),
            'graham': scorer.score_graham(),
            'asset_based': scorer.score_asset_based()
        }

        # Step 3: Save explanations
        self.explanations = scorer.explanations

    def report(self):
        Reporting.print_analysis(self.fit_scores, self.explanations)

    def recommend(self, top_n=2):
        return Reporting.summary(self.fit_scores, top_n)
