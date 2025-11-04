# ml_models/stock_predictor.py
"""
Machine Learning Stock Price Prediction Dashboard
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class StockPricePredictionDashboard:
    """
    ML-based stock price prediction using sector-specific models
    """
    
    def __init__(self, models_dir: str = 'models/'):
        self.models_dir = models_dir
        self.models = {}
        self.sector_mapping = {
            'Technology': 'tech_model.pkl',
            'Healthcare': 'healthcare_model.pkl', 
            'Financial Services': 'financial_model.pkl',
            'Consumer Cyclical': 'consumer_cyclical_model.pkl',
            'Consumer Defensive': 'consumer_defensive_model.pkl',
            'Energy': 'energy_model.pkl',
            'Industrials': 'industrial_model.pkl',
            'Utilities': 'utilities_model.pkl',
            'Real Estate': 'real_estate_model.pkl',
            'Communication Services': 'communication_model.pkl',
            'Basic Materials': 'materials_model.pkl'
        }
        
    def load_sector_models(self):
        """Load sector-specific ML models"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs(self.models_dir, exist_ok=True)
            
            # For demo purposes, we'll create mock models
            # In a real implementation, you would load trained models from files
            logger.info("Loading sector-specific ML models...")
            
            # Create mock models for each sector
            for sector, model_file in self.sector_mapping.items():
                model_path = os.path.join(self.models_dir, model_file)
                
                # For now, we'll use a mock model
                # In production, you would load actual trained models
                self.models[sector] = {
                    'model': f"MockModel_{sector}",
                    'features': ['Basic EPS', 'Free Cash Flow', 'P/E Ratio', 'Share Price', 
                                'Total Revenue', 'Net Income', 'Current Assets', 'Current Liabilities'],
                    'accuracy': np.random.uniform(0.65, 0.85),  # Mock accuracy
                    'loaded': True
                }
                
            logger.info(f"Loaded {len(self.models)} sector models")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
            return False
    
    def predict_stock_movement(self, ticker: str, company_data: pd.DataFrame, sector: str) -> Dict:
        """
        Predict stock price movement using sector-specific ML model
        
        Args:
            ticker: Stock ticker symbol
            company_data: DataFrame with financial metrics
            sector: Company sector
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Find the best matching sector model
            model_sector = self._find_best_sector_match(sector)
            
            if model_sector not in self.models:
                # Use default model if sector-specific model not available
                model_sector = 'Technology'  # Default fallback
            
            # Get prediction from model
            prediction_result = self._make_prediction(ticker, company_data, model_sector)
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Prediction failed for {ticker}: {e}")
            return self._get_fallback_prediction(ticker, sector)
    
    def _find_best_sector_match(self, sector: str) -> str:
        """Find the best matching sector model"""
        sector_lower = sector.lower()
        
        # Map common sector names to our model sectors
        sector_mapping = {
            'technology': 'Technology',
            'healthcare': 'Healthcare',
            'financial': 'Financial Services',
            'banks': 'Financial Services',
            'consumer cyclical': 'Consumer Cyclical',
            'consumer defensive': 'Consumer Defensive',
            'energy': 'Energy',
            'industrials': 'Industrials',
            'utilities': 'Utilities',
            'real estate': 'Real Estate',
            'communication': 'Communication Services',
            'materials': 'Basic Materials'
        }
        
        for key, value in sector_mapping.items():
            if key in sector_lower:
                return value
        
        return 'Technology'  # Default fallback
    
    def _make_prediction(self, ticker: str, company_data: pd.DataFrame, sector: str) -> Dict:
        """Make actual prediction using ML model"""
        try:
            # Extract features from company data
            features = self._extract_features(company_data)
            
            # Mock prediction logic - replace with actual model inference
            # This would normally call model.predict() or model.predict_proba()
            
            # Simulate ML prediction based on financial metrics
            prediction, confidence, expected_change = self._simulate_ml_prediction(features)
            
            return {
                'ticker': ticker,
                'prediction': prediction,
                'confidence': confidence,
                'expected_change': expected_change,
                'model_used': f"{sector} Model",
                'sector': sector,
                'features_used': len(features),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Prediction simulation failed: {e}")
            return self._get_fallback_prediction(ticker, sector)
    
    def _extract_features(self, company_data: pd.DataFrame) -> Dict:
        """Extract and normalize features for ML model"""
        features = {}
        
        # Basic financial metrics
        basic_metrics = ['Basic EPS', 'Free Cash Flow', 'P/E Ratio', 'Share Price', 
                        'Total Revenue', 'Net Income']
        
        for metric in basic_metrics:
            if metric in company_data.columns:
                value = company_data[metric].iloc[0] if not company_data[metric].isna().all() else 0
                features[metric] = float(value)
            else:
                features[metric] = 0.0
        
        # Calculate some derived features
        if features['Basic EPS'] > 0 and features['Share Price'] > 0:
            features['PE_Ratio_Calculated'] = features['Share Price'] / features['Basic EPS']
        else:
            features['PE_Ratio_Calculated'] = 15.0  # Default
        
        if features['Total Revenue'] > 0:
            features['Profit_Margin'] = features['Net Income'] / features['Total Revenue']
        else:
            features['Profit_Margin'] = 0.1  # Default
        
        return features
    
    def _simulate_ml_prediction(self, features: Dict) -> tuple:
        """
        Simulate ML prediction based on financial metrics
        In production, this would be replaced with actual model inference
        """
        # Mock prediction logic based on financial health
        score = 0
        
        # Positive factors
        if features.get('Basic EPS', 0) > 0:
            score += 2
        if features.get('Free Cash Flow', 0) > 0:
            score += 2
        if features.get('Profit_Margin', 0) > 0.1:
            score += 1
        if features.get('PE_Ratio_Calculated', 50) < 25:
            score += 1
        
        # Negative factors
        if features.get('Net Income', 0) < 0:
            score -= 2
        if features.get('Free Cash Flow', 0) < 0:
            score -= 1
        
        # Determine prediction based on score
        if score >= 4:
            prediction = "UP ⬆️"
            expected_change = f"+{np.random.uniform(5, 15):.1f}%"
            confidence = "High"
        elif score >= 2:
            prediction = "UP ⬆️"
            expected_change = f"+{np.random.uniform(2, 8):.1f}%"
            confidence = "Medium"
        elif score >= 0:
            prediction = "FLAT ➡️"
            expected_change = f"{np.random.uniform(-3, 3):.1f}%"
            confidence = "Medium"
        elif score >= -2:
            prediction = "DOWN ⬇️"
            expected_change = f"{np.random.uniform(-8, -2):.1f}%"
            confidence = "Medium"
        else:
            prediction = "DOWN ⬇️"
            expected_change = f"{np.random.uniform(-15, -5):.1f}%"
            confidence = "High"
        
        return prediction, confidence, expected_change
    
    def _get_fallback_prediction(self, ticker: str, sector: str) -> Dict:
        """Provide fallback prediction when ML model fails"""
        return {
            'ticker': ticker,
            'prediction': "FLAT ➡️",
            'confidence': "Low",
            'expected_change': f"{np.random.uniform(-2, 2):.1f}%",
            'model_used': f"Fallback Model",
            'sector': sector,
            'features_used': 0,
            'success': False
        }
    
    def get_available_sectors(self) -> List[str]:
        """Get list of available sector models"""
        return list(self.models.keys())
    
    def get_model_info(self, sector: str) -> Dict:
        """Get information about a specific sector model"""
        if sector in self.models:
            return self.models[sector]
        else:
            return {'error': f'Model for sector {sector} not found'}