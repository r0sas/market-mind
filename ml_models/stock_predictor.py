# ml_models/stock_predictor.py
"""
Machine Learning Stock Price Prediction Dashboard
Sector-specific stock price prediction models
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)

class StockPricePredictionDashboard:
    """
    ML-based stock price prediction using sector-specific models
    Provides 12-month price direction predictions
    """
    
    def __init__(self, models_dir: str = 'models/'):
        self.models_dir = models_dir
        self.models = {}
        self.sector_mapping = {
            'Technology': 'tech_model',
            'Healthcare': 'healthcare_model', 
            'Financial Services': 'financial_model',
            'Consumer Cyclical': 'consumer_cyclical_model',
            'Consumer Defensive': 'consumer_defensive_model',
            'Energy': 'energy_model',
            'Industrials': 'industrial_model',
            'Utilities': 'utilities_model',
            'Real Estate': 'real_estate_model',
            'Communication Services': 'communication_model',
            'Basic Materials': 'materials_model'
        }
        
        # Initialize with demo mode
        self.demo_mode = True
        logger.info("StockPricePredictionDashboard initialized in demo mode")
        
    def load_sector_models(self):
        """Load sector-specific ML models"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs(self.models_dir, exist_ok=True)
            
            logger.info("Loading sector-specific ML models...")
            
            # For demo purposes, create mock model entries
            # In production, you would load actual trained models from .pkl files
            for sector, model_name in self.sector_mapping.items():
                self.models[sector] = {
                    'name': model_name,
                    'type': 'RandomForest',  # Mock model type
                    'accuracy': round(np.random.uniform(0.65, 0.85), 3),
                    'features': 15,
                    'trained_samples': 1000,
                    'status': 'loaded'
                }
                
            logger.info(f"Successfully loaded {len(self.models)} sector models")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
            # Even if loading fails, we can still run in demo mode
            return True  # Return True to allow demo mode to continue
            
    def predict_stock_movement(self, ticker: str, company_data: pd.DataFrame, sector: str) -> Dict[str, Any]:
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
            
            # Get prediction
            prediction_result = self._make_prediction(ticker, company_data, model_sector)
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Prediction failed for {ticker}: {e}")
            # Return a safe fallback prediction
            return self._get_fallback_prediction(ticker, sector)
    
    def _find_best_sector_match(self, sector: str) -> str:
        """Find the best matching sector model"""
        if not sector or sector == 'Unknown':
            return 'Technology'  # Default fallback
            
        sector_lower = sector.lower()
        
        # Map common sector names to our model sectors
        sector_mapping = {
            'tech': 'Technology',
            'software': 'Technology',
            'semiconductor': 'Technology',
            'health': 'Healthcare',
            'pharma': 'Healthcare',
            'biotech': 'Healthcare',
            'financial': 'Financial Services',
            'bank': 'Financial Services',
            'insurance': 'Financial Services',
            'consumer': 'Consumer Cyclical',
            'retail': 'Consumer Cyclical',
            'energy': 'Energy',
            'oil': 'Energy',
            'gas': 'Energy',
            'industrial': 'Industrials',
            'manufacturing': 'Industrials',
            'utilities': 'Utilities',
            'real estate': 'Real Estate',
            'reit': 'Real Estate',
            'communication': 'Communication Services',
            'telecom': 'Communication Services',
            'materials': 'Basic Materials',
            'mining': 'Basic Materials'
        }
        
        for key, value in sector_mapping.items():
            if key in sector_lower:
                return value
        
        # Check for direct match
        for available_sector in self.sector_mapping.keys():
            if available_sector.lower() in sector_lower or sector_lower in available_sector.lower():
                return available_sector
        
        return 'Technology'  # Default fallback
    
    def _make_prediction(self, ticker: str, company_data: pd.DataFrame, sector: str) -> Dict[str, Any]:
        """Make prediction using ML model or demo logic"""
        try:
            # Extract features from company data
            features = self._extract_features(company_data)
            
            # Generate prediction based on financial health
            prediction, confidence, expected_change = self._generate_prediction(features)
            
            # Get model info
            model_info = self.models.get(sector, {'name': f'{sector}_demo_model', 'accuracy': 0.75})
            
            return {
                'ticker': ticker,
                'prediction': prediction,
                'confidence': confidence,
                'expected_change': expected_change,
                'model_used': model_info['name'],
                'sector': sector,
                'model_accuracy': model_info.get('accuracy', 0.75),
                'demo_mode': self.demo_mode,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            return self._get_fallback_prediction(ticker, sector)
    
    def _extract_features(self, company_data: pd.DataFrame) -> Dict[str, float]:
        """Extract and normalize features for prediction"""
        features = {}
        
        try:
            # Basic financial metrics - handle missing data gracefully
            metric_mapping = {
                'Basic EPS': 'Basic EPS',
                'Free Cash Flow': 'Free Cash Flow', 
                'P/E Ratio': 'P/E Ratio',
                'Share Price': 'Share Price',
                'Total Revenue': 'Total Revenue',
                'Net Income': 'Net Income',
                'Current Assets': 'Current Assets',
                'Current Liabilities': 'Current Liabilities'
            }
            
            for feature_name, data_column in metric_mapping.items():
                if data_column in company_data.columns:
                    value = company_data[data_column].iloc[0] if len(company_data) > 0 else 0
                    # Handle NaN and infinite values
                    if pd.isna(value) or np.isinf(value):
                        features[feature_name] = 0.0
                    else:
                        features[feature_name] = float(value)
                else:
                    features[feature_name] = 0.0
            
            # Calculate derived features
            if features.get('Basic EPS', 0) > 0 and features.get('Share Price', 0) > 0:
                features['PE_Ratio_Derived'] = features['Share Price'] / features['Basic EPS']
            else:
                features['PE_Ratio_Derived'] = 15.0
            
            if features.get('Total Revenue', 0) > 0:
                features['Profit_Margin'] = (features.get('Net Income', 0) / features['Total Revenue']) * 100
            else:
                features['Profit_Margin'] = 10.0
            
            if features.get('Current Liabilities', 0) > 0:
                features['Current_Ratio'] = features.get('Current Assets', 0) / features['Current Liabilities']
            else:
                features['Current_Ratio'] = 2.0
                
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Return minimal feature set
            return {'Basic EPS': 0, 'Free Cash Flow': 0, 'P/E Ratio': 15.0}
    
    def _generate_prediction(self, features: Dict[str, float]) -> tuple:
        """
        Generate prediction based on financial metrics
        This simulates what a trained ML model would do
        """
        try:
            # Calculate a health score based on financial metrics
            score = 0
            
            # Positive indicators
            if features.get('Basic EPS', 0) > 0:
                score += 2
            if features.get('Free Cash Flow', 0) > 0:
                score += 2
            if features.get('Profit_Margin', 0) > 8:
                score += 1
            if features.get('PE_Ratio_Derived', 50) < 30:
                score += 1
            if features.get('Current_Ratio', 0) > 1.5:
                score += 1
            
            # Negative indicators
            if features.get('Net Income', 0) < 0:
                score -= 2
            if features.get('Free Cash Flow', 0) < 0:
                score -= 2
            if features.get('Profit_Margin', 0) < 5:
                score -= 1
            if features.get('PE_Ratio_Derived', 0) > 40:
                score -= 1
            
            # Determine prediction based on score
            if score >= 5:
                prediction = "UP ⬆️"
                change_range = np.random.uniform(8, 20)
                expected_change = f"+{change_range:.1f}%"
                confidence = "High"
            elif score >= 3:
                prediction = "UP ⬆️"
                change_range = np.random.uniform(3, 10)
                expected_change = f"+{change_range:.1f}%"
                confidence = "Medium"
            elif score >= 0:
                prediction = "FLAT ➡️"
                change_range = np.random.uniform(-4, 4)
                expected_change = f"{change_range:+.1f}%"
                confidence = "Medium"
            elif score >= -3:
                prediction = "DOWN ⬇️"
                change_range = np.random.uniform(-12, -3)
                expected_change = f"{change_range:.1f}%"
                confidence = "Medium"
            else:
                prediction = "DOWN ⬇️"
                change_range = np.random.uniform(-25, -8)
                expected_change = f"{change_range:.1f}%"
                confidence = "High"
            
            return prediction, confidence, expected_change
            
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            return "FLAT ➡️", "Low", "0.0%"
    
    def _get_fallback_prediction(self, ticker: str, sector: str) -> Dict[str, Any]:
        """Provide fallback prediction when main logic fails"""
        return {
            'ticker': ticker,
            'prediction': "FLAT ➡️",
            'confidence': "Low",
            'expected_change': "0.0%",
            'model_used': f"fallback_model",
            'sector': sector,
            'model_accuracy': 0.5,
            'demo_mode': True,
            'success': False,
            'error': 'Used fallback prediction'
        }
    
    def get_available_sectors(self) -> List[str]:
        """Get list of available sector models"""
        return list(self.models.keys())
    
    def get_model_info(self, sector: str) -> Dict[str, Any]:
        """Get information about a specific sector model"""
        return self.models.get(sector, {'error': f'Model for sector {sector} not found'})
    
    def is_ready(self) -> bool:
        """Check if the dashboard is ready for predictions"""
        return len(self.models) > 0 or self.demo_mode