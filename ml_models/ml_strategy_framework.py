# ============================================
# PHASE 2: EDA & FEATURE ENGINEERING
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class SectorFinancialAnalyzer:
    """
    Analyzes financial metrics by sector to identify predictive features.
    
    Phase 2 of the predictive modeling pipeline.
    """
    
    def __init__(self, combined_df: pd.DataFrame):
        """
        Initialize with comprehensive financial data.
        
        Args:
            combined_df: DataFrame with all companies, sectors, metrics over time
        """
        self.df = combined_df.copy()
        self.sector_insights = {}
        self.feature_importance = {}
        
    def prepare_target_variable(self, horizon: int = 1):
        """
        Create target variable: future price movement.
        
        Args:
            horizon: How many years ahead to predict (default 1)
            
        Returns:
            DataFrame with target variable added
        """
        self.df = self.df.sort_values(['Ticker', 'Sequential_Year'])
        
        # Future price change (%)
        self.df['Target_Price_Change'] = (
            self.df.groupby('Ticker')['Share Price']
            .shift(-horizon) / self.df['Share Price'] - 1
        ) * 100
        
        # Binary classification: Up (1) or Down (0)
        self.df['Target_Direction'] = (self.df['Target_Price_Change'] > 0).astype(int)
        
        # Drop rows without future data (last N years per company)
        self.df = self.df.dropna(subset=['Target_Price_Change'])
        
        return self.df
    
    def calculate_feature_correlations(self, sector: str = None):
        """
        Calculate correlation between features and target for each sector.
        
        Args:
            sector: Specific sector to analyze (None = all sectors)
            
        Returns:
            DataFrame with correlation scores per feature
        """
        if sector:
            sector_df = self.df[self.df['Sector'] == sector].copy()
            sectors_to_analyze = [sector]
        else:
            sectors_to_analyze = self.df['Sector'].unique()
        
        results = {}
        
        for sec in sectors_to_analyze:
            sector_df = self.df[self.df['Sector'] == sec].copy()
            
            # Get numeric columns (exclude identifiers)
            numeric_cols = sector_df.select_dtypes(include=[np.number]).columns
            exclude_cols = ['Ticker', 'Sequential_Year', 'Year', 
                          'Target_Price_Change', 'Target_Direction']
            feature_cols = [c for c in numeric_cols if c not in exclude_cols]
            
            # Calculate correlations
            correlations = {}
            for col in feature_cols:
                # Remove NaN and infinite values
                valid_data = sector_df[[col, 'Target_Price_Change']].replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()
                
                if len(valid_data) > 10:  # Minimum samples
                    corr = valid_data[col].corr(valid_data['Target_Price_Change'])
                    correlations[col] = corr
                else:
                    correlations[col] = 0
            
            results[sec] = pd.Series(correlations).sort_values(ascending=False)
        
        self.feature_importance = results
        return results
    
    def plot_top_features_by_sector(self, top_n: int = 10):
        """
        Visualize top predictive features for each sector.
        """
        if not self.feature_importance:
            self.calculate_feature_correlations()
        
        sectors = list(self.feature_importance.keys())
        n_sectors = len(sectors)
        
        fig, axes = plt.subplots(
            (n_sectors + 1) // 2, 2, 
            figsize=(16, 5 * ((n_sectors + 1) // 2))
        )
        axes = axes.flatten()
        
        for idx, sector in enumerate(sectors):
            top_features = self.feature_importance[sector].head(top_n)
            
            axes[idx].barh(range(len(top_features)), top_features.values)
            axes[idx].set_yticks(range(len(top_features)))
            axes[idx].set_yticklabels(top_features.index)
            axes[idx].set_xlabel('Correlation with Future Price Change')
            axes[idx].set_title(f'{sector} - Top {top_n} Predictive Features')
            axes[idx].axvline(x=0, color='red', linestyle='--', alpha=0.5)
            axes[idx].grid(axis='x', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(sectors), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('sector_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_sector_comparison_heatmap(self):
        """
        Create heatmap showing which features matter for which sectors.
        """
        if not self.feature_importance:
            self.calculate_feature_correlations()
        
        # Get union of all top features across sectors
        all_features = set()
        for sector_features in self.feature_importance.values():
            all_features.update(sector_features.head(15).index)
        
        # Build matrix
        matrix_data = []
        for sector in self.feature_importance.keys():
            row = []
            for feature in sorted(all_features):
                corr = self.feature_importance[sector].get(feature, 0)
                row.append(corr)
            matrix_data.append(row)
        
        heatmap_df = pd.DataFrame(
            matrix_data,
            index=self.feature_importance.keys(),
            columns=sorted(all_features)
        )
        
        plt.figure(figsize=(20, 10))
        sns.heatmap(
            heatmap_df.T,  # Transpose for better readability
            cmap='RdYlGn',
            center=0,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Correlation with Price Change'}
        )
        plt.title('Feature Importance Across Sectors', fontsize=16, pad=20)
        plt.xlabel('Sector', fontsize=12)
        plt.ylabel('Financial Metric', fontsize=12)
        plt.tight_layout()
        plt.savefig('sector_feature_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_feature_stability(self, metric: str, sector: str = None):
        """
        Check if a metric is stable or volatile over time (important for prediction).
        
        Args:
            metric: Feature to analyze
            sector: Specific sector (None = all sectors)
        """
        if sector:
            data = self.df[self.df['Sector'] == sector]
        else:
            data = self.df
        
        # Calculate coefficient of variation by company
        cv_by_company = data.groupby('Ticker')[metric].apply(
            lambda x: x.std() / x.mean() if x.mean() != 0 else np.nan
        ).dropna()
        
        plt.figure(figsize=(12, 6))
        plt.hist(cv_by_company, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(cv_by_company.median(), color='red', linestyle='--', 
                   label=f'Median CV: {cv_by_company.median():.2f}')
        plt.xlabel('Coefficient of Variation')
        plt.ylabel('Number of Companies')
        plt.title(f'{metric} Stability Analysis - {sector or "All Sectors"}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
        
        return {
            'mean_cv': cv_by_company.mean(),
            'median_cv': cv_by_company.median(),
            'stability_score': 1 / (1 + cv_by_company.median())  # Higher = more stable
        }
    
    def identify_lagging_vs_leading_indicators(self, sector: str):
        """
        Determine if features are lagging (historical) or leading (predictive).
        
        Tests correlation with price change at different time lags.
        """
        sector_df = self.df[self.df['Sector'] == sector].copy()
        sector_df = sector_df.sort_values(['Ticker', 'Sequential_Year'])
        
        numeric_cols = sector_df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['Ticker', 'Sequential_Year', 'Year', 
                       'Target_Price_Change', 'Target_Direction']
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        lag_analysis = {}
        
        for feature in feature_cols[:10]:  # Top 10 for speed
            correlations_by_lag = {}
            
            for lag in range(-2, 3):  # -2 years to +2 years
                # Create lagged target
                sector_df[f'Price_Lag_{lag}'] = sector_df.groupby('Ticker')['Share Price'].shift(lag)
                sector_df[f'Price_Change_Lag_{lag}'] = (
                    sector_df[f'Price_Lag_{lag}'] / sector_df['Share Price'] - 1
                ) * 100
                
                valid_data = sector_df[[feature, f'Price_Change_Lag_{lag}']].replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()
                
                if len(valid_data) > 10:
                    corr = valid_data[feature].corr(valid_data[f'Price_Change_Lag_{lag}'])
                    correlations_by_lag[lag] = corr
            
            lag_analysis[feature] = correlations_by_lag
        
        # Plot results
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for feature, lags in lag_analysis.items():
            lag_values = sorted(lags.keys())
            corr_values = [lags[l] for l in lag_values]
            ax.plot(lag_values, corr_values, marker='o', label=feature)
        
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Current Year')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel('Time Lag (years)', fontsize=12)
        ax.set_ylabel('Correlation with Price Change', fontsize=12)
        ax.set_title(f'{sector} - Leading vs Lagging Indicators', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return lag_analysis
    
    def generate_feature_engineering_suggestions(self, sector: str):
        """
        Suggest new features to engineer based on top predictors.
        
        Returns:
            List of suggested engineered features
        """
        if not self.feature_importance:
            self.calculate_feature_correlations()
        
        top_features = self.feature_importance[sector].head(10)
        
        suggestions = {
            'Ratios': [],
            'Growth Rates': [],
            'Trends': [],
            'Composite Scores': []
        }
        
        # Suggest ratios
        if 'Net Income' in top_features.index and 'Total Revenue' in top_features.index:
            suggestions['Ratios'].append('Net_Margin = Net Income / Total Revenue')
        
        if 'Free Cash Flow' in top_features.index and 'Total Revenue' in top_features.index:
            suggestions['Ratios'].append('FCF_to_Revenue = Free Cash Flow / Total Revenue')
        
        # Suggest growth rates
        for feature in ['Basic EPS', 'Free Cash Flow', 'Total Revenue']:
            if feature in top_features.index:
                suggestions['Growth Rates'].append(f'{feature}_YoY_Growth')
                suggestions['Growth Rates'].append(f'{feature}_3Y_CAGR')
        
        # Suggest trends
        suggestions['Trends'].append('EPS_Trend_Slope (3-year regression)')
        suggestions['Trends'].append('P/E_Trend_Direction')
        suggestions['Trends'].append('FCF_Consistency_Score')
        
        # Suggest composite scores
        suggestions['Composite Scores'].append('Profitability_Score (ROE + Net Margin + ROA)')
        suggestions['Composite Scores'].append('Growth_Score (Revenue Growth + EPS Growth)')
        suggestions['Composite Scores'].append('Value_Score (P/E + P/B + P/S inverted)')
        
        return suggestions
    
    def export_feature_importance_report(self, filename: str = 'feature_importance_report.csv'):
        """
        Export comprehensive feature importance analysis to CSV.
        """
        if not self.feature_importance:
            self.calculate_feature_correlations()
        
        # Combine all sectors
        all_data = []
        for sector, features in self.feature_importance.items():
            for feature, corr in features.items():
                all_data.append({
                    'Sector': sector,
                    'Feature': feature,
                    'Correlation': corr,
                    'Abs_Correlation': abs(corr)
                })
        
        report_df = pd.DataFrame(all_data).sort_values(
            ['Sector', 'Abs_Correlation'], 
            ascending=[True, False]
        )
        
        report_df.to_csv(filename, index=False)
        print(f"Feature importance report exported to {filename}")
        
        return report_df


# ============================================
# PHASE 3: MODEL TRAINING
# ============================================

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

class SectorPredictiveModel:
    """
    Trains sector-specific models to predict stock price movements.
    
    Phase 3 of the predictive modeling pipeline.
    """
    
    def __init__(self, sector_df: pd.DataFrame, sector_name: str):
        """
        Initialize with sector-specific data.
        
        Args:
            sector_df: DataFrame for single sector with target variable
            sector_name: Name of the sector
        """
        self.sector = sector_name
        self.df = sector_df.copy()
        self.models = {}
        self.feature_cols = None
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        
    def prepare_features(self, top_n_features: int = 20):
        """
        Select and prepare features for modeling.
        
        Args:
            top_n_features: Number of top features to use
        """
        # Calculate feature importance first
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['Ticker', 'Sequential_Year', 'Year', 
                       'Target_Price_Change', 'Target_Direction', 'Share Price']
        
        self.feature_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        # Remove features with too many NaN
        valid_features = []
        for col in self.feature_cols:
            if self.df[col].notna().sum() / len(self.df) > 0.5:  # At least 50% valid
                valid_features.append(col)
        
        self.feature_cols = valid_features[:top_n_features]
        
        print(f"Selected {len(self.feature_cols)} features for {self.sector}")
        print(f"Features: {self.feature_cols}")
        
        return self.feature_cols
    
    def train_models(self, test_size: int = 2):
        """
        Train multiple models and compare performance.
        
        Args:
            test_size: Number of years to use for testing (time-series split)
        """
        # Prepare data
        X = self.df[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = self.df['Target_Price_Change'].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_cols, index=X.index)
        
        # Time-series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Models to test
        models_to_try = {
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1)
        }
        
        results = {}
        
        for name, model in models_to_try.items():
            print(f"\nTraining {name} for {self.sector}...")
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_scaled, y, 
                cv=tscv, 
                scoring='neg_mean_absolute_error'
            )
            
            # Train on full data
            model.fit(X_scaled, y)
            self.models[name] = model
            
            # Store results
            results[name] = {
                'CV_MAE': -cv_scores.mean(),
                'CV_STD': cv_scores.std(),
                'model': model
            }
            
            print(f"  CV MAE: {-cv_scores.mean():.2f}% (+/- {cv_scores.std():.2f}%)")
        
        # Select best model
        best_name = min(results, key=lambda x: results[x]['CV_MAE'])
        self.best_model = results[best_name]['model']
        self.best_model_name = best_name
        
        print(f"\n🏆 Best model for {self.sector}: {best_name}")
        print(f"   Expected error: ±{results[best_name]['CV_MAE']:.2f}%")
        
        return results
    
    def get_feature_importance(self):
        """
        Get feature importance from best model (if tree-based).
        """
        if self.best_model_name in ['Random Forest', 'Gradient Boosting']:
            importances = self.best_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': self.feature_cols,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            return feature_importance_df
        else:
            # For linear models, use coefficients
            coeffs = self.best_model.coef_
            feature_importance_df = pd.DataFrame({
                'Feature': self.feature_cols,
                'Coefficient': coeffs
            }).sort_values('Coefficient', key=abs, ascending=False)
            
            return feature_importance_df
    
    def predict(self, company_data: pd.DataFrame):
        """
        Predict price change for a company.
        
        Args:
            company_data: DataFrame with single row of company metrics
            
        Returns:
            Dict with prediction and confidence
        """
        # Prepare features
        X = company_data[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.best_model.predict(X_scaled)[0]
        
        # Get prediction from all models for confidence estimate
        all_predictions = []
        for model in self.models.values():
            pred = model.predict(X_scaled)[0]
            all_predictions.append(pred)
        
        std_dev = np.std(all_predictions)
        
        return {
            'predicted_change_pct': prediction,
            'direction': 'UP' if prediction > 0 else 'DOWN',
            'confidence': 'High' if std_dev < 5 else 'Medium' if std_dev < 10 else 'Low',
            'uncertainty': std_dev
        }
    
    def save_model(self, filename: str = None):
        """Save trained model to disk."""
        if filename is None:
            filename = f"{self.sector.replace(' ', '_')}_model.pkl"
        
        model_package = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'sector': self.sector,
            'model_name': self.best_model_name
        }
        
        joblib.dump(model_package, filename)
        print(f"Model saved to {filename}")
    
    @staticmethod
    def load_model(filename: str):
        """Load trained model from disk."""
        return joblib.load(filename)


# ============================================
# PHASE 4: DASHBOARD INTEGRATION
# ============================================

class StockPricePredictionDashboard:
    """
    Integration component for Streamlit dashboard.
    
    Provides simple interface to get predictions for any stock.
    """
    
    def __init__(self, models_dir: str = 'models/'):
        """
        Load all sector models.
        
        Args:
            models_dir: Directory containing saved models
        """
        self.models = {}
        self.models_dir = models_dir
        
    def load_sector_models(self):
        """Load all trained sector models."""
        import os
        import glob
        
        model_files = glob.glob(f"{self.models_dir}/*_model.pkl")
        
        for model_file in model_files:
            sector_name = os.path.basename(model_file).replace('_model.pkl', '').replace('_', ' ')
            self.models[sector_name] = SectorPredictiveModel.load_model(model_file)
            print(f"Loaded model for {sector_name}")
    
    def predict_stock_movement(self, ticker: str, company_data: pd.DataFrame, sector: str):
        """
        Predict if stock will go up or down.
        
        Args:
            ticker: Stock ticker
            company_data: Latest financial metrics for the company
            sector: Company's sector
            
        Returns:
            Dict with prediction, confidence, and explanation
        """
        if sector not in self.models:
            return {
                'prediction': 'UNKNOWN',
                'confidence': 'N/A',
                'message': f'No model available for {sector} sector'
            }
        
        model_package = self.models[sector]
        model = model_package['model']
        scaler = model_package['scaler']
        feature_cols = model_package['feature_cols']
        
        # Prepare data
        X = company_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = scaler.transform(X)
        
        # Predict
        prediction_pct = model.predict(X_scaled)[0]
        
        direction = 'UP ⬆️' if prediction_pct > 5 else 'DOWN ⬇️' if prediction_pct < -5 else 'FLAT ➡️'
        
        return {
            'ticker': ticker,
            'prediction': direction,
            'expected_change': f"{prediction_pct:+.1f}%",
            'confidence': 'Medium',  # Could be enhanced with ensemble variance
            'model_used': model_package['model_name'],
            'sector': sector
        }


# ============================================
# USAGE EXAMPLE
# ============================================

def example_full_pipeline():
    """
    Complete example of the 4-phase pipeline.
    """
    
    # Load your comprehensive data
    combined_df = pd.read_csv('aggregated_data.csv')
    
    # PHASE 2: EDA
    print("="*60)
    print("PHASE 2: EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    analyzer = SectorFinancialAnalyzer(combined_df)
    analyzer.prepare_target_variable(horizon=1)
    
    # Analyze correlations
    correlations = analyzer.calculate_feature_correlations()
    
    # Visualize
    analyzer.plot_top_features_by_sector(top_n=10)
    analyzer.create_sector_comparison_heatmap()
    
    # Export report
    report = analyzer.export_feature_importance_report()
    
    # PHASE 3: MODEL TRAINING
    print("\n✅ Model training complete!")
    
    # PHASE 4: DASHBOARD INTEGRATION
    print("\n" + "="*60)
    print("PHASE 4: DASHBOARD INTEGRATION")
    print("="*60)
    
    dashboard = StockPricePredictionDashboard(models_dir='models/')
    dashboard.load_sector_models()
    
    # Test prediction
    print("\n### Testing Prediction ###")
    
    # Get latest data for a company (example)
    test_ticker = 'AAPL'
    test_company_data = combined_df[
        (combined_df['Ticker'] == test_ticker) & 
        (combined_df['Sequential_Year'] == combined_df['Sequential_Year'].max())
    ].iloc[0:1]
    
    prediction = dashboard.predict_stock_movement(
        ticker=test_ticker,
        company_data=test_company_data,
        sector='Technology'
    )
    
    print(f"\n🎯 Prediction for {prediction['ticker']}:")
    print(f"   Direction: {prediction['prediction']}")
    print(f"   Expected Change: {prediction['expected_change']}")
    print(f"   Confidence: {prediction['confidence']}")
    print(f"   Model: {prediction['model_used']}")
    
    return analyzer, dashboard


if __name__ == "__main__":
    # Run full pipeline
    analyzer, dashboard = example_full_pipeline()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review feature importance reports")
    print("2. Integrate dashboard.predict_stock_movement() into Streamlit")
    print("3. Add prediction display in your valuation results")
    print("\n" + "="*60)
    print("PHASE 3: MODEL TRAINING")
    print("="*60)
    
    sectors = analyzer.df['Sector'].unique()
    
    for sector in sectors:
        print(f"\n### Training model for {sector} ###")
        
        sector_df = analyzer.df[analyzer.df['Sector'] == sector]
        
        if len(sector_df) < 50:  # Skip sectors with too little data
            print(f"Skipping {sector} (insufficient data)")
            continue
        
        model = SectorPredictiveModel(sector_df, sector)
        model.prepare_features(top_n_features=15)
        results = model.train_models()
        
        # Show feature importance
        importance_df = model.get_feature_importance()
        print("\nTop 10 Important Features:")
        print(importance_df.head(10))
        
        # Save model
        model.save_model(f"models/{sector.replace(' ', '_')}_model.pkl")
    
    print("\n✅ Model training complete!")