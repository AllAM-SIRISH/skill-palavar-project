import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Any
import pickle
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveModelEvaluator:
    def __init__(self):
        self.results = {}
        self.metrics_history = []
        
    def evaluate_comprehensive(self, models: Dict, X_test: np.ndarray, y_test: np.ndarray, 
                             feature_names: List[str]) -> Dict:
        """Comprehensive evaluation of all models"""
        print("🔍 Starting Comprehensive Model Evaluation")
        print("=" * 60)
        
        evaluation_results = {}
        
        for model_name, model in models.items():
            print(f"\n📊 Evaluating {model_name.upper()} Model")
            print("-" * 40)
            
            # Get predictions and probabilities
            y_pred = model.predict(X_test)
            
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
            else:
                y_proba = None
            
            # Basic metrics
            metrics = self.calculate_basic_metrics(y_test, y_pred, y_proba)
            
            # Advanced metrics
            advanced_metrics = self.calculate_advanced_metrics(y_test, y_pred, y_proba)
            
            # Business metrics
            business_metrics = self.calculate_business_metrics(y_test, y_pred, y_proba)
            
            # Cross-validation
            cv_metrics = self.perform_cross_validation(model, X_test, y_test)
            
            # Feature importance (if available)
            feature_importance = self.extract_feature_importance(model, feature_names)
            
            # Combine all results
            model_results = {
                'basic_metrics': metrics,
                'advanced_metrics': advanced_metrics,
                'business_metrics': business_metrics,
                'cv_metrics': cv_metrics,
                'feature_importance': feature_importance,
                'predictions': y_pred,
                'probabilities': y_proba
            }
            
            evaluation_results[model_name] = model_results
            
            # Print summary
            self.print_model_summary(model_name, model_results)
        
        self.results = evaluation_results
        return evaluation_results
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_proba: np.ndarray = None) -> Dict:
        """Calculate basic classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                metrics['pr_auc'] = average_precision_score(y_true, y_proba)
            except:
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
        
        return metrics
    
    def calculate_advanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_proba: np.ndarray = None) -> Dict:
        """Calculate advanced evaluation metrics"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'true_negative_rate': tn / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (tp + fn) if (tp + fn) > 0 else 0,
            'false_discovery_rate': fp / (tp + fp) if (tp + fp) > 0 else 0,
            'false_omission_rate': fn / (tn + fn) if (tn + fn) > 0 else 0,
            'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'matthews_corrcoef': self.calculate_matthews_corrcoef(y_true, y_pred)
        }
        
        if y_proba is not None:
            # Find optimal threshold
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            metrics['optimal_threshold'] = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            metrics['optimal_f1'] = f1_scores[optimal_idx]
        
        return metrics
    
    def calculate_business_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_proba: np.ndarray = None) -> Dict:
        """Calculate business-relevant metrics"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Assume business costs (can be adjusted based on real business requirements)
        cost_fp = 10  # Cost of false positive (customer inconvenience)
        cost_fn = 1000  # Cost of false negative (fraud loss)
        cost_tp = 50  # Cost of true positive (investigation)
        cost_tn = 1  # Cost of true negative (normal processing)
        
        total_cost = (fp * cost_fp) + (fn * cost_fn) + (tp * cost_tp) + (tn * cost_tn)
        
        metrics = {
            'total_cost': total_cost,
            'cost_per_transaction': total_cost / len(y_true),
            'fraud_prevention_savings': fn * cost_fn,  # Potential savings from catching fraud
            'customer_inconvenience_cost': fp * cost_fp,
            'investigation_cost': tp * cost_tp,
            'cost_reduction_percentage': ((fn * cost_fn) / total_cost * 100) if total_cost > 0 else 0
        }
        
        # ROI metrics
        if metrics['fraud_prevention_savings'] > 0:
            metrics['roi'] = (metrics['fraud_prevention_savings'] - metrics['investigation_cost']) / metrics['investigation_cost']
        else:
            metrics['roi'] = 0
        
        return metrics
    
    def calculate_matthews_corrcoef(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Matthews correlation coefficient"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def perform_cross_validation(self, model, X: np.ndarray, y: np.ndarray, 
                               cv_folds: int = 5) -> Dict:
        """Perform cross-validation"""
        try:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # For models that support predict_proba
            if hasattr(model, 'predict_proba'):
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
                metric_name = 'roc_auc'
            else:
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
                metric_name = 'f1'
            
            return {
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_metric': metric_name
            }
        except Exception as e:
            print(f"Cross-validation failed: {e}")
            return {'cv_mean': 0.0, 'cv_std': 0.0, 'cv_metric': 'failed'}
    
    def extract_feature_importance(self, model, feature_names: List[str]) -> Dict:
        """Extract feature importance if available"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return {
                'available': True,
                'top_features': feature_imp.head(20).to_dict('records'),
                'feature_importance_df': feature_imp
            }
        else:
            return {'available': False}
    
    def print_model_summary(self, model_name: str, results: Dict):
        """Print model evaluation summary"""
        basic = results['basic_metrics']
        advanced = results['advanced_metrics']
        business = results['business_metrics']
        cv = results['cv_metrics']
        
        print(f"📈 Performance Metrics:")
        print(f"   Accuracy: {basic['accuracy']:.4f}")
        print(f"   Precision: {basic['precision']:.4f}")
        print(f"   Recall: {basic['recall']:.4f}")
        print(f"   F1-Score: {basic['f1_score']:.4f}")
        
        if 'roc_auc' in basic:
            print(f"   ROC AUC: {basic['roc_auc']:.4f}")
            print(f"   PR AUC: {basic['pr_auc']:.4f}")
        
        print(f"\n💰 Business Impact:")
        print(f"   Total Cost: ${business['total_cost']:,.2f}")
        print(f"   Cost per Transaction: ${business['cost_per_transaction']:.2f}")
        print(f"   Fraud Prevention Savings: ${business['fraud_prevention_savings']:,.2f}")
        print(f"   ROI: {business['roi']:.2f}")
        
        print(f"\n🔄 Cross-Validation:")
        print(f"   {cv['cv_metric'].upper()}: {cv['cv_mean']:.4f} (+/- {cv['cv_std']:.4f})")
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Create comparison table of all models"""
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            basic = results['basic_metrics']
            business = results['business_metrics']
            cv = results['cv_metrics']
            
            row = {
                'Model': model_name,
                'Accuracy': basic['accuracy'],
                'Precision': basic['precision'],
                'Recall': basic['recall'],
                'F1-Score': basic['f1_score'],
                'ROC AUC': basic.get('roc_auc', 0.0),
                'Total Cost ($)': business['total_cost'],
                'ROI': business['roi'],
                'CV Score': cv['cv_mean']
            }
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def plot_model_comparison(self, save_path: str = None):
        """Create comprehensive comparison plots"""
        if not self.results:
            print("No results to plot")
            return
        
        # Create comparison table
        comparison_df = self.create_comparison_table()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Performance Metrics', 'Business Impact', 'ROC AUC Comparison',
                          'Precision vs Recall', 'Cost Analysis', 'Model Rankings'],
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}, {"type": "table"}]]
        )
        
        # Performance metrics
        fig.add_trace(
            go.Bar(x=comparison_df['Model'], y=comparison_df['Accuracy'], name='Accuracy', marker_color='lightblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=comparison_df['Model'], y=comparison_df['Precision'], name='Precision', marker_color='lightgreen'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=comparison_df['Model'], y=comparison_df['Recall'], name='Recall', marker_color='lightcoral'),
            row=1, col=1
        )
        
        # Business impact
        fig.add_trace(
            go.Bar(x=comparison_df['Model'], y=comparison_df['Total Cost ($)'], name='Total Cost', marker_color='orange'),
            row=1, col=2
        )
        
        # ROC AUC
        fig.add_trace(
            go.Bar(x=comparison_df['Model'], y=comparison_df['ROC AUC'], name='ROC AUC', marker_color='purple'),
            row=1, col=3
        )
        
        # Precision vs Recall scatter
        fig.add_trace(
            go.Scatter(x=comparison_df['Recall'], y=comparison_df['Precision'], 
                      mode='markers+text', text=comparison_df['Model'], 
                      textposition="top center", name='Models'),
            row=2, col=1
        )
        
        # Cost analysis
        fig.add_trace(
            go.Bar(x=comparison_df['Model'], y=comparison_df['ROI'], name='ROI', marker_color='gold'),
            row=2, col=2
        )
        
        # Rankings table
        fig.add_trace(
            go.Table(
                header=dict(values=['Model', 'F1-Score', 'ROC AUC', 'ROI']),
                cells=dict(values=[comparison_df['Model'], 
                                 comparison_df['F1-Score'].round(4),
                                 comparison_df['ROC AUC'].round(4),
                                 comparison_df['ROI'].round(2)])
            ),
            row=2, col=3
        )
        
        fig.update_layout(height=800, showlegend=False, title_text="Model Comparison Dashboard")
        
        if save_path:
            fig.write_html(save_path)
            print(f"Comparison plot saved to {save_path}")
        
        return fig
    
    def generate_report(self, save_path: str = None) -> str:
        """Generate comprehensive evaluation report"""
        if not self.results:
            return "No evaluation results available"
        
        report = []
        report.append("# 📊 Fraud Detection Model Evaluation Report")
        report.append("=" * 60)
        report.append("")
        
        # Executive summary
        report.append("## 🎯 Executive Summary")
        comparison_df = self.create_comparison_table()
        best_f1 = comparison_df.loc[comparison_df['F1-Score'].idxmax()]
        best_auc = comparison_df.loc[comparison_df['ROC AUC'].idxmax()]
        best_roi = comparison_df.loc[comparison_df['ROI'].idxmax()]
        
        report.append(f"- **Best Overall Performance**: {best_f1['Model']} (F1-Score: {best_f1['F1-Score']:.4f})")
        report.append(f"- **Best ROC AUC**: {best_auc['Model']} (ROC AUC: {best_auc['ROC AUC']:.4f})")
        report.append(f"- **Best ROI**: {best_roi['Model']} (ROI: {best_roi['ROI']:.2f})")
        report.append("")
        
        # Detailed results for each model
        for model_name, results in self.results.items():
            report.append(f"## 📈 {model_name.upper()} Model Results")
            report.append("-" * 40)
            
            basic = results['basic_metrics']
            advanced = results['advanced_metrics']
            business = results['business_metrics']
            
            report.append("### Performance Metrics")
            report.append(f"- Accuracy: {basic['accuracy']:.4f}")
            report.append(f"- Precision: {basic['precision']:.4f}")
            report.append(f"- Recall: {basic['recall']:.4f}")
            report.append(f"- F1-Score: {basic['f1_score']:.4f}")
            
            if 'roc_auc' in basic:
                report.append(f"- ROC AUC: {basic['roc_auc']:.4f}")
                report.append(f"- PR AUC: {basic['pr_auc']:.4f}")
            
            report.append("\n### Business Impact")
            report.append(f"- Total Cost: ${business['total_cost']:,.2f}")
            report.append(f"- Cost per Transaction: ${business['cost_per_transaction']:.2f}")
            report.append(f"- Fraud Prevention Savings: ${business['fraud_prevention_savings']:,.2f}")
            report.append(f"- ROI: {business['roi']:.2f}")
            
            # Feature importance
            if results['feature_importance']['available']:
                report.append("\n### Top Features")
                top_features = results['feature_importance']['top_features'][:10]
                for i, feature in enumerate(top_features, 1):
                    report.append(f"{i}. {feature['feature']}: {feature['importance']:.4f}")
            
            report.append("")
        
        # Recommendations
        report.append("## 💡 Recommendations")
        report.append("-" * 40)
        
        if best_f1['Model'] == best_auc['Model'] == best_roi['Model']:
            report.append(f"🏆 **Recommended Model**: {best_f1['Model']}")
            report.append("This model performs best across all metrics and business criteria.")
        else:
            report.append("🤔 **Model Selection Considerations**:")
            report.append(f"- For balanced performance: {best_f1['Model']}")
            report.append(f"- For best discrimination: {best_auc['Model']}")
            report.append(f"- For best business value: {best_roi['Model']}")
        
        report.append("")
        report.append("### Implementation Recommendations")
        report.append("1. **Ensemble Approach**: Consider combining multiple models for improved robustness")
        report.append("2. **Threshold Optimization**: Adjust detection thresholds based on business risk tolerance")
        report.append("3. **Continuous Monitoring**: Implement model drift detection and periodic retraining")
        report.append("4. **Feature Engineering**: Continue exploring new features based on emerging fraud patterns")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Evaluation report saved to {save_path}")
        
        return report_text

def run_full_evaluation():
    """Run complete model evaluation"""
    print("🚀 Starting Full Model Evaluation")
    print("=" * 60)
    
    try:
        # Load models and data
        with open('fraud_models.pkl', 'rb') as f:
            models = pickle.load(f)
        
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        
        df = pd.read_csv('financial_data_transactions.csv')
        X, y = preprocessor.transform(df)
        
        # Split data for evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize evaluator
        evaluator = ComprehensiveModelEvaluator()
        
        # Run comprehensive evaluation
        results = evaluator.evaluate_comprehensive(
            models, X_test, y_test, preprocessor.feature_columns
        )
        
        # Generate comparison table
        comparison_df = evaluator.create_comparison_table()
        print("\n📊 Model Comparison Table:")
        print(comparison_df.to_string(index=False))
        
        # Generate plots
        fig = evaluator.plot_model_comparison('model_comparison.html')
        
        # Generate report
        report = evaluator.generate_report('evaluation_report.md')
        
        # Save results
        with open('evaluation_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        print("\n✅ Evaluation completed successfully!")
        print("📁 Files generated:")
        print("   - model_comparison.html")
        print("   - evaluation_report.md")
        print("   - evaluation_results.pkl")
        
        return evaluator, results
        
    except FileNotFoundError as e:
        print(f"❌ Missing files: {e}")
        print("Please run data_generator.py, data_preprocessor.py, and fraud_models.py first")
        return None, None

if __name__ == "__main__":
    evaluator, results = run_full_evaluation()
