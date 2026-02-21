import numpy as np
import pandas as pd
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json
import threading
from queue import Queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TransactionResult:
    """Result of fraud detection for a single transaction"""
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    confidence_score: float
    risk_level: str
    model_predictions: Dict[str, float]
    processing_time: float
    timestamp: datetime
    alert_triggered: bool = False

@dataclass
class Alert:
    """Fraud alert information"""
    alert_id: str
    transaction_id: str
    customer_id: str
    risk_level: str
    fraud_probability: float
    timestamp: datetime
    details: Dict[str, Any]

class RealTimeFraudDetector:
    def __init__(self, model_path='fraud_models.pkl', preprocessor_path='preprocessor.pkl'):
        self.models = {}
        self.preprocessor = None
        self.transaction_queue = Queue()
        self.alert_queue = Queue()
        self.is_running = False
        self.processing_thread = None
        self.alert_thread = None
        
        # Risk thresholds
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 0.9
        }
        
        # Performance tracking
        self.stats = {
            'total_transactions': 0,
            'fraud_detected': 0,
            'alerts_triggered': 0,
            'avg_processing_time': 0.0,
            'model_accuracy': {}
        }
        
        # Load models and preprocessor
        self.load_models(model_path, preprocessor_path)
        
    def load_models(self, model_path: str, preprocessor_path: str):
        """Load trained models and preprocessor"""
        try:
            with open(model_path, 'rb') as f:
                self.models = pickle.load(f)
            logger.info(f"Loaded {len(self.models)} models from {model_path}")
            
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            logger.info(f"Loaded preprocessor from {preprocessor_path}")
            
        except FileNotFoundError as e:
            logger.error(f"Could not load models: {e}")
            raise
    
    def preprocess_transaction(self, transaction_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess a single transaction for prediction"""
        # Convert to DataFrame
        df = pd.DataFrame([transaction_data])
        
        # Apply preprocessing
        X, _ = self.preprocessor.transform(df)
        
        return X
    
    def calculate_confidence_score(self, predictions: Dict[str, float]) -> float:
        """Calculate confidence score based on model agreement"""
        if not predictions:
            return 0.0
        
        # Calculate variance among predictions (lower variance = higher confidence)
        values = list(predictions.values())
        variance = np.var(values)
        
        # Convert variance to confidence (inverse relationship)
        confidence = 1.0 / (1.0 + variance)
        
        return confidence
    
    def determine_risk_level(self, fraud_probability: float) -> str:
        """Determine risk level based on fraud probability"""
        if fraud_probability >= self.risk_thresholds['critical']:
            return 'critical'
        elif fraud_probability >= self.risk_thresholds['high']:
            return 'high'
        elif fraud_probability >= self.risk_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def detect_fraud(self, transaction_data: Dict[str, Any]) -> TransactionResult:
        """Detect fraud for a single transaction"""
        start_time = time.time()
        
        try:
            # Preprocess transaction
            X = self.preprocess_transaction(transaction_data)
            
            # Get predictions from all models
            model_predictions = {}
            fraud_probabilities = []
            
            for model_name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)[0]
                    model_predictions[model_name] = prob
                    fraud_probabilities.append(prob)
                elif hasattr(model, 'predict'):
                    pred = model.predict(X)[0]
                    model_predictions[model_name] = float(pred)
                    fraud_probabilities.append(pred)
            
            # Calculate ensemble probability (weighted average)
            if fraud_probabilities:
                ensemble_probability = np.mean(fraud_probabilities)
            else:
                ensemble_probability = 0.0
            
            # Calculate confidence score
            confidence_score = self.calculate_confidence_score(model_predictions)
            
            # Determine risk level
            risk_level = self.determine_risk_level(ensemble_probability)
            
            # Make final decision
            is_fraud = ensemble_probability >= self.risk_thresholds['medium']
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create result
            result = TransactionResult(
                transaction_id=transaction_data.get('transaction_id', 'unknown'),
                is_fraud=is_fraud,
                fraud_probability=ensemble_probability,
                confidence_score=confidence_score,
                risk_level=risk_level,
                model_predictions=model_predictions,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
            # Update statistics
            self.update_stats(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing transaction: {e}")
            processing_time = time.time() - start_time
            
            return TransactionResult(
                transaction_id=transaction_data.get('transaction_id', 'unknown'),
                is_fraud=False,
                fraud_probability=0.0,
                confidence_score=0.0,
                risk_level='error',
                model_predictions={},
                processing_time=processing_time,
                timestamp=datetime.now()
            )
    
    def update_stats(self, result: TransactionResult):
        """Update detection statistics"""
        self.stats['total_transactions'] += 1
        self.stats['avg_processing_time'] = (
            (self.stats['avg_processing_time'] * (self.stats['total_transactions'] - 1) + 
             result.processing_time) / self.stats['total_transactions']
        )
        
        if result.is_fraud:
            self.stats['fraud_detected'] += 1
    
    def should_trigger_alert(self, result: TransactionResult) -> bool:
        """Determine if an alert should be triggered"""
        return result.risk_level in ['high', 'critical'] and result.confidence_score > 0.7
    
    def create_alert(self, result: TransactionResult, transaction_data: Dict[str, Any]) -> Alert:
        """Create a fraud alert"""
        alert = Alert(
            alert_id=f"ALERT_{int(time.time())}_{result.transaction_id}",
            transaction_id=result.transaction_id,
            customer_id=transaction_data.get('customer_id', 'unknown'),
            risk_level=result.risk_level,
            fraud_probability=result.fraud_probability,
            timestamp=result.timestamp,
            details={
                'model_predictions': result.model_predictions,
                'confidence_score': result.confidence_score,
                'transaction_amount': transaction_data.get('amount', 0),
                'merchant_category': transaction_data.get('merchant_category', 'unknown'),
                'location': transaction_data.get('location', 'unknown')
            }
        )
        
        self.stats['alerts_triggered'] += 1
        return alert
    
    def process_transaction_queue(self):
        """Process transactions from the queue"""
        logger.info("Starting transaction processing thread")
        
        while self.is_running:
            try:
                # Get transaction from queue (with timeout)
                transaction_data = self.transaction_queue.get(timeout=1.0)
                
                # Process transaction
                result = self.detect_fraud(transaction_data)
                
                # Check if alert should be triggered
                if self.should_trigger_alert(result):
                    alert = self.create_alert(result, transaction_data)
                    self.alert_queue.put(alert)
                    result.alert_triggered = True
                
                # Log result
                logger.info(f"Processed {result.transaction_id}: "
                           f"Fraud={result.is_fraud}, "
                           f"Risk={result.risk_level}, "
                           f"Prob={result.fraud_probability:.3f}, "
                           f"Time={result.processing_time:.4f}s")
                
                # Mark task as done
                self.transaction_queue.task_done()
                
            except Exception as e:
                if self.is_running:  # Only log if we're supposed to be running
                    logger.error(f"Error in transaction processing: {e}")
        
        logger.info("Transaction processing thread stopped")
    
    def process_alert_queue(self):
        """Process alerts from the queue"""
        logger.info("Starting alert processing thread")
        
        while self.is_running:
            try:
                # Get alert from queue (with timeout)
                alert = self.alert_queue.get(timeout=1.0)
                
                # Handle alert (in real system, this would send notifications, etc.)
                self.handle_alert(alert)
                
                # Mark task as done
                self.alert_queue.task_done()
                
            except Exception as e:
                if self.is_running:  # Only log if we're supposed to be running
                    logger.error(f"Error in alert processing: {e}")
        
        logger.info("Alert processing thread stopped")
    
    def handle_alert(self, alert: Alert):
        """Handle a fraud alert"""
        logger.warning(f"🚨 FRAUD ALERT: {alert.alert_id}")
        logger.warning(f"   Transaction: {alert.transaction_id}")
        logger.warning(f"   Customer: {alert.customer_id}")
        logger.warning(f"   Risk Level: {alert.risk_level}")
        logger.warning(f"   Fraud Probability: {alert.fraud_probability:.3f}")
        logger.warning(f"   Timestamp: {alert.timestamp}")
        logger.warning(f"   Details: {json.dumps(alert.details, indent=2)}")
        
        # In a real system, this would:
        # - Send email/SMS notifications
        # - Block the transaction
        # - Update risk scores
        # - Create tickets for investigation
    
    def start(self):
        """Start the real-time detection system"""
        if self.is_running:
            logger.warning("Detection system is already running")
            return
        
        self.is_running = True
        
        # Start processing threads
        self.processing_thread = threading.Thread(target=self.process_transaction_queue)
        self.alert_thread = threading.Thread(target=self.process_alert_queue)
        
        self.processing_thread.start()
        self.alert_thread.start()
        
        logger.info("Real-time fraud detection system started")
    
    def stop(self):
        """Stop the real-time detection system"""
        if not self.is_running:
            logger.warning("Detection system is not running")
            return
        
        self.is_running = False
        
        # Wait for threads to finish
        if self.processing_thread:
            self.processing_thread.join()
        if self.alert_thread:
            self.alert_thread.join()
        
        logger.info("Real-time fraud detection system stopped")
    
    def submit_transaction(self, transaction_data: Dict[str, Any]):
        """Submit a transaction for processing"""
        if not self.is_running:
            logger.warning("Detection system is not running. Transaction not processed.")
            return
        
        # Add timestamp if not present
        if 'timestamp' not in transaction_data:
            transaction_data['timestamp'] = datetime.now().isoformat()
        
        self.transaction_queue.put(transaction_data)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current system statistics"""
        return {
            **self.stats,
            'queue_sizes': {
                'transactions': self.transaction_queue.qsize(),
                'alerts': self.alert_queue.qsize()
            },
            'is_running': self.is_running,
            'uptime': datetime.now().isoformat() if self.is_running else None
        }
    
    def simulate_real_time_processing(self, transaction_data: Dict[str, Any]) -> TransactionResult:
        """Simulate real-time processing without queue (for testing)"""
        return self.detect_fraud(transaction_data)

# Example usage and testing
if __name__ == "__main__":
    # Initialize detector
    detector = RealTimeFraudDetector()
    
    # Start the detection system
    detector.start()
    
    # Simulate some transactions
    sample_transactions = [
        {
            'transaction_id': 'TEST_001',
            'customer_id': 'CUST_000001',
            'amount': 150.00,
            'merchant_category': 'grocery',
            'location': 'local',
            'transaction_type': 'purchase',
            'device_type': 'mobile',
            'timestamp': datetime.now().isoformat()
        },
        {
            'transaction_id': 'TEST_002',
            'customer_id': 'CUST_000002',
            'amount': 5000.00,
            'merchant_category': 'electronics',
            'location': 'international',
            'transaction_type': 'purchase',
            'device_type': 'web',
            'timestamp': datetime.now().isoformat()
        }
    ]
    
    # Submit transactions
    for transaction in sample_transactions:
        detector.submit_transaction(transaction)
        time.sleep(0.1)  # Small delay between submissions
    
    # Wait for processing
    time.sleep(5)
    
    # Get statistics
    stats = detector.get_statistics()
    print(f"\nSystem Statistics: {json.dumps(stats, indent=2)}")
    
    # Stop the system
    detector.stop()
