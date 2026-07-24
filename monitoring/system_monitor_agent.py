#!/usr/bin/env python3
"""
EarlyStrike System Monitor Agent
Real-time system monitoring and ransomware detection
"""

import os
import time
import psutil
import threading
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from collections import deque
import logging
import sqlite3
from pathlib import Path

class SystemMonitorAgent:
    def __init__(self, log_file="system_monitor.log", db_file="earlystrike_monitoring.db"):
        self.log_file = log_file
        self.db_file = db_file
        self.monitoring = False
        self.monitor_thread = None
        
        # Setup logging first
        self.setup_logging()
        
        # Initialize database
        self.init_database()
        
        # Load detection models
        self.load_models()
        
        # Activity buffer for sequence analysis
        self.activity_buffer = deque(maxlen=100)
        
        # Monitoring intervals
        self.monitor_interval = 1.0  # seconds
        self.batch_interval = 10.0  # seconds for batch processing
        
        # System metrics
        self.system_metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_io': [],
            'network_activity': [],
            'process_count': []
        }
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def init_database(self):
        """Initialize SQLite database for storing monitoring data"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_activities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                process_name TEXT,
                pid INTEGER,
                cpu_usage REAL,
                memory_usage REAL,
                disk_read REAL,
                disk_write REAL,
                network_sent REAL,
                network_recv REAL,
                file_operations TEXT,
                entropy_score REAL,
                threat_level TEXT,
                is_anomaly INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                alert_type TEXT,
                severity TEXT,
                process_name TEXT,
                description TEXT,
                confidence REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def load_models(self):
        """Load detection models"""
        try:
            # Load Autoencoder
            self.autoencoder = tf.keras.models.load_model(
                "autoencoder_ransomware.h5",
                custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
            )
            self.autoencoder_scaler = joblib.load("autoencoder_scaler.pkl")
            self.autoencoder_threshold = joblib.load("autoencoder_threshold.pkl")
            self.autoencoder_loaded = True
            self.logger.info("Autoencoder model loaded successfully")
        except Exception as e:
            self.autoencoder_loaded = False
            self.logger.error(f"Failed to load autoencoder: {e}")
        
        try:
            # Load CNN-BiLSTM
            self.cnn_bilstm = tf.keras.models.load_model("cnn_bilstm_simplified.h5")
            self.cnn_bilstm_scaler = joblib.load("cnn_bilstm_simplified_scaler.pkl")
            self.cnn_bilstm_encoders = joblib.load("cnn_bilstm_simplified_encoders.pkl")
            self.cnn_bilstm_sequence_length = joblib.load("cnn_bilstm_simplified_sequence_length.pkl")
            self.cnn_bilstm_loaded = True
            self.logger.info("CNN-BiLSTM model loaded successfully")
        except Exception as e:
            self.cnn_bilstm_loaded = False
            self.logger.error(f"Failed to load CNN-BiLSTM: {e}")
        
        return self.autoencoder_loaded or self.cnn_bilstm_loaded
    
    def collect_system_metrics(self):
        """Collect real-time system metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
            disk_write = disk_io.write_bytes / (1024 * 1024) if disk_io else 0
            
            # Network I/O
            net_io = psutil.net_io_counters()
            net_sent = net_io.bytes_sent / (1024 * 1024) if net_io else 0
            net_recv = net_io.bytes_recv / (1024 * 1024) if net_io else 0
            
            # Process information
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cpu': proc.info['cpu_percent'],
                        'memory': proc.info['memory_percent']
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Update metrics history
            self.system_metrics['cpu_usage'].append(cpu_percent)
            self.system_metrics['memory_usage'].append(memory.percent)
            self.system_metrics['disk_io'].append(disk_read + disk_write)
            self.system_metrics['network_activity'].append(net_sent + net_recv)
            self.system_metrics['process_count'].append(len(processes))
            
            # Keep only last 100 data points
            for key in self.system_metrics:
                if len(self.system_metrics[key]) > 100:
                    self.system_metrics[key] = self.system_metrics[key][-100:]
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'disk_read': disk_read,
                'disk_write': disk_write,
                'network_sent': net_sent,
                'network_recv': net_recv,
                'processes': processes
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return None
    
    def analyze_process_behavior(self, processes):
        """Analyze process behavior for suspicious patterns"""
        suspicious_processes = []
        
        for proc in processes:
            risk_score = 0
            risk_factors = []
            
            # High CPU usage
            if proc['cpu'] > 80:
                risk_score += 2
                risk_factors.append("High CPU usage")
            
            # High memory usage
            if proc['memory'] > 80:
                risk_score += 1
                risk_factors.append("High memory usage")
            
            # Suspicious process names
            suspicious_names = ['powershell', 'cmd', 'wscript', 'cscript', 'unknown', 'encrypt']
            if any(susp in proc['name'].lower() for susp in suspicious_names):
                risk_score += 3
                risk_factors.append("Suspicious process name")
            
            # System processes with unusual behavior
            system_processes = ['svchost', 'lsass', 'winlogon', 'csrss']
            if any(sys in proc['name'].lower() for sys in system_processes) and proc['cpu'] > 50:
                risk_score += 2
                risk_factors.append("System process high CPU")
            
            if risk_score > 0:
                suspicious_processes.append({
                    'pid': proc['pid'],
                    'name': proc['name'],
                    'cpu': proc['cpu'],
                    'memory': proc['memory'],
                    'risk_score': risk_score,
                    'risk_factors': risk_factors
                })
        
        return suspicious_processes
    
    def calculate_entropy_score(self, processes):
        """Calculate entropy score based on process diversity"""
        if not processes:
            return 0.0
        
        # Count process name frequencies
        process_names = [p['name'] for p in processes]
        unique_processes = len(set(process_names))
        total_processes = len(process_names)
        
        # Calculate entropy
        if total_processes == 0:
            return 0.0
        
        entropy = 0.0
        for name in set(process_names):
            p = process_names.count(name) / total_processes
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def extract_features(self, metrics, suspicious_processes):
        """Extract features for model prediction"""
        if not metrics:
            return None
        
        # Base features
        features = [
            metrics.get('cpu_usage', 0),
            metrics.get('memory_usage', 0),
            len(suspicious_processes),
            1 if any('High CPU' in p.get('risk_factors', []) for p in suspicious_processes) else 0,
            1 if any('Suspicious process' in p.get('risk_factors', []) for p in suspicious_processes) else 0,
            1 if metrics.get('disk_write', 0) > 10 else 0,
            1 if metrics.get('network_sent', 0) > 5 else 0,
            1 if metrics.get('cpu_usage', 0) > 80 else 0,
            1 if self.calculate_entropy_score(metrics.get('processes', [])) > 6.0 else 0,
            1 if any('.encrypted' in p.get('name', '') for p in suspicious_processes) else 0,
            1 if any(p.get('pid', 0) < 1000 for p in suspicious_processes) else 0,
            1 if any('unknown' in p.get('name', '').lower() for p in suspicious_processes) else 0,
            1 if metrics.get('disk_read', 0) > 50 else 0,
            1 if metrics.get('cpu_usage', 0) > 50 else 0,
            1 if self.calculate_entropy_score(metrics.get('processes', [])) > 4.0 else 0
        ]
        
        return features
    
    def predict_with_models(self, features):
        """Make predictions using loaded models"""
        autoencoder_score = 0.5
        cnn_bilstm_score = 0.5
        
        if self.autoencoder_loaded and features:
            try:
                features_scaled = self.autoencoder_scaler.transform([features])
                reconstructed = self.autoencoder.predict(features_scaled, verbose=0)
                reconstruction_error = np.mean(np.square(features_scaled - reconstructed))
                autoencoder_score = reconstruction_error if reconstruction_error > self.autoencoder_threshold else 0.1
            except Exception as e:
                self.logger.error(f"Autoencoder prediction error: {e}")
        
        if self.cnn_bilstm_loaded and len(self.activity_buffer) >= self.cnn_bilstm_sequence_length:
            try:
                # Create sequence from activity buffer
                sequence_data = list(self.activity_buffer)[-self.cnn_bilstm_sequence_length:]
                cnn_bilstm_score = float(self.cnn_bilstm.predict(np.array([sequence_data]), verbose=0)[0][0])
            except Exception as e:
                self.logger.error(f"CNN-BiLSTM prediction error: {e}")
        
        # Combine scores
        combined_score = (autoencoder_score + cnn_bilstm_score) / 2
        return combined_score, autoencoder_score, cnn_bilstm_score
    
    def determine_threat_level(self, combined_score, suspicious_processes):
        """Determine threat level based on scores and behavior"""
        # Base threat from model score
        if combined_score > 0.8:
            base_threat = "CRITICAL"
        elif combined_score > 0.6:
            base_threat = "HIGH"
        elif combined_score > 0.4:
            base_threat = "MEDIUM"
        else:
            base_threat = "LOW"
        
        # Adjust based on suspicious processes
        high_risk_processes = [p for p in suspicious_processes if p['risk_score'] >= 4]
        if high_risk_processes:
            if base_threat == "LOW":
                base_threat = "MEDIUM"
            elif base_threat == "MEDIUM":
                base_threat = "HIGH"
            elif base_threat == "HIGH":
                base_threat = "CRITICAL"
        
        return base_threat
    
    def store_activity(self, metrics, threat_level, model_scores):
        """Store activity in database"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_activities 
                (timestamp, process_name, pid, cpu_usage, memory_usage, disk_read, disk_write, 
                 network_sent, network_recv, file_operations, entropy_score, threat_level, is_anomaly)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics['timestamp'],
                'system_monitor',
                0,
                metrics['cpu_usage'],
                metrics['memory_usage'],
                metrics['disk_read'],
                metrics['disk_write'],
                metrics['network_sent'],
                metrics['network_recv'],
                json.dumps([p['name'] for p in metrics.get('processes', [])]),
                self.calculate_entropy_score(metrics.get('processes', [])),
                threat_level,
                1 if threat_level in ['HIGH', 'CRITICAL'] else 0
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing activity: {e}")
    
    def create_alert(self, alert_type, severity, process_name, description, confidence):
        """Create and store alert"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alerts (timestamp, alert_type, severity, process_name, description, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                alert_type,
                severity,
                process_name,
                description,
                confidence
            ))
            
            conn.commit()
            conn.close()
            
            # Log alert
            self.logger.warning(f"ALERT: {severity} - {description}")
            
        except Exception as e:
            self.logger.error(f"Error creating alert: {e}")
    
    def monitor_cycle(self):
        """Single monitoring cycle"""
        try:
            # Collect system metrics
            metrics = self.collect_system_metrics()
            if not metrics:
                return
            
            # Analyze process behavior
            suspicious_processes = self.analyze_process_behavior(metrics.get('processes', []))
            
            # Extract features for ML models
            features = self.extract_features(metrics, suspicious_processes)
            
            # Make predictions
            if features:
                combined_score, autoencoder_score, cnn_bilstm_score = self.predict_with_models(features)
                
                # Determine threat level
                threat_level = self.determine_threat_level(combined_score, suspicious_processes)
                
                # Store activity
                self.store_activity(metrics, threat_level, {
                    'combined': combined_score,
                    'autoencoder': autoencoder_score,
                    'cnn_bilstm': cnn_bilstm_score
                })
                
                # Create alerts for high-risk processes
                for proc in suspicious_processes:
                    if proc['risk_score'] >= 4:
                        self.create_alert(
                            "SUSPICIOUS_PROCESS",
                            threat_level,
                            proc['name'],
                            f"Process {proc['name']} (PID: {proc['pid']}) - Risk factors: {', '.join(proc['risk_factors'])}",
                            proc['risk_score'] / 10.0
                        )
                
                # Add to activity buffer
                self.activity_buffer.append(features)
                
                # Log monitoring results
                self.logger.info(f"Monitoring cycle - CPU: {metrics['cpu_usage']:.1f}%, "
                               f"Memory: {metrics['memory_usage']:.1f}%, "
                               f"Threat: {threat_level}, "
                               f"Suspicious processes: {len(suspicious_processes)}")
                
        except Exception as e:
            self.logger.error(f"Error in monitoring cycle: {e}")
    
    def start_monitoring(self):
        """Start the monitoring agent"""
        if self.monitoring:
            self.logger.warning("Monitoring is already running")
            return
        
        self.monitoring = True
        self.logger.info("Starting EarlyStrike System Monitor Agent")
        
        def monitor_loop():
            while self.monitoring:
                cycle_start = time.time()
                
                # Perform monitoring cycle
                self.monitor_cycle()
                
                # Calculate sleep time to maintain interval
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self.monitor_interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info(f"System monitoring started with {self.monitor_interval}s interval")
    
    def stop_monitoring(self):
        """Stop the monitoring agent"""
        if not self.monitoring:
            self.logger.warning("Monitoring is not running")
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("System monitoring stopped")
    
    def get_recent_alerts(self, limit=10):
        """Get recent alerts from database"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT timestamp, alert_type, severity, process_name, description, confidence
                FROM alerts
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            alerts = cursor.fetchall()
            conn.close()
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error getting alerts: {e}")
            return []
    
    def get_system_status(self):
        """Get current system status"""
        if not self.system_metrics['cpu_usage']:
            return "Unknown"
        
        avg_cpu = np.mean(self.system_metrics['cpu_usage'][-10:])
        avg_memory = np.mean(self.system_metrics['memory_usage'][-10:])
        
        if avg_cpu > 80 or avg_memory > 80:
            return "CRITICAL"
        elif avg_cpu > 60 or avg_memory > 60:
            return "HIGH"
        elif avg_cpu > 40 or avg_memory > 40:
            return "MEDIUM"
        else:
            return "NORMAL"
    
    def generate_report(self, hours=1):
        """Generate monitoring report"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Get activities from last N hours
            cursor.execute('''
                SELECT COUNT(*) as total_activities,
                       AVG(cpu_usage) as avg_cpu,
                       AVG(memory_usage) as avg_memory,
                       COUNT(CASE WHEN threat_level = 'HIGH' OR threat_level = 'CRITICAL' THEN 1 END) as high_threats
                FROM system_activities
                WHERE timestamp > datetime('now', '-{} hours')
            '''.format(hours))
            
            report_data = cursor.fetchone()
            
            # Get alerts from last N hours
            cursor.execute('''
                SELECT COUNT(*) as alert_count
                FROM alerts
                WHERE timestamp > datetime('now', '-{} hours')
            '''.format(hours))
            
            alert_count = cursor.fetchone()[0]
            
            conn.close()
            
            report = f"""
EarlyStrike Monitoring Report - Last {hours} Hours
============================================
Total Activities: {report_data[0]}
Average CPU Usage: {report_data[1]:.1f}%
Average Memory Usage: {report_data[2]:.1f}%
High Threat Activities: {report_data[3]}
Total Alerts: {alert_count}
System Status: {self.get_system_status()}
============================================
            """
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return "Error generating report"

def main():
    print("EarlyStrike System Monitor Agent")
    print("=" * 40)
    print("Real-time ransomware detection and monitoring")
    print("=" * 40)
    
    # Initialize monitor agent
    agent = SystemMonitorAgent()
    
    # Check if models loaded
    if not (agent.autoencoder_loaded or agent.cnn_bilstm_loaded):
        print("⚠️ Warning: No detection models loaded")
        print("The agent will run in rule-based mode only")
    else:
        print("✅ Detection models loaded successfully")
        if agent.autoencoder_loaded:
            print("  - Autoencoder: Active")
        if agent.cnn_bilstm_loaded:
            print("  - CNN-BiLSTM: Active")
    
    print("\nCommands:")
    print("  start     - Start monitoring")
    print("  stop      - Stop monitoring")
    print("  status    - Show system status")
    print("  alerts    - Show recent alerts")
    print("  report    - Generate monitoring report")
    print("  quit      - Exit")
    
    try:
        while True:
            command = input("\nearlystrike> ").strip().lower()
            
            if command == "start":
                agent.start_monitoring()
                print("Monitoring started. Press Ctrl+C to stop.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    agent.stop_monitoring()
                    print("Monitoring stopped.")
            
            elif command == "stop":
                agent.stop_monitoring()
                print("Monitoring stopped.")
            
            elif command == "status":
                status = agent.get_system_status()
                print(f"System Status: {status}")
                if agent.system_metrics['cpu_usage']:
                    print(f"Current CPU: {agent.system_metrics['cpu_usage'][-1]:.1f}%")
                    print(f"Current Memory: {agent.system_metrics['memory_usage'][-1]:.1f}%")
            
            elif command == "alerts":
                alerts = agent.get_recent_alerts()
                if alerts:
                    print("\nRecent Alerts:")
                    for alert in alerts:
                        print(f"  {alert[0]} - {alert[2]} - {alert[4]}")
                else:
                    print("No recent alerts.")
            
            elif command == "report":
                report = agent.generate_report()
                print(report)
            
            elif command == "quit":
                if agent.monitoring:
                    agent.stop_monitoring()
                print("Goodbye!")
                break
            
            else:
                print("Unknown command. Available: start, stop, status, alerts, report, quit")
    
    except KeyboardInterrupt:
        if agent.monitoring:
            agent.stop_monitoring()
        print("\nMonitoring stopped. Goodbye!")

if __name__ == "__main__":
    main()
