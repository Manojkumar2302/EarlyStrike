import smtplib
import json
import sqlite3
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Optional
import logging
import threading
import time
from queue import Queue
from dataclasses import dataclass
from enum import Enum

class AlertSeverity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AlertType(Enum):
    RANSOMWARE_DETECTED = "RANSOMWARE_DETECTED"
    HIGH_CPU_USAGE = "HIGH_CPU_USAGE"
    SUSPICIOUS_FILE_ACCESS = "SUSPICIOUS_FILE_ACCESS"
    MULTIPLE_ENCRYPTION = "MULTIPLE_ENCRYPTION"
    SYSTEM_ANOMALY = "SYSTEM_ANOMALY"

@dataclass
class Alert:
    timestamp: datetime
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    source: str
    probability: float
    details: Dict
    acknowledged: bool = False
    resolved: bool = False

class AlertSystem:
    """
    Comprehensive alert system for ransomware detection
    
    Features:
    - Multiple alert channels (email, SMS, webhook)
    - Alert escalation and prioritization
    - Alert aggregation and deduplication
    - Historical alert tracking
    - Alert acknowledgment and resolution
    """
    
    def __init__(self, db_path="ransomware_alerts.db"):
        self.db_path = db_path
        self.alert_queue = Queue()
        self.alert_history = []
        self.alert_rules = {}
        self.notification_channels = {}
        
        # Alert thresholds
        self.thresholds = {
            'probability_critical': 0.9,
            'probability_high': 0.8,
            'probability_medium': 0.6,
            'cpu_usage_threshold': 80.0,
            'entropy_threshold': 7.5,
            'alert_aggregation_window': 300  # 5 minutes
        }
        
        # Alert statistics
        self.stats = {
            'total_alerts': 0,
            'critical_alerts': 0,
            'acknowledged_alerts': 0,
            'resolved_alerts': 0,
            'false_positives': 0
        }
        
        # Setup logging
        self.setup_logging()
        
        # Initialize database
        self.init_database()
        
        # Start alert processing thread
        self.alert_thread = threading.Thread(target=self._process_alerts, daemon=True)
        self.alert_thread.start()
    
    def setup_logging(self):
        """Setup logging for alert system"""
        self.logger = logging.getLogger('AlertSystem')
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('alert_system.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def init_database(self):
        """Initialize alert database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    source TEXT,
                    probability REAL,
                    details TEXT,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    resolved BOOLEAN DEFAULT FALSE,
                    acknowledged_at TEXT,
                    resolved_at TEXT,
                    acknowledged_by TEXT,
                    resolved_by TEXT
                )
            ''')
            
            # Create alert_rules table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alert_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    created_at TEXT NOT NULL
                )
            ''')
            
            # Create notification_channels table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS notification_channels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    config TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    created_at TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("Alert database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize alert database: {e}")
    
    def add_email_channel(self, name: str, smtp_server: str, smtp_port: int, 
                         username: str, password: str, recipients: List[str]):
        """Add email notification channel"""
        config = {
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'username': username,
            'password': password,
            'recipients': recipients
        }
        
        self.notification_channels[name] = {
            'type': 'email',
            'config': config,
            'enabled': True
        }
        
        # Store in database
        self._store_notification_channel(name, 'email', config)
        self.logger.info(f"Email notification channel '{name}' added")
    
    def add_webhook_channel(self, name: str, webhook_url: str, headers: Dict = None):
        """Add webhook notification channel"""
        config = {
            'webhook_url': webhook_url,
            'headers': headers or {}
        }
        
        self.notification_channels[name] = {
            'type': 'webhook',
            'config': config,
            'enabled': True
        }
        
        # Store in database
        self._store_notification_channel(name, 'webhook', config)
        self.logger.info(f"Webhook notification channel '{name}' added")
    
    def create_alert(self, alert_type: AlertType, severity: AlertSeverity, 
                    message: str, source: str, probability: float, details: Dict = None):
        """Create a new alert"""
        alert = Alert(
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            message=message,
            source=source,
            probability=probability,
            details=details or {}
        )
        
        # Add to queue for processing
        self.alert_queue.put(alert)
        
        return alert
    
    def _process_alerts(self):
        """Process alerts from queue"""
        while True:
            try:
                if not self.alert_queue.empty():
                    alert = self.alert_queue.get(timeout=1)
                    self._handle_alert(alert)
                else:
                    time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error processing alert: {e}")
    
    def _handle_alert(self, alert: Alert):
        """Handle individual alert"""
        # Check for alert aggregation
        aggregated_alert = self._check_aggregation(alert)
        
        if aggregated_alert:
            # Update existing alert
            self._update_alert(aggregated_alert)
        else:
            # Store new alert
            self._store_alert(alert)
            
            # Update statistics
            self.stats['total_alerts'] += 1
            if alert.severity == AlertSeverity.CRITICAL:
                self.stats['critical_alerts'] += 1
            
            # Send notifications
            self._send_notifications(alert)
            
            # Log alert
            self.logger.warning(f"ALERT: {alert.severity.value} - {alert.message}")
    
    def _check_aggregation(self, new_alert: Alert) -> Optional[Alert]:
        """Check if alert should be aggregated with existing alerts"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check for similar alerts in the aggregation window
            window_start = (datetime.now() - timedelta(seconds=self.thresholds['alert_aggregation_window'])).isoformat()
            
            cursor.execute('''
                SELECT * FROM alerts 
                WHERE alert_type = ? AND severity = ? AND source = ?
                AND timestamp > ? AND acknowledged = FALSE AND resolved = FALSE
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (new_alert.alert_type.value, new_alert.severity.value, new_alert.source, window_start))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                # Return existing alert for aggregation
                existing_alert = Alert(
                    timestamp=datetime.fromisoformat(result[1]),
                    alert_type=AlertType(result[2]),
                    severity=AlertSeverity(result[3]),
                    message=result[4],
                    source=result[5],
                    probability=result[6],
                    details=json.loads(result[7]) if result[7] else {},
                    acknowledged=bool(result[8]),
                    resolved=bool(result[9])
                )
                return existing_alert
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking alert aggregation: {e}")
            return None
    
    def _store_alert(self, alert: Alert):
        """Store alert in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alerts 
                (timestamp, alert_type, severity, message, source, probability, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.timestamp.isoformat(),
                alert.alert_type.value,
                alert.severity.value,
                alert.message,
                alert.source,
                alert.probability,
                json.dumps(alert.details)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store alert: {e}")
    
    def _update_alert(self, alert: Alert):
        """Update existing alert (aggregation)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update timestamp and message to indicate aggregation
            cursor.execute('''
                UPDATE alerts 
                SET timestamp = ?, message = ?, probability = ?
                WHERE alert_type = ? AND severity = ? AND source = ?
                AND acknowledged = FALSE AND resolved = FALSE
            ''', (
                datetime.now().isoformat(),
                f"[AGGREGATED] {alert.message}",
                max(alert.probability, 0.9),  # Increase probability for aggregated alerts
                alert.alert_type.value,
                alert.severity.value,
                alert.source
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to update alert: {e}")
    
    def _send_notifications(self, alert: Alert):
        """Send notifications through configured channels"""
        for channel_name, channel in self.notification_channels.items():
            if not channel['enabled']:
                continue
            
            try:
                if channel['type'] == 'email':
                    self._send_email_notification(alert, channel['config'])
                elif channel['type'] == 'webhook':
                    self._send_webhook_notification(alert, channel['config'])
                
            except Exception as e:
                self.logger.error(f"Failed to send notification via {channel_name}: {e}")
    
    def _send_email_notification(self, alert: Alert, config: Dict):
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = config['username']
            msg['To'] = ', '.join(config['recipients'])
            msg['Subject'] = f"🚨 Ransomware Alert: {alert.severity.value} - {alert.alert_type.value}"
            
            # Create email body
            body = self._create_email_body(alert)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['username'], config['password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email notification sent for {alert.alert_type.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
    
    def _send_webhook_notification(self, alert: Alert, config: Dict):
        """Send webhook notification"""
        try:
            import requests
            
            payload = {
                'timestamp': alert.timestamp.isoformat(),
                'alert_type': alert.alert_type.value,
                'severity': alert.severity.value,
                'message': alert.message,
                'source': alert.source,
                'probability': alert.probability,
                'details': alert.details
            }
            
            response = requests.post(
                config['webhook_url'],
                json=payload,
                headers=config['headers'],
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info(f"Webhook notification sent for {alert.alert_type.value}")
            else:
                self.logger.error(f"Webhook notification failed: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
    
    def _create_email_body(self, alert: Alert) -> str:
        """Create HTML email body"""
        severity_colors = {
            'LOW': '#28a745',
            'MEDIUM': '#ffc107',
            'HIGH': '#fd7e14',
            'CRITICAL': '#dc3545'
        }
        
        color = severity_colors.get(alert.severity.value, '#6c757d')
        
        html = f"""
        <html>
        <body>
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background-color: {color}; color: white; padding: 20px; text-align: center;">
                    <h1>🚨 Ransomware Detection Alert</h1>
                    <h2>{alert.severity.value} - {alert.alert_type.value}</h2>
                </div>
                
                <div style="padding: 20px; background-color: #f8f9fa;">
                    <h3>Alert Details</h3>
                    <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Source:</strong> {alert.source}</p>
                    <p><strong>Probability:</strong> {alert.probability:.3f}</p>
                    <p><strong>Message:</strong> {alert.message}</p>
                    
                    {self._format_details(alert.details)}
                </div>
                
                <div style="padding: 20px; background-color: #e9ecef; text-align: center;">
                    <p>This is an automated alert from the Ransomware Detection System.</p>
                    <p>Please investigate immediately if this is a critical alert.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _format_details(self, details: Dict) -> str:
        """Format alert details for email"""
        if not details:
            return ""
        
        html = "<h3>Additional Details</h3><ul>"
        for key, value in details.items():
            html += f"<li><strong>{key}:</strong> {value}</li>"
        html += "</ul>"
        
        return html
    
    def _store_notification_channel(self, name: str, channel_type: str, config: Dict):
        """Store notification channel in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO notification_channels 
                (name, type, config, created_at)
                VALUES (?, ?, ?, ?)
            ''', (name, channel_type, json.dumps(config), datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store notification channel: {e}")
    
    def acknowledge_alert(self, alert_id: int, acknowledged_by: str):
        """Acknowledge an alert"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE alerts 
                SET acknowledged = TRUE, acknowledged_at = ?, acknowledged_by = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), acknowledged_by, alert_id))
            
            conn.commit()
            conn.close()
            
            self.stats['acknowledged_alerts'] += 1
            self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            
        except Exception as e:
            self.logger.error(f"Failed to acknowledge alert: {e}")
    
    def resolve_alert(self, alert_id: int, resolved_by: str):
        """Resolve an alert"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE alerts 
                SET resolved = TRUE, resolved_at = ?, resolved_by = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), resolved_by, alert_id))
            
            conn.commit()
            conn.close()
            
            self.stats['resolved_alerts'] += 1
            self.logger.info(f"Alert {alert_id} resolved by {resolved_by}")
            
        except Exception as e:
            self.logger.error(f"Failed to resolve alert: {e}")
    
    def get_active_alerts(self, severity: AlertSeverity = None) -> List[Dict]:
        """Get active (unresolved) alerts"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT id, timestamp, alert_type, severity, message, source, 
                       probability, details, acknowledged, resolved
                FROM alerts 
                WHERE resolved = FALSE
            '''
            params = []
            
            if severity:
                query += ' AND severity = ?'
                params.append(severity.value)
            
            query += ' ORDER BY timestamp DESC'
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            conn.close()
            
            alerts = []
            for row in results:
                alerts.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'alert_type': row[2],
                    'severity': row[3],
                    'message': row[4],
                    'source': row[5],
                    'probability': row[6],
                    'details': json.loads(row[7]) if row[7] else {},
                    'acknowledged': bool(row[8]),
                    'resolved': bool(row[9])
                })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Failed to get active alerts: {e}")
            return []
    
    def get_alert_statistics(self) -> Dict:
        """Get alert statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get counts by severity
            cursor.execute('''
                SELECT severity, COUNT(*) FROM alerts 
                WHERE resolved = FALSE 
                GROUP BY severity
            ''')
            severity_counts = dict(cursor.fetchall())
            
            # Get recent alerts (last 24 hours)
            cursor.execute('''
                SELECT COUNT(*) FROM alerts 
                WHERE timestamp > datetime('now', '-1 day')
            ''')
            recent_count = cursor.fetchone()[0]
            
            conn.close()
            
            stats = self.stats.copy()
            stats.update({
                'active_alerts_by_severity': severity_counts,
                'recent_alerts_24h': recent_count
            })
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get alert statistics: {e}")
            return self.stats

# Example usage
if __name__ == "__main__":
    # Initialize alert system
    alert_system = AlertSystem()
    
    # Add email notification channel
    alert_system.add_email_channel(
        name="admin_email",
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        username="admin@company.com",
        password="app_password",
        recipients=["security@company.com", "admin@company.com"]
    )
    
    # Create test alerts
    alert_system.create_alert(
        alert_type=AlertType.RANSOMWARE_DETECTED,
        severity=AlertSeverity.CRITICAL,
        message="Ransomware activity detected in process 'cryptolocker'",
        source="monitoring_system",
        probability=0.95,
        details={
            'process_name': 'cryptolocker',
            'file_path': '/desktop/important.docx.encrypted',
            'entropy': 7.8,
            'cpu_usage': 95.0
        }
    )
    
    # Get active alerts
    active_alerts = alert_system.get_active_alerts()
    print(f"Active alerts: {len(active_alerts)}")
    
    # Get statistics
    stats = alert_system.get_alert_statistics()
    print(f"Alert statistics: {stats}")
    
    # Wait for processing
    time.sleep(2)
    print("Alert system test completed")
