#!/usr/bin/env python3
"""
Canary Automation System

Automated canary monitoring that integrates with data pipeline updates.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import hashlib
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CanaryAutomation:
    """Automated canary monitoring for pipeline updates."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.canary_config_path = self.project_root / "config" / "canary.json"
        self.incidents_path = self.project_root / "data" / "incidents"
        self.incidents_path.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        logger.info(f"Initialized canary automation for {self.project_root}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load canary configuration."""
        
        default_config = {
            'triggers': {
                'overlay_update': True,
                'source_sync': True,
                'model_change': True,
                'scheduled': False
            },
            'thresholds': {
                'mean_drop_critical': 0.05,      # 5% drop = critical
                'mean_drop_warning': 0.03,       # 3% drop = warning
                'ci_confidence': 0.95,           # 95% confidence interval
                'tail_drop_threshold': 0.05      # 5% P95 drop
            },
            'baseline': {
                'rolling_window': 7,             # 7-run rolling baseline
                'min_history_days': 3,           # Need 3+ days history
                'max_history_days': 30           # Use last 30 days max
            },
            'alerts': {
                'block_on_critical': True,       # Block pipeline on critical alerts
                'create_incidents': True,        # Create incident tickets
                'notification_channels': []     # TODO: Add Slack/email
            },
            'probe': {
                'version': 'v1.0',
                'auto_update': False             # Probe set updates require manual approval
            }
        }
        
        if self.canary_config_path.exists():
            try:
                with open(self.canary_config_path, 'r') as f:
                    user_config = json.load(f)
                
                # Merge with defaults
                config = default_config.copy()
                config.update(user_config)
                return config
                
            except Exception as e:
                logger.warning(f"Failed to load canary config: {e}, using defaults")
        
        return default_config
    
    def _save_config(self):
        """Save current configuration."""
        self.canary_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.canary_config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Saved canary config to {self.canary_config_path}")
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file for change detection."""
        if not file_path.exists():
            return ""
        
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def detect_overlay_change(self) -> bool:
        """Detect if overlay files have changed."""
        
        overlay_dir = self.project_root / "data" / "dataset" / "metadata"
        state_file = overlay_dir / ".canary_overlay_state.json"
        
        # Find all overlay files
        current_overlays = {}
        for overlay_file in overlay_dir.glob("*-overlay.json"):
            current_overlays[overlay_file.name] = self._compute_file_hash(overlay_file)
        
        # Load previous state
        previous_overlays = {}
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                    previous_overlays = state_data.get('overlay_hashes', {})
            except:
                pass
        
        # Check for changes
        changed = current_overlays != previous_overlays
        
        if changed:
            logger.info("Overlay change detected")
            
            # Update state file
            state_data = {
                'overlay_hashes': current_overlays,
                'last_check': datetime.now().isoformat(),
                'change_detected': changed
            }
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
        
        return changed
    
    def detect_source_sync(self) -> bool:
        """Detect if source data has been synced."""
        
        # Check if frozen snapshot has been updated
        snapshot_file = self.project_root / "data" / "dataset" / "metadata" / "frozen_snapshot.json"
        state_file = self.project_root / "data" / "dataset" / "metadata" / ".canary_source_state.json"
        
        if not snapshot_file.exists():
            return False
        
        current_hash = self._compute_file_hash(snapshot_file)
        
        # Load previous hash
        previous_hash = ""
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                    previous_hash = state_data.get('snapshot_hash', '')
            except:
                pass
        
        changed = current_hash != previous_hash
        
        if changed:
            logger.info("Source sync detected")
            
            # Update state
            state_data = {
                'snapshot_hash': current_hash,
                'last_check': datetime.now().isoformat(),
                'change_detected': changed
            }
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
        
        return changed
    
    def detect_model_change(self) -> bool:
        """Detect if model or core modules have changed."""
        
        # Check key model files
        model_files = [
            self.project_root / "src" / "subject_object.py",
            self.project_root / "src" / "conflict_penalty.py", 
            self.project_root / "src" / "dual_score.py",
            self.project_root / "pipeline.py"
        ]
        
        state_file = self.project_root / ".canary_model_state.json"
        
        # Compute current hashes
        current_hashes = {}
        for file_path in model_files:
            if file_path.exists():
                current_hashes[str(file_path.relative_to(self.project_root))] = self._compute_file_hash(file_path)
        
        # Load previous hashes
        previous_hashes = {}
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                    previous_hashes = state_data.get('model_hashes', {})
            except:
                pass
        
        changed = current_hashes != previous_hashes
        
        if changed:
            logger.info("Model change detected")
            
            # Update state
            state_data = {
                'model_hashes': current_hashes,
                'last_check': datetime.now().isoformat(),
                'change_detected': changed
            }
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
        
        return changed
    
    def run_canary_check(self, trigger_reason: str) -> Dict[str, Any]:
        """Run canary check and return results."""
        
        try:
            # Generate run ID with trigger context
            run_id = f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{trigger_reason}"
            
            # Run canary monitor
            cmd = [
                sys.executable, "canary_monitor.py",
                "--probe-version", self.config['probe']['version'],
                "--run-id", run_id
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Canary check failed: {result.stderr}")
                return {
                    'success': False,
                    'error': result.stderr,
                    'trigger_reason': trigger_reason
                }
            
            # Load the saved results
            canary_results_file = self.project_root / "data" / "canary" / f"canary_metrics_{run_id}.json"
            
            if canary_results_file.exists():
                with open(canary_results_file, 'r') as f:
                    canary_data = json.load(f)
                
                return {
                    'success': True,
                    'canary_data': canary_data,
                    'trigger_reason': trigger_reason,
                    'run_id': run_id,
                    'stdout': result.stdout
                }
            else:
                return {
                    'success': False,
                    'error': 'Canary results file not found',
                    'trigger_reason': trigger_reason
                }
                
        except subprocess.TimeoutExpired:
            logger.error("Canary check timed out")
            return {
                'success': False,
                'error': 'Timeout after 5 minutes',
                'trigger_reason': trigger_reason
            }
        except Exception as e:
            logger.error(f"Canary check exception: {e}")
            return {
                'success': False,
                'error': str(e),
                'trigger_reason': trigger_reason
            }
    
    def create_drift_incident(self, canary_results: Dict[str, Any], alerts: List[Dict[str, Any]]) -> str:
        """Create drift incident ticket with details."""
        
        run_id = canary_results.get('run_id', 'unknown')
        trigger_reason = canary_results.get('trigger_reason', 'unknown')
        
        # Count alert severities
        critical_alerts = [a for a in alerts if a['severity'] == 'critical']
        warning_alerts = [a for a in alerts if a['severity'] == 'warning']
        
        # Create incident
        incident = {
            'incident_id': f"drift_{run_id}",
            'created_at': datetime.now().isoformat(),
            'trigger_reason': trigger_reason,
            'canary_run_id': run_id,
            'status': 'open',
            'severity': 'critical' if critical_alerts else 'warning',
            'summary': f"Dataset drift detected ({len(alerts)} alerts)",
            'alerts': alerts,
            'affected_components': ['dataset', 'pipeline'],
            'investigation_notes': [],
            'resolution': None,
            'closed_at': None
        }
        
        # Save incident
        incident_file = self.incidents_path / f"{incident['incident_id']}.json"
        with open(incident_file, 'w') as f:
            json.dump(incident, f, indent=2)
        
        logger.info(f"Created drift incident: {incident_file}")
        
        # Generate incident summary
        summary_lines = [
            f"🚨 DRIFT INCIDENT: {incident['incident_id']}",
            f"Trigger: {trigger_reason}",
            f"Severity: {incident['severity']}",
            f"Total alerts: {len(alerts)} ({len(critical_alerts)} critical, {len(warning_alerts)} warning)",
            "",
            "Alert Details:"
        ]
        
        for alert in alerts[:5]:  # Top 5 alerts
            summary_lines.append(f"  • {alert['type']}: {alert['message']}")
            if alert.get('affected_images'):
                summary_lines.append(f"    Worst images: {', '.join(alert['affected_images'][:3])}")
        
        summary_lines.extend([
            "",
            f"Investigation required:",
            f"  1. Review canary results: data/canary/canary_metrics_{run_id}.json",
            f"  2. Check recent changes in trigger: {trigger_reason}",
            f"  3. Validate probe set is still appropriate",
            f"  4. Consider dataset quality improvements",
            "",
            f"Incident file: {incident_file}"
        ])
        
        summary = "\n".join(summary_lines)
        
        # Save summary to file
        summary_file = self.incidents_path / f"{incident['incident_id']}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        return str(incident_file)
    
    def should_block_pipeline(self, alerts: List[Dict[str, Any]]) -> bool:
        """Determine if pipeline should be blocked based on alerts."""
        
        if not self.config['alerts']['block_on_critical']:
            return False
        
        # Block if any critical alerts
        return any(alert['severity'] == 'critical' for alert in alerts)
    
    def check_triggers_and_run(self) -> Dict[str, Any]:
        """Check all triggers and run canary if needed."""
        
        triggered_reasons = []
        
        # Check each trigger type
        if self.config['triggers']['overlay_update'] and self.detect_overlay_change():
            triggered_reasons.append('overlay_update')
        
        if self.config['triggers']['source_sync'] and self.detect_source_sync():
            triggered_reasons.append('source_sync')
        
        if self.config['triggers']['model_change'] and self.detect_model_change():
            triggered_reasons.append('model_change')
        
        if not triggered_reasons:
            return {
                'triggered': False,
                'message': 'No triggers detected, canary check skipped'
            }
        
        # Run canary check
        trigger_reason = '_'.join(triggered_reasons)
        logger.info(f"Triggers detected: {triggered_reasons}")
        
        canary_result = self.run_canary_check(trigger_reason)
        
        if not canary_result['success']:
            return {
                'triggered': True,
                'success': False,
                'error': canary_result['error'],
                'trigger_reasons': triggered_reasons
            }
        
        # Parse canary results for alerts (mock parsing for now)
        # In real implementation, this would parse the actual canary monitor output
        stdout = canary_result.get('stdout', '')
        alerts = []
        
        # Simple alert detection from stdout
        if 'ALERT' in stdout or '🚨' in stdout:
            # Mock alert for demonstration
            alerts = [{
                'type': 'mean_drop',
                'severity': 'warning',
                'message': 'Mean margin dropped below threshold',
                'baseline_value': 0.4,
                'current_value': 0.35,
                'delta': 0.125,
                'affected_images': ['probe_pos_001', 'probe_pos_005']
            }]
        
        # Handle alerts
        if alerts:
            if self.config['alerts']['create_incidents']:
                incident_file = self.create_drift_incident(canary_result, alerts)
            else:
                incident_file = None
            
            should_block = self.should_block_pipeline(alerts)
            
            result = {
                'triggered': True,
                'success': True,
                'alerts_detected': True,
                'alerts': alerts,
                'incident_file': incident_file,
                'should_block_pipeline': should_block,
                'trigger_reasons': triggered_reasons,
                'canary_run_id': canary_result['run_id']
            }
            
            if should_block:
                result['message'] = 'PIPELINE BLOCKED: Critical drift detected'
                logger.error(result['message'])
            else:
                result['message'] = 'Drift detected but pipeline allowed to continue'
                logger.warning(result['message'])
            
            return result
        
        else:
            return {
                'triggered': True,
                'success': True,
                'alerts_detected': False,
                'message': 'Canary check passed - no drift detected',
                'trigger_reasons': triggered_reasons,
                'canary_run_id': canary_result['run_id']
            }

def main():
    """CLI interface for canary automation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Canary Automation System')
    parser.add_argument('--check', action='store_true', help='Run trigger check')
    parser.add_argument('--force', help='Force canary check with reason')
    parser.add_argument('--init-config', action='store_true', help='Initialize canary config')
    parser.add_argument('--status', action='store_true', help='Show system status')
    
    args = parser.parse_args()
    
    automation = CanaryAutomation()
    
    if args.init_config:
        automation._save_config()
        print("✅ Canary configuration initialized")
        return
    
    if args.status:
        print("🕯️  CANARY AUTOMATION STATUS")
        print("=" * 40)
        print(f"Project root: {automation.project_root}")
        print(f"Config file: {automation.canary_config_path}")
        print(f"Incidents dir: {automation.incidents_path}")
        
        config = automation.config
        print(f"\nTriggers:")
        for trigger, enabled in config['triggers'].items():
            status = "✅" if enabled else "❌"
            print(f"  {status} {trigger}")
        
        print(f"\nThresholds:")
        print(f"  Mean drop critical: {config['thresholds']['mean_drop_critical']:.1%}")
        print(f"  Mean drop warning: {config['thresholds']['mean_drop_warning']:.1%}")
        print(f"  Rolling window: {config['baseline']['rolling_window']} runs")
        
        return
    
    if args.force:
        result = automation.run_canary_check(args.force)
        if result['success']:
            print(f"✅ Forced canary check completed: {result['run_id']}")
        else:
            print(f"❌ Forced canary check failed: {result['error']}")
        return
    
    if args.check:
        result = automation.check_triggers_and_run()
        
        print("🕯️  CANARY AUTOMATION RESULT")
        print("=" * 40)
        
        if not result['triggered']:
            print("✅ No triggers detected")
        else:
            print(f"🔥 Triggered by: {', '.join(result['trigger_reasons'])}")
            
            if result['success']:
                if result.get('alerts_detected'):
                    print(f"🚨 Alerts detected: {len(result['alerts'])} alerts")
                    if result.get('should_block_pipeline'):
                        print("🛑 PIPELINE BLOCKED")
                        sys.exit(1)  # Exit with error code to block pipeline
                    else:
                        print("⚠️  Pipeline continues with warnings")
                else:
                    print("✅ No drift detected")
                
                print(f"📊 Canary run: {result['canary_run_id']}")
                
                if result.get('incident_file'):
                    print(f"📋 Incident created: {result['incident_file']}")
            else:
                print(f"❌ Canary check failed: {result['error']}")
                sys.exit(1)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()