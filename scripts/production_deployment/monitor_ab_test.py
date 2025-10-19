#!/usr/bin/env python3
"""
Real-time A/B Test Monitoring Dashboard
Continuous monitoring and analysis during 5K A/B test execution

Usage:
    python monitor_ab_test.py --deployment-id ra_guard_5k_ab_20251017_143000
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from dataclasses import dataclass
import argparse
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MonitoringMetrics:
    """Real-time monitoring metrics"""
    timestamp: datetime
    ndcg_control: float
    ndcg_treatment: float
    ndcg_improvement: float
    latency_p95_control: float
    latency_p95_treatment: float
    error_rate_control: float
    error_rate_treatment: float
    sample_size_control: int
    sample_size_treatment: int
    traffic_allocation_percent: float

class RealTimeABMonitor:
    """Real-time monitoring and analysis for A/B test"""
    
    def __init__(self, deployment_id: str):
        self.deployment_id = deployment_id
        self.start_time = datetime.now()
        self.metrics_history: List[MonitoringMetrics] = []
        self.alerts_triggered: List[Dict] = []
        
    def simulate_real_time_data(self, hours_elapsed: float, traffic_percent: float) -> MonitoringMetrics:
        """Simulate real-time A/B test data based on T3-Verified results"""
        
        # Base performance from T3-Verified validation
        base_ndcg = 75.0  # Baseline nDCG@10
        expected_improvement = 4.24  # Expected RA-Guard improvement
        
        # Simulate realistic variance and temporal effects
        time_factor = 1.0 + 0.1 * np.sin(hours_elapsed * np.pi / 12)  # Diurnal pattern
        noise_control = np.random.normal(0, 0.5)
        noise_treatment = np.random.normal(0, 0.5)
        
        # Control group (baseline)
        ndcg_control = base_ndcg * time_factor + noise_control
        latency_control = 150 + np.random.normal(0, 10)  # ms
        error_rate_control = 0.01 + np.random.normal(0, 0.001)
        
        # Treatment group (RA-Guard)
        # Gradually approach expected improvement as sample size grows
        confidence_factor = min(hours_elapsed / 48, 1.0)  # Full confidence after 48h
        actual_improvement = expected_improvement * confidence_factor + np.random.normal(0, 0.3)
        
        ndcg_treatment = ndcg_control + actual_improvement + noise_treatment
        latency_treatment = latency_control + 25 + np.random.normal(0, 5)  # Slight latency increase
        error_rate_treatment = error_rate_control + 0.002 + np.random.normal(0, 0.0005)
        
        # Sample sizes based on traffic allocation
        base_hourly_users = 500  # Users per hour at full traffic
        sample_control = int(base_hourly_users * hours_elapsed * traffic_percent / 100 * 0.5)
        sample_treatment = int(base_hourly_users * hours_elapsed * traffic_percent / 100 * 0.5)
        
        return MonitoringMetrics(
            timestamp=self.start_time + timedelta(hours=hours_elapsed),
            ndcg_control=ndcg_control,
            ndcg_treatment=ndcg_treatment,
            ndcg_improvement=ndcg_treatment - ndcg_control,
            latency_p95_control=latency_control,
            latency_p95_treatment=latency_treatment,
            error_rate_control=max(0, error_rate_control),
            error_rate_treatment=max(0, error_rate_treatment),
            sample_size_control=sample_control,
            sample_size_treatment=sample_treatment,
            traffic_allocation_percent=traffic_percent
        )
    
    def check_guardrail_violations(self, metrics: MonitoringMetrics) -> List[Dict]:
        """Check for guardrail metric violations"""
        
        violations = []
        
        # Latency guardrail: < +100ms
        latency_increase = metrics.latency_p95_treatment - metrics.latency_p95_control
        if latency_increase > 100:
            violations.append({
                'type': 'latency_violation',
                'severity': 'high' if latency_increase > 200 else 'medium',
                'value': latency_increase,
                'threshold': 100,
                'message': f"Latency increase: +{latency_increase:.1f}ms (threshold: +100ms)"
            })
        
        # Error rate guardrail: < +1%
        error_rate_increase = (metrics.error_rate_treatment - metrics.error_rate_control) * 100
        if error_rate_increase > 1.0:
            violations.append({
                'type': 'error_rate_violation',
                'severity': 'critical' if error_rate_increase > 2.0 else 'high',
                'value': error_rate_increase,
                'threshold': 1.0,
                'message': f"Error rate increase: +{error_rate_increase:.2f}% (threshold: +1.0%)"
            })
        
        # nDCG degradation check
        if metrics.ndcg_improvement < -1.0:
            violations.append({
                'type': 'performance_degradation',
                'severity': 'medium',
                'value': metrics.ndcg_improvement,
                'threshold': -1.0,
                'message': f"nDCG degradation: {metrics.ndcg_improvement:.2f} points"
            })
        
        return violations
    
    def calculate_statistical_power(self, hours_elapsed: float) -> Dict:
        """Calculate current statistical power and significance"""
        
        if len(self.metrics_history) < 2:
            return {'power': 0, 'significance': 1.0, 'sample_size': 0}
        
        # Get recent metrics for analysis
        recent_metrics = self.metrics_history[-10:]  # Last 10 observations
        improvements = [m.ndcg_improvement for m in recent_metrics]
        sample_sizes = [m.sample_size_control + m.sample_size_treatment for m in recent_metrics]
        
        if len(improvements) < 2:
            return {'power': 0, 'significance': 1.0, 'sample_size': 0}
        
        # Statistical analysis
        mean_improvement = np.mean(improvements)
        std_improvement = np.std(improvements)
        current_sample_size = max(sample_sizes)
        
        # Power calculation (simplified)
        effect_size = mean_improvement / max(std_improvement, 0.1)
        power = stats.norm.cdf(effect_size * np.sqrt(current_sample_size / 4) - 1.96)
        
        # Significance test (one-sample t-test against 0)
        if len(improvements) > 1 and std_improvement > 0:
            t_stat, p_value = stats.ttest_1samp(improvements, 0)
        else:
            t_stat, p_value = 0, 1.0
        
        return {
            'power': max(0, min(1, power)),
            'significance': p_value,
            'effect_size': effect_size,
            'mean_improvement': mean_improvement,
            'std_improvement': std_improvement,
            'sample_size': current_sample_size,
            't_statistic': t_stat
        }
    
    def generate_interim_report(self, hours_elapsed: float) -> Dict:
        """Generate interim analysis report"""
        
        if not self.metrics_history:
            return {}
        
        latest_metrics = self.metrics_history[-1]
        statistical_analysis = self.calculate_statistical_power(hours_elapsed)
        
        # Calculate trends
        if len(self.metrics_history) >= 24:  # 24 hours of data
            recent_24h = self.metrics_history[-24:]
            trend_improvement = np.polyfit(range(24), [m.ndcg_improvement for m in recent_24h], 1)[0]
        else:
            trend_improvement = 0
        
        # Success probability estimation
        mean_improvement = statistical_analysis.get('mean_improvement', 0)
        std_improvement = statistical_analysis.get('std_improvement', 1)
        
        if mean_improvement > 0 and std_improvement > 0:
            success_prob = 1 - stats.norm.cdf(2.97, mean_improvement, std_improvement)
        else:
            success_prob = 0.5
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'hours_elapsed': hours_elapsed,
            'current_status': {
                'traffic_allocation': f"{latest_metrics.traffic_allocation_percent}%",
                'sample_size_total': latest_metrics.sample_size_control + latest_metrics.sample_size_treatment,
                'ndcg_improvement': f"+{latest_metrics.ndcg_improvement:.2f} points",
                'statistical_power': f"{statistical_analysis['power']*100:.1f}%",
                'statistical_significance': statistical_analysis['significance']
            },
            'performance_summary': {
                'mean_improvement_24h': f"+{mean_improvement:.2f} points",
                'improvement_trend': f"{'↗️' if trend_improvement > 0 else '↘️'} {trend_improvement:.3f}/hour",
                'success_probability': f"{success_prob*100:.1f}%",
                'expected_final_result': f"+{mean_improvement:.2f} points"
            },
            'guardrail_status': {
                'latency_impact': f"+{latest_metrics.latency_p95_treatment - latest_metrics.latency_p95_control:.1f}ms",
                'error_rate_impact': f"+{(latest_metrics.error_rate_treatment - latest_metrics.error_rate_control)*100:.3f}%",
                'guardrails_healthy': len(self.check_guardrail_violations(latest_metrics)) == 0
            },
            'recommendation': self._generate_recommendation(statistical_analysis, latest_metrics)
        }
        
        return report
    
    def _generate_recommendation(self, statistical_analysis: Dict, latest_metrics: MonitoringMetrics) -> str:
        """Generate recommendation based on current data"""
        
        violations = self.check_guardrail_violations(latest_metrics)
        power = statistical_analysis.get('power', 0)
        significance = statistical_analysis.get('significance', 1.0)
        improvement = statistical_analysis.get('mean_improvement', 0)
        
        if violations:
            return "⚠️ CAUTION: Guardrail violations detected. Consider rollback if violations persist."
        
        if improvement < 0:
            return "❌ CONCERN: Negative performance trend. Monitor closely and prepare for potential rollback."
        
        if power < 0.8:
            return "📊 CONTINUE: Insufficient statistical power. Continue test to reach adequate sample size."
        
        if significance > 0.05:
            return "📈 CONTINUE: Positive trend but not yet significant. Continue test."
        
        if improvement >= 2.97:
            return "✅ SUCCESS: Strong positive results with statistical significance. Test on track for success."
        
        return "📊 MONITOR: Test progressing normally. Continue monitoring."

class ABTestDashboard:
    """Generate visual dashboard for A/B test monitoring"""
    
    def __init__(self, monitor: RealTimeABMonitor):
        self.monitor = monitor
        
    def generate_dashboard(self, output_dir: Path) -> Path:
        """Generate comprehensive visual dashboard"""
        
        if len(self.monitor.metrics_history) < 2:
            logger.warning("Insufficient data for dashboard generation")
            return None
        
        # Setup plotting
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'RA-Guard 5K A/B Test - Real-time Dashboard\nDeployment ID: {self.monitor.deployment_id}', 
                     fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        timestamps = [m.timestamp for m in self.monitor.metrics_history]
        hours_elapsed = [(t - self.monitor.start_time).total_seconds() / 3600 for t in timestamps]
        
        ndcg_improvements = [m.ndcg_improvement for m in self.monitor.metrics_history]
        latency_deltas = [m.latency_p95_treatment - m.latency_p95_control for m in self.monitor.metrics_history]
        error_rate_deltas = [(m.error_rate_treatment - m.error_rate_control) * 100 for m in self.monitor.metrics_history]
        sample_sizes = [m.sample_size_control + m.sample_size_treatment for m in self.monitor.metrics_history]
        traffic_allocations = [m.traffic_allocation_percent for m in self.monitor.metrics_history]
        
        # Plot 1: nDCG Improvement Over Time
        axes[0, 0].plot(hours_elapsed, ndcg_improvements, 'b-', linewidth=2, label='nDCG Improvement')
        axes[0, 0].axhline(y=2.97, color='g', linestyle='--', label='Success Threshold')
        axes[0, 0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
        axes[0, 0].set_title('nDCG@10 Improvement Over Time')
        axes[0, 0].set_xlabel('Hours Elapsed')
        axes[0, 0].set_ylabel('nDCG Points')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Sample Size Growth
        axes[0, 1].plot(hours_elapsed, sample_sizes, 'g-', linewidth=2, label='Total Sample Size')
        axes[0, 1].axhline(y=5000, color='r', linestyle='--', label='Target Sample Size')
        axes[0, 1].set_title('Sample Size Growth')
        axes[0, 1].set_xlabel('Hours Elapsed')
        axes[0, 1].set_ylabel('Users')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Latency Impact
        axes[1, 0].plot(hours_elapsed, latency_deltas, 'orange', linewidth=2, label='Latency Delta')
        axes[1, 0].axhline(y=100, color='r', linestyle='--', label='Guardrail Threshold')
        axes[1, 0].set_title('Latency Impact (Treatment - Control)')
        axes[1, 0].set_xlabel('Hours Elapsed')
        axes[1, 0].set_ylabel('Latency Increase (ms)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Error Rate Impact
        axes[1, 1].plot(hours_elapsed, error_rate_deltas, 'red', linewidth=2, label='Error Rate Delta')
        axes[1, 1].axhline(y=1.0, color='r', linestyle='--', label='Guardrail Threshold')
        axes[1, 1].set_title('Error Rate Impact (Treatment - Control)')
        axes[1, 1].set_xlabel('Hours Elapsed')
        axes[1, 1].set_ylabel('Error Rate Increase (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: Traffic Allocation
        axes[2, 0].step(hours_elapsed, traffic_allocations, 'purple', linewidth=2, label='Traffic %')
        axes[2, 0].set_title('Traffic Allocation Over Time')
        axes[2, 0].set_xlabel('Hours Elapsed')
        axes[2, 0].set_ylabel('Traffic Allocation (%)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 6: Statistical Power
        powers = []
        for i in range(len(self.monitor.metrics_history)):
            if i > 10:  # Need some data history
                temp_history = self.monitor.metrics_history[:i+1]
                recent = temp_history[-10:]
                improvements = [m.ndcg_improvement for m in recent]
                if len(improvements) > 1 and np.std(improvements) > 0:
                    effect_size = np.mean(improvements) / np.std(improvements)
                    power = stats.norm.cdf(effect_size * np.sqrt(len(improvements) * 100) - 1.96)
                    powers.append(max(0, min(1, power)))
                else:
                    powers.append(0)
            else:
                powers.append(0)
        
        if powers:
            axes[2, 1].plot(hours_elapsed, powers, 'brown', linewidth=2, label='Statistical Power')
            axes[2, 1].axhline(y=0.8, color='g', linestyle='--', label='80% Power Threshold')
            axes[2, 1].set_title('Statistical Power Over Time')
            axes[2, 1].set_xlabel('Hours Elapsed')
            axes[2, 1].set_ylabel('Statistical Power')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_file = output_dir / f'ab_test_dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return dashboard_file

def simulate_ab_test_execution(deployment_id: str, simulation_hours: int = 336) -> Dict:
    """Simulate complete A/B test execution (14 days = 336 hours)"""
    
    logger.info(f"Simulating A/B test execution for {simulation_hours} hours...")
    
    # Initialize monitor
    monitor = RealTimeABMonitor(deployment_id)
    dashboard = ABTestDashboard(monitor)
    
    # Ramp schedule (matches deployment plan)
    ramp_schedule = [
        (24, 1),     # Day 1: 1%
        (48, 5),     # Day 2: 5%
        (72, 10),    # Day 3: 10%
        (96, 25),    # Day 4: 25%
        (336, 50)    # Day 5-14: 50%
    ]
    
    # Simulate hour by hour
    reports = []
    dashboard_files = []
    
    output_dir = Path(f'monitoring/{deployment_id}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for hour in range(simulation_hours):
        # Determine current traffic allocation
        traffic_percent = 1  # Default
        for hour_threshold, percent in ramp_schedule:
            if hour < hour_threshold:
                traffic_percent = percent
                break
        
        # Generate metrics
        metrics = monitor.simulate_real_time_data(hour, traffic_percent)
        monitor.metrics_history.append(metrics)
        
        # Check for violations
        violations = monitor.check_guardrail_violations(metrics)
        monitor.alerts_triggered.extend(violations)
        
        # Generate reports at key intervals
        if hour % 24 == 0 or hour in [72, 168, 240]:  # Daily + key milestones
            report = monitor.generate_interim_report(hour)
            reports.append({
                'hour': hour,
                'day': hour // 24 + 1,
                'report': report
            })
            
            # Generate dashboard
            if hour > 24:  # Need some data history
                dashboard_file = dashboard.generate_dashboard(output_dir)
                if dashboard_file:
                    dashboard_files.append(str(dashboard_file))
        
        # Simulate potential early stopping for severe violations
        critical_violations = [v for v in violations if v['severity'] == 'critical']
        if critical_violations and hour > 48:  # Allow 48h buffer
            logger.warning(f"Critical violations detected at hour {hour}. Recommending early termination.")
            break
    
    # Final analysis
    final_report = monitor.generate_interim_report(len(monitor.metrics_history))
    final_dashboard = dashboard.generate_dashboard(output_dir)
    
    # Generate test summary
    all_improvements = [m.ndcg_improvement for m in monitor.metrics_history[-48:]]  # Last 48 hours
    final_sample_size = monitor.metrics_history[-1].sample_size_control + monitor.metrics_history[-1].sample_size_treatment
    
    final_analysis = {
        'test_completion_status': 'COMPLETED' if len(monitor.metrics_history) >= 336 else 'EARLY_TERMINATION',
        'total_runtime_hours': len(monitor.metrics_history),
        'final_sample_size': final_sample_size,
        'mean_improvement': np.mean(all_improvements),
        'improvement_ci_95': [
            np.percentile(all_improvements, 2.5),
            np.percentile(all_improvements, 97.5)
        ],
        'statistical_significance': stats.ttest_1samp(all_improvements, 0)[1] if len(all_improvements) > 1 else 1.0,
        'success_achieved': np.mean(all_improvements) >= 2.97,
        'guardrail_violations_total': len(monitor.alerts_triggered),
        'recommendation': 'DEPLOY_TO_PRODUCTION' if np.mean(all_improvements) >= 2.97 and len([v for v in monitor.alerts_triggered if v['severity'] == 'critical']) == 0 else 'DO_NOT_DEPLOY'
    }
    
    return {
        'simulation_metadata': {
            'deployment_id': deployment_id,
            'simulation_completed': datetime.now().isoformat(),
            'total_hours_simulated': len(monitor.metrics_history),
            'reports_generated': len(reports),
            'dashboards_created': len(dashboard_files)
        },
        'interim_reports': reports,
        'final_analysis': final_analysis,
        'final_report': final_report,
        'dashboard_files': dashboard_files,
        'alerts_triggered': monitor.alerts_triggered,
        'metrics_summary': {
            'total_data_points': len(monitor.metrics_history),
            'mean_improvement': np.mean([m.ndcg_improvement for m in monitor.metrics_history]),
            'final_sample_size': final_sample_size,
            'test_duration_days': len(monitor.metrics_history) / 24
        }
    }

def main():
    """Main execution function for A/B test monitoring"""
    
    parser = argparse.ArgumentParser(description='Monitor RA-Guard 5K A/B Test')
    parser.add_argument('--deployment-id', required=True, help='Deployment ID to monitor')
    parser.add_argument('--simulation', action='store_true', help='Run full test simulation')
    parser.add_argument('--hours', type=int, default=336, help='Hours to simulate (default: 336 = 14 days)')
    
    args = parser.parse_args()
    
    if args.simulation:
        # Run full simulation
        logger.info("Running complete A/B test simulation...")
        
        results = simulate_ab_test_execution(args.deployment_id, args.hours)
        
        # Save simulation results
        output_file = Path(f'monitoring/{args.deployment_id}/simulation_results.json')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        final_analysis = results['final_analysis']
        
        print(f"\n🔬 A/B Test Simulation Results")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"Deployment ID: {args.deployment_id}")
        print(f"Test Duration: {final_analysis['total_runtime_hours']} hours ({final_analysis['total_runtime_hours']/24:.1f} days)")
        print(f"Final Sample Size: {final_analysis['final_sample_size']:,} users")
        
        print(f"\n📊 Performance Results:")
        print(f"  Mean Improvement: +{final_analysis['mean_improvement']:.2f} nDCG points")
        print(f"  95% Confidence Interval: [{final_analysis['improvement_ci_95'][0]:.2f}, {final_analysis['improvement_ci_95'][1]:.2f}]")
        print(f"  Statistical Significance: p = {final_analysis['statistical_significance']:.4f}")
        print(f"  Success Achieved: {'✅ YES' if final_analysis['success_achieved'] else '❌ NO'}")
        
        print(f"\n⚠️  Safety Monitoring:")
        print(f"  Total Alerts: {final_analysis['guardrail_violations_total']}")
        print(f"  Test Completion: {final_analysis['test_completion_status']}")
        
        print(f"\n🚀 Final Recommendation: {final_analysis['recommendation']}")
        
        print(f"\n📂 Results saved to: {output_file}")
        print(f"📊 Dashboards: {len(results['dashboard_files'])} files generated")
        
    else:
        # Real-time monitoring mode (would connect to live data)
        logger.info("Real-time monitoring mode not implemented in simulation")
        print("Use --simulation flag to run test simulation")
    
    logger.info("A/B test monitoring completed")

if __name__ == "__main__":
    main()