#!/usr/bin/env python3
"""
Source Governance CI Pipeline

Integrates source governance checks into CI pipeline:
- Quota enforcement per domain/photographer
- Reputation-based probation and blocking
- Pre-off-topic gate filtering
- Fail-fast policies with detailed reporting

Usage:
    # Check a batch before processing
    python source_governance_ci.py --batch-file candidates.json --check-only
    
    # Full pipeline with governance
    python source_governance_ci.py --batch-file candidates.json --apply-governance
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from source_governance import SourceGovernance
from pre_offtopic_gate import PreOffTopicGate

logger = logging.getLogger(__name__)

class SourceGovernanceCI:
    """CI integration for source governance pipeline"""
    
    def __init__(self, config_path: str = "config/governance.json"):
        self.config_path = config_path
        self.governance = SourceGovernance(config_path=config_path)
        self.pre_gate = PreOffTopicGate()
        
    def process_batch(self, batch_file: str, apply_governance: bool = False) -> Dict[str, Any]:
        """Process a batch of candidate items through governance pipeline"""
        
        # Load candidate items
        with open(batch_file, 'r') as f:
            items = json.load(f)
        
        logger.info(f"Processing batch of {len(items)} items")
        
        results = {
            "input_count": len(items),
            "timestamp": datetime.now().isoformat(),
            "stages": {},
            "final_items": [],
            "quarantined": [],
            "governance_report": {},
            "success": True,
            "errors": []
        }
        
        try:
            # Stage 1: Pre-off-topic gate
            gate_results = self._apply_pre_gate(items)
            results["stages"]["pre_gate"] = gate_results
            
            passed_items = gate_results["passed_items"]
            results["quarantined"].extend(gate_results["quarantined"])
            
            # Stage 2: Source governance checks
            if apply_governance:
                gov_results = self._apply_governance_checks(passed_items)
                results["stages"]["governance"] = gov_results
                results["governance_report"] = gov_results["evaluation"]
                
                # Check if pipeline should fail
                if not gov_results["evaluation"]["pass"]:
                    results["success"] = False
                    results["errors"].extend([f["message"] for f in gov_results["evaluation"]["failures"]])
                    
                passed_items = gov_results["approved_items"]
            
            # Stage 3: Final results
            results["final_items"] = passed_items
            results["final_count"] = len(passed_items)
            
            # Log source events if applying governance
            if apply_governance:
                self._log_source_events(items, results)
                
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            results["success"] = False
            results["errors"].append(str(e))
            
        return results
    
    def _apply_pre_gate(self, items: List[Dict]) -> Dict[str, Any]:
        """Apply pre-off-topic gate filtering"""
        logger.info("Applying pre-off-topic gate...")
        
        passed = []
        quarantined = []
        
        for item in items:
            # Extract similarities and detections
            sims = {
                'cocktail': item.get('sim_cocktail', 0.0),
                'not_cocktail': item.get('sim_not_cocktail', 0.0)
            }
            detections = item.get('detected_objects', [])
            
            # Evaluate with gate
            gate_result = self.pre_gate.evaluate(item, sims, detections)
            
            if gate_result.discard:
                quarantined.append({
                    'item_id': item.get('id'),
                    'reason': gate_result.reason,
                    'details': gate_result.details,
                    'stage': 'pre_gate'
                })
                logger.info(f"Pre-gate quarantined {item.get('id')}: {gate_result.reason}")
            else:
                passed.append(item)
        
        return {
            "input_count": len(items),
            "passed_count": len(passed),
            "quarantined_count": len(quarantined),
            "passed_items": passed,
            "quarantined": quarantined
        }
    
    def _apply_governance_checks(self, items: List[Dict]) -> Dict[str, Any]:
        """Apply source governance quotas and reputation checks"""
        logger.info("Applying source governance checks...")
        
        # Evaluate batch against CI policies
        evaluation = self.governance.evaluate_batch_for_ci(items)
        
        # Filter items based on governance rules
        approved_items = []
        rejected_items = []
        
        # Group items by source for quota checking
        source_items = {}
        for item in items:
            source_id = item.get('source_id', 'unknown')
            if source_id not in source_items:
                source_items[source_id] = []
            source_items[source_id].append(item)
        
        # Check each source
        for source_id, source_item_list in source_items.items():
            stats = self.governance.get_source_stats(source_id)
            
            # Skip blocked sources
            if stats.status == "blocked":
                rejected_items.extend(source_item_list)
                logger.warning(f"Rejected {len(source_item_list)} items from blocked source {source_id}")
                continue
                
            # Apply quota limits (sample implementation)
            for item in source_item_list:
                domain = item.get('domain', 'unknown')
                quota_check = self.governance.check_quota(items, source_id, domain)
                
                if quota_check.allowed:
                    approved_items.append(item)
                else:
                    rejected_items.append(item)
                    logger.warning(f"Quota rejected {item.get('id')}: {quota_check.reason}")
        
        return {
            "evaluation": evaluation,
            "input_count": len(items),
            "approved_count": len(approved_items),
            "rejected_count": len(rejected_items),
            "approved_items": approved_items,
            "rejected_items": rejected_items
        }
    
    def _log_source_events(self, original_items: List[Dict], results: Dict[str, Any]):
        """Log events to source governance system"""
        
        # Log submissions
        for item in original_items:
            source_id = item.get('source_id', 'unknown')
            self.governance.log_event(
                source_id, 
                "submitted", 
                item.get('id'),
                metadata={"domain": item.get('domain')}
            )
        
        # Log quarantined items as off-topic
        for quarantined in results["quarantined"]:
            if 'source_id' in quarantined.get('item', {}):
                source_id = quarantined['item']['source_id']
                self.governance.log_event(
                    source_id,
                    "off_topic",
                    quarantined.get('item_id'),
                    reason=quarantined['reason']
                )
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report for CI pipeline"""
        
        status_emoji = "✅" if results["success"] else "❌"
        
        html = f"""
        <!DOCTYPE html>
        <html><head><title>Source Governance CI Report</title></head><body>
        <h1>Source Governance CI Report {status_emoji}</h1>
        <p>Generated: {results['timestamp']}</p>
        
        <h2>Pipeline Summary</h2>
        <ul>
        <li>Input items: {results['input_count']}</li>
        <li>Final items: {results['final_count']}</li>
        <li>Success: {'Yes' if results['success'] else 'No'}</li>
        </ul>
        """
        
        # Stage results
        if "pre_gate" in results["stages"]:
            gate = results["stages"]["pre_gate"]
            html += f"""
            <h3>Pre-Off-Topic Gate</h3>
            <ul>
            <li>Input: {gate['input_count']} items</li>
            <li>Passed: {gate['passed_count']} items</li>
            <li>Quarantined: {gate['quarantined_count']} items</li>
            </ul>
            """
            
        if "governance" in results["stages"]:
            gov = results["stages"]["governance"]
            html += f"""
            <h3>Source Governance</h3>
            <ul>
            <li>Input: {gov['input_count']} items</li>
            <li>Approved: {gov['approved_count']} items</li>
            <li>Rejected: {gov['rejected_count']} items (quotas/blocking)</li>
            </ul>
            """
        
        # Errors
        if results["errors"]:
            html += "<h3>❌ Errors</h3><ul>"
            for error in results["errors"]:
                html += f"<li>{error}</li>"
            html += "</ul>"
        
        # Quarantined items
        if results["quarantined"]:
            html += "<h3>🚫 Quarantined Items</h3><table border='1'>"
            html += "<tr><th>Item ID</th><th>Reason</th><th>Stage</th></tr>"
            for q in results["quarantined"]:
                html += f"<tr><td>{q.get('item_id', 'N/A')}</td><td>{q['reason']}</td><td>{q.get('stage', 'unknown')}</td></tr>"
            html += "</table>"
        
        # Source governance details
        if "governance_report" in results and results["governance_report"]:
            gov_report = results["governance_report"]
            if "source_stats" in gov_report:
                html += "<h3>Source Details</h3><table border='1'>"
                html += "<tr><th>Source</th><th>Domain</th><th>Status</th><th>Items</th><th>Off-topic%</th></tr>"
                
                for source_id, stats in gov_report["source_stats"].items():
                    status_emoji = {"ok": "✅", "probation": "⚠️", "blocked": "❌"}[stats["status"]]
                    html += f"""
                    <tr>
                        <td>{source_id}</td>
                        <td>{stats.get('domain', 'N/A')}</td>
                        <td>{status_emoji} {stats['status']}</td>
                        <td>{stats.get('item_count', 0)}</td>
                        <td>{stats.get('off_topic_rate', 0):.1%}</td>
                    </tr>
                    """
                html += "</table>"
        
        html += "</body></html>"
        return html

def main():
    parser = argparse.ArgumentParser(description="Source Governance CI Pipeline")
    parser.add_argument("--batch-file", required=True, help="JSON file with candidate items")
    parser.add_argument("--config", default="config/governance.json", help="Governance configuration")
    parser.add_argument("--check-only", action="store_true", help="Check only, don't apply governance")
    parser.add_argument("--report-file", help="Output HTML report file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run pipeline
    ci = SourceGovernanceCI(config_path=args.config)
    results = ci.process_batch(args.batch_file, apply_governance=not args.check_only)
    
    # Generate report
    if args.report_file:
        report_html = ci.generate_report(results)
        with open(args.report_file, 'w') as f:
            f.write(report_html)
        logger.info(f"Report saved to: {args.report_file}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"SOURCE GOVERNANCE CI RESULTS")
    print(f"{'='*50}")
    print(f"Status: {'✅ PASS' if results['success'] else '❌ FAIL'}")
    print(f"Input: {results['input_count']} items")
    print(f"Final: {results['final_count']} items")
    print(f"Quarantined: {len(results['quarantined'])} items")
    
    if results["errors"]:
        print(f"\nErrors:")
        for error in results["errors"]:
            print(f"  - {error}")
    
    # Exit code for CI
    sys.exit(0 if results["success"] else 1)

if __name__ == "__main__":
    main()