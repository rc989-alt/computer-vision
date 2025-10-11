#!/usr/bin/env python3
"""
Source Governance System - Quotas, Reputation, and Quality Control

Manages source quality through:
- Per-domain and per-photographer quotas
- Rolling 30-day reputation metrics (off-topic rate, duplicate rate, etc.)
- Automatic probation and blocking based on quality thresholds
- CI integration with fail-fast policies

Usage:
    governance = SourceGovernance()
    governance.evaluate_submission_batch(items)
    governance.update_reputation_from_reviews(review_decisions)
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import os

logger = logging.getLogger(__name__)

@dataclass
class SourceStats:
    source_id: str
    domain_name: str
    photographer_id: str
    submitted: int
    accepted: int
    rejected: int
    off_topic: int
    duplicate: int
    broken_url: int
    off_topic_rate: float
    dup_rate: float
    broken_url_rate: float
    status: str  # ok, probation, blocked
    probation_until: Optional[str] = None

@dataclass
class QuotaCheck:
    allowed: bool
    reason: str
    current_count: int
    limit: int
    details: Dict[str, Any]

class SourceGovernance:
    """Source governance with quotas, reputation tracking, and quality control"""
    
    def __init__(self, db_path: str = "source_governance.db", config_path: Optional[str] = None):
        self.db_path = db_path
        self.config = self._load_config(config_path)
        self._init_database()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load governance configuration"""
        default_config = {
            "quotas": {
                "domain_cap_per_run": 10,
                "photographer_cap_per_domain_per_week": 2,
                "min_photographers_per_domain": 3
            },
            "reputation": {
                "probation_thresholds": {
                    "off_topic_rate": 0.20,
                    "dup_rate": 0.10, 
                    "broken_url_rate": 0.02,
                    "min_submissions": 10
                },
                "block_thresholds": {
                    "off_topic_rate": 0.35,
                    "min_submissions": 20
                },
                "probation_days": 30,
                "rolling_window_days": 30
            },
            "ci_policies": {
                "max_broken_url_rate_new": 0.01,  # 1%
                "max_duplicate_rate_new": 0.00,   # 0% - no duplicates allowed
                "fail_on_blocked_sources": True,
                "fail_on_quota_exceeded": True
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                user_config = json.load(f)
                # Deep merge configs
                for key in user_config:
                    if key in default_config and isinstance(default_config[key], dict):
                        default_config[key].update(user_config[key])
                    else:
                        default_config[key] = user_config[key]
                        
        return default_config
        
    def _init_database(self):
        """Initialize SQLite database for source tracking"""
        with sqlite3.connect(self.db_path) as conn:
            # Enable foreign keys and WAL mode for better concurrency
            conn.execute('PRAGMA foreign_keys = ON')
            conn.execute('PRAGMA journal_mode = WAL')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS source_stats (
                    source_id TEXT PRIMARY KEY,
                    domain_name TEXT NOT NULL,
                    photographer_id TEXT NOT NULL,
                    status TEXT DEFAULT 'ok',
                    probation_until TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS source_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,  -- submitted, accepted, rejected, off_topic, duplicate, broken_url
                    item_id TEXT,
                    reason TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT  -- JSON blob for additional context
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_source_events_source_timestamp 
                ON source_events(source_id, timestamp)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_source_events_type_timestamp
                ON source_events(event_type, timestamp)
            ''')
            
            # Commit the changes explicitly
            conn.commit()
    
    def get_source_stats(self, source_id: str, days_back: int = 30) -> SourceStats:
        """Get rolling statistics for a source over specified time window"""
        cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            # Get basic source info
            source_row = conn.execute('''
                SELECT domain_name, photographer_id, status, probation_until
                FROM source_stats WHERE source_id = ?
            ''', (source_id,)).fetchone()
            
            if not source_row:
                # Create new source entry
                return SourceStats(
                    source_id=source_id,
                    domain_name="unknown",
                    photographer_id="unknown", 
                    submitted=0, accepted=0, rejected=0,
                    off_topic=0, duplicate=0, broken_url=0,
                    off_topic_rate=0.0, dup_rate=0.0, broken_url_rate=0.0,
                    status="ok"
                )
                
            domain_name, photographer_id, status, probation_until = source_row
            
            # Get event counts in rolling window
            events = conn.execute('''
                SELECT event_type, COUNT(*) 
                FROM source_events 
                WHERE source_id = ? AND timestamp >= ?
                GROUP BY event_type
            ''', (source_id, cutoff)).fetchall()
            
            event_counts = dict(events)
            submitted = event_counts.get('submitted', 0)
            accepted = event_counts.get('accepted', 0)
            rejected = event_counts.get('rejected', 0)
            off_topic = event_counts.get('off_topic', 0)
            duplicate = event_counts.get('duplicate', 0)
            broken_url = event_counts.get('broken_url', 0)
            
            # Calculate rates (avoid division by zero)
            off_topic_rate = off_topic / max(submitted, 1)
            dup_rate = duplicate / max(submitted, 1) 
            broken_url_rate = broken_url / max(submitted, 1)
            
            return SourceStats(
                source_id=source_id,
                domain_name=domain_name,
                photographer_id=photographer_id,
                submitted=submitted,
                accepted=accepted,
                rejected=rejected,
                off_topic=off_topic,
                duplicate=duplicate,
                broken_url=broken_url,
                off_topic_rate=off_topic_rate,
                dup_rate=dup_rate,
                broken_url_rate=broken_url_rate,
                status=status,
                probation_until=probation_until
            )
    
    def check_quota(self, items: List[Dict], source_id: str, domain: str) -> QuotaCheck:
        """Check if submission respects quotas"""
        config = self.config["quotas"]
        
        # Count items by domain in this batch
        domain_items = [item for item in items if item.get("domain") == domain]
        domain_count = len(domain_items)
        
        # Check domain cap per run
        if domain_count > config["domain_cap_per_run"]:
            return QuotaCheck(
                allowed=False,
                reason="domain_cap_exceeded",
                current_count=domain_count,
                limit=config["domain_cap_per_run"],
                details={"domain": domain, "items": [item["id"] for item in domain_items]}
            )
            
        # Check photographer weekly cap for this domain
        photographer_id = self._extract_photographer_id(source_id)
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            weekly_count = conn.execute('''
                SELECT COUNT(*) FROM source_events se
                JOIN source_stats ss ON se.source_id = ss.source_id
                WHERE ss.photographer_id = ? AND ss.domain_name = ? 
                AND se.event_type = 'submitted' AND se.timestamp >= ?
            ''', (photographer_id, domain, week_ago)).fetchone()[0]
            
        if weekly_count + domain_count > config["photographer_cap_per_domain_per_week"]:
            return QuotaCheck(
                allowed=False,
                reason="photographer_weekly_cap_exceeded",
                current_count=weekly_count + domain_count,
                limit=config["photographer_cap_per_domain_per_week"],
                details={"photographer_id": photographer_id, "domain": domain}
            )
            
        return QuotaCheck(
            allowed=True,
            reason="quota_ok",
            current_count=domain_count,
            limit=config["domain_cap_per_run"],
            details={}
        )
    
    def update_reputation(self, source_id: str, domain: str, photographer_id: str):
        """Update source reputation and status based on recent performance"""
        stats = self.get_source_stats(source_id)
        config = self.config["reputation"]
        
        # Update source_stats table
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO source_stats 
                (source_id, domain_name, photographer_id, status, probation_until, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (source_id, domain, photographer_id, stats.status, stats.probation_until, 
                  datetime.now().isoformat()))
        
        # Check if source should be put on probation or blocked
        new_status = self._calculate_status(stats, config)
        
        if new_status != stats.status:
            self._update_source_status(source_id, new_status)
            logger.info(f"Source {source_id} status changed: {stats.status} -> {new_status}")
            
    def _calculate_status(self, stats: SourceStats, config: Dict) -> str:
        """Calculate new status based on reputation metrics"""
        prob_th = config["probation_thresholds"]
        block_th = config["block_thresholds"]
        
        # Need minimum submissions to make decisions
        if stats.submitted < prob_th["min_submissions"]:
            return "ok"
            
        # Check for blocking conditions
        if (stats.submitted >= block_th["min_submissions"] and 
            stats.off_topic_rate > block_th["off_topic_rate"]):
            return "blocked"
            
        # Check for probation conditions
        if (stats.off_topic_rate > prob_th["off_topic_rate"] or
            stats.dup_rate > prob_th["dup_rate"] or 
            stats.broken_url_rate > prob_th["broken_url_rate"]):
            return "probation"
            
        # Check if probation period has expired
        if stats.status == "probation" and stats.probation_until:
            probation_end = datetime.fromisoformat(stats.probation_until)
            if datetime.now() > probation_end:
                return "ok"
                
        return stats.status
    
    def _update_source_status(self, source_id: str, new_status: str):
        """Update source status in database"""
        probation_until = None
        if new_status == "probation":
            probation_end = datetime.now() + timedelta(days=self.config["reputation"]["probation_days"])
            probation_until = probation_end.isoformat()
            
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE source_stats 
                SET status = ?, probation_until = ?, updated_at = ?
                WHERE source_id = ?
            ''', (new_status, probation_until, datetime.now().isoformat(), source_id))
    
    def log_event(self, source_id: str, event_type: str, item_id: str = None, 
                  reason: str = None, metadata: Dict = None):
        """Log an event for reputation tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO source_events 
                (source_id, event_type, item_id, reason, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (source_id, event_type, item_id, reason, 
                  json.dumps(metadata) if metadata else None,
                  datetime.now().isoformat()))
    
    def evaluate_batch_for_ci(self, items: List[Dict]) -> Dict[str, Any]:
        """Evaluate batch against CI policies - return pass/fail with details"""
        config = self.config["ci_policies"]
        
        # Group items by source and domain
        source_domains = defaultdict(list)
        for item in items:
            source_id = item.get("source_id", "unknown")
            domain = item.get("domain", "unknown")
            source_domains[(source_id, domain)].append(item)
        
        results = {
            "pass": True,
            "failures": [],
            "warnings": [],
            "source_stats": {},
            "domain_stats": {}
        }
        
        # Check each source/domain combination
        for (source_id, domain), source_items in source_domains.items():
            stats = self.get_source_stats(source_id)
            
            # Check if source is blocked
            if config["fail_on_blocked_sources"] and stats.status == "blocked":
                results["pass"] = False
                results["failures"].append({
                    "type": "blocked_source",
                    "source_id": source_id,
                    "domain": domain,
                    "message": f"Source {source_id} is blocked but submitted {len(source_items)} items"
                })
            
            # Check quotas
            quota_check = self.check_quota(items, source_id, domain)
            if config["fail_on_quota_exceeded"] and not quota_check.allowed:
                results["pass"] = False
                results["failures"].append({
                    "type": "quota_exceeded",
                    "source_id": source_id,
                    "domain": domain,
                    "message": f"Quota exceeded: {quota_check.reason}",
                    "details": quota_check.details
                })
                
            results["source_stats"][source_id] = {
                "domain": domain,
                "status": stats.status,
                "item_count": len(source_items),
                "off_topic_rate": stats.off_topic_rate,
                "dup_rate": stats.dup_rate,
                "broken_url_rate": stats.broken_url_rate
            }
        
        # Domain-level stats
        domain_counts = Counter(item.get("domain", "unknown") for item in items)
        results["domain_stats"] = dict(domain_counts)
        
        return results
    
    def _extract_photographer_id(self, source_id: str) -> str:
        """Extract photographer ID from source ID (customize based on your format)"""
        # Assuming source_id format like "unsplash_photographer123" or similar
        if "_" in source_id:
            return source_id.split("_", 1)[1]
        return source_id
    
    def generate_governance_report(self, items: List[Dict]) -> str:
        """Generate HTML report for CI pipeline"""
        evaluation = self.evaluate_batch_for_ci(items)
        
        html = f"""
        <html><head><title>Source Governance Report</title></head><body>
        <h1>Source Governance Report</h1>
        <p>Generated: {datetime.now().isoformat()}</p>
        
        <h2>Overall Status: {'✅ PASS' if evaluation['pass'] else '❌ FAIL'}</h2>
        
        <h3>Batch Summary</h3>
        <ul>
        <li>Total items: {len(items)}</li>
        <li>Unique sources: {len(evaluation['source_stats'])}</li>
        <li>Domains: {', '.join(evaluation['domain_stats'].keys())}</li>
        </ul>
        """
        
        if evaluation["failures"]:
            html += "<h3>❌ Failures</h3><ul>"
            for failure in evaluation["failures"]:
                html += f"<li><strong>{failure['type']}</strong>: {failure['message']}</li>"
            html += "</ul>"
            
        if evaluation["warnings"]:
            html += "<h3>⚠️ Warnings</h3><ul>"
            for warning in evaluation["warnings"]:
                html += f"<li>{warning}</li>"
            html += "</ul>"
            
        html += "<h3>Source Details</h3><table border='1'>"
        html += "<tr><th>Source</th><th>Domain</th><th>Status</th><th>Items</th><th>Off-topic%</th><th>Dup%</th></tr>"
        
        for source_id, stats in evaluation["source_stats"].items():
            status_emoji = {"ok": "✅", "probation": "⚠️", "blocked": "❌"}[stats["status"]]
            html += f"""
            <tr>
                <td>{source_id}</td>
                <td>{stats['domain']}</td>
                <td>{status_emoji} {stats['status']}</td>
                <td>{stats['item_count']}</td>
                <td>{stats['off_topic_rate']:.1%}</td>
                <td>{stats['dup_rate']:.1%}</td>
            </tr>
            """
            
        html += "</table></body></html>"
        return html

def demo_governance():
    """Demo the governance system with sample data"""
    # Use temp file instead of memory for better reliability
    import tempfile
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    
    governance = SourceGovernance(db_path=temp_db.name)
    
    # Simulate some source events
    sources = [
        ("unsplash_photographer1", "blue_tropical"),
        ("unsplash_photographer2", "black_charcoal"), 
        ("unsplash_photographer3", "red_berry")
    ]
    
    # Log some events to build reputation
    for source_id, domain in sources:
        photographer_id = source_id.split("_")[1]
        
        # Simulate submissions and outcomes
        for i in range(15):
            governance.log_event(source_id, "submitted", f"item_{i}")
            
            # photographer2 has quality issues
            if "photographer2" in source_id:
                if i % 3 == 0:  # 33% off-topic rate
                    governance.log_event(source_id, "off_topic", f"item_{i}", "not_cocktail")
                else:
                    governance.log_event(source_id, "accepted", f"item_{i}")
            else:
                governance.log_event(source_id, "accepted", f"item_{i}")
                
        governance.update_reputation(source_id, domain, photographer_id)
    
    # Check governance evaluation
    test_items = [
        {"id": "test1", "source_id": "unsplash_photographer1", "domain": "blue_tropical"},
        {"id": "test2", "source_id": "unsplash_photographer2", "domain": "black_charcoal"},  # Should be on probation
        {"id": "test3", "source_id": "unsplash_photographer3", "domain": "red_berry"}
    ]
    
    evaluation = governance.evaluate_batch_for_ci(test_items)
    print("Governance Evaluation:")
    print(f"Pass: {evaluation['pass']}")
    print(f"Failures: {len(evaluation['failures'])}")
    
    for source_id, stats in evaluation['source_stats'].items():
        print(f"  {source_id}: {stats['status']} ({stats['off_topic_rate']:.1%} off-topic)")

if __name__ == "__main__":
    demo_governance()