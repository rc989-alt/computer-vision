#!/usr/bin/env python3
"""
RA-Guard Configuration Audit System
Validates images against YAML/DSL compliance rules
"""

import yaml
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image
import numpy as np

class ComplianceAuditor:
    """Audit gallery against YAML/DSL rules"""
    
    def __init__(self, gallery_dir: str = "pilot_gallery"):
        self.gallery_dir = Path(gallery_dir)
        self.db_path = self.gallery_dir / "candidate_library.db"
        
        # Load compliance rules
        self.rules = self._load_compliance_rules()
    
    def _load_compliance_rules(self) -> Dict:
        """Load YAML/DSL compliance configuration"""
        
        # Default rules for cocktails domain
        default_rules = {
            'cocktails': {
                'require': [
                    {'subject': 'cocktail', 'confidence': 0.7},
                    {'subject': 'glass', 'confidence': 0.5},
                ],
                'forbid': [
                    {'type': 'watermark', 'threshold': 0.1},
                    {'type': 'text_overlay', 'coverage': 0.05},
                    {'type': 'logo', 'size_ratio': 0.02},
                    {'type': 'nsfw_content', 'confidence': 0.1}
                ],
                'relations': [
                    {'type': 'size_ratio', 'subject': 'cocktail', 'min_ratio': 0.1},
                    {'type': 'position', 'subject': 'glass', 'constraint': 'centered'},
                ],
                'thresholds': {
                    'min_resolution': 512,
                    'max_aspect_ratio': 3.0,
                    'min_subject_ratio': 0.05,
                    'delta_e_threshold': 15.0
                }
            },
            'flowers': {
                'require': [
                    {'subject': 'flower', 'confidence': 0.8},
                ],
                'forbid': [
                    {'type': 'artificial', 'confidence': 0.3},
                    {'type': 'plastic', 'confidence': 0.2},
                ],
                'thresholds': {
                    'min_resolution': 512,
                    'natural_score': 0.7
                }
            },
            'professional': {
                'require': [
                    {'subject': 'person', 'confidence': 0.9},
                    {'subject': 'business_attire', 'confidence': 0.6},
                ],
                'forbid': [
                    {'type': 'casual_attire', 'confidence': 0.7},
                    {'type': 'inappropriate', 'confidence': 0.1},
                ],
                'thresholds': {
                    'min_resolution': 800,
                    'professionalism_score': 0.8
                }
            }
        }
        
        # Try to load from file, fallback to defaults
        config_path = Path("config/compliance_rules.yaml")
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        else:
            # Save defaults
            config_path.parent.mkdir(exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(default_rules, f, indent=2)
            return default_rules
    
    def audit_gallery(self) -> Dict:
        """Run complete compliance audit"""
        
        print("🔍 RA-GUARD COMPLIANCE AUDIT")
        print("=" * 40)
        
        # Get all candidates
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, url_path, domain, det_cache FROM candidates')
        candidates = cursor.fetchall()
        conn.close()
        
        audit_results = {
            'total_images': len(candidates),
            'violations': [],
            'compliance_rate': {},
            'domain_stats': {}
        }
        
        for id, url_path, domain, det_cache_str in candidates:
            if not Path(url_path).exists():
                continue
                
            # Parse detection cache
            try:
                det_cache = json.loads(det_cache_str) if det_cache_str else {}
            except:
                det_cache = {}
            
            # Audit this image
            violations = self._audit_single_image(id, url_path, domain, det_cache)
            
            if violations:
                audit_results['violations'].extend(violations)
            
            # Update domain stats
            if domain not in audit_results['domain_stats']:
                audit_results['domain_stats'][domain] = {
                    'total': 0,
                    'violations': 0,
                    'compliance_rate': 0.0
                }
            
            audit_results['domain_stats'][domain]['total'] += 1
            if violations:
                audit_results['domain_stats'][domain]['violations'] += 1
        
        # Calculate compliance rates
        for domain, stats in audit_results['domain_stats'].items():
            if stats['total'] > 0:
                stats['compliance_rate'] = (stats['total'] - stats['violations']) / stats['total']
        
        overall_violations = len(audit_results['violations'])
        overall_compliance = (len(candidates) - overall_violations) / len(candidates) if candidates else 0
        audit_results['overall_compliance_rate'] = overall_compliance
        
        return audit_results
    
    def _audit_single_image(self, id: str, url_path: str, domain: str, det_cache: Dict) -> List[Dict]:
        """Audit single image against domain rules"""
        
        violations = []
        
        if domain not in self.rules:
            return violations
        
        domain_rules = self.rules[domain]
        
        try:
            img = Image.open(url_path)
            width, height = img.size
            
            # Check resolution requirements
            min_res = domain_rules.get('thresholds', {}).get('min_resolution', 512)
            if min(width, height) < min_res:
                violations.append({
                    'id': id,
                    'type': 'resolution',
                    'rule': f'min_resolution >= {min_res}',
                    'actual': min(width, height),
                    'severity': 'error'
                })
            
            # Check aspect ratio
            aspect_ratio = max(width, height) / min(width, height)
            max_aspect = domain_rules.get('thresholds', {}).get('max_aspect_ratio', 3.0)
            if aspect_ratio > max_aspect:
                violations.append({
                    'id': id,
                    'type': 'aspect_ratio',
                    'rule': f'aspect_ratio <= {max_aspect}',
                    'actual': aspect_ratio,
                    'severity': 'warning'
                })
            
            # Check required subjects
            for req in domain_rules.get('require', []):
                subject = req['subject']
                min_conf = req['confidence']
                
                # Check if subject detected with sufficient confidence
                subject_found = False
                for detection in det_cache.get('detections', []):
                    if detection.get('class') == subject and detection.get('confidence', 0) >= min_conf:
                        subject_found = True
                        break
                
                if not subject_found:
                    violations.append({
                        'id': id,
                        'type': 'missing_subject',
                        'rule': f'require {subject} >= {min_conf}',
                        'actual': 'not_found',
                        'severity': 'error'
                    })
            
            # Check forbidden elements
            for forbid in domain_rules.get('forbid', []):
                forbid_type = forbid['type']
                
                if forbid_type in ['watermark', 'text_overlay', 'logo']:
                    # Simple check - could be enhanced with actual detection
                    threshold = forbid.get('threshold', forbid.get('coverage', 0.05))
                    
                    # Mock detection for demo
                    detected_coverage = self._mock_detect_overlay(img, forbid_type)
                    
                    if detected_coverage > threshold:
                        violations.append({
                            'id': id,
                            'type': 'forbidden_content',
                            'rule': f'forbid {forbid_type} > {threshold}',
                            'actual': detected_coverage,
                            'severity': 'error'
                        })
        
        except Exception as e:
            violations.append({
                'id': id,
                'type': 'processing_error',
                'rule': 'image_processable',
                'actual': str(e),
                'severity': 'error'
            })
        
        return violations
    
    def _mock_detect_overlay(self, img: Image, overlay_type: str) -> float:
        """Mock overlay detection - replace with real detection"""
        
        # Simple mock based on image properties
        img_array = np.array(img)
        
        if overlay_type == 'watermark':
            # Check for semi-transparent regions (mock)
            if len(img_array.shape) == 3:
                # Look for uniform regions that might be watermarks
                std_dev = np.std(img_array)
                return min(0.02, (100 - std_dev) / 10000)  # Lower std might indicate overlays
            
        elif overlay_type == 'text_overlay':
            # Check for high contrast regions (mock)
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
                edges = np.abs(np.diff(gray, axis=0)).sum() + np.abs(np.diff(gray, axis=1)).sum()
                normalized_edges = edges / (gray.shape[0] * gray.shape[1])
                return min(0.01, normalized_edges / 1000)
        
        return 0.0  # No overlay detected
    
    def save_audit_report(self, audit_results: Dict, output_path: str = None):
        """Save audit report to file"""
        
        if output_path is None:
            output_path = self.gallery_dir / "compliance_audit_report.json"
        
        with open(output_path, 'w') as f:
            json.dump(audit_results, f, indent=2)
        
        return output_path

def main():
    auditor = ComplianceAuditor("pilot_gallery")
    
    print("🔍 Running compliance audit on pilot gallery...")
    results = auditor.audit_gallery()
    
    # Save report
    report_path = auditor.save_audit_report(results)
    
    # Display summary
    print(f"\n📊 AUDIT SUMMARY:")
    print(f"   • Total images: {results['total_images']}")
    print(f"   • Overall compliance: {results['overall_compliance_rate']:.1%}")
    print(f"   • Total violations: {len(results['violations'])}")
    
    print(f"\n📋 BY DOMAIN:")
    for domain, stats in results['domain_stats'].items():
        print(f"   • {domain}: {stats['compliance_rate']:.1%} ({stats['total'] - stats['violations']}/{stats['total']})")
    
    if results['violations']:
        print(f"\n⚠️  VIOLATIONS FOUND:")
        violation_types = {}
        for v in results['violations']:
            v_type = v['type']
            violation_types[v_type] = violation_types.get(v_type, 0) + 1
        
        for v_type, count in violation_types.items():
            print(f"   • {v_type}: {count} images")
    
    print(f"\n💾 Full report saved to: {report_path}")

if __name__ == "__main__":
    main()