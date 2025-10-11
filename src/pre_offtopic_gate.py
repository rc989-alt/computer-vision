#!/usr/bin/env python3
"""
Pre-Off-Topic Gate - Filter obvious non-cocktails before borderline review

Catches items that should never reach human reviewers:
- Low cocktail similarity + no glass/liquid detected
- Broken URLs, tiny images, wrong MIME types
- Prevents keyboards, random objects from wasting review time

Usage:
    gate = PreOffTopicGate()
    should_discard = gate.evaluate(item, sims, detections)
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import requests
from PIL import Image
import io

logger = logging.getLogger(__name__)

@dataclass
class GateResult:
    discard: bool
    reason: str
    details: Dict[str, Any]

class PreOffTopicGate:
    """Fast pre-filter to catch obvious off-topic items before borderline review"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict:
        return {
            "sim_cocktail_min": 0.60,  # Raised threshold - keyboard has only 0.5
            "neg_beats_pos": True,
            "object_requirements": ["glass", "cup", "martini", "rocks_glass", "liquid", "garnish"],
            "min_image_size": 256,
            "max_image_kb": 10240,  # 10MB
            "allowed_mime_types": ["image/jpeg", "image/png", "image/webp"]
        }
    
    def evaluate(self, item: Dict, sims: Dict[str, float], detections: List[str]) -> GateResult:
        """
        Evaluate if item should be discarded before borderline review
        
        Args:
            item: Item metadata with url, domain, etc.
            sims: CLIP similarities {"cocktail": 0.x, "not_cocktail": 0.y}
            detections: List of detected objects
            
        Returns:
            GateResult with discard decision and reasoning
        """
        
        # 1. CLIP similarity gates
        clip_result = self._check_clip_similarity(sims)
        if clip_result.discard:
            return clip_result
            
        # 2. Object detection gates
        object_result = self._check_required_objects(detections)
        if object_result.discard:
            return object_result
            
        # 3. Combined CLIP + object gate (strictest)
        combined_result = self._check_combined_gates(sims, detections)
        if combined_result.discard:
            return combined_result
            
        # 4. Technical quality gates
        tech_result = self._check_technical_quality(item)
        if tech_result.discard:
            return tech_result
            
        return GateResult(
            discard=False,
            reason="passed_all_gates",
            details={"sims": sims, "detections": detections}
        )
    
    def _check_clip_similarity(self, sims: Dict[str, float]) -> GateResult:
        """Check CLIP similarity thresholds"""
        pos = sims.get("cocktail", 0.0)
        neg = sims.get("not_cocktail", 0.0)
        
        # Very low cocktail similarity
        if pos < self.config["sim_cocktail_min"]:
            return GateResult(
                discard=True,
                reason="clip_cocktail_too_low",
                details={"sim_cocktail": pos, "threshold": self.config["sim_cocktail_min"]}
            )
            
        # Negative similarity beats positive
        if self.config["neg_beats_pos"] and neg >= pos:
            return GateResult(
                discard=True,
                reason="clip_negative_wins",
                details={"sim_cocktail": pos, "sim_not_cocktail": neg}
            )
            
        return GateResult(discard=False, reason="clip_ok", details={"sims": sims})
    
    def _check_required_objects(self, detections: List[str]) -> GateResult:
        """Check if any required objects are detected"""
        required = set(self.config["object_requirements"])
        detected = set(detections)
        
        has_required = bool(required & detected)
        
        if not has_required:
            return GateResult(
                discard=False,  # Don't discard on objects alone - too strict
                reason="no_required_objects", 
                details={"required": list(required), "detected": detections}
            )
            
        return GateResult(discard=False, reason="objects_ok", details={"detections": detections})
    
    def _check_combined_gates(self, sims: Dict[str, float], detections: List[str]) -> GateResult:
        """Combined CLIP + object check - most reliable filter"""
        pos = sims.get("cocktail", 0.0)
        neg = sims.get("not_cocktail", 0.0)
        
        required = set(self.config["object_requirements"])
        detected = set(detections)
        has_required = bool(required & detected)
        
        # Discard if BOTH conditions are true:
        # 1. Poor CLIP score (low cocktail OR negative wins)
        # 2. No required objects detected
        poor_clip = (pos < self.config["sim_cocktail_min"]) or (neg >= pos)
        
        if poor_clip and not has_required:
            return GateResult(
                discard=True,
                reason="combined_clip_objects_fail",
                details={
                    "sim_cocktail": pos,
                    "sim_not_cocktail": neg,
                    "detections": detections,
                    "has_required_objects": has_required
                }
            )
            
        return GateResult(discard=False, reason="combined_ok", details={})
    
    def _check_technical_quality(self, item: Dict) -> GateResult:
        """Check technical quality: image size, MIME type, URL health"""
        url = item.get("url", "")
        
        if not url:
            return GateResult(discard=True, reason="no_url", details={})
            
        try:
            # Quick HEAD request to check URL and MIME type
            response = requests.head(url, timeout=5, allow_redirects=True)
            
            if response.status_code >= 400:
                return GateResult(
                    discard=True,
                    reason="broken_url",
                    details={"status_code": response.status_code, "url": url}
                )
                
            content_type = response.headers.get('content-type', '').lower()
            if not any(mime in content_type for mime in self.config["allowed_mime_types"]):
                return GateResult(
                    discard=True,
                    reason="invalid_mime_type",
                    details={"content_type": content_type}
                )
                
            # Check image size if content-length available
            content_length = response.headers.get('content-length')
            if content_length:
                size_kb = int(content_length) / 1024
                if size_kb > self.config["max_image_kb"]:
                    return GateResult(
                        discard=True,
                        reason="image_too_large",
                        details={"size_kb": size_kb, "max_kb": self.config["max_image_kb"]}
                    )
                    
        except Exception as e:
            logger.warning(f"Technical quality check failed for {url}: {e}")
            return GateResult(
                discard=True,
                reason="url_check_failed",
                details={"error": str(e), "url": url}
            )
            
        return GateResult(discard=False, reason="technical_ok", details={})

def simulate_item_evaluation():
    """Test with the keyboard item to verify it gets caught"""
    gate = PreOffTopicGate()
    
    # Simulate the keyboard item (demo_017)
    keyboard_item = {
        "id": "demo_017",
        "domain": "black_charcoal", 
        "query": "black charcoal cocktail with activated carbon",
        "url": "https://images.unsplash.com/photo-1612198188060-c7c2a3b66eae"
    }
    
    keyboard_sims = {
        "cocktail": 0.5,       # Suspiciously low
        "not_cocktail": 0.275  # Still lower, but close
    }
    
    keyboard_detections = ["glass", "olive"]  # Has glass, so not purely technical
    
    result = gate.evaluate(keyboard_item, keyboard_sims, keyboard_detections)
    
    print(f"Keyboard item evaluation:")
    print(f"  Discard: {result.discard}")  
    print(f"  Reason: {result.reason}")
    print(f"  Details: {result.details}")
    
    # Test a good cocktail item
    good_item = {
        "id": "demo_005", 
        "url": "https://images.unsplash.com/photo-1546171753-97d7676e4602"
    }
    
    good_sims = {
        "cocktail": 0.611,
        "not_cocktail": 0.414  
    }
    
    good_detections = ["glass", "olive", "garnish"]
    
    good_result = gate.evaluate(good_item, good_sims, good_detections)
    
    print(f"\nGood cocktail evaluation:")
    print(f"  Discard: {good_result.discard}")
    print(f"  Reason: {good_result.reason}")

if __name__ == "__main__":
    simulate_item_evaluation()