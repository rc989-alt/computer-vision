#!/usr/bin/env python3
"""
Create demo grid images showing baseline vs enhanced system comparisons.
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_demo_grid(title, baseline_score, enhanced_score, filename):
    """Create a side-by-side comparison grid image."""
    
    # Image dimensions
    width, height = 1024, 512
    left_width = width // 2
    
    # Colors
    bg_color = (45, 55, 75)  # Dark blue-gray
    baseline_color = (180, 100, 100)  # Muted red
    enhanced_color = (100, 180, 120)  # Green
    text_color = (255, 255, 255)  # White
    divider_color = (80, 90, 110)  # Light gray
    
    # Create image
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw divider line
    draw.line([(left_width, 0), (left_width, height)], fill=divider_color, width=3)
    
    # Try to use a system font, fallback to default
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        score_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except:
        title_font = ImageFont.load_default()
        score_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    
    # Draw title (centered across both sides)
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (width - title_width) // 2
    draw.text((title_x, 30), title, fill=text_color, font=title_font)
    
    # Left side - Baseline
    baseline_label = "Baseline (CLIP-only)"
    baseline_score_text = f"Score: {baseline_score:.2f}"
    
    label_bbox = draw.textbbox((0, 0), baseline_label, font=label_font)
    label_width = label_bbox[2] - label_bbox[0]
    label_x = (left_width - label_width) // 2
    draw.text((label_x, 100), baseline_label, fill=text_color, font=label_font)
    
    # Baseline score box
    score_box_width = 200
    score_box_height = 80
    score_box_x = (left_width - score_box_width) // 2
    score_box_y = 200
    
    draw.rectangle([
        (score_box_x, score_box_y),
        (score_box_x + score_box_width, score_box_y + score_box_height)
    ], fill=baseline_color, outline=divider_color, width=2)
    
    score_bbox = draw.textbbox((0, 0), baseline_score_text, font=score_font)
    score_width = score_bbox[2] - score_bbox[0]
    score_height = score_bbox[3] - score_bbox[1]
    score_x = score_box_x + (score_box_width - score_width) // 2
    score_y = score_box_y + (score_box_height - score_height) // 2
    draw.text((score_x, score_y), baseline_score_text, fill=text_color, font=score_font)
    
    # Baseline issues
    issues = ["❌ No glass validation", "❌ Basic garnish detection", "❌ Color conflicts possible"]
    for i, issue in enumerate(issues):
        draw.text((20, 320 + i * 30), issue, fill=baseline_color, font=label_font)
    
    # Right side - Enhanced
    enhanced_label = "Enhanced (+Region Control)"
    enhanced_score_text = f"Score: {enhanced_score:.2f}"
    improvement = enhanced_score - baseline_score
    improvement_text = f"(+{improvement:.2f})"
    
    label_bbox = draw.textbbox((0, 0), enhanced_label, font=label_font)
    label_width = label_bbox[2] - label_bbox[0]
    label_x = left_width + (left_width - label_width) // 2
    draw.text((label_x, 100), enhanced_label, fill=text_color, font=label_font)
    
    # Enhanced score box
    score_box_x = left_width + (left_width - score_box_width) // 2
    
    draw.rectangle([
        (score_box_x, score_box_y),
        (score_box_x + score_box_width, score_box_y + score_box_height)
    ], fill=enhanced_color, outline=divider_color, width=2)
    
    # Enhanced score and improvement
    combined_text = f"{enhanced_score:.2f} {improvement_text}"
    score_bbox = draw.textbbox((0, 0), combined_text, font=score_font)
    score_width = score_bbox[2] - score_bbox[0]
    score_height = score_bbox[3] - score_bbox[1]
    score_x = score_box_x + (score_box_width - score_width) // 2
    score_y = score_box_y + (score_box_height - score_height) // 2
    draw.text((score_x, score_y), combined_text, fill=text_color, font=score_font)
    
    # Enhanced improvements
    improvements = ["✅ Glass validation", "✅ Advanced garnish detection", "✅ Conflict prevention"]
    for i, improvement in enumerate(improvements):
        draw.text((left_width + 20, 320 + i * 30), improvement, fill=enhanced_color, font=label_font)
    
    # Save image
    img.save(filename)
    print(f"Created {filename}")

def main():
    """Create demo grid images."""
    
    # Create output directory
    os.makedirs('runs/report', exist_ok=True)
    
    # Demo cases from our samples.json
    demo_cases = [
        ("Pink Floral Cocktail", 0.73, 0.92, "runs/report/grid_001_pink_floral.png"),
        ("Golden Whiskey Old Fashioned", 0.81, 0.95, "runs/report/grid_002_whiskey_classic.png"),
        ("Blue Tropical Cocktail", 0.71, 0.88, "runs/report/grid_003_tropical_blue.png"),
        ("Clear Martini with Olive", 0.82, 0.94, "runs/report/grid_004_martini_classic.png"),
        ("Rainbow Layered Cocktail", 0.74, 0.91, "runs/report/grid_005_rainbow_layers.png")
    ]
    
    for title, baseline, enhanced, filename in demo_cases:
        create_demo_grid(title, baseline, enhanced, filename)
    
    print(f"\nCreated {len(demo_cases)} demo grid images in runs/report/")

if __name__ == "__main__":
    main()