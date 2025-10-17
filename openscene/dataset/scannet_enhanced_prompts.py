"""
Enhanced prompt templates for ScanNet-20 categories.
Each category has multiple short descriptions (≤4 words) to improve discrimination.

Design principles:
1. Include category name itself
2. Multi-dimensional features (function, material, shape, location)
3. Avoid mentioning other categories (prevent semantic pollution)
4. Keep descriptions short and distinctive
"""

SCANNET_20_ENHANCED_PROMPTS = {
    'wall': [
        'wall',                      # base
        'vertical flat surface',     # shape + orientation
        'room boundary wall',        # function
        'painted wall structure',    # material + property
    ],
    
    'floor': [
        'floor',                     # base
        'horizontal walking surface', # function + orientation
        'ground level floor',        # position
        'flat floor area',          # shape
    ],
    
    'cabinet': [
        'cabinet',                   # base
        'storage cabinet furniture', # function + type
        'enclosed cabinet space',    # property
        'wooden cabinet structure',  # material
    ],
    
    'bed': [
        'bed',                       # base
        'sleeping bed furniture',    # function
        'mattress bed frame',        # component
        'bedroom bed area',          # location + context
    ],
    
    'chair': [
        'chair',                     # base
        'seating chair furniture',   # function
        'four-legged chair',         # structure (注意：避免提及其他家具)
        'backrest chair seat',       # component (区别凳子)
    ],
    
    'sofa': [
        'sofa',                      # base
        'cushioned sofa furniture',  # property
        'multi-seat sofa',           # capacity (区别单人椅)
        'living room sofa',          # location
    ],
    
    'table': [
        'table',                     # base
        'flat table surface',        # shape + property
        'dining table furniture',    # function
        'horizontal table top',      # orientation + component
    ],
    
    'door': [
        'door',                      # base
        'hinged door panel',         # mechanism + type
        'entrance door opening',     # function
        'wooden door frame',         # material
    ],
    
    'window': [
        'window',                    # base
        'glass window pane',         # material + component
        'transparent window opening', # property + function
        'outdoor view window',       # function
    ],
    
    'bookshelf': [
        'bookshelf',                 # base
        'book storage shelves',      # function
        'vertical bookshelf unit',   # orientation + structure
        'multi-tier bookshelf',      # structure
    ],
    
    'picture': [
        'picture',                   # base
        'wall-mounted picture frame', # location + type
        'decorative picture art',    # function + category
        'framed picture hanging',    # component + state
    ],
    
    'counter': [
        'counter',                   # base
        'kitchen counter surface',   # location + type
        'workspace counter top',     # function + component
        'long counter area',         # shape + property
    ],
    
    'desk': [
        'desk',                      # base
        'office desk furniture',     # context + type (区别 table)
        'workspace desk surface',    # function (强调工作)
        'writing desk area',         # function (区别餐桌)
    ],
    
    'curtain': [
        'curtain',                   # base
        'hanging curtain fabric',    # state + material
        'window curtain drape',      # location + type
        'textile curtain panel',     # material + component
    ],
    
    'refrigerator': [
        'refrigerator',              # base
        'cooling refrigerator appliance', # function + type
        'tall refrigerator unit',    # shape + structure
        'kitchen refrigerator door', # location + component
    ],
    
    'shower curtain': [
        'shower curtain',            # base
        'waterproof shower curtain', # property + type
        'bathroom shower curtain',   # location
        'hanging shower barrier',    # state + function
    ],
    
    'toilet': [
        'toilet',                    # base
        'ceramic toilet fixture',    # material + type
        'bathroom toilet bowl',      # location + component
        'sanitary toilet unit',      # property + structure
    ],
    
    'sink': [
        'sink',                      # base
        'washing sink basin',        # function + component
        'water sink fixture',        # function + type
        'porcelain sink bowl',       # material + component
    ],
    
    'bathtub': [
        'bathtub',                   # base
        'bathing tub fixture',       # function + type
        'large bathtub basin',       # size + component
        'bathroom bathtub area',     # location
    ],
    
    'otherfurniture': [
        'otherfurniture',            # base (保持原样)
        'other furniture item',      # category
        'miscellaneous furniture',   # property
        'unspecified furniture piece', # property
    ],
}


def get_enhanced_prompts_for_training(categories, alpha=0.5):
    """
    Get enhanced prompts for training with anchored ensemble strategy.
    
    Args:
        categories: list of category names
        alpha: weight for baseline prompt (0.5 = equal weight)
               baseline_weight = alpha, enhanced_weight = (1-alpha)
    
    Returns:
        prompts_dict: {category: [baseline_prompt, enhanced_prompts]}
    """
    prompts_dict = {}
    
    for cat in categories:
        baseline = f"{cat}"
        enhanced = SCANNET_20_ENHANCED_PROMPTS.get(cat, [cat])
        
        prompts_dict[cat] = {
            'baseline': baseline,
            'enhanced': enhanced,
            'alpha': alpha  # for weighted averaging
        }
    
    return prompts_dict


def get_baseline_prompts_for_testing(categories):
    """
    Get baseline prompts for testing (same as original).
    
    Args:
        categories: list of category names
    
    Returns:
        prompts: list of baseline prompts
    """
    # Special handling for scannet
    prompts = []
    for cat in categories:
        if cat == 'otherfurniture':
            prompts.append('other')  # keep original behavior
        else:
            prompts.append(f"a {cat} in a scene")
    
    return prompts


# Example usage and statistics
if __name__ == "__main__":
    from label_constants import SCANNET_LABELS_20
    
    print("=" * 80)
    print("ScanNet-20 Enhanced Prompts Summary")
    print("=" * 80)
    
    for i, cat in enumerate(SCANNET_LABELS_20, 1):
        prompts = SCANNET_20_ENHANCED_PROMPTS[cat]
        print(f"\n{i:2d}. {cat:20s} ({len(prompts)} prompts)")
        print(f"    Baseline:  a {cat} in a scene")
        print(f"    Enhanced:")
        for j, p in enumerate(prompts, 1):
            word_count = len(p.split())
            print(f"      {j}. {p:30s} ({word_count} words)")
    
    print("\n" + "=" * 80)
    print("Strategy: Anchored Ensemble")
    print("  - Training:   weighted_avg(baseline, enhanced_multi_desc)")
    print("  - Testing:    baseline only (fair comparison)")
    print("  - Alpha:      0.5 (equal weight, adjustable)")
    print("=" * 80)

