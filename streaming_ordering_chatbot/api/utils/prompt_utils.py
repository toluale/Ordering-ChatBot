from typing import Optional, Tuple


def build_overlay_cache_key(brand: Optional[str], style: Optional[str], template_path: Optional[str]) -> tuple[str, str, str]:
    """Create a stable cache key for brand/style/template overlays."""
    return (brand or "", (style or "default").lower(), template_path or "")


def make_overlay_parts(
    brand_instructions: Optional[str],
    style_instructions: Optional[str],
) -> Tuple[str, str]:
    """Build brand prefix and style suffix strings for overlaying a system prompt."""
    brand_prefix = f"{brand_instructions}\n\nBASE INSTRUCTIONS:\n" if brand_instructions else ""
    style_suffix = f"\n\nCONVERSATION STYLE:\n{style_instructions}" if style_instructions else ""
    return brand_prefix, style_suffix


def enhance_prompt_with_parts(system_prompt: str, brand_prefix: str, style_suffix: str) -> str:
    """Apply overlay parts to a system prompt."""
    return f"{brand_prefix}{system_prompt}{style_suffix}"
