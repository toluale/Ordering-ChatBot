from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from dataclasses import dataclass
import hashlib

@dataclass
class ScenarioConfig:
    """Configuration for a test conversation scenario."""
    text: str
    type: str
    id: Optional[str] = None
    expected_outcomes: Optional[List[str]] = None
    
    def __post_init__(self):
        if not self.id:
            # Generate a stable ID from the text if none provided
            self.id = hashlib.md5(self.text.encode()).hexdigest()[:8]


class ScenarioLoader:
    """Loads test scenarios from configuration files."""
    
    @staticmethod
    def load_scenarios(file_path: Path) -> List[ScenarioConfig]:
        """Load scenarios from a JSON file.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {file_path}")
            
        with open(file_path) as f:
            data = json.load(f)
            
        scenarios_data = data.get("scenarios", data) if isinstance(data, dict) else data
        
        scenarios = []
        for scenario in scenarios_data:
            if isinstance(scenario, str):
                scenarios.append(ScenarioConfig(
                    text=scenario,
                    type="general"
                ))
            else:
                scenarios.append(ScenarioConfig(
                    text=scenario.get("text"),
                    type=scenario.get("type", "general"),
                    id=scenario.get("id"),
                    expected_outcomes=scenario.get("expected_outcomes")
                ))
                
        return scenarios
