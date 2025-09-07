import json
import re
from typing import Union, Dict, List, Any

import re
import json
from typing import Any, Dict, List, Union

def parse_json_response(response: Any) -> Union[Dict, List, None]:
    """
    Parse a JSON response from various formats, including JSON embedded in text.

    Args:
    response (Any): The input to parse, can be a string or any other type.

    Returns:
    Union[Dict, List, None]: Parsed JSON object or None if parsing fails.
    """
    if not isinstance(response, str):
        return response if isinstance(response, (dict, list)) else None
    
    if not response.strip():
        return None
    
    def extract_content_between_markers(text: str, start_marker: str, end_marker: str) -> List[str]:
        pattern = f"{re.escape(start_marker)}(.*?){re.escape(end_marker)}"
        return re.findall(pattern, text, re.DOTALL)
    
    def safe_json_parse(json_string: str) -> Union[Dict, List, None]:
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            return None
    
    # Remove common prefixes
    response = re.sub(r'^(json|JSON|Json):\s*', '', response.strip())
    
    # Try to extract JSON from various markers
    for start, end in [('```json', '```'), ('```', '```'), ('{', '}'), ('[', ']')]:
        extracted = extract_content_between_markers(response, start, end)
        if extracted:
            parsed = safe_json_parse(extracted[0])
            if parsed is not None:
                return parsed
    
    # If no markers found, try to extract JSON content embedded in text
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    json_matches = re.findall(json_pattern, response)
    for json_match in json_matches:
        parsed = safe_json_parse(json_match)
        if parsed is not None:
            return parsed
    
    # If no embedded JSON found, try to parse the entire response
    return safe_json_parse(response)

# Example usage and testing
def test_parse_json_response():
    test_cases = [
        '```json\n{"name": "John", "age": 30}\n```',
        '```\n["apple", "banana", "cherry"]\n```',
        'json: {"x": 1, "y": 2}',
        '{"invalid": "json",}',
        '[1, 2, 3]',
        'Plain text',
        '',
        None,
        {'already': 'a dict'},
        ['already', 'a', 'list'],
        '   {"spaces": "around"}   ',
        'JSON: {"uppercase": "prefix"}',
        '{"a_valid": "json"}',
        "[{\"Newsid\":\"5301f0c6-00ed-48c4-bc18-bd4dc492860c\",\"NewsSubj\":\"Intimation Under Regulation 30\",\"Newsdt\":\"12/10/2024\"},{\"Newsid\":\"0e106038-3bb7-4997-aa5e-b45ee43537be\",\"NewsSubj\":\"Compliances-Certificate under Reg. 74 (5) of SEBI (DP) Regulations, 2018\",\"Newsdt\":\"07/10/2024\"},{\"Newsid\":\"11a0305a-18de-41a5-b974-7bc45157bf95\",\"NewsSubj\":\"Intimation Under Regulation 30\",\"Newsdt\":\"07/10/2024\"},{\"Newsid\":\"48a8babf-9313-47b5-ad4a-dd497ff4948b\",\"NewsSubj\":\"Update\",\"Newsdt\":\"07/10/2024\"},{\"Newsid\":\"c9d56fd6-dbc2-48d7-8caa-3d1851c48981\",\"NewsSubj\":\"Announcement under Regulation 30 (LODR)-Dividend Updates\",\"Newsdt\":\"04/10/2024\"},{\"Newsid\":\"67b780df-2f7a-418f-aead-5d419a9b8102\",\"NewsSubj\":\"Closure of Trading Window\",\"Newsdt\":\"30/09/2024\"}]",
        'some text.. {    \"event_name\": \"None of the given list related event is availabe in text.\",    \"event_date\": \"2024-09-30\",    \"short_desc\": \"Certificate under Regulation 74(5) of the SEBI (Depositories and Participants) Regulations, 2018 for the quarter ended September 30, 2024.\"} this is the required json'
    ]

    for i, case in enumerate(test_cases, 1):
        result = parse_json_response(case)
        print(f"Test Case {i}:")
        print(f"Input: {case}")
        print(f"Output: {result}")
        print(f"Type: {type(result)}")
        print()

# if __name__ == "__main__":
#     test_parse_json_response()