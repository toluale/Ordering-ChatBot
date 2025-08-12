import json
from pathlib import Path

# Load the test results
results_file = 'evaluation_results/order_evaluation/order_flow_test_results_20250805_153101.json'
with open(results_file, 'r') as f:
    data = json.load(f)

results = data['results']
total_tests = len(results)

# Categorize results
passed_items = []
partially_passed_items = []
failed_items = []

for result in results:
    expected = set(result['expected_items'])
    extracted = set(result['extracted_items'])
    
    if expected == extracted:
        # Perfect match
        passed_items.append(result)
    elif len(extracted) > 0 and expected.intersection(extracted):
        # Some items extracted correctly
        partially_passed_items.append(result)
    else:
        # No items extracted or completely wrong
        failed_items.append(result)

# Calculate percentages
passed_count = len(passed_items)
partially_passed_count = len(partially_passed_items)
failed_count = len(failed_items)

passed_percentage = (passed_count / total_tests) * 100
partially_passed_percentage = (partially_passed_count / total_tests) * 100
failed_percentage = (failed_count / total_tests) * 100

print(f'EVALUATION METRICS ANALYSIS')
print(f'==========================')
print(f'Total Test Cases: {total_tests}')
print(f'')
print(f'PASSED (Perfect Match): {passed_count} tests ({passed_percentage:.1f}%)')
print(f'PARTIALLY PASSED (Some Items Correct): {partially_passed_count} tests ({partially_passed_percentage:.1f}%)')
print(f'FAILED (No Correct Items): {failed_count} tests ({failed_percentage:.1f}%)')
print(f'')
print(f'DETAILED BREAKDOWN:')
print(f'==================')

print(f'')
print(f'PASSED TESTS ({passed_count} cases):')
for i, result in enumerate(passed_items[:5], 1):
    print(f'{i}. "{result["input_message"]}"')
    print(f'   Expected: {result["expected_items"]}')
    print(f'   Extracted: {result["extracted_items"]}')

if len(passed_items) > 5:
    print(f'   ... and {len(passed_items) - 5} more')

print(f'')
print(f'PARTIALLY PASSED TESTS ({partially_passed_count} cases):')
for i, result in enumerate(partially_passed_items, 1):
    expected = set(result['expected_items'])
    extracted = set(result['extracted_items'])
    missing = expected - extracted
    extra = extracted - expected
    print(f'{i}. "{result["input_message"]}"')
    print(f'   Expected: {result["expected_items"]}')
    print(f'   Extracted: {result["extracted_items"]}')
    print(f'   Missing: {list(missing)}')
    if extra:
        print(f'   Extra: {list(extra)}')

print(f'')
print(f'FAILED TESTS ({failed_count} cases):')
for i, result in enumerate(failed_items, 1):
    print(f'{i}. "{result["input_message"]}"')
    print(f'   Expected: {result["expected_items"]}')
    print(f'   Extracted: {result["extracted_items"]}')

print(f'')
print(f'COMMON FAILURE PATTERNS:')
print(f'========================')

# Analyze failure patterns
black_bean_failures = [r for r in failed_items if 'black bean' in r['input_message'].lower()]
complex_order_failures = [r for r in failed_items if len(r['expected_items']) > 2]
customization_failures = [r for r in failed_items if any(word in r['input_message'].lower() for word in ['with', 'no', 'extra', 'double'])]

print(f'Black Bean Burger Recognition Failures: {len(black_bean_failures)} cases')
print(f'Complex Multi-Item Order Failures: {len(complex_order_failures)} cases')
print(f'Customization Handling Failures: {len(customization_failures)} cases')
