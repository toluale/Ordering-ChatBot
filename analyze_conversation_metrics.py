import json
from pathlib import Path

# Load the end-to-end evaluation results
results_file = 'evaluation_results/overall_evaluation_results/gpt4o/20250729_151143_end-to-end_evaluation_results.json'
with open(results_file, 'r') as f:
    data = json.load(f)

conversations = data['conversations']
total_conversations = len(conversations)

# Initialize metric accumulators
metrics_totals = {
    'groundedness': 0,
    'relevance': 0,
    'coherence': 0,
    'fluency': 0
}

# Collect all scores for detailed analysis
all_scores = {
    'groundedness': [],
    'relevance': [],
    'coherence': [],
    'fluency': []
}

print("END-TO-END EVALUATION METRICS ANALYSIS")
print("=====================================")
print(f"Total Conversations Analyzed: {total_conversations}")
print()

# Process each conversation
for i, conversation in enumerate(conversations, 1):
    query = conversation['query']
    metrics = conversation['metrics']
    
    print(f"Conversation {i}: \"{query[:50]}{'...' if len(query) > 50 else ''}\"")
    
    # Extract scores for each metric
    for metric_name in metrics_totals.keys():
        score = metrics[metric_name]['score']
        reason = metrics[metric_name]['reason']
        
        metrics_totals[metric_name] += score
        all_scores[metric_name].append(score)
        
        print(f"  {metric_name.capitalize()}: {score}/5 - {reason[:80]}{'...' if len(reason) > 80 else ''}")
    
    print()

# Calculate averages
averages = {}
for metric_name, total in metrics_totals.items():
    averages[metric_name] = total / total_conversations

print("AVERAGE SCORES SUMMARY")
print("=====================")
for metric_name, avg_score in averages.items():
    scores = all_scores[metric_name]
    min_score = min(scores)
    max_score = max(scores)
    
    print(f"{metric_name.upper()}: {avg_score:.2f}/5.00")
    print(f"  Range: {min_score} - {max_score}")
    print(f"  Distribution: {[scores.count(i) for i in range(1, 6)]}")
    print()

# Overall performance analysis
overall_average = sum(averages.values()) / len(averages)
print(f"OVERALL AVERAGE ACROSS ALL METRICS: {overall_average:.2f}/5.00")
print()

# Performance categories
print("PERFORMANCE BREAKDOWN BY SCORE:")
print("===============================")
for score in range(5, 0, -1):
    count_by_metric = {}
    for metric_name in metrics_totals.keys():
        count_by_metric[metric_name] = all_scores[metric_name].count(score)
    
    total_count = sum(count_by_metric.values())
    percentage = (total_count / (total_conversations * 4)) * 100  # 4 metrics per conversation
    
    print(f"Score {score}/5: {total_count} instances ({percentage:.1f}%)")
    for metric_name, count in count_by_metric.items():
        if count > 0:
            print(f"  {metric_name.capitalize()}: {count}")
    print()

# Identify strengths and weaknesses
print("STRENGTHS AND AREAS FOR IMPROVEMENT:")
print("===================================")

# Sort metrics by average score
sorted_metrics = sorted(averages.items(), key=lambda x: x[1], reverse=True)

print("Strongest Metrics:")
for metric_name, avg_score in sorted_metrics[:2]:
    print(f"  {metric_name.capitalize()}: {avg_score:.2f}/5.00")

print("\nAreas for Improvement:")
for metric_name, avg_score in sorted_metrics[-2:]:
    print(f"  {metric_name.capitalize()}: {avg_score:.2f}/5.00")

# Analyze conversations with lowest scores
print("\nCONVERSATIONS NEEDING ATTENTION:")
print("===============================")

conversation_scores = []
for i, conversation in enumerate(conversations):
    metrics = conversation['metrics']
    total_score = sum(metrics[metric]['score'] for metric in metrics_totals.keys())
    avg_score = total_score / len(metrics_totals)
    conversation_scores.append((i, conversation['query'], avg_score, total_score))

# Sort by average score
conversation_scores.sort(key=lambda x: x[2])

print("Lowest Performing Conversations:")
for i, (idx, query, avg_score, total_score) in enumerate(conversation_scores[:3]):
    print(f"{i+1}. \"{query[:60]}{'...' if len(query) > 60 else ''}\"")
    print(f"   Average Score: {avg_score:.2f}/5.00 (Total: {total_score}/20)")
    
    # Show specific weak areas
    conv_metrics = conversations[idx]['metrics']
    weak_metrics = [(metric, data['score']) for metric, data in conv_metrics.items() if data['score'] < 4]
    if weak_metrics:
        print(f"   Weak Areas: {', '.join([f'{metric}({score})' for metric, score in weak_metrics])}")
    print()
