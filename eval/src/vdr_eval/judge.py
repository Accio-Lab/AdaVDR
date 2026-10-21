"""Judge sharded VideoDR multi-turn predictions with the official VideoDR prompt."""
from __future__ import annotations
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from openai import OpenAI
from .services_config import load_services
load_services()
SYSTEM_MESSAGE = 'You are an expert answer evaluator with strong analytical capabilities. \nYour role is to carefully analyze and compare answers using a systematic evaluation framework. \nYou should apply critical thinking to determine correctness based on semantic equivalence and factual accuracy, \nrather than relying solely on exact textual matching. Always provide clear, detailed reasoning for your judgments.'

def get_judge_prompt(question: str, standard_answer: str, model_answer: str) -> str:
    """
    Generate the prompt for judging if a model answer is correct.

    Args:
        question: The question being asked
        standard_answer: The correct/standard answer
        model_answer: The answer from the model to be evaluated

    Returns:
        The formatted prompt string
    """
    prompt = f"""You are an expert answer evaluator. Your task is to determine whether a model's answer is correct by comparing it with a standard answer.\n\nQuestion: {question}\n\nStandard Answer: {standard_answer}\n\nModel Answer: {model_answer}\n\n**Evaluation Framework:**\n\n1. **Identify the answer type**: Determine whether the answer is numerical, categorical, descriptive, or a combination. Consider what type of information the question is asking for.\n\n2. **Extract core information**: Identify the essential information in both answers, ignoring formatting, stylistic differences, and non-essential details.\n\n3. **Semantic equivalence assessment**: \n   - For numerical answers: Check if the values are equivalent considering units, precision, and representation formats\n   - For categorical answers: Check if they refer to the same entity or concept, accounting for synonyms and alternative names\n   - For descriptive answers: Check if they convey the same meaning and key information\n\n4. **Completeness check**: Determine if the model answer contains all critical information required by the question, or if missing information would make the answer incorrect.\n\n5. **Contextual relevance**: Consider the question's intent and whether the model answer adequately addresses what was asked.\n\n**Decision Principles:**\n- Prioritize semantic correctness over exact textual matching\n- Allow reasonable variations in format, units, and expression\n- Be strict about factual accuracy and completeness of essential information\n- Distinguish between related but distinct entities or concepts\n- Consider the question's context and requirements when making judgments\n\n**Output Format:**\nProvide your judgment in JSON format:\n{{\n    "is_correct": true/false,\n    "reasoning": "Your detailed reasoning explaining: (1) the answer type identified, (2) how you compared the answers, (3) what differences you found (if any), and (4) why you concluded the answer is correct or incorrect (in English)"\n}}\n"""
    return prompt

def parse_judgment(text: str) -> tuple[bool, str]:
    text = text.strip()
    if '```json' in text:
        text = text.split('```json', 1)[1].split('```', 1)[0].strip()
    elif '```' in text:
        text = text.split('```', 1)[1].split('```', 1)[0].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        (start, end) = (text.find('{'), text.rfind('}'))
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start:end + 1])
    value = payload.get('is_correct')
    if not isinstance(value, bool):
        raise ValueError(f'is_correct is not boolean: {value!r}')
    return (value, str(payload.get('reasoning', '')))

def judge_prediction(row: dict[str, Any]) -> tuple[str, str]:
    """Use the raw final assistant turn when no answer was parsed or turns were exhausted."""
    prediction = str(row.get('prediction') or '')
    turn_count = int(row.get('turn_count') or 0)
    if prediction.strip() and turn_count < 21:
        return (prediction, 'parsed_answer')
    for message in reversed(row.get('messages') or row.get('trajectory') or []):
        if message.get('role') == 'assistant':
            content = message.get('content', '')
            if isinstance(content, str) and content.strip():
                return (content, 'last_assistant_output')
    for item in reversed(row.get('trace') or []):
        response = str(item.get('response') or '')
        if response.strip():
            return (response, 'last_assistant_output')
    return (prediction, 'empty')

def judge_one(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    sample = row.get('sample', {})
    question = str(sample.get('question', ''))
    gold = str(row.get('gold', sample.get('answer', '')))
    (prediction, prediction_source) = judge_prediction(row)
    if not prediction.strip():
        return {'id': str(row.get('id', '')), 'question': question, 'gold': gold, 'prediction': prediction, 'is_correct': False, 'reasoning': 'Answer is empty', 'prediction_source': prediction_source, 'attempt': 0}
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    prompt = get_judge_prompt(question, gold, prediction)
    last_error = ''
    raw_attempts = []
    for attempt in range(1, args.retries + 1):
        try:
            response = client.chat.completions.create(model=args.model, messages=[{'role': 'system', 'content': SYSTEM_MESSAGE}, {'role': 'user', 'content': prompt}], temperature=0.1, max_tokens=args.max_tokens)
            raw = response.choices[0].message.content or ''
            raw_attempts.append({'attempt': attempt, 'raw_judgment': raw})
            (correct, reasoning) = parse_judgment(raw)
            return {'id': str(row.get('id', '')), 'question': question, 'gold': gold, 'prediction': prediction, 'is_correct': correct, 'reasoning': reasoning, 'prediction_source': prediction_source, 'attempt': attempt, 'max_tokens': args.max_tokens, 'raw_judgment': raw, 'raw_attempts': raw_attempts}
        except Exception as error:
            last_error = f'{type(error).__name__}: {error}'
            raw_attempts.append({'attempt': attempt, 'error': last_error})
            time.sleep(min(2 ** attempt, 10))
    return {'id': str(row.get('id', '')), 'question': question, 'gold': gold, 'prediction': prediction, 'is_correct': None, 'reasoning': '', 'prediction_source': prediction_source, 'attempt': args.retries, 'error': last_error, 'raw_attempts': raw_attempts}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--services-config', type=Path, help='Private service YAML; environment values take priority.')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--base-url', default=os.environ.get('JUDGE_BASE_URL', ''))
    parser.add_argument('--api-key', default=os.environ.get('JUDGE_API_KEY', 'EMPTY'))
    parser.add_argument('--model', default=os.environ.get('JUDGE_MODEL', ''))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--max-tokens', type=int, default=500)
    parser.add_argument('--expected-samples', type=int, default=100)
    args = parser.parse_args()
    if not args.base_url or not args.model:
        parser.error('Configure judge.base_url and judge.model in the service YAML, or pass --base-url and --model')
    (input_dir, output_dir) = (Path(args.input_dir), Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(input_dir.glob('predictions*.jsonl')):
        for line in path.read_text().split('\n'):
            if line.strip():
                row = json.loads(line)
                rows[str(row['id'])] = row
    if len(rows) != args.expected_samples:
        raise RuntimeError(f'expected {args.expected_samples} unique predictions, found {len(rows)}')
    results_path = output_dir / 'results.jsonl'
    completed: dict[str, dict[str, Any]] = {}
    if results_path.exists():
        for line in results_path.read_text().split('\n'):
            if line.strip():
                result = json.loads(line)
                if result.get('is_correct') is not None:
                    completed[str(result['id'])] = result
    pending = [row for (key, row) in rows.items() if key not in completed]
    lock = threading.Lock()
    with results_path.open('a') as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(judge_one, row, args) for row in pending]
        for (index, future) in enumerate(as_completed(futures), 1):
            result = future.result()
            with lock:
                handle.write(json.dumps(result, ensure_ascii=False) + '\n')
                handle.flush()
            completed[str(result['id'])] = result
            print(f"[{index}/{len(pending)}] id={result['id']} correct={result['is_correct']}", flush=True)
    final = list(completed.values())
    judged = [row for row in final if row.get('is_correct') is not None]
    correct = [row for row in judged if row.get('is_correct') is True]
    summary = {'samples': len(rows), 'judged': len(judged), 'correct': len(correct), 'accuracy': len(correct) / len(rows), 'judge_coverage': len(judged) / len(rows), 'base_url': args.base_url, 'model': args.model, 'official_prompt': str(Path(__file__).resolve())}
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
