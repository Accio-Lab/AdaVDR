"""Multi-turn evaluation of this video QA benchmark through an existing OpenAI-compatible model server."""
from __future__ import annotations
import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from openai import OpenAI
DEFAULT_PROTOCOL = Path(os.environ.get('PROTOCOL_FILE', str(Path(__file__).resolve().parents[2] / 'protocols' / 'protocol.jsonl')))
from . import tools
from collections import Counter
from urllib.parse import parse_qs, urlparse
VISIT_PAGE_SUMMARY_BASE_URL = tools.BASE_URL
VISIT_PAGE_SUMMARY_MODEL = tools.MODEL
summarize_visit_page = tools.summarize_visit_page
os.environ.setdefault('EVAL_MEDIA_FILE_URI', '1')

def load_samples(path: Path, video_dir: Path | None=None, *, shard_id: int=0, num_shards: int=1, offset: int=0, limit: int | None=None) -> list[dict[str, Any]]:
    """Load this benchmark's QA records and resolve downloaded local videos."""
    if num_shards < 1 or not 0 <= shard_id < num_shards:
        raise ValueError('Require num_shards > 0 and 0 <= shard_id < num_shards')
    if offset < 0 or (limit is not None and limit < 0):
        raise ValueError('offset and limit must be nonnegative')
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        records = pq.read_table(path).to_pylist()
    elif path.suffix == '.json':
        records = json.loads(path.read_text())
    elif path.suffix == '.jsonl':
        records = [json.loads(line) for line in path.read_text().split('\n') if line.strip()]
    else:
        raise ValueError('Input must be .parquet, .jsonl, or .json')
    if not isinstance(records, list):
        raise ValueError('Annotations must be a list of QA records')
    prepared = []
    seen = set()
    for (index, row) in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f'Input row {index} is not an object')
        identifier = row.get('uid') if row.get('uid') is not None else row.get('id')
        if identifier is None or not str(identifier).strip():
            raise ValueError(f'Input row {index} needs uid or id')
        identifier = str(identifier)
        if identifier in seen:
            raise ValueError(f'Duplicate question ID: {identifier}')
        seen.add(identifier)
        if not isinstance(row.get('question'), str) or not row['question'].strip():
            raise ValueError(f'Question is missing for {identifier}')
        answer = row.get('answer')
        if answer is None:
            answer = row.get('ground_truth')
        if answer is None or not str(answer).strip():
            raise ValueError(f'Reference answer is missing for {identifier}')
        if index % num_shards == shard_id:
            prepared.append((identifier, row, str(answer)))
    prepared = prepared[offset:]
    if limit is not None:
        prepared = prepared[:limit]
    samples = []
    for (identifier, row, answer) in prepared:
        explicit = row.get('video_path') or row.get('video')
        candidates = []
        if isinstance(explicit, str) and explicit:
            if '://' in explicit:
                raise ValueError(f'{identifier}: provide a local video file, not a webpage URL')
            local = Path(explicit)
            candidates.append(local if local.is_absolute() else path.parent / local)
        if video_dir is not None:
            ids = [str(row.get('video_id') or ''), identifier]
            youtube_id = parse_qs(urlparse(row.get('youtube_url') or '').query).get('v', [''])[0]
            ids.append(youtube_id)
            for video_id in dict.fromkeys(ids):
                if video_id and Path(video_id).name == video_id:
                    candidates.extend((video_dir / f'{video_id}{suffix}' for suffix in ('.mp4', '.webm', '.mkv', '.mov')))
        video = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
        if video is None:
            raise FileNotFoundError(f'{identifier}: no local video found; set video_path or --video-dir')
        samples.append({'id': identifier, 'video': video, 'raw': {**row, 'answer': answer}})
    return samples

def write_summary(output: Path, summary_path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in output.read_text().split('\n') if line.strip()]
    tool_counts = Counter((tool for row in rows for tool in row.get('tools', [])))
    valid = [row for row in rows if row.get('prediction')]
    correct = sum((row.get('correct', False) for row in rows))

    def tool_payload_success(payload: dict[str, Any]) -> bool:
        if payload.get('ok') is False or payload.get('error'):
            return False
        if payload.get('image_search_result', {}).get('error', '').startswith(('visual_grounding_required', 'visual_grounding_image_missing', 'spatio_grounding_required', 'spatio_grounding_crop_missing', 'external_image_search_keys_missing')):
            return False
        return any((key in payload for key in ('video_segment', 'frame', 'bbox', 'image_search_result', 'search_results', 'webpage'))) or payload.get('ok') is True
    summary = {'samples': len(rows), 'valid_answers': len(valid), 'correct': correct, 'accuracy': correct / len(rows) if rows else 0.0, 'valid_answer_rate': len(valid) / len(rows) if rows else 0.0, 'average_turns': sum((row.get('turn_count', 0) for row in rows)) / len(rows) if rows else 0.0, 'tool_calls': dict(tool_counts), 'tool_success_rate': sum((tool_payload_success(x) for row in rows for x in row.get('tool_results', []))) / max(1, sum((len(row.get('tool_results', [])) for row in rows)))}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary

def load_protocol(path: Path) -> tuple[str, list[dict[str, Any]]]:
    with path.open(encoding='utf-8') as handle:
        record = json.loads(handle.readline())
    system = next((message['content'] for message in record['messages'] if message.get('role') == 'system'))
    tools = record['tools']
    if isinstance(tools, str):
        tools = json.loads(tools)
    return (system, tools)

def media_url(path: str) -> str:
    resolved = Path(path).resolve()
    if os.environ.get('EVAL_MEDIA_FILE_URI') == '1':
        return resolved.as_uri()
    return str(resolved)

def openai_content(content: Any) -> Any:
    """Convert ms-swift media items to OpenAI multimodal content items."""
    if isinstance(content, str):
        return content
    converted = []
    for item in content or []:
        if 'text' in item:
            converted.append({'type': 'text', 'text': str(item['text'])})
        elif 'video' in item:
            converted.append({'type': 'video_url', 'video_url': {'url': media_url(str(item['video']))}, **{key: item[key] for key in ('fps', 'min_frames', 'max_frames', 'min_pixels', 'max_pixels', 'total_pixels') if key in item}})
        elif 'image' in item:
            converted.append({'type': 'image_url', 'image_url': {'url': media_url(str(item['image']))}, **{key: item[key] for key in ('min_pixels', 'max_pixels') if key in item}})
        else:
            raise ValueError(f'unsupported multimodal content item: {item}')
    return converted

def initial_messages(sample: dict[str, Any], args: argparse.Namespace, system_prompt: str) -> list[dict[str, Any]]:
    """Use the same open-ended video prompt and media budgets as the source run."""
    content = [{'video': str(sample['video']), 'fps': args.fps, 'min_pixels': args.min_pixels, 'max_pixels': args.max_pixels, 'min_frames': args.min_frames, 'max_frames': args.max_frames, 'total_pixels': args.total_pixels}, {'text': f"Question: {sample['raw']['question']}\n\nUse the video tools if useful, then return a concise answer."}]
    return [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': openai_content(content)}]

def parsed_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get('function') or {}
    raw_name = str(function.get('name', ''))
    arguments = function.get('arguments', {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    name_match = re.match('\\s*([A-Za-z_][A-Za-z0-9_.-]*)', raw_name)
    name = name_match.group(1) if name_match else raw_name.strip()
    for (key, value) in re.findall('<parameter=([^>\\s]+)>\\s*(.*?)\\s*</parameter>', raw_name, re.DOTALL):
        if key not in arguments or arguments.get(key) in (None, ''):
            arguments[key] = value.strip()
    if isinstance(arguments, dict):
        cleaned_arguments: dict[str, Any] = {}
        for (key, value) in arguments.items():
            key_match = re.match('\\s*([A-Za-z_][A-Za-z0-9_.-]*)', str(key))
            cleaned_key = key_match.group(1) if key_match else str(key).strip()
            cleaned_arguments[cleaned_key] = value
        arguments = cleaned_arguments
        for key in ('temporal_span', 'bbox'):
            value = arguments.get(key)
            if not isinstance(value, str):
                continue
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                normalized = value.strip().strip('[]')
                parts = [part.strip().strip('"\'') for part in normalized.split(',', 1)]
                decoded = parts if len(parts) == 2 else None
            if isinstance(decoded, list):
                arguments[key] = decoded
    return {'name': name, 'arguments': arguments}

def assistant_message(response_message: Any) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    dumped = response_message.model_dump(exclude_none=True)
    message: dict[str, Any] = {'role': 'assistant', 'content': dumped.get('content') or ''}
    reasoning_value = dumped.get('reasoning_content') or dumped.get('reasoning')
    if reasoning_value:
        message['reasoning_content'] = reasoning_value
    calls = dumped.get('tool_calls') or []
    if calls:
        message['tool_calls'] = calls
    reasoning = str(reasoning_value or '').strip()
    content = str(dumped.get('content') or '').strip()
    raw = (f'<think>\n{reasoning}\n</think>\n\n' if reasoning else '') + content
    return (message, raw, calls)

def tool_response(call: dict[str, Any], tool_call_id: str, video: Path, work_dir: Path, sample_id: str, turn: int, history: list[dict[str, Any]], question: str, step: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if call.get('name') == 'web_search':
        arguments = call.get('arguments') if isinstance(call.get('arguments'), dict) else {}
        if not str(arguments.get('query', '')).strip():
            call = {**call, 'arguments': {**arguments, 'query': question}}
    (response, payload) = tools.tool_result(call, video, work_dir, sample_id, turn, history, question=question)
    if call.get('name') == 'visit_page':
        webpage = payload.get('webpage', {})
        if webpage.get('title') != 'tool_error':
            original = dict(webpage)
            webpage['raw_content'] = webpage.get('content', '')
            try:
                summary = summarize_visit_page(question, step, original)
                webpage['raw_content'] = webpage.get('content', '')
                webpage['content'] = summary
                webpage['summary_model'] = VISIT_PAGE_SUMMARY_MODEL
                webpage['summary_base_url'] = VISIT_PAGE_SUMMARY_BASE_URL
            except Exception as error:
                webpage['summary_error'] = type(error).__name__
                webpage['content'] = 'tool_error: full-page summary failed; no complete page summary is available.'
    response['tool_call_id'] = tool_call_id
    if call.get('name') == 'visit_page' and payload.get('webpage', {}).get('title') != 'tool_error':
        page = payload['webpage']
        response['content'] = [{'text': f"Page title: {page.get('title', '')}\n\n{page.get('content', '')}"}]
    response['content'] = openai_content(response.get('content'))
    return (response, payload)

def summarize(output: Path) -> dict[str, Any]:
    return write_summary(output, output.with_suffix('.summary.json'))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--services-config', type=Path, help='Private service YAML; environment values take priority.')
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--video-dir', type=Path)
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument('--model-name', default=os.environ.get('MODEL_NAME', ''))
    parser.add_argument('--api-base', default=os.environ.get('API_BASE', ''))
    parser.add_argument('--api-key', default=os.environ.get('EVAL_API_KEY', 'EMPTY'))
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--media-dir')
    parser.add_argument('--max-turns', type=int, default=20)
    parser.add_argument('--concurrency', type=int, default=8, help='Number of sample trajectories concurrently handled by one GPU server.')
    parser.add_argument('--request-timeout', type=float, default=600.0, help='Maximum seconds for one OpenAI/vLLM request; zero disables the timeout.')
    parser.add_argument('--request-retries', type=int, default=0, help='OpenAI client retries per request; keep at zero for bounded eval latency.')
    parser.add_argument('--require-web-search', action='store_true')
    parser.add_argument('--require-tool-use', action='store_true')
    parser.add_argument('--disable-forced-final', action='store_true')
    parser.add_argument('--sample-offset', type=int, default=0)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=-1)
    parser.add_argument('--presence-penalty', type=float, default=0.0)
    parser.add_argument('--repetition-penalty', type=float, default=1.0)
    parser.add_argument('--fps', type=float, default=2)
    parser.add_argument('--min-pixels', type=int, default=3584)
    parser.add_argument('--max-pixels', type=int, default=401408)
    parser.add_argument('--min-frames', type=int, default=4)
    parser.add_argument('--max-frames', type=int, default=256)
    parser.add_argument('--total-pixels', type=int, default=19267584)
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()
    if not args.api_base or not args.model_name:
        parser.error('Configure inference.base_url and inference.model in the service YAML, or pass --api-base and --model-name')
    (system_prompt, registered_tools) = load_protocol(args.protocol)
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    media_dir = Path(args.media_dir) if args.media_dir else output.parent / 'tool_media'
    samples = load_samples(args.input, args.video_dir, shard_id=args.shard_id, num_shards=args.num_shards, offset=args.sample_offset, limit=args.max_samples)
    if args.resume and output.exists():
        done = {str(json.loads(line)['id']) for line in output.read_text().split('\n') if line.strip()}
        samples = [sample for sample in samples if str(sample['id']) not in done]
    thread_local = threading.local()

    def get_client() -> OpenAI:
        client = getattr(thread_local, 'client', None)
        if client is None:
            client = OpenAI(base_url=args.api_base.rstrip('/') if args.api_base.rstrip('/').endswith('/v1') else args.api_base.rstrip('/') + '/v1', api_key=args.api_key, timeout=None if args.request_timeout <= 0 else args.request_timeout, max_retries=args.request_retries)
            thread_local.client = client
        return client

    def evaluate_sample(sample: dict[str, Any]) -> dict[str, Any]:
        client = get_client()
        messages = initial_messages(sample, args, system_prompt)
        trace: list[dict[str, Any]] = []
        tool_names: list[str] = []
        results: list[dict[str, Any]] = []
        prediction = None
        inference_error = ''
        remaining_generation_tokens = args.max_new_tokens
        for turn in range(1, args.max_turns + 1):
            if remaining_generation_tokens <= 0:
                break
            try:
                completion = client.chat.completions.create(model=args.model_name, messages=messages, tools=registered_tools, tool_choice='auto', parallel_tool_calls=False, max_tokens=remaining_generation_tokens, temperature=args.temperature, top_p=args.top_p, presence_penalty=args.presence_penalty, extra_body={'top_k': args.top_k, 'repetition_penalty': args.repetition_penalty, 'chat_template_kwargs': {'enable_thinking': False}}, seed=args.seed)
                used_tokens = getattr(completion.usage, 'completion_tokens', None)
                if used_tokens is not None:
                    remaining_generation_tokens -= int(used_tokens)
                (model_message, raw, calls) = assistant_message(completion.choices[0].message)
            except Exception as error:
                inference_error = f'{type(error).__name__}: {error}'
                trace.append({'turn': turn, 'error': inference_error})
                break
            trace.append({'turn': turn, 'response': raw, 'tool_calls': calls})
            prediction = tools.answer_value(raw)
            if prediction and args.require_tool_use and (not tool_names):
                messages.append(model_message)
                messages.append({'role': 'user', 'content': 'You must actually call at least one appropriate tool before answering. Choose the tool yourself and return a structured tool call now; do not merely describe the intended action.'})
                prediction = None
                continue
            if prediction and args.require_web_search and ('web_search' not in tool_names):
                messages.append(model_message)
                messages.append({'role': 'user', 'content': 'You must call web_search at least once before giving the final answer. Search now.'})
                prediction = None
                continue
            if prediction:
                messages.append(model_message)
                break
            if not calls:
                messages.append(model_message)
                break
            parsed_calls = [parsed_call(raw_call) for raw_call in calls]
            answer_calls = [call for call in parsed_calls if call.get('name') == 'answer']
            if answer_calls:
                arguments = answer_calls[0].get('arguments') or {}
                if isinstance(arguments, dict):
                    candidate = next((str(arguments[key]).strip() for key in ('answer', 'final_answer', 'content', 'text', 'value') if arguments.get(key) not in (None, '')), '')
                else:
                    candidate = str(arguments).strip()
                if candidate:
                    prediction = candidate
                    trace[-1]['answer_tool_compat'] = True
                    messages.append(model_message)
                    break
            messages.append(model_message)
            for (call_index, raw_call) in enumerate(calls, 1):
                call = parsed_calls[call_index - 1]
                call_id = str(raw_call.get('id') or f'call_{turn:04d}_{call_index:03d}')
                (response, payload) = tool_response(call, call_id, Path(sample['video']), media_dir, str(sample['id']), turn, results, str(sample['raw'].get('question', '')), raw)
                messages.append(response)
                tool_names.append(str(call.get('name', '')))
                results.append(payload)
        if not prediction and (not inference_error) and (not args.disable_forced_final):
            final_format = '<answer>concise final answer</answer>'
            messages.append({'role': 'user', 'content': f'No more tools. Output only {final_format}.'})
            try:
                completion = client.chat.completions.create(model=args.model_name, messages=messages, max_tokens=args.max_new_tokens, temperature=0.0, seed=args.seed, extra_body={'chat_template_kwargs': {'enable_thinking': False}})
                (model_message, raw, calls) = assistant_message(completion.choices[0].message)
                trace.append({'turn': len(trace) + 1, 'response': raw, 'tool_calls': calls, 'forced_final': True})
                messages.append(model_message)
                prediction = tools.answer_value(raw)
                if not prediction:
                    retry_messages = [{'role': 'system', 'content': 'Answer directly without reasoning or tools.'}, {'role': 'user', 'content': str(sample['raw'].get('question', '')) + f'\nOutput only {final_format}.'}]
                    retry = client.chat.completions.create(model=args.model_name, messages=retry_messages, max_tokens=256, temperature=0.0, seed=args.seed, extra_body={'chat_template_kwargs': {'enable_thinking': False}})
                    (retry_message, retry_raw, retry_calls) = assistant_message(retry.choices[0].message)
                    trace.append({'turn': len(trace) + 1, 'response': retry_raw, 'tool_calls': retry_calls, 'forced_final_retry': True})
                    messages.append(retry_message)
                    prediction = tools.answer_value(retry_raw)
            except Exception as error:
                inference_error = f'{type(error).__name__}: {error}'
                trace.append({'turn': len(trace) + 1, 'error': inference_error, 'forced_final': True})
        gold = str(sample['raw']['answer'])
        is_correct = (prediction or '').strip().casefold() == gold.strip().casefold()
        record = {'id': sample['id'], 'sample': sample['raw'], 'video': str(sample['video']), 'prediction': prediction, 'gold': gold, 'correct': is_correct, 'turn_count': len(trace), 'tools': tool_names, 'tool_results': results, 'trace': trace, 'messages': messages, 'trajectory': messages, 'format': 'qwen3.5-native-openai-tool-calls'}
        if inference_error:
            record['error'] = inference_error
        return record
    with output.open('a' if args.resume else 'w') as handle:
        completed = 0
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = {pool.submit(evaluate_sample, sample): sample for sample in samples}
            for future in as_completed(futures):
                sample = futures[future]
                try:
                    record = future.result()
                except Exception as error:
                    record = {'id': sample['id'], 'sample': sample['raw'], 'video': str(sample['video']), 'prediction': None, 'gold': str(sample['raw'].get('answer', '')), 'correct': False, 'turn_count': 0, 'tools': [], 'tool_results': [], 'trace': [], 'messages': [], 'trajectory': [], 'format': 'qwen3.5-native-openai-tool-calls', 'error': f'{type(error).__name__}: {error}'}
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                handle.flush()
                completed += 1
                print(f"[{completed}/{len(samples)}] {record['id']} pred={record.get('prediction')} gold={record.get('gold')} tools={record.get('tools', [])}", flush=True)
                if completed % max(1, args.concurrency) == 0 or completed == len(samples):
                    summarize(output)
if __name__ == '__main__':
    main()
