"""Load private service settings without embedding deployment details in code."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2].parent / '.local' / 'services.yaml'
FIELDS = {
    'inference': {'base_url': 'API_BASE', 'model': 'MODEL_NAME', 'api_key': 'EVAL_API_KEY'},
    'ydc': {'endpoint': 'YDC_SEARCH_ENDPOINT', 'api_key': 'YDC_API_KEY'},
    'serper': {'search_endpoint': 'SERPER_SEARCH_ENDPOINT', 'image_endpoint': 'SERPER_IMAGE_ENDPOINT', 'scrape_endpoint': 'SERPER_SCRAPE_ENDPOINT', 'api_key': 'SERPER_API_KEY'},
    'serpapi': {'endpoint': 'SERPAPI_ENDPOINT', 'api_key': 'SERPAPI_API_KEY'},
    'oss': {'endpoint': 'OSS_ENDPOINT', 'bucket': 'OSS_BUCKET', 'public_base_url': 'OSS_PUBLIC_BASE_URL', 'access_key_id': 'OSS_ACCESS_KEY_ID', 'access_key_secret': 'OSS_ACCESS_KEY_SECRET', 'prefix': 'OSS_PREFIX', 'object_acl': 'OSS_OBJECT_ACL'},
    'summary': {'base_url': 'VISIT_PAGE_SUMMARY_BASE_URL', 'model': 'VISIT_PAGE_SUMMARY_MODEL', 'api_key': 'VISIT_PAGE_SUMMARY_API_KEY'},
    'judge': {'base_url': 'JUDGE_BASE_URL', 'model': 'JUDGE_MODEL', 'api_key': 'JUDGE_API_KEY'},
}


def load_services(path: str | Path | None = None) -> Path | None:
    """Explicit CLI path, environment path, then local default; env values win."""
    if not os.environ.get('SERPER_API_KEY') and os.environ.get('SERP_API_KEY'):
        os.environ['SERPER_API_KEY'] = os.environ['SERP_API_KEY']
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--services-config')
    args, _ = parser.parse_known_args()
    explicit = path or args.services_config or os.environ.get('VDR_SERVICES_CONFIG')
    config = Path(explicit).expanduser() if explicit else DEFAULT_CONFIG
    if not config.is_file():
        if explicit:
            raise FileNotFoundError(f'Service configuration does not exist: {config}')
        return None
    data = yaml.safe_load(config.read_text(encoding='utf-8')) or {}
    if not isinstance(data, dict) or data.get('version', 1) != 1:
        raise ValueError('Service configuration must be a mapping with version: 1')
    for section, values in data.items():
        if section == 'version':
            continue
        if section not in FIELDS or not isinstance(values, dict):
            raise ValueError(f'Invalid service section: {section}')
        for key, value in values.items():
            if key not in FIELDS[section]:
                raise ValueError(f'Unknown service setting: {section}.{key}')
            if value is None or value == '':
                continue
            if not isinstance(value, str):
                raise ValueError(f'Service setting must be a string: {section}.{key}')
            name = FIELDS[section][key]
            if not os.environ.get(name):
                os.environ[name] = value
    return config
