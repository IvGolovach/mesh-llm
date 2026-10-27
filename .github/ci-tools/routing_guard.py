#!/usr/bin/env python3
"""Reject unclassified jobs and changes to a reviewed runner assignment."""
import argparse
import json
import re
from pathlib import Path
import yaml


def workflows(root):
    for path in sorted((root / '.github/workflows').glob('*')):
        if path.suffix not in ('.yml', '.yaml'):
            continue
        data = yaml.safe_load(path.read_text()) or {}
        for name, job in data.get('jobs', {}).items():
            yield str(path.relative_to(root)) + ':' + name, job, data


def assignment(job):
    if 'uses' in job:
        return {'uses': job['uses']}
    route={'runs-on': job.get('runs-on')}
    if 'matrix' in job.get('strategy', {}):
        route['matrix']=job['strategy']['matrix']
    return route


def check(root):
    policy = json.loads((root / '.github/ci-routing.json').read_text())
    if policy.get('schema') != 1 or policy.get('mode') not in ('private', 'public-hub'):
        raise ValueError('Unsupported CI routing policy')
    errors, seen = [], set()
    entries = policy.get('jobs', {})
    for key, job, workflow in workflows(root):
        seen.add(key)
        entry = entries.get(key)
        if not entry:
            errors.append(f'{key}: unclassified job; run ci onboard and review its platform')
            continue
        if entry.get('assignment') != assignment(job):
            errors.append(f'{key}: runner assignment changed without updating its reviewed policy')
        platform = entry.get('platform')
        route = json.dumps(assignment(job)).lower()
        if platform == 'linux':
            if policy['mode'] != 'private' or 'unraid-ci' not in route or 'linux' not in route or 'x64' not in route or 'macos' in route:
                errors.append(f'{key}: ordinary Linux CI must use the Unraid pool')
            profiles = [p for p in ['ci-light', 'ci-standard', 'ci-heavy', 'ci-mac-admission'] if p in route]
            if len(profiles) != 1:
                errors.append(f'{key}: choose exactly one resource profile')
            events = workflow.get('on', workflow.get(True, {}))
            if isinstance(events, str):
                events = [events]
            if 'pull_request_target' in events:
                errors.append(f'{key}: privileged pull_request_target is not a home-runner entrypoint')
        elif platform == 'apple':
            if 'macos' not in route or 'self-hosted' not in route or policy['mode'] != 'private':
                errors.append(f'{key}: native Apple checks require the private Mac runner')
            if not entry.get('reason', '').strip():
                errors.append(f'{key}: explain the Apple SDK requirement')
        elif platform == 'public-hosted':
            if policy['mode'] != 'public-hub' or 'self-hosted' in route or 'unraid-ci' in route:
                errors.append(f'{key}: public PR jobs must remain outside home runners')
        elif platform == 'reusable':
            if not re.fullmatch(r'IvGolovach/unraid-ci/\.github/workflows/ci-(linux|apple)\.yml@[0-9a-f]{40}', job.get('uses', '')):
                errors.append(f'{key}: shared CI must be pinned to a reviewed full commit SHA')
        elif platform == 'exception':
            if len(entry.get('reason', '').strip()) < 20:
                errors.append(f'{key}: a specific platform/credential exception needs a reason')
        else:
            errors.append(f'{key}: unknown platform {platform}')
    for key in sorted(set(entries) - seen):
        errors.append(f'{key}: obsolete policy entry')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', nargs='?', default='.')
    args = parser.parse_args()
    try:
        errors = check(Path(args.path).resolve())
    except (OSError, ValueError, yaml.YAMLError) as exc:
        errors = [str(exc)]
    for error in errors:
        print(error)
    if errors:
        raise SystemExit(1)
    print('CI routing policy: all jobs classified; reviewed routes unchanged')


if __name__ == '__main__':
    main()
