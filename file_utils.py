import json
import pickle
import random


def add_lang_by_task(target_str, task, sub_task):
    if task == 'summarize':
        target_str = '<en> ' + target_str
    elif task == 'refine':
        target_str = '<java> ' + target_str
    elif task == 'translate':
        if sub_task == 'java-cs':
            target_str = '<c_sharp> ' + target_str
        else:
            target_str = '<java> ' + target_str
    elif task == 'concode':
        target_str = '<java> ' + target_str
    elif task == 'defect':
        target_str = target_str
    return target_str


def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item

    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        if args.sub_task != 'none':
            source_str = "{} {}: {}".format(args.task, args.sub_task, example.source)
        else:
            source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source

    source_str = source_str.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    assert source_ids.count(tokenizer.eos_token_id) == 1
    if stage == 'test':
        target_ids = []
    else:
        target_str = example.target
        if args.add_lang_ids:
            target_str = add_lang_by_task(example.target, args.task, args.sub_task)
        if args.task in ['defect', 'clone']:
            if target_str == 0:
                target_str = 'false'
            elif target_str == 1:
                target_str = 'true'
            else:
                raise NameError
        target_str = target_str.replace('</s>', '<unk>')
        target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length',
                                      truncation=True)
        assert target_ids.count(tokenizer.eos_token_id) == 1

    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
        url=example.url
    )


def convert_clone_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
        target_str = "{}: {}".format(args.task, example.target)
    else:
        source_str = example.source
        target_str = example.target
    code1 = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    code2 = tokenizer.encode(target_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    source_ids = code1 + code2
    return CloneInputFeatures(example_index, source_ids, example.label, example.url1, example.url2)


def random_indices(source, percentage):
    assert 0 <= percentage <= 1
    tokens = source.split()
    ids = list(set([random.randint(0, len(tokens) - 1) for _ in range(int(len(tokens) * percentage))]))
    return ids


def _switch_token(tokens: list, ids: list):
    ids = ids[:-1] if len(ids) % 2 == 1 else ids
    for idx1, idx2 in zip(ids[:len(ids) // 2], list(reversed(ids))[:len(ids) // 2]):
        tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]
    return tokens


def _replace_token(tokens: list, ids: list):
    for idx in ids:
        tokens[idx] = tokens[random.randint(0, len(tokens) - 1)]
    return tokens


def _delete_token(tokens: list, ids: list):
    _tokens = []
    for idx, token in enumerate(tokens):
        if idx in ids:
            continue
        _tokens.append(tokens[idx])
    return tokens


def _add_token(tokens: list, ids: list):
    _tokens = []
    for idx, token in enumerate(tokens):
        if idx in ids:
            _tokens.append(tokens[random.randint(0, len(tokens) - 1)])
        _tokens.append(tokens[idx])
    return tokens


def _prepare_corrupt_str(item):
    example, example_index, tokenizer, args = item
    # perform obfuscation

    # perform noising
    # sum: 20%  switch: 5% replace 5% deletion 5% addition 5%

    code_tokens = example.source.split()
    choice = random.choice('123')
    switch_ids = random_indices(example.source, 0.01)
    # code_tokens = _switch_token(code_tokens, switch_ids)

    replace_ids = random_indices(example.source, random.random() / 10)
    code_tokens = _replace_token(code_tokens, replace_ids)

    deletion_ids = random_indices(example.source, random.random() / 10)
    code_tokens = _delete_token(code_tokens, deletion_ids)

    addition_ids = random_indices(example.source, random.random() / 10)
    code_tokens = _add_token(code_tokens, addition_ids)

    example.source = ' '.join(code_tokens)

    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        corrupt_str = "{}: {}".format(args.task, example.source)
    else:
        corrupt_str = example.source

    return corrupt_str


def convert_defect_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source

    corrupt_str = _prepare_corrupt_str(item)
    corrupt_code_ids = tokenizer.encode(corrupt_str, max_length=args.max_source_length, padding='max_length', truncation=True)

    code_ids = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)

    feat = [DefectInputFeatures(example_index, code_ids, example.target, 0)]
    for _ in range(args.corrupt_instance):
        feat.append(DefectInputFeatures(example_index, corrupt_code_ids, -100, 1))

    return feat



class CloneInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label,
                 url1,
                 url2
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label
        self.url1 = url1
        self.url2 = url2


class DefectInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label,
                 corrupt_label
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label
        self.corrupt_label = corrupt_label


class InputFeatures(object):
    """A single training/test features for an example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task


class CloneExample(object):
    """A single training/test example."""

    def __init__(self,
                 code1,
                 code2,
                 label,
                 url1,
                 url2
                 ):
        self.source = code1
        self.target = code2
        self.label = label
        self.url1 = url1
        self.url2 = url2


def read_translate_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            src = line1.strip()
            trg = line2.strip()
            examples.append(
                Example(
                    idx=idx,
                    source=src,
                    target=trg,
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_refine_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0

    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip(),
                    target=line2.strip(),
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_concode_examples(filename, data_num):
    """Read examples from filename."""
    examples = []

    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=x["code"].strip()
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_summarize_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def remove_comment(code_str):
    source = code_str
    # source = source.replace('\n', '\n\n')
    lines = source.split('\\n')
    for i, line in enumerate(lines):
        if '//' in line:
            line = line[:line.find('//')]
        if '/*' and '*/' in lines[i]:
            line = line[:line.find('/*')] + line[line.find('*/') + 2:]
        lines[i] = line
    return '\\n'.join(lines)


def read_defect_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            # line = remove_comment(line).strip()
            line = line.strip()
            try:
                js = json.loads(line)
                code = ' '.join(js['func'].split())
                examples.append(
                    Example(
                        idx=js['idx'],
                        source=code,
                        target=js['target']
                    )
                )
            except:
                print(idx)
            if idx + 1 == data_num:
                break
    return examples


def read_clone_examples(filename, data_num):
    """Read examples from filename."""
    index_filename = filename
    url_to_code = {}
    with open('/'.join(index_filename.split('/')[:-1]) + '/data.jsonl') as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())
            url_to_code[js['idx']] = code

    data = []
    with open(index_filename) as f:
        idx = 0
        for line in f:
            line = line.strip()
            url1, url2, label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label == '0':
                label = 0
            else:
                label = 1
            data.append(CloneExample(url_to_code[url1], url_to_code[url2], label, url1, url2))
            idx += 1
            if idx == data_num:
                break
    return data
