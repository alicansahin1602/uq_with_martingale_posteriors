"""Mock all active sampling backends; exercise real message and score handling."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from martingale_consistency_and_posteriors.utils import model_calls as providers
from martingale_consistency_and_posteriors.utils.prompt_types import RenderedPrompt
from tools.preview_prompt_strategies import BASE_PROMPT, five_step_preview, preview_configs


def api_response():
    alternatives = [SimpleNamespace(token=label, logprob=float(np.log(p)))
                    for label, p in zip('ABCD', [0.1, 0.6, 0.2, 0.1])]
    return SimpleNamespace(choices=[SimpleNamespace(
        logprobs=SimpleNamespace(content=[SimpleNamespace(token='B', top_logprobs=alternatives)]),
        message=SimpleNamespace(content='B'), finish_reason='stop',
    )])


class TinyTokenizer:
    """Record chat inputs and return tiny tensors, without a tokenizer download."""
    name_or_path = 'test-tokenizer'
    pad_token = '<pad>'
    pad_token_id = 7

    def encode(self, text, **kwargs):
        return ['ABCDE'.index(text.strip().upper())]

    def apply_chat_template(self, conversations, **kwargs):
        self.conversations = conversations
        self.options = kwargs
        return {'input_ids': torch.ones((len(conversations), 2), dtype=torch.long),
                'attention_mask': torch.ones((len(conversations), 2), dtype=torch.long)}

    def convert_ids_to_tokens(self, token_id):
        return str(token_id)

    def decode(self, ids, **kwargs):
        return ' '.join(map(str, ids))


class TinyModel:
    """CPU-only deterministic logits used to verify the existing scoring code."""
    device = torch.device('cpu')

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        logits = torch.arange(8, dtype=torch.float).repeat(input_ids.shape[0], 2, 1)
        return SimpleNamespace(logits=logits)


class PromptProviderTests(unittest.TestCase):
    def test_openai_and_deepseek_all_formats_messages_and_logs(self):
        for factory in (providers._build_openai_martingale_sampling_provider,
                        providers._build_deepseek_martingale_sampling_provider):
            with self.subTest(provider=factory.__name__), tempfile.TemporaryDirectory() as directory:
                client = MagicMock()
                client.chat.completions.create.return_value = api_response()
                path = str(Path(directory) / 'responses.jsonl')
                with patch.object(providers.openai, 'OpenAI', return_value=client):
                    get_probs = factory('mock-model', list('ABCDE'), api='unused', raw_log_path=path)
                    legacy_scores, legacy_probs = get_probs([BASE_PROMPT])
                    self.assertEqual(legacy_probs[0, 4], 0)
                    for config in preview_configs().values():
                        for step in five_step_preview(config)['steps']:
                            system, user = [m['content'] for m in step['messages']]
                            request = RenderedPrompt(system, user + '\nE) Not an answer choice', tuple('ABCD'))
                            scores, probs = get_probs([request])
                            np.testing.assert_array_equal(scores, legacy_scores)
                            np.testing.assert_array_equal(probs, legacy_probs)
                            sent = client.chat.completions.create.call_args.kwargs
                            self.assertEqual(sent['messages'], [
                                {'role': 'system', 'content': system},
                                {'role': 'user', 'content': request.user_text}])
                            self.assertEqual(sent['temperature'], 0.5)
                    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
                    self.assertEqual(len(records), 96)
                    self.assertEqual(records[-1]['valid_labels'], list('ABCD'))
                    self.assertEqual(records[-1]['system_prompt'], system)

    def test_local_all_formats_and_unchanged_assistant_prefill(self):
        tokenizer = TinyTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'local.jsonl')
            get_probs = providers._build_local_hf_martingale_sampling_provider(
                'tiny', TinyModel(), tokenizer, list('ABCDE'), raw_log_path=path)
            scores_old, probs_old = get_probs([BASE_PROMPT])
            for config in preview_configs().values():
                requests = [RenderedPrompt(step['messages'][0]['content'], step['messages'][1]['content'], tuple('ABCD'))
                            for step in five_step_preview(config)['steps']]
                scores, probs = get_probs(requests)
                np.testing.assert_allclose(scores, np.repeat(scores_old, 5, axis=0))
                np.testing.assert_allclose(probs, np.repeat(probs_old, 5, axis=0))
                self.assertTrue((probs[:, 4] == 0).all())
                for request, conversation in zip(requests, tokenizer.conversations):
                    self.assertEqual(conversation, [
                        {'role': 'system', 'content': request.system_text},
                        {'role': 'user', 'content': request.user_text},
                        {'role': 'assistant', 'content': 'Answer: '},
                    ])
                self.assertTrue(tokenizer.options['continue_final_message'])
                self.assertFalse(tokenizer.options['add_generation_prompt'])
            records = [json.loads(line) for line in Path(path).read_text().splitlines()]
            self.assertEqual(len(records), 96)
            self.assertIn('system_prompt', records[-1])


if __name__ == '__main__':
    unittest.main()
