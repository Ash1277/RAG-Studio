"""Quiz parser, batching, rate-limit backoff, and partial preservation tests."""
import json
import re
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _test_shims import install_fastapi_shims
install_fastapi_shims()

import rag_core as stub
import api
import quiz_service


class QuizParserTests(unittest.TestCase):
    def test_plain_json(self):
        raw = '{"questions":[{"question":"Q?","options":["A","B","C","D"],"answer_index":1}]}'
        quiz, error = api.parse_quiz_response(raw, 1)
        self.assertIsNone(error)
        self.assertEqual(quiz[0]["answer_index"], 1)

    def test_fenced_json_and_letter_answer(self):
        raw = '```json\n{"questions":[{"question":"Q?","options":["A","B","C","D"],"answer":"C"}]}\n```'
        quiz, error = api.parse_quiz_response(raw, 1)
        self.assertIsNone(error)
        self.assertEqual(quiz[0]["answer_index"], 2)

    def test_array_and_option_objects(self):
        raw = '[{"question":"Q?","options":[{"label":"A"},{"label":"B"},{"label":"C"},{"label":"D"}],"correct_answer_index":3}]'
        quiz, error = api.parse_quiz_response(raw, 1)
        self.assertIsNone(error)
        self.assertEqual(quiz[0]["answer_index"], 3)

    def test_answer_text_matching_option_casefold(self):
        raw = '{"questions":[{"question":"Which protocol?","options":["HTTP","FTP","SSH","SMTP"],"answer":"ssh"}]}'
        quiz, error = api.parse_quiz_response(raw, 1)
        self.assertIsNone(error)
        self.assertEqual(quiz[0]["answer_index"], 2)

    def test_boolean_answer_index_rejected(self):
        raw = '{"questions":[{"question":"Q?","options":["A","B","C","D"],"answer_index":true}]}'
        quiz, error = api.parse_quiz_response(raw, 1)
        self.assertIsNone(quiz)
        self.assertIn("unambiguous", error)

    def test_empty_or_wrong_count_rejected(self):
        quiz, error = api.parse_quiz_response('{"questions":[]}', 1)
        self.assertIsNone(quiz)
        self.assertTrue(error)


class QuizBatchTests(unittest.TestCase):
    def test_generation_splits_large_quizzes_into_small_valid_batches(self):
        calls = []
        def fake_answer(instruction, question, docs, max_tokens=1600):
            count = int(re.search(r"Create exactly (\d+)", instruction).group(1))
            calls.append((count, len(docs), max_tokens))
            items = [
                {"question": f"Q{len(calls)}-{i}?", "options": ["A", "B", "C", "D"], "answer_index": 0}
                for i in range(count)
            ]
            return json.dumps({"questions": items})

        original = stub.answer_with_context
        stub.answer_with_context = fake_answer
        try:
            quiz = api.generate_quiz_batches([{"source": "notes", "content": "evidence"}], 7)
        finally:
            stub.answer_with_context = original
        self.assertEqual(len(quiz), 7)
        self.assertEqual([c[0] for c in calls], [3, 3, 1])
        self.assertTrue(all(c[2] == 2600 for c in calls))

    def test_partial_batch_preservation_on_later_batch_failure(self):
        calls = []
        def fake_answer(instruction, question, docs, max_tokens=1600):
            calls.append(instruction)
            if len(calls) == 1:
                items = [
                    {"question": f"Q{i}?", "options": ["A", "B", "C", "D"], "answer_index": 1}
                    for i in range(3)
                ]
                return json.dumps({"questions": items})
            raise RuntimeError("429 Rate limit reached on second batch")

        original = stub.answer_with_context
        stub.answer_with_context = fake_answer
        try:
            questions, warning = quiz_service.generate_quiz_batches(
                [{"source": "notes", "content": "evidence"}], 6, allow_partial=True
            )
        finally:
            stub.answer_with_context = original
        self.assertEqual(len(questions), 3)
        self.assertIsNotNone(warning)
        self.assertIn("Returned 3 of 6", warning)

    def test_rate_limit_transient_backoff_succeeds(self):
        attempts = []
        def fake_answer(instruction, question, docs, max_tokens=1600):
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("HTTP 429 rate limit exceeded")
            return '{"questions":[{"question":"Q?","options":["A","B","C","D"],"answer_index":0}]}'

        original = stub.answer_with_context
        stub.answer_with_context = fake_answer
        try:
            quiz = quiz_service.generate_quiz_batches([{"source": "notes", "content": "evidence"}], 1)
        finally:
            stub.answer_with_context = original
        self.assertEqual(len(quiz), 1)
        self.assertEqual(len(attempts), 2)


class QuizRepairRetryTests(unittest.TestCase):
    def test_quiz_batch_gets_second_repair_attempt_before_failure(self):
        original = stub.answer_with_context
        calls = []
        def fake_answer(instruction, question, docs, max_tokens=1600):
            calls.append((instruction, max_tokens))
            if len(calls) < 3:
                return '{"not_questions": []}'
            return '{"questions":[{"question":"Q?","options":["A","B","C","D"],"answer_index":0}]}'

        stub.answer_with_context = fake_answer
        try:
            quiz = api.generate_quiz_batches([{"source": "lecture", "content": "evidence"}], 1)
        finally:
            stub.answer_with_context = original
        self.assertEqual(len(quiz), 1)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[-1][1], 3000)

    def test_study_important_questions_framed_as_study_prompts_not_exam_predictions(self):
        original_retrieve = stub.retrieve
        original_answer = stub.answer_with_context
        captured = {}
        stub.retrieve = lambda *a, **kw: [{"source": "Doc1", "content": "Key topic"}]
        def fake_answer(instruction, question, docs, max_tokens=1600):
            captured["instruction"] = instruction
            captured["question"] = question
            return "### Study Prompt 1..."
        stub.answer_with_context = fake_answer
        try:
            res = api.study(api.StudyRequest(source="all", kind="questions", count=5))
        finally:
            stub.retrieve = original_retrieve
            stub.answer_with_context = original_answer
        self.assertIn("never claim they are guaranteed exam predictions", captured["instruction"])
        self.assertIn("Study Prompt 1", res["result"])


if __name__ == "__main__":
    unittest.main()
