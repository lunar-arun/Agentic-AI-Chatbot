import unittest
from types import SimpleNamespace

import _stubs


def tok(text):
    return SimpleNamespace(message=SimpleNamespace(content=text))


def chunk(score, text="Some policy text.", source="shipping_policy.md"):
    return {"text": text, "source": source, "score": score}


class ParsingTests(unittest.TestCase):
    def setUp(self):
        self.fake = _stubs.install()
        import agents
        self.a = agents

    def test_parse_variants(self):
        p = self.a._parse_decision
        self.assertEqual(
            p("DECISION: RETRIEVE\nQUERY: \"return window\""),
            {"type": "RETRIEVE", "query": "return window"})
        self.assertEqual(
            p("<think>hmm</think>DECISION: ANSWER\nRESPONSE: Hi!\nLine2"),
            {"type": "ANSWER", "response": "Hi!\nLine2"})
        self.assertEqual(
            p("decision: handoff\nreason: wants refund")["type"], "HANDOFF")
        self.assertEqual(p("just chatting"),
                         {"type": "ANSWER", "response": "just chatting"})
        self.assertEqual(p("DECISION: HANDOFF")["reason"],
                         "Customer needs further assistance.")

    def test_think_stream_filter(self):
        f = lambda parts: "".join(
            self.a._filter_think_stream(iter([tok(t) for t in parts])))
        self.assertEqual(f(["Hello ", "there, friend!"]), "Hello there, friend!")
        self.assertEqual(f(["<think>", "reason", "</think>", "\n\nAnswer"]), "Answer")
        self.assertEqual(f(["<think>x</think>Answer here"]), "Answer here")
        self.assertEqual(f(["Hi"]), "Hi")
        self.assertEqual(f(["", "Hello world, again"]), "Hello world, again")

    def test_prepare_history(self):
        a = self.a
        hist = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "_Escalating to Agent 2: x_",
             "agent": "Agent 1", "handoff_reason": "x"},
            {"role": "assistant", "content": "❌ Error in Agent 1: boom"},
            {"role": "assistant", "content": "a2", "agent": "Agent 2",
             "handoff_reason": None},
        ]
        self.assertEqual(
            a._prepare_history(hist),
            [{"role": "user", "content": "q1"},
             {"role": "assistant", "content": "a2"}])
        long = [{"role": "user", "content": str(i)} for i in range(50)]
        self.assertEqual(len(a._prepare_history(long)), a.MAX_HISTORY_MESSAGES)
        self.assertEqual(a._prepare_history(long)[-1]["content"], "49")


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.fake = _stubs.install()
        import agents
        self.a = agents
        self.retrievals = []
        self.script = []

        def fake_retrieve(q, *args, **kw):
            self.retrievals.append(q)
            return self.script.pop(0)
        agents.retrieve = fake_retrieve

    def test_greeting_answers_directly(self):
        self.fake.chat_replies += ["DECISION: ANSWER\nRESPONSE: Hello!"]
        r = self.a.run_agent1_loop("hi", [])
        self.assertEqual(r["action"], "answer")
        self.assertEqual(r["direct_answer"], "Hello!")
        self.assertEqual(self.retrievals, [])

    def test_immediate_handoff(self):
        self.fake.chat_replies += ["DECISION: HANDOFF\nREASON: order lookup"]
        r = self.a.run_agent1_loop("where is order 5?", [])
        self.assertEqual((r["action"], r["reason"]), ("handoff", "order lookup"))

    def test_retrieve_sufficient(self):
        self.fake.chat_replies += [
            "DECISION: RETRIEVE\nQUERY: return window", "SUFFICIENT"]
        self.script += [[chunk(0.8), chunk(0.3)]]
        r = self.a.run_agent1_loop("how long to return?", [])
        self.assertEqual(r["action"], "answer")
        self.assertEqual(len(r["context"]), 1)          # below-threshold dropped
        self.assertEqual(r["trace"][0]["judgment"], "LLM judged: SUFFICIENT")

    def test_insufficient_judgment_is_not_misread_as_sufficient(self):
        self.fake.chat_replies += [
            "DECISION: RETRIEVE\nQUERY: q1", "INSUFFICIENT - missing details",
            "DECISION: RETRIEVE\nQUERY: q2", "SUFFICIENT"]
        self.script += [[chunk(0.7)], [chunk(0.9)]]
        r = self.a.run_agent1_loop("question", [])
        self.assertEqual(r["action"], "answer")
        self.assertEqual(self.retrievals, ["q1", "q2"])

    def test_max_iterations_hands_off_without_wasted_llm_call(self):
        a = self.a
        n = a.MAX_RETRIEVAL_ITERATIONS
        self.fake.chat_replies += [
            f"DECISION: RETRIEVE\nQUERY: query {i}" for i in range(n)]
        self.script += [[chunk(0.1)] for _ in range(n)]
        r = a.run_agent1_loop("obscure", [])
        self.assertEqual(r["action"], "handoff")
        self.assertIn("max iterations", r["trace"][-1]["action"])
        self.assertEqual(len(self.fake.chat_calls), n)   # no extra decision call
        self.assertEqual(len(self.retrievals), n)

    def test_direct_answer_after_failed_search_is_not_allowed(self):
        self.fake.chat_replies += [
            "DECISION: RETRIEVE\nQUERY: q1",
            "DECISION: ANSWER\nRESPONSE: made up policy",   # must be ignored
            "SUFFICIENT"]
        self.script += [[chunk(0.1)], [chunk(0.9)]]
        r = self.a.run_agent1_loop("What is the policy?", [])
        self.assertEqual(r["action"], "answer")
        self.assertIsNone(r["direct_answer"])
        self.assertEqual(self.retrievals, ["q1", "What is the policy?"])

    def test_repeated_query_is_not_re_run(self):
        a = self.a
        self.fake.chat_replies += [
            "DECISION: RETRIEVE\nQUERY: same", "DECISION: RETRIEVE\nQUERY: Same ",
            "DECISION: HANDOFF\nREASON: nothing found"]
        self.script += [[chunk(0.1)]]
        r = a.run_agent1_loop("x", [])
        self.assertEqual(self.retrievals, ["same"])
        self.assertEqual(r["action"], "handoff")
        self.assertIn("Repeated query", r["trace"][1]["judgment"])

    def test_user_quotes_and_no_think_are_sanitised(self):
        self.fake.chat_replies += ["DECISION: ANSWER\nRESPONSE: ok"]
        self.a.run_agent1_loop('say "hi" /no_think now', [])
        prompt = self.fake.chat_calls[0][-1]["content"]
        self.assertIn("say 'hi'  now", prompt)
        self.assertEqual(prompt.count("/no_think"), 1)   # only our own suffix


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.fake = _stubs.install()
        import agents
        self.a = agents

    def test_answer_prompt_survives_braces_in_context(self):
        self.fake.chat_replies += [[tok("Fine"), tok(" answer here!!")]]
        ctx = [chunk(0.9, text="Use {braces} and {0} freely")]
        out = "".join(self.a.stream_agent1_answer("q", [], ctx))
        self.assertEqual(out, "Fine answer here!!")
        system = self.fake.chat_calls[0][0]["content"]
        self.assertIn("Use {braces} and {0} freely", system)

    def test_agent2_prompt_has_no_fabrication_and_has_contact(self):
        self.fake.chat_replies += [[tok("Please share your order number.")]]
        hist = [{"role": "user", "content": "cancel my order"}]
        "".join(self.a.stream_agent2("cancel request", hist))
        system = self.fake.chat_calls[0][0]["content"]
        self.assertIn("NEVER claim", system)
        self.assertIn("1-800-555-GREEN", system)
        self.assertNotIn("{contact}", system)
        self.assertIn("cancel request", self.fake.chat_calls[0][-1]["content"])


if __name__ == "__main__":
    unittest.main()
