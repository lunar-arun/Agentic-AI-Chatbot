import unittest

import _stubs  # noqa: F401  (sets up sys.path)
from validation import filter_input, escape_html, inline_code_safe


class FilterInputTests(unittest.TestCase):
    def test_accepts_normal_question(self):
        self.assertIsNone(filter_input("How long does standard shipping take?"))

    def test_rejects_empty_and_whitespace(self):
        self.assertIn("enter a message", filter_input(""))
        self.assertIn("enter a message", filter_input("   \n "))

    def test_rejects_too_long(self):
        self.assertIn("too long", filter_input("a b " * 400))

    def test_accepts_exactly_max_length(self):
        self.assertIsNone(filter_input(("word " * 200)[:1000]))

    def test_rejects_repeated_characters(self):
        self.assertIn("spam", filter_input("hellooooooooo"))
        self.assertIn("spam", filter_input("😀" * 9))

    def test_rejects_low_variety(self):
        self.assertIn("spam", filter_input("abababab"))

    def test_short_low_variety_is_allowed(self):
        self.assertIsNone(filter_input("ok ok"))

    def test_rejects_bare_url(self):
        self.assertIn("link", filter_input("https://example.com/order/123"))

    def test_url_with_question_is_allowed(self):
        self.assertIsNone(filter_input("Is https://example.com/x my order?"))


class EscapeTests(unittest.TestCase):
    def test_escape_html(self):
        self.assertEqual(
            escape_html('<script>alert("x")</script>'),
            "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;",
        )

    def test_inline_code_safe(self):
        self.assertEqual(inline_code_safe("a`b\nc"), "a'b c")


if __name__ == "__main__":
    unittest.main()
