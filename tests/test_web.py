import unittest

from web_app import render_page


class WebPageTests(unittest.TestCase):
    def test_form_escapes_user_text(self):
        page = render_page("<script>alert(1)</script>")
        self.assertNotIn("<script>alert(1)</script>", page)
        self.assertIn("&lt;script&gt;", page)

    def test_page_has_a_planning_form(self):
        page = render_page()
        self.assertIn('name="request"', page)
        self.assertIn("Plan my week", page)


if __name__ == "__main__":
    unittest.main()
