import unittest

from idioms.idion import parse_entry, parse_example

PAGE = (
    "<h1>αφήνω στη μέση κάτι\nαφήνω μισό κάτι</h1><h2>Meaning</h2><h3>αφήνω κάτι ανολοκλήρωτο</h3>"
    "<p>Greek</p><h4>to leave something unfinished</h4><p>English</p>"
    "<h2>Corpus Examples</h2><ul>"
    "<li>Αφέθηκε η κουβέντα μας στη μέση. acceptable  http://example.gr/a</li>"
    "<li>Κατέβηκα και το άφησα στη μέση του τραπεζιού acceptable 1 https://books.google.nl/x?a=1&amp;b=2</li>"
    "<li>Με άφησε στους πέντε δρόμους. acceptable  http://example.gr/b# sent_id = x\n# text = ...\n1\tΜε</li>"
    "<li>Αφήνω το στη μέση unacceptable  introspection</li>"
    "</ul><h2>Relations</h2>"
)


class TestIdionParsing(unittest.TestCase):
    def test_example_formats(self):
        self.assertEqual(parse_example("Αφέθηκε. acceptable  http://example.gr/a"),
                         ("Αφέθηκε.", "acceptable", "http://example.gr/a"))
        self.assertEqual(parse_example("Το άφησα acceptable 1 https://b.nl/x?a=1&amp;b=2"),
                         ("Το άφησα", "acceptable", "https://b.nl/x?a=1&b=2"))
        self.assertEqual(parse_example("Μένει. acceptable  http://e.gr/b# sent_id = x\n1\tΜε")[2], "http://e.gr/b")
        self.assertIsNone(parse_example("κάτι χωρίς κρίση")[1])

    def test_entry(self):
        entry = parse_entry(PAGE)
        self.assertEqual(entry["headword"], "αφήνω στη μέση κάτι")
        self.assertEqual(entry["variants"], ["αφήνω στη μέση κάτι", "αφήνω μισό κάτι"])
        self.assertEqual(entry["meaning_en"], "to leave something unfinished")
        self.assertEqual([j for _, j, _ in entry["examples"]], ["acceptable"] * 3 + ["unacceptable"])


if __name__ == "__main__":
    unittest.main()
