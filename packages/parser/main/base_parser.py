from tree_sitter import Language, Parser, Tree


class BaseParser:
    def __init__(self, lang: object):
        self.language = Language(lang)
        self.parser = Parser(self.language)

    def parse_code(self, source_code: str) -> Tree:
        tree = self.parser.parse(bytes(source_code, "utf-8"))
        return tree
