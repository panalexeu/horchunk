from tiktoken import get_encoding


class Chunk:

    def __init__(
            self,
            splits: list[str]
    ):
        self.splits = splits

    @property
    def size(self) -> int:
        """The number of elements in the ``splits`` list."""
        return len(self.splits)

    @property
    def chars(self) -> int:
        """The number of chars in the chunk."""
        return sum(len(split) for split in self.splits)

    @property
    def tokens(self) -> int:
        """
        The total number of tokens in the chunk based on OpenAI's ``cl100k_base`` encoding.
        """
        enc = get_encoding('cl100k_base')
        return sum(len(enc.encode(split)) for split in self.splits)

    def __repr__(self) -> str:
        # forming fancy splits list string
        str_splits = ''
        for i, split in enumerate(self.splits):
            str_splits += f'"{split}", '
        str_splits = str_splits[:-2]

        # final repr template
        str_template = f'<Chunk size={self.size}, chars={self.chars}, tokens={self.tokens}, splits=[{str_splits}]/>'

        return str_template
