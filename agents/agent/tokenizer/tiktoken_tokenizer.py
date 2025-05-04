import concurrent.futures
from typing import List

import tiktoken


class TikToken:
    def __init__(
            self,
            model_name: str = "o200k_base",
    ):
        try:
            self.model_name = model_name
            self.encoding = tiktoken.get_encoding(model_name)
        except Exception as e:
            raise ValueError(
                f"Failed to initialize tokenizer with model '{model_name}': {str(e)}"
            )

    def encode(self, string: str) -> str:
        return self.encoding.encode(string)

    def decode(self, tokens: List[int]) -> str:
        return self.encoding.decode(tokens)

    def count_tokens(self, string: str) -> int:
        """
        Count the number of tokens in the input string using multithreading (thread-safe).
        """
        def count_tokens_in_chunk(chunk):
            return len(self.encoding.encode(chunk))

        # Split the string into chunks for parallel processing
        chunks = [
            string[i: i + 1000] for i in range(0, len(string), 1000)
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(count_tokens_in_chunk, chunks))

        return sum(results)
