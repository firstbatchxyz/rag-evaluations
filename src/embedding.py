import json

import replicate


class BGELarge:
    def __init__(self):
        self.replicate = replicate

    def encode(self, text):
        output = self.replicate.run(
            "nateraw/bge-large-en-v1.5:9cf9f015a9cb9c61d1a2610659cdac4a4ca222f2d3707a68517b18c198a9add1",
            input={
                "texts": json.dumps(text),
                "convert_to_numpy": False,
                "normalize_embeddings": True
            }
        )
        return output[0]

