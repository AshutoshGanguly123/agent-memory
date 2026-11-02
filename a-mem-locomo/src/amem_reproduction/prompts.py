import copy
import json
from pathlib import Path


class Prompts:
    def __init__(self, root):
        root = Path(root)
        self.note_template = (root / "configs/prompts/note.txt").read_text()
        self.evolution_template = (root / "configs/prompts/evolution.txt").read_text()
        self.schemas = json.loads((root / "configs/schemas.json").read_text())

    def note(self, content):
        return self.note_template + content, self.schemas["note"]

    def evolution(self, note, neighbor_text, indices):
        prompt = self.evolution_template.format(
            context=note["context"], content=note["content"], keywords=note["keywords"],
            nearest_neighbors_memories=neighbor_text, neighbor_number=len(indices))
        schema = copy.deepcopy(self.schemas["evolution"])
        schema["properties"]["actions"]["items"]["enum"] = ["strengthen", "update_neighbor"]
        if indices:
            schema["properties"]["suggested_connections"]["items"]["enum"] = indices
        else:
            schema["properties"]["suggested_connections"]["maxItems"] = 0
        return prompt, schema

    def rewrite(self, question):
        return (f'''Given the following question, generate several keywords, using 'cosmos' as the separator.
                Question: {question}

                Format your response as a JSON object with a "keywords" field containing the selected text.
                Example response format:
                {{"keywords": "keyword1, keyword2, keyword3"}}''', self.schemas["rewrite"])

    def answer(self, question, category, context):
        if category == 2:
            prompt = f'''Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.
Question: {question} Short answer:'''
        else:
            prompt = f'''Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.
Question: {question} Short answer:'''
        if category == 5:
            # The generative abstention track never exposes the dataset distractor.
            prompt += '\nIf the requested information is not in the context, answer "No information available".'
        return prompt, self.schemas["answer"]
