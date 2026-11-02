"""Ordered A-Mem notes, author-code evolution and retrieval, atomic checkpoints."""
import copy
import json
import time
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .storage import atomic_json, digest


class Memory:
    def __init__(self, embedder, prompts, journal, config, state=None):
        self.embedder, self.prompts, self.journal, self.config = embedder, prompts, journal, config
        self.state = state or {"notes": [], "index_documents": [], "embeddings": [],
                               "evolution_count": 0, "frozen": False, "build_seconds": 0.0}
        self.notes = self.state["notes"]

    def search(self, query, k):
        if not self.notes:
            return []
        scores = cosine_similarity(self.embedder.encode([query]),
                                   np.asarray(self.state["embeddings"], dtype=np.float32))[0]
        return [int(i) for i in np.argsort(scores)[-k:][::-1]]

    def neighbor_text(self, indices):
        text = ""
        for i in indices:
            n = self.notes[i]
            text += ("memory index:" + str(i) + "\t talk start time:" + n["timestamp"] +
                     "\t memory content: " + n["content"] + "\t memory context: " + n["context"] +
                     "\t memory keywords: " + str(n["keywords"]) + "\t memory tags: " + str(n["tags"]) + "\n")
        return text

    def add(self, turn, prefix):
        if self.state["frozen"]:
            raise RuntimeError("QA memory is frozen")
        started = time.perf_counter()
        content = "Speaker " + turn["speaker"] + "says : " + turn["text"]
        enriched, note_call = self.journal.complete("note", prefix + "/note", *self.prompts.note(content))
        note = dict(enriched, id=turn["dia_id"], content=content, timestamp=turn["timestamp"],
                    speaker=turn["speaker"], session=turn["session"], links=[],
                    importance_score=1.0, retrieval_count=0, last_accessed=turn["timestamp"],
                    evolution_history=[], category="Uncategorized")
        note["context"] = note["context"] or "General"
        indices = self.search(content, self.config["write_k"])
        decision, evolve_call = self.journal.complete(
            "evolution", prefix + "/evolution",
            *self.prompts.evolution(note, self.neighbor_text(indices), indices))
        before = {str(i): copy.deepcopy(self.notes[i]) for i in indices}
        if decision["should_evolve"]:
            for action in decision["actions"]:
                if action == "strengthen":
                    # Integers are stable insertion indices, as in the released code.
                    if any(i not in indices for i in decision["suggested_connections"]):
                        raise ValueError("Evolution linked outside fetched neighbors")
                    note["links"].extend(decision["suggested_connections"])
                    note["tags"] = decision["tags_to_update"]
                elif action == "update_neighbor":
                    tags, contexts = decision["new_tags_neighborhood"], decision["new_context_neighborhood"]
                    for j in range(min(len(indices), len(tags))):
                        old = self.notes[indices[j]]
                        old["tags"] = tags[j]
                        if j < len(contexts):
                            old["context"] = contexts[j]
            self.state["evolution_count"] += 1
        self.notes.append(note)
        document = ("content:" + note["content"] + " context:" + note["context"] + " keywords: " +
                    ", ".join(note["keywords"]) + " tags: " + ", ".join(note["tags"]))
        self.state["index_documents"].append(document)
        self.state["embeddings"].append(self.embedder.encode([document])[0].tolist())
        refreshed = bool(decision["should_evolve"] and self.state["evolution_count"] %
                         self.config["evolution_refresh_interval"] == 0)
        if refreshed:
            documents = [n["content"] + " , " + n["context"] + " " + " ".join(n["keywords"]) +
                         " " + " ".join(n["tags"]) for n in self.notes]
            self.state["index_documents"] = documents
            # Encode each note separately, as in authors' consolidation routine.
            self.state["embeddings"] = [self.embedder.encode([d])[0].tolist() for d in documents]
        elapsed = time.perf_counter() - started
        self.state["build_seconds"] += elapsed
        return {"turn_id": turn["dia_id"], "note_call_id": note_call, "evolution_call_id": evolve_call,
                "neighbor_indices": indices, "neighbor_ids": [self.notes[i]["id"] for i in indices],
                "decision": decision, "before_neighbors": before,
                "after_neighbors": {str(i): copy.deepcopy(self.notes[i]) for i in indices},
                "new_note": copy.deepcopy(note), "index_refreshed": refreshed,
                "partial_neighbor_arrays": (len(decision["new_tags_neighborhood"]) != len(indices) or
                                            len(decision["new_context_neighborhood"]) != len(indices)),
                "latency_seconds": elapsed, "footprint": self.footprint()}

    def retrieve(self, query):
        k = self.config["read_k"]
        primary = self.search(query, k)
        expanded, j = [], 0
        for i in primary:
            expanded.append(i)
            for neighbor in self.notes[i]["links"]:
                expanded.append(neighbor)
                # Deliberately preserve author's global counter / append-before-cap behavior.
                if j >= k:
                    break
                j += 1
        context = ""
        for i in expanded:
            n = self.notes[i]
            context += ("talk start time:" + n["timestamp"] + "memory content: " + n["content"] +
                        "memory context: " + n["context"] + "memory keywords: " + str(n["keywords"]) +
                        "memory tags: " + str(n["tags"]) + "\n")
        return context, {"primary_indices": primary, "expanded_indices": expanded,
                         "primary_ids": [self.notes[i]["id"] for i in primary],
                         "supplied_ids": [self.notes[i]["id"] for i in expanded],
                         "duplicate_count": len(expanded) - len(set(expanded))}

    def footprint(self):
        size = lambda value: len(json.dumps(value, ensure_ascii=False).encode())
        return {"notes": len(self.notes), "directed_edges": sum(len(n["links"]) for n in self.notes),
                "notes_json_bytes": size(self.notes), "index_documents_json_bytes": size(self.state["index_documents"]),
                "embedding_float32_bytes": int(np.asarray(self.state["embeddings"], dtype=np.float32).nbytes)}

    def fingerprint(self):
        return digest({k: v for k, v in self.state.items() if k != "build_seconds"})

    def save(self, path):
        atomic_json(path, self.state)
