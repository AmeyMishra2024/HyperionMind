from app.memory import MemoryStore


def test_memories_are_separated_by_person() -> None:
    store = MemoryStore()
    store.add("Person 1", "user", "My favorite color is orange.")
    store.add("Person 2", "user", "I like blue.")
    assert store.context("Person 1") == [{"role": "user", "content": "My favorite color is orange."}]
    assert store.context("Person 2") == [{"role": "user", "content": "I like blue."}]


def test_memory_is_bounded() -> None:
    store = MemoryStore()
    for index in range(30):
        store.add("Person 1", "user", f"message {index}")
    context = store.context("Person 1")
    assert len(context) == 24
    assert context[0]["content"] == "message 6"


def test_forget_removes_only_selected_person() -> None:
    store = MemoryStore()
    store.add("Person 1", "user", "one")
    store.add("Person 2", "user", "two")
    assert store.forget("Person 1") is True
    assert "Person 1" not in store.snapshot()
    assert "Person 2" in store.snapshot()

