from app.classifier.training import stratified_split


def test_stratified_split_is_reproducible_and_keeps_each_class_in_all_splits():
    samples = [
        (f"class-{class_index}/{sample_index}.jpg", class_index)
        for class_index in range(5)
        for sample_index in range(20)
    ]

    first = stratified_split(samples, seed=42)
    second = stratified_split(samples, seed=42)

    assert first == second
    assert [len(split) for split in first] == [80, 10, 10]
    assert set(first[0]).isdisjoint(first[1])
    assert set(first[0]).isdisjoint(first[2])
    assert set(first[1]).isdisjoint(first[2])
    for split in first:
        assert {samples[index][1] for index in split} == set(range(5))
