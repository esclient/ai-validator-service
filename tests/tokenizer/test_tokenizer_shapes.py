def test_tokenizer_truncates_to_max_length_128(real_tokenizer):
    out = real_tokenizer(
        "token " * 400, return_tensors="pt", max_length=128, truncation=True
    )
    assert out["input_ids"].shape == (1, 128)
    assert out["attention_mask"].shape == (1, 128)
