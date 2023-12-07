from utils.cleaning import remove_special_symbols


def test_remove_special_symbols():
    # Test with an empty string
    assert remove_special_symbols([""]) == [""]

    # Test with a string containing no special symbols
    assert remove_special_symbols(["Hello World 123"]) == ["Hello World"]

    # Test with a string containing special symbols and numbers
    assert remove_special_symbols(["@Remove!Special#Symbols123"]) == ["RemoveSpecialSymbols"]

    # Test with multiple strings
    assert remove_special_symbols(["String1 with special characters!", "String2 with 123 numbers"]) == [
        "String with special characters", "String with numbers"]

    # Test with a string containing only whitespace
    assert remove_special_symbols(["  \t  "]) == [""]
