import pytest
import argparse
from unittest.mock import patch
from io import StringIO
from src.inference import main


@pytest.mark.parametrize("input_text, expected_output", [
    ("Hello! My name is Jeremy.", (0, 0, 0, 0, 0, 0)),
    ("You are a fool!", (1, 0, 0, 0, 1, 0)),
    ("How dare you!", (1, 0, 0, 0, 0, 0))
])
def test_main(input_text, expected_output):
    with patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(text=input_text)):
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            main()
            output = mock_stdout.getvalue().strip().split('\n')

    # Check if the printed output matches the expected values
    assert f"toxic: {bool(expected_output[0])}" in output
    assert f"severe toxic: {bool(expected_output[1])}" in output
    assert f"obscene: {bool(expected_output[2])}" in output
    assert f"threat: {bool(expected_output[3])}" in output
    assert f"insult: {bool(expected_output[4])}" in output
    assert f"identity hate: {bool(expected_output[5])}" in output
