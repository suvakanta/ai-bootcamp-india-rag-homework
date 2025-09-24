"""
Unit tests for eval.py module.

This module contains comprehensive unit tests for all functions in the eval.py file,
including test cases for normal operation, edge cases, and error conditions.
"""

import unittest
import json
import os
import tempfile
import pandas as pd
from unittest.mock import patch, mock_open, MagicMock
from datasets import Dataset, Features, Value, Sequence

# Import the functions to test
import eval


class TestEvalModule(unittest.TestCase):
    """Test cases for the eval module."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_csv_data = pd.DataFrame({
            'question': ['What is AI?', 'What is ML?'],
            'contexts': [['AI is artificial intelligence'], ['ML is machine learning']],
            'ground_truth': ['AI stands for artificial intelligence', 'ML stands for machine learning']
        })
        
        self.sample_json_data = [
            {
                "question": "What is AI?",
                "answer": "AI is artificial intelligence",
                "contexts": ["AI is artificial intelligence"]
            },
            {
                "question": "What is ML?", 
                "answer": "ML is machine learning",
                "contexts": ["ML is machine learning"]
            }
        ]
        
        self.invalid_json_data = [
            {
                "question": "What is AI?",
                "answer": "AI is artificial intelligence"
                # Missing required 'contexts' field
            }
        ]

    def test_test_dataset_success(self):
        """Test test_dataset function with valid CSV file."""
        with patch('eval.load_dataset') as mock_load_dataset:
            # Mock the dataset structure
            mock_dataset = MagicMock()
            mock_train_data = MagicMock()
            mock_train_data.remove_columns.return_value = "processed_dataset"
            mock_dataset.__getitem__.return_value = mock_train_data
            mock_load_dataset.return_value = mock_dataset
            
            result = eval.test_dataset()
            
            # Verify load_dataset was called with correct parameters
            mock_load_dataset.assert_called_once_with('csv', data_files='./eval/eval_dataset.csv')
            mock_train_data.remove_columns.assert_called_once_with(['Unnamed: 0'])
            self.assertEqual(result, "processed_dataset")

    def test_test_dataset_file_not_found(self):
        """Test test_dataset function when CSV file doesn't exist."""
        with patch('eval.load_dataset') as mock_load_dataset:
            mock_load_dataset.side_effect = FileNotFoundError("File not found")
            
            with self.assertRaises(FileNotFoundError):
                eval.test_dataset()

    def test_eval_dataset_success(self):
        """Test eval_dataset function with valid data."""
        test_data = [
            {
                "question": "What is AI?",
                "answer": "AI is artificial intelligence",
                "contexts": ["AI context"],
                "ground_truth": "AI is artificial intelligence"
            }
        ]
        
        result = eval.eval_dataset(test_data)
        
        # Verify it's a Dataset object
        self.assertIsInstance(result, Dataset)
        
        # Verify the data structure
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['question'], "What is AI?")
        self.assertEqual(result[0]['answer'], "AI is artificial intelligence")
        self.assertEqual(result[0]['contexts'], ["AI context"])
        self.assertEqual(result[0]['ground_truth'], "AI is artificial intelligence")

    def test_eval_dataset_empty_data(self):
        """Test eval_dataset function with empty data."""
        result = eval.eval_dataset([])
        
        self.assertIsInstance(result, Dataset)
        self.assertEqual(len(result), 0)

    def test_eval_dataset_missing_fields(self):
        """Test eval_dataset function with missing required fields."""
        incomplete_data = [
            {
                "question": "What is AI?",
                "answer": "AI is artificial intelligence"
                # Missing 'contexts' and 'ground_truth'
            }
        ]
        
        with self.assertRaises(KeyError):
            eval.eval_dataset(incomplete_data)

    def test_validate_json_valid_data(self):
        """Test validate_json function with valid JSON data."""
        valid_data = [
            {
                "question": "What is AI?",
                "answer": "AI is artificial intelligence",
                "contexts": ["AI context"]
            }
        ]
        
        with patch('builtins.print'):  # Suppress print output
            result = eval.validate_json(valid_data)
            
        self.assertTrue(result)

    def test_validate_json_invalid_data(self):
        """Test validate_json function with invalid JSON data."""
        invalid_data = [
            {
                "question": "What is AI?",
                "answer": "AI is artificial intelligence"
                # Missing required 'contexts' field
            }
        ]
        
        with patch('builtins.print'):  # Suppress print output
            result = eval.validate_json(invalid_data)
            
        self.assertFalse(result)

    def test_validate_json_wrong_structure(self):
        """Test validate_json function with wrong data structure."""
        wrong_structure = {
            "not": "an array"
        }
        
        with patch('builtins.print'):  # Suppress print output
            result = eval.validate_json(wrong_structure)
            
        self.assertFalse(result)

    def test_validate_json_additional_properties(self):
        """Test validate_json function with additional properties."""
        data_with_extra = [
            {
                "question": "What is AI?",
                "answer": "AI is artificial intelligence",
                "contexts": ["AI context"],
                "extra_field": "not allowed"
            }
        ]
        
        with patch('builtins.print'):  # Suppress print output
            result = eval.validate_json(data_with_extra)
            
        self.assertFalse(result)

    @patch('eval.evaluate')
    @patch('os.makedirs')
    def test_run_ragas_eval_success(self, mock_makedirs, mock_evaluate):
        """Test run_ragas_eval function with successful evaluation."""
        # Mock the evaluation results using MagicMock to properly handle dictionary behavior
        mock_eval_results = MagicMock()
        values_list = [0.9, 0.8, 0.85, 0.82, 0.75]
        mock_eval_results.values.return_value = values_list
        mock_eval_results.__len__.return_value = len(values_list)
        mock_eval_results.to_pandas.return_value = pd.DataFrame([{
            'faithfulness': 0.9,
            'answer_relevancy': 0.8,
            'context_recall': 0.85,
            'context_precision': 0.82,
            'answer_correctness': 0.75
        }])
        
        mock_evaluate.return_value = mock_eval_results
        
        # Mock dataset
        mock_dataset = MagicMock()
        
        with patch('builtins.open', mock_open()):
            with patch('json.dumps') as mock_json_dumps:
                mock_json_dumps.return_value = '{"test": "data"}'
                
                result = eval.run_ragas_eval(mock_dataset)
                
                # Verify evaluate was called
                mock_evaluate.assert_called_once_with(mock_dataset, eval.metrics)
                
                # Verify the result structure
                self.assertIn('results', result)
                self.assertIn('ragas_score', result)
                self.assertEqual(result['results'], mock_eval_results)
                # Verify the calculated score
                expected_score = sum(values_list) / len(values_list)
                self.assertEqual(result['ragas_score'], expected_score)

    @patch('eval.evaluate')
    def test_run_ragas_eval_evaluation_error(self, mock_evaluate):
        """Test run_ragas_eval function when evaluation fails."""
        mock_evaluate.side_effect = Exception("Evaluation failed")
        mock_dataset = MagicMock()
        
        with self.assertRaises(Exception):
            eval.run_ragas_eval(mock_dataset)

    @patch('eval.run_ragas_eval')
    @patch('eval.eval_dataset')
    @patch('eval.test_dataset')
    @patch('eval.validate_json')
    def test_process_file_success(self, mock_validate, mock_test_dataset, 
                                  mock_eval_dataset, mock_run_ragas):
        """Test process_file function with valid file processing."""
        # Setup mocks
        mock_validate.return_value = True
        
        mock_dataset = MagicMock()
        mock_df = pd.DataFrame({
            'question': ['What is AI?', 'What is ML?'],
            'ground_truth': ['AI answer', 'ML answer']
        })
        mock_dataset.to_pandas.return_value = mock_df
        mock_test_dataset.return_value = mock_dataset
        
        mock_eval_dataset.return_value = MagicMock()
        mock_run_ragas.return_value = {"test": "results"}
        
        json_content = [
            {"question": "What is AI?", "answer": "test", "contexts": ["test"]},
            {"question": "What is ML?", "answer": "test", "contexts": ["test"]}
        ]
        
        with patch('builtins.open', mock_open(read_data=json.dumps(json_content))):
            with patch('json.loads', return_value=json_content):
                with patch('json.dumps'):
                    with patch('builtins.print'):
                        result = eval.process_file('test_file.json')
                        
        # Verify all functions were called
        mock_validate.assert_called_once()
        mock_test_dataset.assert_called_once()
        mock_eval_dataset.assert_called_once()
        mock_run_ragas.assert_called_once()

    @patch('eval.validate_json')
    def test_process_file_invalid_json(self, mock_validate):
        """Test process_file function with invalid JSON."""
        mock_validate.return_value = False
        
        json_content = [{"invalid": "data"}]
        
        with patch('builtins.open', mock_open(read_data=json.dumps(json_content))):
            with patch('json.loads', return_value=json_content):
                with patch('builtins.print'):
                    result = eval.process_file('test_file.json')
                    
        # Should not proceed with evaluation when JSON is invalid
        mock_validate.assert_called_once()

    def test_process_file_question_mismatch(self):
        """Test process_file function when questions don't match."""
        json_content = [
            {"question": "Different question", "answer": "test", "contexts": ["test"]}
        ]
        
        mock_dataset = MagicMock()
        mock_df = pd.DataFrame({
            'question': ['What is AI?'],
            'ground_truth': ['AI answer']
        })
        mock_dataset.to_pandas.return_value = mock_df
        
        with patch('eval.validate_json', return_value=True):
            with patch('eval.test_dataset', return_value=mock_dataset):
                with patch('builtins.open', mock_open(read_data=json.dumps(json_content))):
                    with patch('json.loads', return_value=json_content):
                        with patch('builtins.print'):
                            result = eval.process_file('test_file.json')
                            
        # Should return False when questions don't match
        self.assertFalse(result)

    def test_process_file_file_not_found(self):
        """Test process_file function when file doesn't exist."""
        with patch('builtins.open', side_effect=FileNotFoundError()):
            with self.assertRaises(FileNotFoundError):
                eval.process_file('nonexistent_file.json')

    def test_process_file_invalid_json_format(self):
        """Test process_file function with malformed JSON."""
        with patch('builtins.open', mock_open(read_data='invalid json')):
            with patch('json.loads', side_effect=json.JSONDecodeError("msg", "doc", 0)):
                with self.assertRaises(json.JSONDecodeError):
                    eval.process_file('test_file.json')

    @patch('eval.process_file')
    @patch('os.path.isfile')
    @patch('os.path.abspath')
    @patch('os.path.expanduser')
    @patch('argparse.ArgumentParser.parse_args')
    @patch('eval.load_dotenv')
    def test_main_success(self, mock_load_dotenv, mock_parse_args, 
                         mock_expanduser, mock_abspath, mock_isfile, mock_process_file):
        """Test main function with valid arguments."""
        # Setup mocks
        mock_args = MagicMock()
        mock_args.file = 'test_file.json'
        mock_parse_args.return_value = mock_args
        
        mock_expanduser.return_value = 'expanded_path'
        mock_abspath.return_value = '/absolute/path/test_file.json'
        mock_isfile.return_value = True
        mock_process_file.return_value = None
        
        # Call main
        eval.main()
        
        # Verify calls
        mock_load_dotenv.assert_called_once_with(override=True)
        mock_expanduser.assert_called_once_with('test_file.json')
        mock_abspath.assert_called_once_with('expanded_path')
        mock_isfile.assert_called_once_with('/absolute/path/test_file.json')
        mock_process_file.assert_called_once_with('/absolute/path/test_file.json')

    @patch('os.path.isfile')
    @patch('os.path.abspath')
    @patch('os.path.expanduser')
    @patch('argparse.ArgumentParser.parse_args')
    @patch('eval.load_dotenv')
    def test_main_file_not_found(self, mock_load_dotenv, mock_parse_args,
                                 mock_expanduser, mock_abspath, mock_isfile):
        """Test main function when file doesn't exist."""
        # Setup mocks
        mock_args = MagicMock()
        mock_args.file = 'nonexistent_file.json'
        mock_parse_args.return_value = mock_args
        
        mock_expanduser.return_value = 'expanded_path'
        mock_abspath.return_value = '/absolute/path/nonexistent_file.json'
        mock_isfile.return_value = False
        
        with patch('builtins.print'):
            eval.main()
        
        # Verify file check was performed
        mock_isfile.assert_called_once_with('/absolute/path/nonexistent_file.json')

    def test_schema_definition(self):
        """Test that the JSON schema is properly defined."""
        # Verify schema exists and has correct structure
        self.assertIsInstance(eval.schema, dict)
        self.assertEqual(eval.schema['type'], 'array')
        self.assertIn('items', eval.schema)
        
        # Verify required fields
        items_schema = eval.schema['items']
        self.assertEqual(set(items_schema['required']), {'question', 'answer', 'contexts'})
        self.assertFalse(items_schema['additionalProperties'])

    def test_metrics_definition(self):
        """Test that metrics are properly defined."""
        self.assertIsInstance(eval.metrics, list)
        self.assertEqual(len(eval.metrics), 5)
        
        # Check that all expected metrics are present
        # Note: Ragas metric objects don't have __name__ attribute, so we check the class names
        metric_names = [metric.__class__.__name__.lower() for metric in eval.metrics]
        expected_metrics = ['faithfulness', 'answerrelevancy', 'contextrecall', 
                          'contextprecision', 'answercorrectness']
        for expected in expected_metrics:
            self.assertTrue(any(expected in name for name in metric_names))


class TestIntegration(unittest.TestCase):
    """Integration tests for the eval module."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.csv_file = os.path.join(self.test_dir, 'test_dataset.csv')
        self.json_file = os.path.join(self.test_dir, 'test_input.json')
        
        # Create test CSV file
        test_csv_data = pd.DataFrame({
            'question': ['What is AI?'],
            'contexts': [['AI is artificial intelligence']],
            'ground_truth': ['AI stands for artificial intelligence'],
            'Unnamed: 0': [0]
        })
        test_csv_data.to_csv(self.csv_file, index=False)
        
        # Create test JSON file
        test_json_data = [
            {
                "question": "What is AI?",
                "answer": "AI is artificial intelligence",
                "contexts": ["AI is artificial intelligence"]
            }
        ]
        with open(self.json_file, 'w') as f:
            json.dump(test_json_data, f)

    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.test_dir)

    @patch('eval.load_dataset')
    def test_integration_test_dataset_and_eval_dataset(self, mock_load_dataset):
        """Integration test for test_dataset and eval_dataset functions."""
        # Mock load_dataset to return our test data
        mock_dataset = MagicMock()
        mock_train_data = MagicMock()
        
        # Create a simple mock that returns our test data structure
        mock_dataset.__getitem__.return_value = mock_train_data
        mock_train_data.remove_columns.return_value = Dataset.from_dict({
            'question': ['What is AI?'],
            'contexts': [['AI is artificial intelligence']],
            'ground_truth': ['AI stands for artificial intelligence']
        })
        
        mock_load_dataset.return_value = mock_dataset
        
        # Test the integration
        test_data = eval.test_dataset()
        
        # Prepare data for eval_dataset
        eval_data = [
            {
                "question": "What is AI?",
                "answer": "AI is artificial intelligence",
                "contexts": ["AI is artificial intelligence"],
                "ground_truth": "AI stands for artificial intelligence"
            }
        ]
        
        result_dataset = eval.eval_dataset(eval_data)
        
        self.assertIsInstance(result_dataset, Dataset)
        self.assertEqual(len(result_dataset), 1)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)