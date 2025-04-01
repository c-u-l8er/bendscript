defmodule EmergenceTestAssertions do
  @moduledoc """
  Assertion functions for the emergence test
  """

  import ExUnit.Assertions
  import EmergenceTestHelper

  # Verify seed structure
  def verify_seed(classifier, vocabulary) do
    assert length(classifier.model.parameters) == length(vocabulary)
    assert length(classifier.model.heuristics) > 0
  end

  # Verify learning occurred
  def verify_learning(original_classifier, trained_classifier, test_text, expected_category) do
    # Check that weights changed
    original_weights = extract_parameter_values(original_classifier.model.parameters)
    trained_weights = extract_parameter_values(trained_classifier.model.parameters)
    assert original_weights != trained_weights, "Expected weights to change after training"

    # Check classification works as expected
    {category, _} = TextClassifier.classify(trained_classifier, test_text)
    assert category == expected_category,
      "Classification test failed: expected #{expected_category}, got #{category}"
  end

  # Verify canopy test cases
  def verify_canopy_tests(classifier, test_cases) do
    correct_count = count_correct(classifier, test_cases)
    assert correct_count > 0, "Expected to get at least some test cases correct"
  end

  # Verify meta-learning improvement
  def verify_meta_learning(pre_classifier, post_classifier, test_cases) do
    pre_correct = count_correct(pre_classifier, test_cases)
    post_correct = count_correct(post_classifier, test_cases)

    assert post_correct >= pre_correct,
      "Meta-learning should improve or maintain performance"
  end

  # Verify important feature identification
  def verify_feature_identification(classifier, important_words) do
    features = TextClassifier.important_features(classifier)
    feature_words = Enum.map(features, fn {word, _weight} -> word end)

    assert Enum.any?(important_words, &(&1 in feature_words)),
      "At least one important word should be identified as a key feature"
  end

  # Verify reborn classifier performance
  def verify_reborn_performance(reborn_classifier, seed_classifier, test_cases) do
    reborn_correct = count_correct(reborn_classifier, test_cases)
    seed_correct = count_correct(seed_classifier, test_cases)

    assert reborn_correct >= seed_correct,
      "Reborn classifier should perform at least as well as seed classifier"

    # Print results
    IO.puts("\n--- Emergent Foundation Model Evolution Results ---")
    IO.puts("Seed classifier → Branch classifier → Network classifier → Meta classifier → Ecosystem classifier → Reborn classifier")
    IO.puts("Complex test accuracy: #{reborn_correct}/#{length(test_cases)} (Reborn) vs #{seed_correct}/#{length(test_cases)} (Seed)")
    IO.puts("The system has successfully evolved from 0→1 to A→Z and back to a new beginning\n")
  end

  # Helper to count correct classifications
  defp count_correct(classifier, test_cases) do
    Enum.count(test_cases, fn {text, expected} ->
      {category, _} = TextClassifier.classify(classifier, %{original_text: text})
      category == expected
    end)
  end
end
