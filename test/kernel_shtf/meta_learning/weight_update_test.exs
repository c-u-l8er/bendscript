defmodule WeightUpdateTest do
  use ExUnit.Case
  import EmergenceTestHelper

  @moduledoc """
  Tests specifically for verifying that weights are properly updated during training
  """

  test "classifier weights change after training" do
    # Create a simple classifier
    vocabulary = ["good", "bad", "like", "dislike"]
    categories = ["positive", "negative"]
    classifier = TextClassifier.new(vocabulary, categories)

    # Extract initial weights
    initial_weights = extract_parameter_values(classifier.model.parameters)

    # Create simple training documents with explicit features
    training_docs = [
      # Make sure we include original_text to fix issues with text extraction
      TextClassifier.TextData.document("good like", %{original_text: "good like"}, "positive"),
      TextClassifier.TextData.document("bad dislike", %{original_text: "bad dislike"}, "negative"),
      # Add a few more variations to ensure learning
      TextClassifier.TextData.document("very good", %{original_text: "very good"}, "positive"),
      TextClassifier.TextData.document("really bad", %{original_text: "really bad"}, "negative")
    ]

    # Train with multiple epochs to ensure weights change
    trained_classifier = TextClassifier.train(classifier, training_docs, 5)

    # Extract updated weights
    updated_weights = extract_parameter_values(trained_classifier.model.parameters)

    # Print weights for debugging if needed
    IO.puts("Initial weights: #{inspect(initial_weights)}")
    IO.puts("Updated weights: #{inspect(updated_weights)}")

    # Verify weights changed
    assert initial_weights != updated_weights,
      "Weights should change during training"

    # Verify positive words have higher weights for positive category
    {_category, _score} = TextClassifier.classify(trained_classifier, "good")

    # Extract specific word weights
    good_weight = Map.get(updated_weights, "good", 0)
    bad_weight = Map.get(updated_weights, "bad", 0)

    # Check that positive words have higher weights than negative words
    assert good_weight > bad_weight,
      "Positive words should have higher weights than negative words"
  end

  test "features are properly extracted from text" do
    # Create vocabulary
    vocabulary = ["good", "bad", "happy", "sad"]

    # Test text
    text = "I am very good and happy today"

    # Extract features
    features = TextClassifier.extract_features(text, vocabulary)

    # Verify counts are correct
    assert Map.get(features, "good") == 1
    assert Map.get(features, "happy") == 1
    assert Map.get(features, "bad") == 0
    assert Map.get(features, "sad") == 0
  end

  test "original_text is preserved in features" do
    # Create vocabulary
    vocabulary = ["good", "bad"]

    # Test text with original_text field
    text = %{original_text: "this is good"}

    # Extract features
    features = TextClassifier.extract_features(text, vocabulary)

    # Verify original_text is preserved
    assert Map.get(features, :original_text) == "this is good"
  end
end
