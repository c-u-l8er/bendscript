defmodule TextClassifierExampleTest do
  use ExUnit.Case

  test "can train and classify text correctly with custom configuration" do
    # Define initial vocabulary and categories
    vocabulary = [
      "price",
      "cost",
      "money",
      "expensive",
      "cheap",
      "budget",
      "quality",
      "excellent",
      "poor",
      "good",
      "bad",
      "satisfied",
      "shipping",
      "delivery",
      "fast",
      "slow",
      "delay",
      "arrived",
      "return",
      "refund",
      "customer",
      "service",
      "help",
      "support"
    ]

    categories = ["price", "quality", "shipping", "service"]

    # Define custom configuration
    custom_config = %{
      key_terms: %{
        "price" => ["price", "cost", "money", "expensive", "cheap", "budget"],
        "quality" => ["quality", "excellent", "poor", "good", "bad", "satisfied"],
        "shipping" => ["shipping", "delivery", "fast", "slow", "delay", "arrived"],
        "service" => ["service", "customer", "help", "support", "refund", "return"]
      },
      boost_weights: %{
        # Higher weight for positive matches
        positive_match: 0.25,
        # Stronger penalty for negative matches
        negative_match: -0.2,
        # Stronger category-specific boost
        category_boost: 0.35,
        # Faster decay for unused words
        unused_decay: -0.02,
        # Higher multiplier for category-specific terms
        category_multiplier: 2.0
      },
      confidence_values: %{
        positive_match: 0.85,
        negative_match: 0.75,
        category_boost: 0.95,
        unused_decay: 0.35
      },
      parameters: %{
        # Higher initial weights
        initial_weight: 0.15,
        # Faster learning rate
        update_rate: 0.15,
        # Less momentum to allow faster adaptation
        momentum: 0.8,
        # Higher pruning threshold
        prune_threshold: 0.05
      }
    }

    # Create classifier with custom configuration
    classifier = TextClassifier.new(vocabulary, categories, custom_config)

    # Create training documents
    training_docs = [
      # Price related reviews
      TextClassifier.TextData.document(
        "The price is too expensive for what you get.",
        %{},
        "price"
      ),
      TextClassifier.TextData.document(
        "Very affordable and good value for money.",
        %{},
        "price"
      ),
      TextClassifier.TextData.document(
        "Cheaper than other options in the market.",
        %{},
        "price"
      ),
      TextClassifier.TextData.document(
        "The cost is reasonable for the quality.",
        %{},
        "price"
      ),

      # Quality related reviews
      TextClassifier.TextData.document(
        "Excellent quality product, very satisfied.",
        %{},
        "quality"
      ),
      TextClassifier.TextData.document(
        "Poor quality, broke after first use.",
        %{},
        "quality"
      ),
      TextClassifier.TextData.document(
        "The quality exceeded my expectations.",
        %{},
        "quality"
      ),
      TextClassifier.TextData.document(
        "Good build quality, feels durable.",
        %{},
        "quality"
      ),

      # Shipping related reviews
      TextClassifier.TextData.document(
        "Fast shipping, arrived ahead of schedule.",
        %{},
        "shipping"
      ),
      TextClassifier.TextData.document(
        "Delivery was delayed by several days.",
        %{},
        "shipping"
      ),
      TextClassifier.TextData.document(
        "The shipping was quick and well-packaged.",
        %{},
        "shipping"
      ),
      TextClassifier.TextData.document(
        "Slow delivery time, took forever to arrive.",
        %{},
        "shipping"
      ),

      # Customer service related reviews
      TextClassifier.TextData.document(
        "Customer service was very helpful.",
        %{},
        "service"
      ),
      TextClassifier.TextData.document(
        "Poor support experience, no help at all.",
        %{},
        "service"
      ),
      TextClassifier.TextData.document(
        "Excellent customer service and support team.",
        %{},
        "service"
      ),
      TextClassifier.TextData.document(
        "Refund process was easy with good service.",
        %{},
        "service"
      )
    ]

    # Train the classifier
    trained_classifier = TextClassifier.train(classifier, training_docs, 10)

    # Test classification
    test_cases = [
      {"This product is too expensive for what it offers.", "price"},
      {"The quality is excellent, I'm very satisfied with my purchase.", "quality"},
      {"Shipping was very fast, arrived in just two days.", "shipping"},
      {"Customer service helped me with my refund request.", "service"}
    ]

    # Test each case
    for {text, expected_category} <- test_cases do
      {category, _score} = TextClassifier.classify(trained_classifier, text)

      assert category == expected_category,
             "Expected '#{text}' to be classified as '#{expected_category}', but got '#{category}'"
    end

    # Verify model has important features
    important_words = TextClassifier.important_features(trained_classifier)
    assert length(important_words) > 0, "Model should have identified important features"

    # Verify that key words from the config are in the important features
    key_words =
      Enum.flat_map(custom_config.key_terms, fn {_category, words} -> words end)
      |> Enum.uniq()
      # Just check a few key words
      |> Enum.take(4)

    important_word_keys = Enum.map(important_words, fn {word, _weight} -> word end)

    for word <- key_words do
      assert word in important_word_keys, "Expected '#{word}' to be an important feature"
    end
  end

  test "classifier works with minimal configuration" do
    # Minimal vocabulary and categories
    vocabulary = ["good", "bad", "fast", "slow"]
    categories = ["positive", "negative"]

    # Create classifier with default configuration
    classifier = TextClassifier.new(vocabulary, categories)

    # Simple training data
    training_docs = [
      TextClassifier.TextData.document("This is good.", %{}, "positive"),
      TextClassifier.TextData.document("That was bad.", %{}, "negative"),
      TextClassifier.TextData.document("Very fast service.", %{}, "positive"),
      TextClassifier.TextData.document("Too slow response.", %{}, "negative")
    ]

    # Train classifier
    trained_classifier = TextClassifier.train(classifier, training_docs, 5)

    # Test cases
    {category, _} = TextClassifier.classify(trained_classifier, "It was good and fast")
    assert category == "positive"

    {category, _} = TextClassifier.classify(trained_classifier, "Too bad and slow")
    assert category == "negative"
  end
end
