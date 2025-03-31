defmodule TextClassifierExampleTest do
  use ExUnit.Case

  test "can train and classify text correctly" do
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

    # Create classifier
    classifier = TextClassifier.new(vocabulary, categories)

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

    # Verify some key words are in the important features
    key_words = ["price", "quality", "shipping", "service"]
    important_word_keys = Enum.map(important_words, fn {word, _weight} -> word end)

    for word <- key_words do
      assert word in important_word_keys, "Expected '#{word}' to be an important feature"
    end
  end
end
