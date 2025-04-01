defmodule EmergenceTestData do
  @moduledoc """
  Test data for the meta-learning emergence tests
  """

  # Simple test documents
  def simple_docs do
    [
      %{
        variant: :document,
        text: "good happy like",
        features: %{
          "good" => 1,
          "happy" => 1,
          "like" => 1,
          original_text: "good happy like"
        },
        category: "positive"
      },
      %{
        variant: :document,
        text: "bad sad dislike",
        features: %{
          "bad" => 1,
          "sad" => 1,
          "dislike" => 1,
          original_text: "bad sad dislike"
        },
        category: "negative"
      },
      %{
        variant: :document,
        text: "very good and happy",
        features: %{
          "good" => 1,
          "happy" => 1,
          original_text: "very good and happy"
        },
        category: "positive"
      },
      %{
        variant: :document,
        text: "really bad and sad",
        features: %{
          "bad" => 1,
          "sad" => 1,
          original_text: "really bad and sad"
        },
        category: "negative"
      }
    ]
  end

  # Complex test documents with mixed signals
  def complex_docs do
    [
      # Simple examples
      TextClassifier.TextData.document("good happy like", %{original_text: "good happy like"}, "positive"),
      TextClassifier.TextData.document("bad sad dislike", %{original_text: "bad sad dislike"}, "negative"),

      # Complex examples with mixed signals
      TextClassifier.TextData.document("good product but bad service",
        %{original_text: "good product but bad service"}, "negative"),
      TextClassifier.TextData.document("bad experience but happy with resolution",
        %{original_text: "bad experience but happy with resolution"}, "positive"),

      # Examples with negation
      TextClassifier.TextData.document("not good at all", %{original_text: "not good at all"}, "negative"),
      TextClassifier.TextData.document("not bad actually", %{original_text: "not bad actually"}, "positive")
    ]
  end

  # Expanded test documents for more categories
  def expanded_docs do
    [
      # Very positive
      TextClassifier.TextData.document("excellent product, I love it and highly recommend",
        %{original_text: "excellent product, I love it and highly recommend"}, "very_positive"),
      TextClassifier.TextData.document("absolutely love this, it's beautiful and fast",
        %{original_text: "absolutely love this, it's beautiful and fast"}, "very_positive"),

      # Positive
      TextClassifier.TextData.document("good product, I like it",
        %{original_text: "good product, I like it"}, "positive"),
      TextClassifier.TextData.document("happy with my purchase",
        %{original_text: "happy with my purchase"}, "positive"),

      # Neutral
      TextClassifier.TextData.document("average product, not good or bad",
        %{original_text: "average product, not good or bad"}, "neutral"),
      TextClassifier.TextData.document("expected more but it's okay",
        %{original_text: "expected more but it's okay"}, "neutral"),

      # Negative
      TextClassifier.TextData.document("bad experience, I dislike it",
        %{original_text: "bad experience, I dislike it"}, "negative"),
      TextClassifier.TextData.document("sad that I bought this, avoid it",
        %{original_text: "sad that I bought this, avoid it"}, "negative"),

      # Very negative
      TextClassifier.TextData.document("terrible product, I hate it completely",
        %{original_text: "terrible product, I hate it completely"}, "very_negative"),
      TextClassifier.TextData.document("absolutely ugly and expensive waste of money",
        %{original_text: "absolutely ugly and expensive waste of money"}, "very_negative")
    ]
  end

  # Negation test documents
  def negation_docs do
    [
      TextClassifier.TextData.document("not good", %{original_text: "not good"}, "negative"),
      TextClassifier.TextData.document("not bad", %{original_text: "not bad"}, "positive"),
      TextClassifier.TextData.document("not happy", %{original_text: "not happy"}, "negative"),
      TextClassifier.TextData.document("not sad", %{original_text: "not sad"}, "positive"),
      TextClassifier.TextData.document("don't like", %{original_text: "don't like"}, "negative"),
      TextClassifier.TextData.document("don't dislike", %{original_text: "don't dislike"}, "positive")
    ]
  end

  # Context test documents
  def context_docs do
    [
      TextClassifier.TextData.document(
        "product quality is excellent",
        %{original_text: "product quality is excellent"},
        "very_positive"
      ),
      TextClassifier.TextData.document(
        "service is slow",
        %{original_text: "service is slow"},
        "negative"
      ),
      TextClassifier.TextData.document(
        "price is expensive but quality is good",
        %{original_text: "price is expensive but quality is good"},
        "positive"
      ),
      TextClassifier.TextData.document(
        "delivery was fast but product is terrible",
        %{original_text: "delivery was fast but product is terrible"},
        "negative"
      )
    ]
  end

  # Minimal training set
  def minimal_training do
    [
      # Just 5 examples covering the range
      TextClassifier.TextData.document(
        "wonderful amazing perfect product",
        %{original_text: "wonderful amazing perfect product"},
        "very_positive"
      ),
      TextClassifier.TextData.document(
        "good reliable value",
        %{original_text: "good reliable value"},
        "positive"
      ),
      TextClassifier.TextData.document(
        "neither good nor bad",
        %{original_text: "neither good nor bad"},
        "neutral"
      ),
      TextClassifier.TextData.document(
        "disappointing and frustrating",
        %{original_text: "disappointing and frustrating"},
        "negative"
      ),
      TextClassifier.TextData.document(
        "terrible broken waste of money",
        %{original_text: "terrible broken waste of money"},
        "very_negative"
      )
    ]
  end

  # Canopy test cases
  def canopy_test_cases do
    [
      {"This product is excellent but expensive", "positive"},
      {"Not terrible, but definitely not good either", "neutral"},
      {"I don't hate it, but I can't recommend it", "neutral"},
      {"Good product with terrible customer service", "negative"},
      {"The price is cheap but quality is excellent", "very_positive"}
    ]
  end

  # Negation test cases
  def negation_test_cases do
    [
      {"This is not good", "negative"},
      {"This is not bad", "positive"},
      {"I do not hate it", "positive"},
      {"I do not love it", "negative"}
    ]
  end

  # Context test cases
  def context_test_cases do
    [
      {"The product quality is excellent but the price is expensive", "positive"},
      {"Service is fast but product quality is bad", "negative"},
      {"Cheap price but terrible quality", "negative"},
      {"Slow delivery but excellent product", "positive"}
    ]
  end

  # Final test cases
  def final_test_cases do
    [
      {"This product is not bad, actually quite good quality", "positive"},
      {"The price is expensive but the product quality is excellent", "very_positive"},
      {"Fast delivery service but product quality is terrible", "negative"},
      {"Not happy with this overpriced and disappointing purchase", "very_negative"},
      {"It's neither wonderful nor terrible, just a reliable bargain", "neutral"}
    ]
  end
end
