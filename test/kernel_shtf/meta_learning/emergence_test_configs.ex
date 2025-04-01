defmodule EmergenceTestConfigs do
  @moduledoc """
  Configuration data for the emergence tests
  """

  # Basic vocabulary and categories
  def basic_vocabulary, do: ["good", "bad", "like", "dislike", "happy", "sad"]
  def basic_categories, do: ["positive", "negative"]

  # Expanded vocabulary and categories
  def expanded_vocabulary do
    basic_vocabulary() ++ ["excellent", "terrible", "love", "hate",
      "recommend", "avoid", "fast", "slow",
      "expensive", "cheap", "beautiful", "ugly"]
  end

  def expanded_categories do
    ["very_positive", "positive", "neutral", "negative", "very_negative"]
  end

  # Reborn vocabulary
  def reborn_vocabulary do
    expanded_vocabulary() ++ [
      "wonderful", "amazing", "disappointing", "frustrating",
      "perfect", "broken", "reliable", "flimsy",
      "value", "overpriced", "bargain", "waste"
    ]
  end

  # Expanded configuration
  def expanded_config do
    %{
      key_terms: %{
        "very_positive" => ["excellent", "love", "recommend"],
        "positive" => ["good", "happy", "like", "recommend"],
        "neutral" => [],
        "negative" => ["bad", "sad", "dislike", "avoid"],
        "very_negative" => ["terrible", "hate", "avoid"]
      },
      boost_weights: %{
        positive_match: 0.3,
        negative_match: -0.25,
        category_boost: 0.4,
        unused_decay: -0.03,
        category_multiplier: 2.5
      }
    }
  end

  # Create optimal config from learned weights
  def create_optimal_config(optimal_weights) do
    # Prepare key terms for optimal configuration
    filtered_weights = optimal_weights
      |> Enum.filter(fn {_, w} -> abs(w) > 0.1 end)

    key_terms_by_category = filtered_weights
      |> Enum.group_by(
        fn {word, weight} ->
          cond do
            weight > 0.3 -> "very_positive"
            weight > 0.1 -> "positive"
            weight > -0.1 -> "neutral"
            weight > -0.3 -> "negative"
            true -> "very_negative"
          end
        end,
        fn {word, _} -> word end
      )

    # Ensure we have at least some entries in each category
    key_terms_map = Enum.reduce(expanded_categories(), %{}, fn category, acc ->
      terms = Map.get(key_terms_by_category, category, [])
      # Add default terms if empty
      terms = if terms == [], do: [category], else: terms
      Map.put(acc, category, terms)
    end)

    # Initialize with optimal configuration
    %{
      key_terms: key_terms_map,
      boost_weights: %{
        positive_match: 0.35,
        negative_match: -0.3,
        category_boost: 0.5,
        unused_decay: -0.02,
        category_multiplier: 3.0
      },
      confidence_values: %{
        positive_match: 0.9,
        negative_match: 0.85,
        category_boost: 0.95,
        unused_decay: 0.4
      },
      parameters: %{
        # Use learned knowledge for initial settings
        initial_weight: 0.2,
        update_rate: 0.15,
        momentum: 0.8,
        prune_threshold: 0.03
      },
      priorities: %{
        confidence_update: 0.8,
        oscillation_damping: 0.85,
        feature_boosting: 0.9,
        feature_pruning: 0.6
      }
    }
  end

  # Create a function that initializes parameters based on learned weights
  def create_learned_initialization(optimal_weights) do
    fn word ->
      weight = Map.get(optimal_weights, word, 0.1)

      # Analyze word importance and assign appropriate update rate
      update_rate = cond do
        abs(weight) > 0.5 -> 0.2  # Important words learn faster
        abs(weight) > 0.2 -> 0.15 # Moderately important words
        true -> 0.1              # Less important words
      end

      KernelShtf.MetaLearning.new_parameter(word, weight, update_rate)
    end
  end
end
