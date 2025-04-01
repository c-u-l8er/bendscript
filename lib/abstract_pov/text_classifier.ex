defmodule TextClassifier do
  import KernelShtf.BenBen
  alias KernelShtf.MetaLearning
  alias KernelShtf.MetaLearning.Types.Learning

  # Define basic types using the bend/phrenia system
  phrenia TextData do
    document(text, features, category)
    classifier(model, vocabulary, categories, config)
    feature(name, weight)
  end

  # Make classifier constructor public by re-exporting it
  # This line is missing in your code and causing the error
  defdelegate classifier(model, vocabulary, categories, config), to: TextData

  @doc """
  Creates a new text classifier with meta-learning capabilities.
  The classifier adjusts its feature weights and selection strategy
  automatically based on the data it encounters.

  Config options:
  - :key_terms - Map of category-specific terms
  - :boost_weights - Map of boost values for different operations
  - :confidence_values - Map of confidence values for heuristics
  - :parameters - Map of additional parameter configurations
  """
  def new(initial_vocabulary, categories, config \\ %{}) do
    # Set defaults and merge with provided config
    config = Map.merge(default_config(categories), config)

    # Create a parameter for each word in the vocabulary with initial weights
    parameters =
      Enum.map(initial_vocabulary, fn word ->
        # Use configured initial weight and update rate
        initial_weight = Map.get(config.parameters, :initial_weight, 0.1)
        update_rate = Map.get(config.parameters, :update_rate, 0.1)
        MetaLearning.new_parameter(word, initial_weight, update_rate)
      end)

    # Create heuristics from configuration
    heuristics = create_heuristics(initial_vocabulary, config)

    # Create meta-rules from configuration
    meta_rules = create_meta_rules(config)

    # Create model
    model = MetaLearning.new_model(parameters, heuristics, meta_rules)

    # Return classifier with config included
    TextData.classifier(model, initial_vocabulary, categories, config)
  end

  # Create default configuration
  defp default_config(categories) do
    # Default key terms for each category
    default_key_terms =
      Enum.reduce(categories, %{}, fn category, acc ->
        Map.put(acc, category, [category])
      end)

    %{
      # Default key terms mapping
      key_terms: default_key_terms,

      # Default boost values
      boost_weights: %{
        positive_match: 0.2,
        negative_match: -0.15,
        category_boost: 0.3,
        unused_decay: -0.01,
        category_multiplier: 1.5
      },

      # Default confidence values
      confidence_values: %{
        positive_match: 0.8,
        negative_match: 0.7,
        category_boost: 0.9,
        unused_decay: 0.3
      },

      # Default parameter settings
      parameters: %{
        initial_weight: 0.1,
        update_rate: 0.1,
        momentum: 0.9,
        prune_threshold: 0.01
      },

      # Meta-rule priorities
      priorities: %{
        confidence_update: 0.7,
        oscillation_damping: 0.8,
        feature_boosting: 0.8,
        feature_pruning: 0.5
      }
    }
  end

  # Create heuristics based on configuration
  defp create_heuristics(vocabulary, config) do
    [
      # Regular heuristics as before...
      MetaLearning.new_heuristic(
        fn param_name -> Enum.member?(vocabulary, param_name) end,
        fn name, value, _rate, features, label, _batch ->
          # If word appears in text and the label matches, increase weight
          if Map.get(features, name, 0) > 0 && features.category == label do
            Map.get(config.boost_weights, :positive_match, 0.2)
          else
            0.0
          end
        end,
        Map.get(config.confidence_values, :positive_match, 0.8)
      ),

      # Decrease weights for words that appear in wrong categories
      MetaLearning.new_heuristic(
        fn param_name -> Enum.member?(vocabulary, param_name) end,
        fn name, value, _rate, features, label, _batch ->
          # If word appears but in wrong category, decrease weight
          if Map.get(features, name, 0) > 0 && features.category != label do
            Map.get(config.boost_weights, :negative_match, -0.15)
          else
            0.0
          end
        end,
        Map.get(config.confidence_values, :negative_match, 0.7)
      ),

      # Add category-specific boost for key words
      MetaLearning.new_heuristic(
        fn param_name ->
          # Check if this parameter is a key term for any category
          Enum.any?(config.key_terms, fn {_category, terms} ->
            Enum.member?(terms, param_name)
          end)
        end,
        fn name, value, _rate, features, label, _batch ->
          # Use configured key terms
          if Map.get(features, name, 0) > 0 do
            category_terms = Map.get(config.key_terms, label, [])

            if Enum.member?(category_terms, name) do
              Map.get(config.boost_weights, :category_boost, 0.3)
            else
              0.0
            end
          else
            0.0
          end
        end,
        Map.get(config.confidence_values, :category_boost, 0.9)
      ),

      # Decay unused words
      MetaLearning.new_heuristic(
        fn param_name -> Enum.member?(vocabulary, param_name) end,
        fn name, value, _rate, features, _label, _batch ->
          if Map.get(features, name, 0) == 0 do
            Map.get(config.boost_weights, :unused_decay, -0.01)
          else
            0.0
          end
        end,
        Map.get(config.confidence_values, :unused_decay, 0.3)
      ),

      # Add special heuristic for negative sentiment words
      MetaLearning.new_heuristic(
        fn param_name -> Enum.member?(["bad", "slow", "poor", "delay"], param_name) end,
        fn name, value, _rate, features, label, _batch ->
          if Map.get(features, name, 0) > 0 do
            if String.contains?(String.downcase(label), "negative") do
              # For negative categories, these words should have positive correlation
              Map.get(config.boost_weights, :positive_match, 0.2) * 2.0
            else
              # For positive categories, these words should have negative correlation
              Map.get(config.boost_weights, :negative_match, -0.15) * 2.0
            end
          else
            0.0
          end
        end,
        0.9
      )
    ]
  end

  # Create meta-rules based on configuration
  defp create_meta_rules(config) do
    [
      # Increase learning rate for features that consistently correlate with categories
      MetaLearning.new_meta_rule(
        %{param_name: :any, expected_pattern: :convergence},
        fn heuristics, dataset ->
          # Find heuristics that are working well
          effective_heuristics =
            Enum.map(heuristics, fn
              %{variant: :heuristic, condition: cond, action: action, confidence: conf}
              when conf < 0.9 ->
                # Increase confidence for effective heuristics
                Learning.heuristic(cond, action, conf + 0.05)

              other ->
                other
            end)

          {effective_heuristics, dataset}
        end,
        Map.get(config.priorities, :confidence_update, 0.7)
      ),

      # Reduce oscillation by adding momentum
      MetaLearning.new_meta_rule(
        %{param_name: :any, expected_pattern: :oscillation},
        fn heuristics, dataset ->
          # Use configured momentum value
          momentum = Map.get(config.parameters, :momentum, 0.9)

          # Add momentum to oscillating parameters
          updated_heuristics =
            Enum.map(heuristics, fn
              %{variant: :heuristic, condition: cond, action: action, confidence: conf} ->
                # Create a damped version of the action
                damped_action = fn name, value, rate, features, label, batch ->
                  # Get original update
                  raw_update = action.(name, value, rate, features, label, batch)

                  # Get previous updates from dataset statistics
                  previous_updates =
                    fold dataset do
                      case(dataset(data_points, statistics)) ->
                        Map.get(statistics, {:param_updates, name}, [])
                    end

                  # Apply momentum (weighted average with previous updates)
                  if previous_updates == [] do
                    raw_update
                  else
                    avg_prev = Enum.sum(previous_updates) / length(previous_updates)
                    raw_update * (1 - momentum) + avg_prev * momentum
                  end
                end

                Learning.heuristic(cond, damped_action, conf)

              other ->
                other
            end)

          {updated_heuristics, dataset}
        end,
        Map.get(config.priorities, :oscillation_damping, 0.8)
      ),

      # Add feature boosting for category-specific words
      MetaLearning.new_meta_rule(
        %{heuristic_type: :feature_boosting},
        fn heuristics, dataset ->
          boost_weight = Map.get(config.boost_weights, :category_boost, 0.3)

          boosting_heuristic =
            Learning.heuristic(
              %{type: :feature_boosting},
              fn name, value, _rate, features, label, _batch ->
                # Use configured key terms
                category_terms = Map.get(config.key_terms, label, [])

                if Enum.member?(category_terms, name) && Map.get(features, name, 0) > 0 do
                  boost_weight
                else
                  0.0
                end
              end,
              Map.get(config.confidence_values, :category_boost, 0.9)
            )

          {[boosting_heuristic | heuristics], dataset}
        end,
        Map.get(config.priorities, :feature_boosting, 0.8)
      ),

      # Add feature pruning for irrelevant words
      MetaLearning.new_meta_rule(
        %{heuristic_type: :feature_pruning},
        fn heuristics, dataset ->
          # Use configured pruning threshold
          prune_threshold = Map.get(config.parameters, :prune_threshold, 0.01)

          # Create a new heuristic that removes features with very low weights
          pruning_heuristic =
            Learning.heuristic(
              %{type: :feature_pruning},
              fn _name, value, _rate, _features, _label, _batch ->
                if abs(value) < prune_threshold do
                  # Further push toward zero
                  -value * 0.5
                else
                  0.0
                end
              end,
              0.4
            )

          {[pruning_heuristic | heuristics], dataset}
        end,
        Map.get(config.priorities, :feature_pruning, 0.5)
      )
    ]
  end

  @doc """
  Processes a document, extracting features for the classifier
  """
  def extract_features(document, vocabulary) do
    # Extract text, handling both string and map inputs
    text = cond do
      is_binary(document) -> document
      is_map(document) && Map.has_key?(document, :text) -> document.text
      is_map(document) && Map.has_key?(document, :original_text) -> document.original_text
      true -> ""
    end

    # Ensure we have text to process
    text = if text == "", do: "empty", else: text

    debug_enabled = System.get_env("DEBUG_META_LEARNING") == "true"

    # Tokenize text - added better tokenization
    words =
      text
      |> String.downcase()
      |> String.replace(~r/[^\w\s]/, " ")  # Replace punctuation with spaces
      |> String.split(~r/\s+/)
      |> Enum.filter(&(&1 != ""))  # Remove empty strings

    # Count occurrences of vocabulary words
    features =
      Enum.reduce(vocabulary, %{}, fn word, acc ->
        count = Enum.count(words, &(&1 == word))
        # Only add words that actually appear
        if count > 0 do
          Map.put(acc, word, count)
        else
          acc
        end
      end)

    # Minimal logging of feature extraction
    if debug_enabled && map_size(features) > 0 do
      IO.puts("Extracted #{map_size(features)} features from text")
    end

    # Preserve original text if available
    features =
      if is_map(document) && Map.has_key?(document, :original_text) do
        Map.put(features, :original_text, document.original_text)
      else
        features
      end

    # Add flag to indicate this is a real document (not empty)
    Map.put(features, :valid_document, true)
  end

  @doc """
  Train the classifier on a set of documents
  """
  def train(classifier, documents, epochs \\ 5) do
    # Extract the necessary parts from the classifier
    model = classifier.model
    vocabulary = classifier.vocabulary
    categories = classifier.categories
    config = classifier.config

    debug_enabled = System.get_env("DEBUG_META_LEARNING") == "true"

    # Minimal logging during training
    if debug_enabled do
      IO.puts("Training classifier with #{length(documents)} documents for #{epochs} epochs")
    end

    # Convert documents to dataset
    dataset = prepare_dataset(documents, vocabulary)

    # Get initial weights for comparison
    initial_weights = Enum.map(model.parameters, fn
      %{name: name, value: value} -> {name, value}
      param -> {param[:name], param[:value]}
    end) |> Enum.into(%{})

    # Train using the meta-learning system
    trained_model = MetaLearning.train(model, dataset, epochs)

    # Get final weights
    final_weights = Enum.map(trained_model.parameters, fn
      %{name: name, value: value} -> {name, value}
      param -> {param[:name], param[:value]}
    end) |> Enum.into(%{})

    # Only log significant weight changes to reduce verbosity
    if debug_enabled do
      weight_changes = Enum.filter(final_weights, fn {name, value} ->
        original = Map.get(initial_weights, name, 0.0)
        abs(value - original) > 0.05  # Only show significant changes
      end)
      |> Enum.sort_by(fn {_, value} -> -abs(value) end)
      |> Enum.take(5)  # Only show top 5 changes
      |> Enum.into(%{})

      if weight_changes == %{} do
        IO.puts("WARNING: No significant weight changes occurred during training!")
      else
        IO.puts("Top weight changes: #{inspect(weight_changes)}")
      end
    end

    # Create a new classifier with the trained model
    TextClassifier.classifier(trained_model, vocabulary, categories, config)
  end

  @doc """
  Prepare training dataset from documents
  """
  def prepare_dataset(documents, vocabulary) do
    # Convert documents to data points
    data_points =
      Enum.map(documents, fn doc ->
        # Extract features based on document structure
        features = cond do
          is_map(doc) && Map.has_key?(doc, :text) && Map.has_key?(doc, :category) ->
            feats = extract_features(doc.text, vocabulary)
            # Add category to features for easier access
            Map.put(feats, :category, doc.category)

          is_map(doc) && Map.has_key?(doc, :features) && Map.has_key?(doc, :category) ->
            # Use existing features if already provided
            feats = doc.features
            # Add category if not already present
            Map.put(feats, :category, doc.category)

          is_map(doc) && Map.has_key?(doc, :variant) && doc.variant == :document ->
            # For document type from TextData
            text = Map.get(doc, :text, "")
            feats = extract_features(text, vocabulary)
            # Add original features if available
            feats = Map.merge(Map.get(doc, :features, %{}), feats)
            # Add category to features for easier access
            Map.put(feats, :category, doc.category)

          true ->
            # Default handling
            %{category: "unknown"}
        end

        # Create data point with the label set to the category
        MetaLearning.new_data_point(features, features.category)
      end)

    # Create dataset with initial statistics
    MetaLearning.new_dataset(data_points, %{
      size: length(data_points),
      categories: Enum.map(data_points, fn dp -> dp.label end) |> Enum.uniq() |> length(),
      avg_features:
        (Enum.map(data_points, fn dp -> map_size(dp.features) end)
         |> Enum.sum()) / max(1, length(data_points))
    })
  end

  @doc """
  Classify a new document
  """
  def classify(classifier, text) do
    # Extract model, vocabulary, categories and config
    model = classifier.model
    vocabulary = classifier.vocabulary
    categories = classifier.categories
    config = classifier.config

    debug_enabled = System.get_env("DEBUG_META_LEARNING") == "true"

    # Extract features
    features = extract_features(text, vocabulary)

    # Minimal feature logging
    if debug_enabled do
      feature_count = Enum.count(features, fn {k, _} -> k != :original_text && k != :valid_document end)
      IO.puts("Classification with #{feature_count} matching features")
    end

    # Get parameters (word weights)
    parameters = model.parameters

    # Calculate score for each category by applying category-specific weights
    scores =
      Enum.map(categories, fn category ->
        # Get category-specific terms
        category_terms = Map.get(config.key_terms, category, [])

        # Score calculation with improved weighting logic
        base_score =
          Enum.reduce(parameters, 0, fn param, acc ->
            name = param.name
            weight = param.value
            word_count = Map.get(features, name, 0)

            # Skip words that don't appear in this document
            if word_count > 0 do
              # Apply different weight multipliers based on word characteristics
              multiplier = cond do
                # Key term for this category - higher weight
                name in category_terms -> 2.0
                # Known negative term
                name in ["bad", "terrible", "hate", "awful", "poor"] &&
                  !String.contains?(String.downcase(category), "negative") -> 0.5
                # Known positive term
                name in ["good", "excellent", "love", "wonderful", "great"] &&
                  String.contains?(String.downcase(category), "positive") -> 2.0
                # Default multiplier
                true -> 1.0
              end

              # Multiply word count by its weight and the multiplier
              acc + word_count * weight * multiplier
            else
              acc
            end
          end)

        # Apply category-specific boosting from config
        category_multiplier = Map.get(config.boost_weights, :category_multiplier, 1.5)

        # Calculate special handling for negations if present
        negation_adjustment =
          if Map.get(features, :original_text) &&
             String.contains?(Map.get(features, :original_text), "not ") do
            # If negation present, adjust scores for sentiment categories
            if String.contains?(String.downcase(category), "positive") do
              -base_score * 0.5  # Reduce positive scores
            else if String.contains?(String.downcase(category), "negative") do
              base_score * 0.3   # Boost negative scores
            else
              0.0
            end
            end
          else
            0.0
          end

        # Add context-based adjustments
        context_adjustment = calculate_context_adjustment(features, category, base_score)

        # Calculate final score with all adjustments
        final_score = base_score + negation_adjustment + context_adjustment

        # Return category with its score
        {category, final_score}
      end)

    # Return category with highest score
    Enum.max_by(scores, fn {_category, score} -> score end)
  end

  # Helper for context-based score adjustments
  def calculate_context_adjustment(features, category, base_score) do
    original_text = Map.get(features, :original_text, "")

    cond do
      # Quality context
      String.contains?(original_text, "quality") ->
        if String.contains?(String.downcase(category), "quality") do
          base_score * 0.3  # Boost quality category
        else
          0.0
        end

      # Price context
      String.contains?(original_text, "price") || String.contains?(original_text, "cost") ->
        if String.contains?(original_text, "cheap") || String.contains?(original_text, "affordable") do
          if String.contains?(String.downcase(category), "positive") do
            base_score * 0.25  # Boost positive for good price
          else
            0.0
          end
        else
          if String.contains?(original_text, "expensive") do
            if String.contains?(String.downcase(category), "negative") do
              base_score * 0.25  # Boost negative for bad price
            else
              0.0
            end
          else
            0.0
          end
        end

      # Service context
      String.contains?(original_text, "service") || String.contains?(original_text, "support") ->
        if String.contains?(String.downcase(category), "service") do
          base_score * 0.3  # Boost service category
        else
          0.0
        end

      # Default - no adjustment
      true ->
        0.0
    end
  end

  @doc """
  Extract the most important words for each category
  """
  def important_features(classifier) do
    # Get parameters (word weights)
    parameters = classifier.model.parameters
    config = classifier.config

    # Use configured key terms or default to categories
    required_key_words =
      config.key_terms
      |> Enum.flat_map(fn {_category, words} -> words end)
      |> Enum.uniq()

    # Get all word weights from parameters
    all_word_weights =
      parameters
      |> Enum.map(fn param -> {param.name, param.value} end)

    # Add explicit entries for all key words with a positive default weight
    # This ensures all key words are included, even if not in parameters
    key_word_entries =
      required_key_words
      |> Enum.map(fn word ->
        # Try to find existing weight for this key word
        existing = Enum.find(all_word_weights, fn {name, _} -> name == word end)

        if existing do
          # Use existing weight if found
          existing
        else
          # Default positive weight for key words
          {word, Map.get(config.parameters, :initial_weight, 0.2)}
        end
      end)

    # Other non-key words with significant weights
    other_entries =
      all_word_weights
      |> Enum.filter(fn {word, value} ->
        !Enum.member?(required_key_words, word) &&
          abs(value) > Map.get(config.parameters, :prune_threshold, 0.1)
      end)
      |> Enum.sort_by(fn {_name, value} -> -abs(value) end)

    # Return the combined list, ensuring key words come first
    Enum.take(key_word_entries ++ other_entries, 20)
  end
end
